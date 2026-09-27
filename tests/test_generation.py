import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from production_rag.generation import (
    GenerationError,
    TransformersGenerator,
    generate,
    validate_answer,
)

EVIDENCE = [{"citation": 1, "text": "Annual leave is 20 days.", "source": "policy.md", "page": 1}]


@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "20 days [7]", "citations": [7], "abstained": False},
        {"answer": "20 days", "citations": [1], "abstained": False},
        {"answer": "20 days [1]", "citations": [], "abstained": False},
        {"answer": "20 days [1]", "citations": [True], "abstained": False},
        {"answer": "20 days [1]", "citations": [1], "abstained": False, "extra": "bad"},
    ],
)
def test_invalid_citations_are_rejected(payload):
    with pytest.raises(ValueError):
        validate_answer(payload, EVIDENCE)


def test_abstention_discards_model_claims():
    result = validate_answer(
        {"answer": "unsupported claim", "citations": [8], "abstained": True}, EVIDENCE
    )
    assert result.abstained and result.citations == []
    assert "unsupported claim" not in result.answer


def test_python_generator_contract_and_untrusted_evidence(cfg):
    cfg.generator = "transformers"
    backend = Mock()
    backend.complete.return_value = json.dumps(
        {"answer": "20 days [1]", "citations": [1], "abstained": False}
    )
    result = generate("How much leave?", EVIDENCE, cfg, backend)
    assert result.answer == "20 days [1]"
    messages = backend.complete.call_args.args[0]
    assert "untrusted" in messages[0]["content"]
    assert json.loads(messages[1]["content"])["evidence"] == EVIDENCE


@pytest.mark.parametrize(
    "output", ["not json", "{}", '{"answer":"20","citations":[],"abstained":false}']
)
def test_malformed_generation_fails_closed(cfg, output):
    cfg.generator = "transformers"
    backend = Mock()
    backend.complete.return_value = output
    with pytest.raises(GenerationError, match="invalid or uncited"):
        generate("How much leave?", EVIDENCE, cfg, backend)


def test_no_evidence_never_loads_model(cfg):
    cfg.generator = "transformers"
    backend = Mock()
    assert generate("quantum chromodynamics", [], cfg, backend).abstained
    backend.complete.assert_not_called()


def test_python_generation_error_propagates(cfg):
    cfg.generator = "transformers"
    backend = Mock()
    backend.complete.side_effect = GenerationError("Model could not load")
    with pytest.raises(GenerationError, match="could not load"):
        generate("How much leave?", EVIDENCE, cfg, backend)


def fake_runtime(cfg, input_length=20, output_ids=None):
    adapter = TransformersGenerator(cfg)
    tensor = Mock()
    tensor.shape = (1, input_length)
    tensor.to.return_value = tensor
    adapter.tokenizer = Mock(eos_token_id=2)
    adapter.tokenizer.apply_chat_template.return_value = "formatted prompt"
    adapter.tokenizer.return_value = {"input_ids": tensor, "attention_mask": tensor}
    adapter.tokenizer.decode.return_value = (
        '{"answer":"20 days [1]","citations":[1],"abstained":false}'
    )

    class Output:
        def __getitem__(self, index):
            assert index == (0, slice(input_length, None))
            return SimpleNamespace(tolist=lambda: output_ids if output_ids is not None else [42, 2])

    adapter.model = Mock()
    adapter.model.config = SimpleNamespace(max_position_embeddings=4096)
    adapter.model.generation_config = SimpleNamespace(eos_token_id=[2, 3])
    adapter.model.generate.return_value = Output()
    adapter.torch = SimpleNamespace(inference_mode=nullcontext)
    return adapter


def test_adapter_decodes_only_generated_tokens(cfg):
    adapter = fake_runtime(cfg)
    assert "20 days" in adapter.complete([{"role": "user", "content": "hello"}])
    adapter.tokenizer.decode.assert_called_once_with([42, 2], skip_special_tokens=True)
    kwargs = adapter.model.generate.call_args.kwargs
    assert kwargs["do_sample"] is False
    assert kwargs["max_time"] == cfg.generation_seconds
    assert kwargs["max_new_tokens"] == cfg.max_new_tokens


def test_oversized_prompt_rejected_before_generation(cfg):
    adapter = fake_runtime(cfg, input_length=4000)
    with pytest.raises(GenerationError, match="exceeds model context"):
        adapter.complete([])
    adapter.model.generate.assert_not_called()


@pytest.mark.parametrize("ids", [[], [42, 43]])
def test_incomplete_model_generation_rejected(cfg, ids):
    adapter = fake_runtime(cfg, output_ids=ids)
    with pytest.raises(GenerationError, match="truncated"):
        adapter.complete([])


def test_python_model_loaded_once(cfg, monkeypatch):
    import sys

    torch = SimpleNamespace(float32="float32")
    tokenizer_class = Mock()
    model_class = Mock()
    model = model_class.from_pretrained.return_value.to.return_value
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=tokenizer_class, AutoModelForCausalLM=model_class),
    )
    adapter = TransformersGenerator(cfg)
    adapter.load()
    adapter.load()
    model_class.from_pretrained.assert_called_once()
    assert model_class.from_pretrained.call_args.kwargs["use_safetensors"] is True
    assert model_class.from_pretrained.call_args.kwargs["trust_remote_code"] is False
    model.eval.assert_called_once()
