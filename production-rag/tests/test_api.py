import pytest
from fastapi.testclient import TestClient

from production_rag.api import create_app
from production_rag.generation import GenerationError


def test_auth_input_limits_and_chat(service, cfg):
    with TestClient(create_app(cfg, service)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.post("/v1/chat", json={"question": "annual leave"}).status_code == 401
        headers = {"X-API-Key": cfg.api_key}
        assert client.get("/health/ready", headers=headers).status_code == 200
        response = client.post("/v1/chat", headers=headers, json={"question": "annual leave"})
        assert response.status_code == 200
        assert "20" in response.json()["answer"]
        assert len(response.headers["X-Request-ID"]) == 32
        assert client.post("/v1/chat", headers=headers, json={"question": "   "}).status_code == 422
        assert (
            client.post("/v1/chat", headers=headers, json={"question": "x" * 20_000}).status_code
            == 413
        )
        assert "rag_queries_total 1" in client.get("/metrics", headers=headers).text


def test_rate_limit(service, cfg):
    cfg.requests_per_minute = 1
    with TestClient(create_app(cfg, service)) as client:
        headers = {"X-API-Key": cfg.api_key}
        assert (
            client.post("/v1/chat", headers=headers, json={"question": "annual leave"}).status_code
            == 200
        )
        response = client.post("/v1/chat", headers=headers, json={"question": "annual leave"})
        assert response.status_code == 429 and response.headers["Retry-After"] == "60"


def test_no_secret_refuses_to_start(cfg):
    cfg.api_key = ""
    with pytest.raises(RuntimeError, match="RAG_API_KEY"), TestClient(create_app(cfg)):
        pass


def test_provider_failure_returns_502_and_releases_slot(service, cfg, monkeypatch):
    def unavailable(*args):
        raise GenerationError("Model unavailable or timed out")

    monkeypatch.setattr(service, "ask", unavailable)
    cfg.concurrent_queries = 1
    with TestClient(create_app(cfg, service)) as client:
        for _ in range(2):
            response = client.post(
                "/v1/chat", headers={"X-API-Key": cfg.api_key}, json={"question": "annual leave"}
            )
            assert response.status_code == 502


def test_unready_empty_index(cfg):
    with TestClient(create_app(cfg)) as client:
        assert client.get("/health/ready", headers={"X-API-Key": cfg.api_key}).status_code == 503
