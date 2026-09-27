import argparse
import importlib.util
import json
import secrets
import sys
from pathlib import Path

from .config import Settings
from .evaluation import evaluate
from .generation import GenerationError
from .service import RAGService


def main():
    parser = argparse.ArgumentParser(description="Local-first RAG application")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Index a file or recursively index a directory")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--source", help="Stable source ID; only for a single file")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, choices=range(1, 9), default=4)
    commands.add_parser("list")
    delete = commands.add_parser("delete")
    delete.add_argument("source")
    backup = commands.add_parser("backup")
    backup.add_argument("destination", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dataset", type=Path, default=Path("eval/questions.jsonl"))
    evaluation.add_argument("--output", type=Path)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--load-model", action="store_true", help="Download/load the local LLM")
    commands.add_parser("init-env", help="Create a private .env with a random API key")
    serve = commands.add_parser("serve", help="Start the Python API on localhost")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        if args.command == "init-env":
            template = Path(".env.example").read_text(encoding="utf-8")
            with Path(".env").open("x", encoding="utf-8") as out:
                out.write(
                    template.replace("RAG_API_KEY=", "RAG_API_KEY=" + secrets.token_urlsafe(32))
                )
            print(
                "Created .env with a private random API key. Existing files are never overwritten."
            )
            return 0
        if args.command == "serve":
            import uvicorn

            uvicorn.run(
                "production_rag.api:create_app",
                factory=True,
                host="127.0.0.1",
                port=args.port,
                workers=1,
                access_log=False,
            )
            return 0
        settings = Settings()
        service = RAGService(settings)
        if args.command == "ingest":
            if args.path.is_dir():
                if args.source:
                    raise ValueError("--source is for a single file only")
                files = sorted(
                    p
                    for p in args.path.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}
                )
                if not files:
                    raise ValueError("No supported documents found")
                result = [service.ingest(p, p.relative_to(args.path).as_posix()) for p in files]
            else:
                result = service.ingest(args.path, args.source)
        elif args.command == "ask":
            if not 3 <= len(args.question.strip()) <= 2000:
                raise ValueError("Question must contain 3–2000 characters")
            result = service.ask(args.question, args.top_k)
        elif args.command == "list":
            result = service.store.documents()
        elif args.command == "delete":
            result = {"deleted": service.store.delete(args.source)}
        elif args.command == "backup":
            service.store.backup(args.destination)
            result = {"backup": str(args.destination)}
        elif args.command == "evaluate":
            result = evaluate(service, args.dataset)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] == result["cases"] else 1
        else:
            result = {
                "retrieval": settings.retrieval,
                "generator": settings.generator,
                "documents": len(service.store.documents()),
                "api_key_configured": len(settings.api_key) >= 24,
            }
            if settings.generator == "transformers":
                result["llm_model"] = settings.llm_model
                result["llm_dependencies_installed"] = all(
                    importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
                )
                result["model_load_checked"] = False
                if args.load_model:
                    service.generator.load()
                    result["model_load_checked"] = True
        print(json.dumps(result, indent=2))
        return 0
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        GenerationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
