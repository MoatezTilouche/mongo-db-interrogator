"""Embedding comparison CLI. See docs/embedding-benchmark.md; no live defaults."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.embedding_benchmark.data import prepare, read_json, starter, validate, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("starter", help="Expand synthetic multilingual data, offline")
    demo.add_argument("--seed", default=str(ROOT / "evaluation/embedding_benchmark_seed.json"))
    demo.add_argument("--out", required=True)
    demo.add_argument("--corpus-language", choices=("en", "fr", "ar", "aeb", "de", "es"), default="en")
    prep = commands.add_parser("prepare", help="Combine frozen local exports with reviewed labels, offline")
    prep.add_argument("--snapshot", required=True, help="Generation snapshot: catalog/profiles/rules")
    prep.add_argument("--templates", required=True, help="Decoded templates array, or {templates:[...]}")
    prep.add_argument("--labels", required=True)
    prep.add_argument("--out", required=True)
    check = commands.add_parser("check", help="Validate IDs, splits, gold labels and model arms, offline")
    live = commands.add_parser("run", help="Only command that contacts embedding servers")
    for item in (check, live):
        item.add_argument("--dataset", required=True)
        item.add_argument("--config", default=str(ROOT / "evaluation/embedding_benchmark_config.json"))
    live.add_argument("--base-url", default="http://localhost:11434/v1")
    live.add_argument("--protocol", choices=("ollama", "openai"), default="ollama")
    live.add_argument("--timeout", type=float, default=120)
    live.add_argument("--output-dir", required=True)
    live.add_argument("--resume", action="store_true")
    live.add_argument("--max-cases", type=int, help="Smoke only; never use its scores to pick a winner")
    args = parser.parse_args(argv)
    try:
        if args.command in {"starter", "prepare"}:
            if Path(args.out).exists():
                raise ValueError("Output exists; choose a new filename")
            dataset = (starter(read_json(args.seed), args.corpus_language) if args.command == "starter" else
                       prepare(read_json(args.snapshot), read_json(args.labels), read_json(args.templates)))
            write_json(args.out, dataset)
            print(f"Wrote {len(dataset['cases'])} cases to {args.out}")
        else:
            dataset, config = read_json(args.dataset), read_json(args.config)
            snapshot, documents = validate(dataset, config)
            if args.command == "check":
                print(json.dumps({"valid": True, "cases": len(dataset["cases"]), "documents": len(documents),
                                  "models": [arm["id"] for arm in config["arms"]]}, indent=2))
            else:
                from scripts.embedding_benchmark.provider import EmbeddingProvider
                from scripts.embedding_benchmark.runner import run
                provider = EmbeddingProvider(args.base_url, args.protocol, args.timeout)
                result = run(dataset, config, provider, args.output_dir, resume=args.resume, max_cases=args.max_cases)
                print(f"{result['status']}: {Path(args.output_dir) / 'report.json'}")
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        print(f"Benchmark stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
