"""Run STS-B original-test evaluations with SBERT or TypeSafe Jev."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from benchmarks.metrics import similarity_metrics
from benchmarks.providers import (
    Provider,
    ScalarScoreResult,
    Score,
    ScoreResult,
    SentenceTransformerScoreProvider,
    TypeSafeProvider,
)
from benchmarks.runner import run_benchmark

SCHEMA_VERSION = 1
BENCHMARK = "stsb"
DATASET_ID = "mteb/stsbenchmark-sts"
DATASET_CONFIG = "default"
DATASET_REVISION = "96943a16ea6a35129e253c659081cb59daf81b30"
DATASET_SPLIT = "test"
EXPECTED_ROWS = 1_379
ROW_DIGEST = "3e30964a29dd599b3c30fe108303e6858340a9fd82a4ac2a1da33757411c4060"
MODEL = "sentence-transformers/stsb-bert-base"
MODEL_REVISION = "73822fe64c1f91818b34e4b5d23fca7092f6dcc8"
MODEL_CARD = f"https://huggingface.co/{MODEL}"
PAPER = "https://aclanthology.org/D19-1410/"
JEV_MODEL = "jev-1.13.0"
ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Rate the semantic similarity of the two sentences using the rubric."
RUBRIC = (
    "The sentences are completely dissimilar.",
    "The sentences are on the same topic but are not equivalent.",
    "The sentences share some details but are not equivalent.",
    "The sentences are roughly equivalent, but important information differs.",
    "The sentences are mostly equivalent, with only minor differences.",
    "The sentences are completely equivalent in meaning.",
)


def row_digest(dataset: Any) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        value = [row["sentence1"], row["sentence2"], f"{float(row['score']):.10f}"]
        digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_dataset(dataset: Any) -> None:
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} STS-B test rows, got {len(dataset)}")
    actual = row_digest(dataset)
    if actual != ROW_DIGEST:
        raise ValueError(f"STS-B test row digest mismatch: {actual}")


def dataset_identity() -> dict[str, Any]:
    return {
        "id": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "rows": EXPECTED_ROWS,
        "row_digest": ROW_DIGEST,
        "score_scale": "0..5",
    }


def identity(provider: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "dataset": dataset_identity(),
        "instruction": INSTRUCTION,
        "rubric": list(RUBRIC),
        "provider": provider,
    }
    if provider == "sentence-transformers":
        value.update(
            {
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "model_card": MODEL_CARD,
                "model_card_status": "deprecated",
                "paper": PAPER,
                "score": "cosine_similarity",
            }
        )
    elif provider == "typesafe":
        value["model"] = JEV_MODEL
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return value


def _record(row_id: int, row: dict[str, Any], question: Score, inference: Any) -> dict[str, Any]:
    if not isinstance(inference.result, (ScoreResult, ScalarScoreResult)):
        raise ValueError("provider returned a non-score result")
    return {
        "dataset_id": row_id,
        "state": f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
        "question": question.as_dict(),
        "expected": float(row["score"]),
        "result": inference.result.as_dict(),
        "provider_metadata": inference.metadata,
    }


def evaluate_sentence_transformer(provider: Any, row_id: int, row: dict[str, Any]) -> dict[str, Any]:
    question = Score(INSTRUCTION, RUBRIC)
    return _record(row_id, row, question, provider.infer_pair(row["sentence1"], row["sentence2"]))


def evaluate_typesafe(provider: Provider, row_id: int, row: dict[str, Any]) -> dict[str, Any]:
    question = Score(INSTRUCTION, RUBRIC)
    state = f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}"
    return _record(row_id, row, question, provider.infer(state, question))


def run(
    dataset: Any,
    provider: Any,
    provider_name: str,
    output: Path,
    *,
    limit: int | None,
    resume: bool,
    concurrency: int,
) -> dict[str, Any]:
    validate_dataset(dataset)
    evaluate: Callable[..., dict[str, Any]]
    if provider_name == "sentence-transformers":
        evaluate = evaluate_sentence_transformer
        slug = "stsb-bert-base"
        dependencies = {
            "datasets": version("datasets"),
            "sentence-transformers": version("sentence-transformers"),
            "torch": version("torch"),
            "transformers": version("transformers"),
        }
    elif provider_name == "typesafe":
        evaluate = evaluate_typesafe
        slug = JEV_MODEL
        dependencies = {"datasets": version("datasets"), "typesafe-sdk": version("typesafe-sdk")}
    else:
        raise ValueError(f"unsupported provider: {provider_name}")
    return run_benchmark(
        dataset,
        provider,
        output,
        identity=identity(provider_name),
        evaluate=evaluate,
        metrics=similarity_metrics,
        result_path=ROOT / "results" / BENCHMARK / provider_name / f"{slug}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
        dependencies=dependencies,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("sentence-transformers", "typesafe"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.output is None:
        slug = "stsb-bert-base" if args.provider == "sentence-transformers" else JEV_MODEL
        args.output = ROOT / "runs" / BENCHMARK / args.provider / slug
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION, split=DATASET_SPLIT)
    if args.provider == "sentence-transformers":
        provider: Any = SentenceTransformerScoreProvider(MODEL, MODEL_REVISION)
    else:
        load_dotenv(ROOT / ".env")
        provider = TypeSafeProvider(JEV_MODEL)
    try:
        result = run(
            dataset,
            provider,
            args.provider,
            args.output,
            limit=args.limit,
            resume=args.resume,
            concurrency=args.concurrency,
        )
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            close()
    print(
        json.dumps(
            {"status": result["status"], "evaluated": result["evaluated"], "total": result["total"], **result["metrics"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
