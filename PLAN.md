# Jev Benchmarks Implementation Plan

## Objective

Build a reproducible evaluation harness for Jev and Jev-like classifiers. The
harness evaluates existing model artifacts; it is not a model training or
optimization pipeline.

The first milestone implements BANKING77 against the first-party TypeSafe Jev
API. SST-2, STS-B, and local Jev-like providers are explicitly deferred.

## Agreed evaluation policy

- Evaluate `jev-1.13.0`, not a moving model alias.
- Use the first-party TypeSafe Python SDK and `TYPESAFE_API_KEY`.
- Treat the run as zero-shot: do not train, fine-tune, or include labeled
  examples in the prompt.
- Write and freeze the task instruction without inspecting evaluation answers.
- Give the model the original BANKING77 label names without added label
  descriptions.
- Use the canonical BANKING77 test split for inference.
- Report top-1 accuracy as the headline metric.
- Retain mean confidence, negative log loss, and expected calibration error in
  run metadata for possible later analysis.
- Keep per-example run artifacts out of Git. Commit aggregate results only
  after a complete run.

The frozen BANKING77 instruction is:

> Classify this banking customer request by choosing the most appropriate
> BANKING77 intent label.

## Definition of a Jev-like model

A Jev-like model receives state `S`, question `Q`, and candidate answers
`A1...An`, then assigns probability mass to those candidates. Choice, Noul,
and Score are inference-layer adaptations of that common operation. The model
is intended for automation rather than open-ended assistance.

The first future Jev-like provider should be
[`daseinlabs/open-jev`](https://github.com/daseinlabs/open-jev). Its initial
Apple-silicon limitation is acceptable. Do not implement it in this milestone,
but avoid coupling the benchmark to TypeSafe response objects.

## Minimal repository structure

```text
benchmarks/
  __init__.py
  banking77.py
  providers.py
tests/
  test_banking77.py
  test_providers.py
README.md
PLAN.md
pyproject.toml
uv.lock
.gitignore
```

Do not add a larger package hierarchy, configuration framework, model training
code, or remote-inference infrastructure until a concrete benchmark or
provider requires it.

## File responsibilities

### `benchmarks/providers.py`

Define the provider-independent inference boundary:

- Tagged Choice, Score, and Noul question types.
- Normalized result types for each question type.
- A minimal provider interface accepting state plus a typed question.
- A TypeSafe adapter using the official SDK and pinned Jev model.
- Conversion from TypeSafe response objects into normalized results.
- Separation of normalized results from provider-specific metadata.

Normalized result shapes:

```json
{"type":"choice","choice":"label","probabilities":{"label":0.8}}
{"type":"score","score":1.7,"probabilities":{"0":0.1,"1":0.2,"2":0.7},"legend":{"0":"low","1":"medium","2":"high"}}
{"type":"noul","noul":0.9}
```

Provider-specific confidence, usage, cost, latency, and raw response details
belong in a separate metadata object. Do not impose TypeSafe's definition of
confidence on other providers.

Validate provider responses at this boundary. In particular, candidates must
match the submitted answer space and probabilities must be finite and form a
valid distribution within a reasonable floating-point tolerance.

### `benchmarks/banking77.py`

Own all BANKING77-specific behavior:

- CLI argument parsing.
- Pinned Hugging Face dataset ID and revision.
- Loading the canonical test split through `datasets`.
- The frozen instruction and original label set.
- Creating one Choice question per utterance.
- Calling the selected provider.
- Incrementally writing run artifacts.
- Progress reporting, bounded retries, and concurrency.
- Resume validation and completed-ID skipping.
- Top-1 accuracy and secondary run-level diagnostics.
- Automatic publication of a complete aggregate result.

It must not contain TypeSafe SDK calls or depend on TypeSafe response classes.
Hugging Face's normal local dataset cache is sufficient; do not build another
cache.

## Dependencies

- Python 3.11 or newer.
- `uv` for environment and lock-file management.
- Hugging Face `datasets` for BANKING77 loading and caching.
- The official TypeSafe Python SDK for inference.
- Prefer the Python standard library for the CLI, JSONL handling, metrics, and
  tests unless a dependency already required by the project directly provides
  the needed behavior.

Pin dependency resolution in `uv.lock`. Pin the BANKING77 dataset to a concrete
Hugging Face revision in code rather than following `main`.

Dataset source:
[`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77). It has
10,003 training examples, 3,080 test examples, and 77 intents. Do not load or
use the training split in the benchmark.

## CLI contract

The default provider, model, and run directory are inferred. A five-example
live smoke run should require only:

```bash
TYPESAFE_API_KEY=... uv run python -m benchmarks.banking77 --limit 5
```

Defaults:

```text
provider: typesafe
model: jev-1.13.0
output: runs/banking77/typesafe/jev-1.13.0/
concurrency: 1
```

Supported options:

- `--limit N`: evaluate the first `N` examples in pinned dataset order.
- `--resume`: validate the existing run identity and skip completed IDs.
- `--concurrency N`: allow parallel API calls within account limits.
- `--output PATH`: override the inferred run directory for exploratory runs.

`--limit` is not part of the run identity. These commands must let a user
continue the smoke run into a complete run:

```bash
uv run python -m benchmarks.banking77 --limit 5
uv run python -m benchmarks.banking77 --resume
```

If a run directory already exists and `--resume` was not supplied, fail with a
clear instruction instead of overwriting it.

## Run artifacts

Ignore `runs/` in Git. Each run directory contains exactly:

```text
run.json
predictions.jsonl
```

### `run.json`

Combine immutable metadata and mutable summary information in one document.
Include at least:

- Schema version.
- Benchmark name.
- Dataset ID, pinned revision, and split.
- Exact instruction and complete candidate set.
- Provider and pinned model.
- Relevant SDK and dependency versions.
- Repository revision when available.
- Start and update timestamps.
- `partial`, `complete`, or `failed` status.
- Evaluated and total example counts.
- Accuracy and secondary diagnostics.
- Aggregate provider usage and cost when available.

Keep an explicit immutable identity section. Resume must compare that identity
against the requested run before appending anything. Update `run.json`
atomically so interruption cannot leave malformed JSON.

### `predictions.jsonl`

Flush one record after every completed example. Each record includes:

- Dataset example ID.
- Exact input text sent as state.
- Complete normalized question.
- Expected label.
- Normalized result.
- Provider-specific metadata, including latency and available usage fields.

Example:

```json
{
  "dataset_id": 42,
  "state": "Why was I charged for withdrawing cash?",
  "question": {
    "type": "choice",
    "instructions": "Classify this banking customer request by choosing the most appropriate BANKING77 intent label.",
    "criteria": ["activate_my_card", "cash_withdrawal_charge"]
  },
  "expected": "cash_withdrawal_charge",
  "result": {
    "type": "choice",
    "choice": "cash_withdrawal_charge",
    "probabilities": {
      "activate_my_card": 0.01,
      "cash_withdrawal_charge": 0.81
    }
  },
  "provider_metadata": {
    "provider": "typesafe",
    "model": "jev-1.13.0",
    "confidence": 0.73,
    "latency_ms": 184,
    "usage": {}
  }
}
```

The real record must contain all 77 candidates and their returned
probabilities; the shortened example above is illustrative only.

## Resume, retries, and failure behavior

- Read existing prediction IDs when resuming and never submit them again.
- Validate schema version, dataset revision, split, instruction, candidates,
  provider, and model before resuming.
- Permit a run started with `--limit 5` to continue without a limit.
- Retry rate limits and transient server errors with bounded backoff, honoring
  `Retry-After` when the SDK exposes it.
- After retries are exhausted, retain all completed records, mark the run
  failed or partial, and exit nonzero.
- Do not publish a result unless every canonical test example succeeded
  exactly once.

## Metrics

The headline metric is top-1 accuracy:

```text
correct predictions / evaluated examples
```

Also compute and store at run level:

- Mean predicted confidence, defined as the mean of the largest normalized
  candidate probability for each example.
- Multiclass negative log loss from the probability assigned to the gold
  label, clipping probabilities to at least `1e-15` before taking the
  logarithm.
- Expected calibration error using 10 equal-width bins over `[0, 1]`, weighted
  by the number of examples in each non-empty bin.
- Mean provider-native confidence as a separate provider statistic when the
  provider returns one.

A limited run may contain these metrics, but must be marked incomplete with
`evaluated` and `total` counts. Partial metrics are never eligible for the
README results table or automatic publication.

## Automatic result publication

After all 3,080 examples succeed, atomically write the aggregate result to an
inferred path:

```text
results/banking77/typesafe/jev-1.13.0.json
```

The published file contains aggregate provenance, counts, metrics, and usage,
but not per-example predictions.

If the path already exists:

- Do nothing when the new content is identical.
- Refuse to overwrite it when the content differs, and report the conflict.

Do not require a publication flag. Do not modify the README automatically.

Future providers follow the same convention:

```text
results/<benchmark>/<provider>/<model>.json
```

Sanitize provider and model identifiers before using them as path components.

## README requirements

Use this structure:

```markdown
# Jev Benchmarks

One sentence explaining that this repository reproducibly evaluates Jev and
Jev-like classifiers without training them.

## Results

| Benchmark | Metric | BERT-Base | Jev | Jev-like |
| --- | --- | ---: | ---: | ---: |
| BANKING77 | Accuracy | 93.02 | — | — |
| SST-2 | Accuracy | 93.5 | — | — |
| STS-B | Spearman | 85.8 | — | — |

## Usage

Setup, tests, the five-example BANKING77 smoke run, resume, and complete-run
instructions.
```

Explain that the BERT figures are published supervised references, not
zero-shot results. The SST-2 and STS-B figures are hidden GLUE test results;
future locally reproducible Jev runs will use their public validation splits
and therefore must be labeled clearly rather than presented as perfectly
matched comparisons.

Sources:

- BANKING77 BERT reference, 93.02 accuracy:
  [SPACE-2, Table 2](https://aclanthology.org/2022.coling-1.46.pdf).
- SST-2 and STS-B BERT-base references, 93.5 accuracy and 85.8 Spearman:
  [BERT, Table 1](https://arxiv.org/pdf/1810.04805).

Mark BANKING77 as implemented and SST-2/STS-B as planned in the usage text.

## Tests and verification

Use offline tests with a fake provider. CI must never call a paid or
authenticated service.

Cover at least:

- Choice, Score, and Noul normalization.
- Invalid provider probability distributions.
- TypeSafe response conversion using fixtures or constructed SDK objects.
- Accuracy, negative log loss, and expected calibration error.
- Incremental prediction writing.
- Atomic `run.json` updates.
- Identity validation.
- Resume without duplicate submissions.
- Expansion from a limited run to a complete run.
- Partial and failed run handling.
- Publication only after complete success.
- Identical publication no-op and conflicting publication refusal.

Use the documented `--limit 5` command as the manual live end-to-end check.
The milestone is complete without performing the full paid run.

## Acceptance criteria

The first milestone is complete when:

1. `uv sync` creates the locked Python 3.11+ environment.
2. The offline test command documented in the README passes without an API
   key or network inference.
3. With a valid `TYPESAFE_API_KEY`, `--limit 5` evaluates five canonical test
   examples and creates valid ignored run artifacts.
4. Re-running with `--resume --limit 5` makes no duplicate API calls.
5. Re-running with `--resume` is capable of continuing toward all 3,080 test
   examples.
6. A partial run does not create a committed-result candidate under
   `results/`.
7. A simulated complete run in tests automatically creates the correctly
   named aggregate result.
8. README provenance, reference metrics, scope, and commands match the actual
   implementation.

## Explicit non-goals for this milestone

- Executing the full 3,080-call paid benchmark.
- SST-2 or STS-B benchmark code.
- OpenJev or another local-model adapter.
- Training, fine-tuning, prompt optimization, or hyperparameter search.
- Modal or other remote-inference infrastructure.
- A generic plugin/configuration framework.
- Live API calls in automated tests or CI.
- Automatic README editing after a result is published.
