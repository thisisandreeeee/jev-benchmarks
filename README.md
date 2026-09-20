# Jev Benchmarks

Reproducible evaluations of Jev and Jev-like classifiers without training them.

## Results

Direct same-split comparisons recomputed by this repository:

| Benchmark | Supervised checkpoint              | Metric   | Baseline | Jev zero-shot |
| --------- | ---------------------------------- | -------- | -------: | ------------: |
| BANKING77 | SPACE-2 `state_epoch_51`           | Accuracy |    94.77 |         79.90 |
| SST-2     | `philschmid/roberta-large-sst2`    | Accuracy |    96.44 |         94.50 |
| STS-B     | `cross-encoder/stsb-roberta-large` | Spearman |    91.44 |         89.12 |

## Usage

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

Setup the environment:

```bash
uv sync
uv run pytest
```

Run a small sample from each benchmark:

```bash
uv run python -m benchmarks.banking77 --provider typesafe --limit 5
uv run python -m benchmarks.banking77 --provider space-2 --limit 5
uv run python -m benchmarks.sst2 --provider typesafe --limit 5
uv run python -m benchmarks.sst2 --provider huggingface --limit 5
uv run python -m benchmarks.stsb --provider sentence-transformers --limit 5
uv run python -m benchmarks.stsb --provider typesafe --limit 5
```

Remove `--limit 5` to start a complete benchmark, or replace it with `--resume`
to finish a sample run in the same output directory. Commands using
`--provider typesafe` require valid TypeSafe credentials in `.env` and incur
API usage. The supervised commands run locally.

### SPACE-2 release setup

Download the pinned author archives and extract only the BANKING77 files:

```bash
mkdir -p .cache/space2
curl -L 'https://drive.usercontent.google.com/download?id=10QEEMNsjO5rH0ZRsJBj9zkDc5ozxc3Ch&export=download&confirm=t' -o .cache/space2/outputs.zip
curl -L 'https://drive.usercontent.google.com/download?id=1ocwnuOLxB3VzngeWZsm59IRrhEv22Scx&export=download&confirm=t' -o .cache/space2/data.zip
unzip .cache/space2/outputs.zip 'outputs/banking/*' -d .cache/space2
unzip .cache/space2/data.zip 'data/pre_train/AnPreDial/single_turn/banking/test.json' -d .cache/space2
curl -L 'https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt' -o .cache/space2/vocab.txt
```

The BANKING77 entry point verifies every archive and extracted-file digest
against `manifests/space2-banking77.json`, proves author/canonical row
alignment, and checks three local checkpoint predictions against the authors'
saved output before running.

The consolidated `benchmarks.stsb` entry point evaluates the original
1,379-row STS Benchmark test split. Select `sentence-transformers` for the
supervised baseline or `typesafe` for Jev; both providers use the same rows.

Use `--concurrency N` to run TypeSafe inference requests in parallel. SPACE-2
uses internal batching and requires the default concurrency of one.
Use `--output PATH` to write raw run artifacts somewhere other than the
default `runs/` directory.

## Backlog

- Analyse jev calibration (NLL, ECE)
- Remove PLAN.md and EVALUATION.md
