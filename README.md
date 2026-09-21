# Jev Benchmarks

Reproducible, same-split comparisons of Jev 1.13.0 zero-shot inference against
frozen supervised checkpoints. Jev runs through the TypeSafe API; supervised
models run locally. This repository evaluates existing models, it does not train
or fine-tune them.

## Results

All scores use a 0–100 scale. Δ is Jev minus the supervised baseline.

| Benchmark | Split (rows)           | Supervised checkpoint              | Metric   |                                                            Baseline |                                                 Jev |      Δ |
| --------- | ---------------------- | ---------------------------------- | -------- | ------------------------------------------------------------------: | --------------------------------------------------: | -----: |
| BANKING77 | test (3,080)           | SPACE-2 `state_epoch_51`           | Accuracy |              [94.77](results/banking77/space-2/state_epoch_51.json) | [79.90](results/banking77/typesafe/jev-1.13.0.json) | −14.87 |
| CLINC150  | test, in-scope (4,500) | SPACE-2 `state_epoch_27`           | Accuracy |               [97.80](results/clinc150/space-2/state_epoch_27.json) |  [91.96](results/clinc150/typesafe/jev-1.13.0.json) |  −5.84 |
| HWU64     | test (1,076)           | SPACE-2 `state_epoch_25`           | Accuracy |                  [94.24](results/hwu64/space-2/state_epoch_25.json) |     [83.09](results/hwu64/typesafe/jev-1.13.0.json) | −11.15 |
| SST-2     | validation (872)       | `philschmid/roberta-large-sst2`    | Accuracy |           [96.44](results/sst2/huggingface/roberta-large-sst2.json) |      [94.50](results/sst2/typesafe/jev-1.13.0.json) |  −1.95 |
| STS-B     | test (1,379)           | `cross-encoder/stsb-roberta-large` | Spearman | [91.44](results/stsb/sentence-transformers/stsb-roberta-large.json) |      [89.12](results/stsb/typesafe/jev-1.13.0.json) |  −2.32 |

See [EVALUATION.md](EVALUATION.md) for definitions, provenance requirements,
and limitations.

## Quick start

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run python -m benchmarks.sst2 --provider huggingface --limit 5
```

The first local run downloads its pinned dataset and model. To run Jev, set
`TYPESAFE_API_KEY` in `.env` and use the `typesafe` provider; these calls incur
API usage:

```bash
uv run python -m benchmarks.sst2 --provider typesafe --limit 5
```

## Running benchmarks

Use `uv run python -m MODULE --provider PROVIDER --limit 5` with a pair below:

| Benchmark | Module                 | Local provider          | Jev provider |
| --------- | ---------------------- | ----------------------- | ------------ |
| BANKING77 | `benchmarks.banking77` | `space-2`               | `typesafe`   |
| CLINC150  | `benchmarks.clinc150`  | `space-2`               | `typesafe`   |
| HWU64     | `benchmarks.hwu64`     | `space-2`               | `typesafe`   |
| SST-2     | `benchmarks.sst2`      | `huggingface`           | `typesafe`   |
| STS-B     | `benchmarks.stsb`      | `sentence-transformers` | `typesafe`   |

Remove `--limit 5` for a complete run. To continue a limited or interrupted
run, rerun the same command with `--resume` and without `--limit`. TypeSafe runs
accept `--concurrency N`; SPACE-2 requires the default concurrency of one. Use
`--output PATH` to override the default ignored `runs/` directory.

### SPACE-2 setup

The three intent benchmarks require the pinned SPACE-2 release files.

<details>
<summary>Download and extract them</summary>

```bash
mkdir -p .cache/space2
curl -L 'https://drive.usercontent.google.com/download?id=10QEEMNsjO5rH0ZRsJBj9zkDc5ozxc3Ch&export=download&confirm=t' -o .cache/space2/outputs.zip
curl -L 'https://drive.usercontent.google.com/download?id=1ocwnuOLxB3VzngeWZsm59IRrhEv22Scx&export=download&confirm=t' -o .cache/space2/data.zip
unzip .cache/space2/outputs.zip 'outputs/banking/*' 'outputs/clinc/*' 'outputs/hwu/*' -d .cache/space2
unzip .cache/space2/data.zip 'data/pre_train/AnPreDial/single_turn/banking/test.json' 'data/pre_train/AnPreDial/single_turn/clinc/test.json' 'data/pre_train/AnPreDial/single_turn/hwu/test.json' -d .cache/space2
curl -L 'https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt' -o .cache/space2/vocab.txt
```

</details>

Before inference, each entry point verifies file digests, dataset alignment,
and three predictions against the authors' saved output. CLINC150 excludes OOS
rows because its released checkpoint has no OOS output. HWU64 reports raw
checkpoint top-1 accuracy; the authors' postprocessed 94.33% is retained only
as release context in [`manifests/space2-hwu64.json`](manifests/space2-hwu64.json).

## Repository layout

- `benchmarks/`: benchmark entry points and shared evaluation code
- `manifests/`: pinned SPACE-2 release provenance and digests
- `results/`: committed aggregate results
- `runs/`: ignored raw predictions and resumable run state
