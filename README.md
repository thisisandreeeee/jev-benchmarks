# Jev Benchmarks

Reproducible, same-split comparisons of Jev 1.13.0 zero-shot inference against
frozen supervised checkpoints. Jev runs through the TypeSafe API; supervised
models run locally. This repository evaluates existing models, it does not train
or fine-tune them.

## Results

All scores use a 0–100 scale. Δ is Jev minus the supervised baseline. The
interval is a paired percentile 95% bootstrap over the evaluation rows; each Δ
links to its comparison artifact.

| Benchmark | Rows  | Supervised checkpoint              | Metric   |                                                            Baseline |                                                 Jev |                                                                       Δ | Δ 95% CI         |
| --------- | ----- | ---------------------------------- | -------- | -------------------------------------------------------------------: | ---------------------------------------------------: | -----------------------------------------------------------------------: | ---------------- |
| BANKING77 | 3,080 | SPACE-2 `state_epoch_51`           | Accuracy |              [94.77](results/banking77/space-2/state_epoch_51.json) | [79.90](results/banking77/typesafe/jev-1.13.0.json) | [−14.87](results/banking77/comparisons/jev-1.13.0--state_epoch_51.json) | [−16.27, −13.47] |
| CLINC150  | 4,500 | SPACE-2 `state_epoch_27`           | Accuracy |               [97.80](results/clinc150/space-2/state_epoch_27.json) |  [91.96](results/clinc150/typesafe/jev-1.13.0.json) |   [−5.84](results/clinc150/comparisons/jev-1.13.0--state_epoch_27.json) | [−6.64, −5.07]   |
| HWU64     | 1,076 | SPACE-2 `state_epoch_25`           | Accuracy |                  [94.24](results/hwu64/space-2/state_epoch_25.json) |     [83.09](results/hwu64/typesafe/jev-1.13.0.json) |     [−11.15](results/hwu64/comparisons/jev-1.13.0--state_epoch_25.json) | [−13.38, −9.01]  |
| SST-2     | 872   | `philschmid/roberta-large-sst2`    | Accuracy |           [96.44](results/sst2/huggingface/roberta-large-sst2.json) |      [94.50](results/sst2/typesafe/jev-1.13.0.json) |   [−1.95](results/sst2/comparisons/jev-1.13.0--roberta-large-sst2.json) | [−3.44, −0.46]   |
| STS-B     | 1,379 | `cross-encoder/stsb-roberta-large` | Spearman | [91.44](results/stsb/sentence-transformers/stsb-roberta-large.json) |      [89.21](results/stsb/typesafe/jev-1.13.0.json) |   [−2.23](results/stsb/comparisons/jev-1.13.0--stsb-roberta-large.json) | [−3.36, −1.10]   |

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

### Statistical comparison

`benchmarks.compare` builds a deterministic paired-bootstrap artifact from two
completed run directories. It loads no model and contacts no network:

```bash
uv run python -m benchmarks.compare \
  --left runs/sst2/typesafe/jev-1.13.0 \
  --right runs/sst2/huggingface/roberta-large-sst2
```

Both runs must be complete and must score the same ordered rows. The command
verifies the shared dataset identity and row digest, recomputes both headline
metrics from raw predictions, reports `left - right` with a paired percentile
95% interval, and writes `results/<benchmark>/comparisons/`. Use `--samples`,
`--seed`, `--confidence`, and `--output` to override the defaults.

Committed artifacts for every benchmark live under
`results/<benchmark>/comparisons/`, and each Δ in the table above links to one.

## Limitations

- The compared systems differ in architecture, parameter count, pretraining
  data, compute, and release date.
- Public benchmark contamination in Jev cannot be excluded.
- SST-2 uses the public GLUE validation split, which was also the supervised
  checkpoint's development split; that row is a same-split reproduction rather
  than an untouched-test evaluation.
- SPACE-2 reports averages over multiple seeds, while its released files may
  contain selected checkpoints; only the recomputed checkpoint scores appear in
  the table.
- Jev is nondeterministic at the sub-tenth-point scale. A repeat of the STS-B
  evaluation under an identical identity shifted Spearman by 0.088 points
  (89.12 to 89.21) with identical token usage. The STS-B row and its comparison
  use the rerun; sub-0.1-point differences are not precise.
- Every locally recomputed run was made from a non-clean source worktree; paid
  legacy Jev runs predate the `repository_clean` field.

See [EVALUATION.md](EVALUATION.md) for the frozen protocol.

## Repository layout

- `benchmarks/`: benchmark entry points and shared evaluation code
- `manifests/`: pinned SPACE-2 release provenance and digests
- `results/`: committed aggregate results
- `runs/`: ignored raw predictions and resumable run state

## Backlog

- Add providers for Kev, Laya
- Add BART zero-shot provider
- Analyse calibration with ECE
