# Jev Benchmarks

Reproducible, same-split comparisons of Jev 1.13.0 zero-shot inference against
frozen supervised checkpoints. Jev runs through the TypeSafe API; supervised
models run locally. This repository evaluates existing models, it does not train
or fine-tune them.

## Results

Frozen supervised checkpoints outperform Jev on all five benchmarks. The
smallest gaps are 1.95 points on SST-2 and 2.23 points on STS-B; the largest
is 14.87 points on BANKING77.

Jev outperforms the local zero-shot NLI baseline by 12.76-26.77 points on all
three intent benchmarks.

On SST-2, Jev leads the zero-shot baseline by 1.49 points, but the paired 95%
interval includes zero. Every other comparison interval excludes zero.

| Benchmark | Rows  | Jev task | Metric   |                                                          Supervised |                                                 Jev |                                                          Zero-shot |                                                                                    Jev vs supervised |                                                                                    Jev vs zero-shot |
| --------- | ----- | -------- | -------- | ------------------------------------------------------------------: | --------------------------------------------------: | -----------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------: |
| BANKING77 | 3,080 | Choice   | Accuracy |              [94.77](results/banking77/space-2-state_epoch_51.json) | [79.90](results/banking77/typesafe-jev-1.13.0.json) | [67.14](results/banking77/nli-deberta-v3-large-zeroshot-v2.0.json) |             [−14.87](results/banking77/comparisons/typesafe-jev-1.13.0--space-2-state_epoch_51.json) | [12.76](results/banking77/comparisons/typesafe-jev-1.13.0--nli-deberta-v3-large-zeroshot-v2.0.json) |
| CLINC150  | 4,500 | Choice   | Accuracy |               [97.80](results/clinc150/space-2-state_epoch_27.json) |  [91.96](results/clinc150/typesafe-jev-1.13.0.json) |  [67.58](results/clinc150/nli-deberta-v3-large-zeroshot-v2.0.json) |               [−5.84](results/clinc150/comparisons/typesafe-jev-1.13.0--space-2-state_epoch_27.json) |  [24.38](results/clinc150/comparisons/typesafe-jev-1.13.0--nli-deberta-v3-large-zeroshot-v2.0.json) |
| HWU64     | 1,076 | Choice   | Accuracy |                  [94.24](results/hwu64/space-2-state_epoch_25.json) |     [83.09](results/hwu64/typesafe-jev-1.13.0.json) |     [56.32](results/hwu64/nli-deberta-v3-large-zeroshot-v2.0.json) |                 [−11.15](results/hwu64/comparisons/typesafe-jev-1.13.0--space-2-state_epoch_25.json) |     [26.77](results/hwu64/comparisons/typesafe-jev-1.13.0--nli-deberta-v3-large-zeroshot-v2.0.json) |
| SST-2     | 872   | Noul     | Accuracy |           [96.44](results/sst2/huggingface-roberta-large-sst2.json) |      [94.50](results/sst2/typesafe-jev-1.13.0.json) |      [93.00](results/sst2/nli-deberta-v3-large-zeroshot-v2.0.json) |           [−1.95](results/sst2/comparisons/typesafe-jev-1.13.0--huggingface-roberta-large-sst2.json) |       [1.49](results/sst2/comparisons/typesafe-jev-1.13.0--nli-deberta-v3-large-zeroshot-v2.0.json) |
| STS-B     | 1,379 | Score    | Spearman | [91.44](results/stsb/sentence-transformers-stsb-roberta-large.json) |      [89.21](results/stsb/typesafe-jev-1.13.0.json) |                                                                  — | [−2.23](results/stsb/comparisons/typesafe-jev-1.13.0--sentence-transformers-stsb-roberta-large.json) |                                                                                                   — |

- **BANKING77:** Classifies online-banking customer requests into 77 intents;
  the supervised checkpoint is SPACE-2 `state_epoch_51`.
- **CLINC150:** Classifies virtual-assistant requests into 150 in-scope intents;
  the supervised checkpoint is SPACE-2 `state_epoch_27`, and OOS rows are
  excluded because the released checkpoint has no OOS output.
- **HWU64:** Classifies home-assistant requests into 64 intents; the supervised
  checkpoint is SPACE-2 `state_epoch_25`, scored by raw checkpoint top-1 output.
- **SST-2:** Predicts positive or negative sentiment for movie-review sentences;
  the supervised checkpoint is `philschmid/roberta-large-sst2`.
- **STS-B:** Scores semantic similarity between sentence pairs; the supervised
  checkpoint is `cross-encoder/stsb-roberta-large`.

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

Remove `--limit 5` for a complete run. To continue a limited or interrupted
run, rerun the same command with `--resume` and without `--limit`. TypeSafe runs
accept `--concurrency N`; SPACE-2 and `nli` require the default concurrency of
one. Use `--output PATH` to override the default ignored `runs/` directory.

Published results are named
`results/<benchmark>/<provider>-<model>.json`; paired comparison artifacts live
under `results/<benchmark>/comparisons/`.

## Running benchmarks

### Choice

BANKING77, CLINC150, and HWU64 compare the local SPACE-2 checkpoint, Jev, and
the local zero-shot NLI baseline. Replace the module in these commands with
`benchmarks.banking77`, `benchmarks.clinc150`, or `benchmarks.hwu64`:

```bash
uv run python -m benchmarks.banking77 --provider space-2 --limit 5
uv run python -m benchmarks.banking77 --provider typesafe --limit 5
uv run python -m benchmarks.banking77 --provider nli --limit 5
```

SPACE-2 requires the release files described below. TypeSafe requires
`TYPESAFE_API_KEY`; the other providers run locally.

### Noul

SST-2 compares the local RoBERTa checkpoint, Jev's `Noul` output, and the local
zero-shot NLI baseline:

```bash
uv run python -m benchmarks.sst2 --provider huggingface --limit 5
uv run python -m benchmarks.sst2 --provider typesafe --limit 5
uv run python -m benchmarks.sst2 --provider nli --limit 5
```

### Score

STS-B compares the local SentenceTransformers cross-encoder with Jev's
six-level `Score` rubric. There is no NLI baseline for this task:

```bash
uv run python -m benchmarks.stsb --provider sentence-transformers --limit 5
uv run python -m benchmarks.stsb --provider typesafe --limit 5
```

## SPACE-2 setup

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

## Evaluation

Jev is evaluated zero-shot: it receives the frozen task instruction, output
rubric, and complete candidate-label set, but no benchmark examples or
parameter updates. The supervised systems are frozen, author-released
checkpoints trained on labeled benchmark data. Direct comparisons use the same
pinned, ordered rows and the same metric implementation; published paper
scores are context only.

Dataset revisions, splits, row counts and digests, model revisions or file
digests, prompts and label mappings, dependency versions, and repository state
are recorded in the result artifacts. Complete runs publish aggregate results;
per-example predictions remain under ignored `runs/` paths, and all five frozen
benchmarks are published regardless of outcome.

### Statistical comparison

`benchmarks.compare` builds a deterministic paired-bootstrap artifact from two
completed run directories. It loads no model and contacts no network:

```bash
uv run python -m benchmarks.compare \
  --left runs/sst2/typesafe-jev-1.13.0 \
  --right runs/sst2/huggingface-roberta-large-sst2
```

Both runs must be complete and must score the same ordered rows. The command
verifies the shared dataset identity and row digest, recomputes both headline
metrics from raw predictions, reports `left - right` with a paired percentile
95% interval, and writes `results/<benchmark>/comparisons/`. Use `--samples`,
`--seed`, `--confidence`, and `--output` to override the defaults.

Committed artifacts for every benchmark live under
`results/<benchmark>/comparisons/`, and each Δ in the table above links to one.

## Limitations

- These results compare specific deployed systems, not training methods or
  parameter efficiency. The systems differ in architecture, scale, training
  data, compute, supervision, and release date.
- Public benchmark contamination in Jev cannot be excluded.
- SST-2 uses the public GLUE validation split, which was also the supervised
  checkpoint's development split; that row is a same-split reproduction rather
  than an untouched-test evaluation.
- Bootstrap intervals quantify variation from resampling the fixed evaluation
  rows. They do not capture model, checkpoint, prompt, or benchmark-selection
  uncertainty, so the results should not be generalized beyond these pinned
  systems and splits.

## Repository layout

- `benchmarks/`: benchmark entry points and shared evaluation code
- `manifests/`: pinned SPACE-2 release provenance and digests
- `results/`: committed aggregate results
- `runs/`: ignored raw predictions and resumable run state

## Backlog

- Add providers for Kev, Laya
- Analyse calibration with ECE
