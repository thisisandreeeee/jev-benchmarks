# Jev Benchmarks

Reproducible evaluations of Jev and Jev-like classifiers without training them.

## Results

Direct same-split comparisons recomputed by this repository:

| Benchmark | Supervised checkpoint              | Metric   | Baseline | Jev zero-shot |
| --------- | ---------------------------------- | -------- | -------: | ------------: |
| SST-2     | `philschmid/roberta-large-sst2`    | Accuracy |    96.44 |         94.50 |
| STS-B     | `cross-encoder/stsb-roberta-large` | Spearman |    91.44 |         89.12 |

Published historical references and existing Jev evaluations:

| Benchmark | Metric   | BERT-Base |   Jev | Jev-like |
| --------- | -------- | --------: | ----: | -------: |
| BANKING77 | Accuracy |     93.02 | 79.90 |        — |
| SST-2     | Accuracy |      93.5 | 94.50 |        — |
| STS-B     | Spearman |      85.8 |     — |        — |

The BERT figures are published supervised references, not zero-shot results:

- [BERT, Table 1](https://arxiv.org/pdf/1810.04805)
- [SPACE-2, Table 2](https://aclanthology.org/2022.coling-1.46.pdf)

The supervised values above were locally recomputed from pinned checkpoint
revisions. The previous DistilBERT and SBERT aggregate artifacts are retained
as historical results.

SST-2 uses the public GLUE validation split. Its BERT reference uses the hidden
test split, so those figures are not directly comparable.
See [EVALUATION.md](EVALUATION.md) for the frozen comparison protocol and its
limitations.

## Jev question types

| Question type | Benchmarks | What they evaluate              |
| ------------- | ---------- | ------------------------------- |
| `Choice`      | BANKING77  | Intent classification           |
| `Noul`        | SST-2      | Binary sentiment classification |
| `Score`       | STS-B      | Semantic similarity scoring     |

## Usage

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

Setup the environment:

```bash
uv sync
uv run pytest
```

Run the benchmarks:

```bash
# Five-example live smoke test
uv run python -m benchmarks.banking77 --limit 5

# Continue the same run through all remaining examples
uv run python -m benchmarks.banking77 --resume
```

## Backlog

- Analyse jev calibration (NLL, ECE)
