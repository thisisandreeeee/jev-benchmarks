# Jev Benchmarks

Reproducible evaluations of Jev and Jev-like classifiers without training them.

## Results

| Benchmark | Metric   | BERT-Base |   Jev | Jev-like |
| --------- | -------- | --------: | ----: | -------: |
| BANKING77 | Accuracy |     93.02 | 79.90 |        — |
| SST-2     | Accuracy |      93.5 |     — |        — |
| STS-B     | Spearman |      85.8 |     — |        — |

The BERT figures are published supervised references, not zero-shot results:

- [BERT, Table 1](https://arxiv.org/pdf/1810.04805)
- [SPACE-2, Table 2](https://aclanthology.org/2022.coling-1.46.pdf)

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
