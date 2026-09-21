# Evaluation protocol

This repository compares frozen Jev inference with frozen, author-released
supervised checkpoints. It does not compare training methods or claim that
zero-shot learning is generally better than supervised learning.

## Terms

- **Jev zero-shot**: no benchmark examples or parameter updates are supplied
  during evaluation. The task instruction, output rubric, and complete set of
  candidate labels may be supplied.
- **Supervised checkpoint**: a released model whose parameters were trained
  using labeled examples from the benchmark's training split.
- **Comparable result**: both systems predict the same pinned evaluation rows
  and are scored by the same repository metric implementation.

Zero-shot does not mean that Jev's upstream training data is known to exclude
the benchmark. All selected datasets are public, so contamination cannot be
ruled out.

## Frozen benchmark plan

| Output   | Benchmark | Evaluation split            | Supervised checkpoint              |
| -------- | --------- | --------------------------- | ---------------------------------- |
| `Choice` | BANKING77 | official test               | SPACE-2 author release             |
| `Choice` | CLINC150  | official test               | SPACE-2 author release             |
| `Choice` | HWU64     | official test               | SPACE-2 author release             |
| `Noul`   | SST-2     | GLUE validation             | `philschmid/roberta-large-sst2`    |
| `Score`  | STS-B     | original STS Benchmark test | `cross-encoder/stsb-roberta-large` |

The checkpoint and dataset revisions must be pinned before inference. Model
files distributed outside a revisioned registry must also have a recorded
SHA-256 digest.

The BANKING77, SST-2, and former GLUE-validation STS-B Jev results were
inspected before this protocol was written. BANKING77 and SST-2 remain
retrospective; the original STS Benchmark test run was performed after the
protocol was frozen. Future CLINC150 and HWU64 runs are confirmatory.

## Procedure

1. Freeze the dataset revision, evaluation split, prompt or model input
   transformation, checkpoint revision, metric, and expected row count.
2. Run the supervised checkpoint locally without parameter updates. Where the
   authors provide an evaluation procedure, first check that the downloaded
   checkpoint behaves consistently with its documented result.
3. Run Jev on the identical ordered rows without labeled examples. Prompts
   must not be changed after inspecting evaluation predictions or labels.
4. Retain per-example predictions outside Git and publish aggregate results
   only after every canonical evaluation row succeeds exactly once.
5. Report each system's metric, their paired difference, and a paired-bootstrap
   95% confidence interval. If Jev inference is nondeterministic, report the
   run count and variation across runs.
6. Publish every benchmark selected above; do not select results based on
   whether they favor either system.

## Provenance

Every aggregate result must identify:

- dataset ID, configuration, revision, split, and row count;
- provider, model ID, and immutable checkpoint revision or digest;
- prompt, label mapping, or score transformation;
- dependency versions and repository revision;
- whether the source worktree was clean when inference began; and
- the original publication or model card for an external checkpoint.

Published paper scores are historical context, not direct comparison values.
The direct comparison table contains only scores recomputed by this repository
on identical evaluation rows.

## Known limitations

- The compared systems differ in architecture, parameter count, pretraining
  data, compute, and release date.
- Public benchmark contamination in Jev cannot be excluded.
- SST-2 uses the public GLUE validation split, which was also the supervised
  checkpoint's development split; that comparison is a same-split
  reproduction rather than an untouched-test evaluation.
- SPACE-2 reports averages over multiple seeds, while its released files may
  contain selected checkpoints. Recomputed checkpoint scores and published
  paper scores must therefore be reported separately.
