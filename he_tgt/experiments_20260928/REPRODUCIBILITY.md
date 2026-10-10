# Reproducing the revision audit

This folder contains the label audit, retained-model subgroup analysis and the
separately refitted DA-TGT / DT-GNN joint-training comparison. It does not modify
the original dataset files, archived models or DA-TGT architecture.

## Inputs and environment

The externally retained `environment.json` records the actual runtime and library
versions. Fits used two threads; some independent fits were scheduled concurrently.
Their wall times are not controlled inference-speed measurements. This historical
audit is separate from the current no-head inference entry.

Two fusion CSVs are required, with exact SHA-256 values specified in `common.py`:

- `sr_com_causal_v2.csv`
- `tecupsec_information_physical_causal_v3.csv`

The retained-checkpoint replay additionally requires the original archive folders
`hgan_causal_multiseed_ns_corrected_20260907` and `paired_te_main_20260831`, including
their result files and the IGCPS calibration locks/prediction files. These original
inputs are not distributed inside the compact evidence supplement. Their local
source locations and hashes are recorded in the metadata and verification files.

In the retained setup, `SOURCE` in `common.py` points to the original experimental
folder. For a fresh reproduction elsewhere, place the input files under an
appropriate folder and set `SOURCE` before starting, in a separate working copy
without completed-run manifests. Do not alter source paths or training code and
then resume the supplied completed runs: the manifest checks intentionally reject
changes to locked code. Generated caches and final checkpoints/predictions remain
available in the full local experiment folder, but are omitted from the compact
supplement to avoid duplicating large binary files.

## Execution

From this folder, using the recorded environment:

```text
python analyze_archived.py --stage all
python fair_control.py
python verify_experiment_outputs.py
python build_evidence_report.py
```

The serial control can instead be scheduled with:

```text
python parallel_fair_control.py --dataset IGCPS
python parallel_fair_control.py --dataset TE-CUP-SEC
```

Never run the serial and parallel coordinators on the same dataset at the same
time. They use identical fitting, selection and final aggregation functions.
Completed results are reused. An interrupted fit without its final result record
is restarted from its stated seed rather than resumed from an incomplete optimizer
state. The restarted IGCPS DT-GNN seed-22 losses matched all 15 recorded serial
epochs; this is recorded in `postrun_verification.json`.

## External evidence files (not distributed in this source-only branch)

- `data_cache/<dataset>/label_statistics.csv`: full and partition-specific class
  counts, eligible counts, distinct reference sets, modal shares, entropy, and
  source-endpoint variability.
- `data_cache/<dataset>/reference_set_frequencies.csv`: explicit set membership
  and frequency for every class and partition.
- `archived_analysis/<dataset>/`: replay verification, three-metric subgroup
  summaries and per-seed scores, and raw/adjusted class-detection metrics.
- `fair_control/<dataset>/`: manifests, implementation checks, selection locks,
  validation histories, final fit records, class metrics, aggregate/paired results,
  own-probability lookup results and post-run verification.
- `EVIDENCE_REPORT.md`: human-readable tables generated from the result files.
- `MODEL_DECISION.md`: the Chinese model-change assessment after both datasets
  complete.
- `manuscript_qa.json`: checks against the local manuscript and response files.
  This editorial check requires the neighboring `ESWA-round2` folder and is not
  necessary to reproduce the experiments themselves.

## Interpretation boundaries

All counts are fused windows, not independent incidents. All eligible reference
sets are fixed within class under the retained construction rules. New controls
omit probability adjustment and exclude unannotated attacks from negative node
supervision, so they must not replace the archived main-table values silently.
Five training seeds quantify conditional repeatability on previously inspected
test tails, not generalization to new plants. Tie-aware expected Top-1 uses an
absolute score tolerance of `1e-8`. No initiating-node/path annotations or Jetson
timing records are supplied by these experiments.

The subsequent `source_support_audit/` supplement traces IGCPS fusion rows to
the original physical file. Its script uses the original local source files and
the existing sibling experiment cache. It identifies source-sample reuse at
partition boundaries and a single observed NM label run across all partitions.
This does not alter the frozen experiments, quantify score inflation, or provide
independently verified event/node annotations. The supplement's review queue is
unreviewed evidence organization, not a training-label file. Large snapshots of
the manuscript taken before that audit are deliberately excluded from the zip.
