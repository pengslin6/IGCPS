# HE-TGT: History-Enhanced Type Graph Transformer

Source code for **HE-TGT: A History-Enhanced Type Graph Transformer for Anomaly Detection and Reference-Node Ranking in Industrial Cyber–Physical Systems**. Current source revision: **10 October 2026**.

The current manuscript is `HE-TGT.tex`. Its graph backbone has **no learned node-scoring head or node-supervised loss**. Detection uses a typed graph Transformer, fixed history refinement and a development-fitted class-to-reference lookup. Earlier dated implementations remain available as historical comparison code.

## Current implementation

| Role | Source |
|---|---|
| Current inference entry | [`he_tgt/run_he_tgt.py`](he_tgt/run_he_tgt.py) |
| No-head graph definition | [`model.py`](he_tgt/no_head_20261009/model.py) |
| Graph/history inference and reference lookup | [`runtime.py`](he_tgt/no_head_20261009/runtime.py) |
| Five-seed graph refits and former-model replay | [`experiment.py`](he_tgt/no_head_20261009/experiment.py) |
| Strictly past history construction | [`history.py`](he_tgt/detection_history_final_20260929/retained_runtime/history.py) |
| Multiclass history features and fitting | [`unified.py`](he_tgt/unified_detection_20260929/unified.py) |
| Universal class-pair fitting/refinement | [`pair_model.py`](he_tgt/unified_detection_20260929/pair_model.py) |
| Source and inference verification | [`verify_release.py`](he_tgt/verify_release.py) |

History uses at most eight strictly earlier observations from the same directed endpoint pair within 30 seconds. Equal-time observations cannot enter one another's histories. The fixed history weight is 0.25, pair-tree weight 1.0, and pair-refinement strength 0.5. Final class probabilities, rather than learned node scores, determine the reference ranking. Reference sets are scenario-derived investigation targets; they do not independently identify causal initiators.

## Which results were newly computed?

The 9 October study fitted ten new no-head graph models: two datasets with five graph seeds each. Multiclass history and class-pair banks were kept fixed. The ten former graph checkpoints were replayed, and the nine detection baselines retain their earlier fits. Updated detection/ranking metrics, component removals, confusion matrices and calibration diagnostics derive from the new no-head predictions. This does not constitute a new acquisition or a full refit of every component.

This source release and numerical replay do not establish device-latency or peak-memory measurements. Parameter reduction alone is not a latency measurement.

## Source-only distribution

Raw/fused observations, graph/history caches, checkpoints, fitted tree banks, prediction arrays, experiment result summaries and sample-level reports are excluded from the current branch. Necessary source configurations and a code-hash manifest remain. Local experiment records are retained separately; removal from the current branch does not rewrite Git history.

See [`he_tgt/REPRODUCING.md`](he_tgt/REPRODUCING.md) for external artifact layout and commands. Missing trained artifacts cannot be reconstructed from code hashes. Use matching library versions for serialized tree models.

```bash
python -m pip install -r he_tgt/requirements.txt
python he_tgt/verify_release.py
python he_tgt/run_he_tgt.py --help
```

## Historical implementations and data preparation

The [`code/`](code/) directory contains the earlier graph-baseline adapters and the [IGCPS](code/build_igcps_causal_fusion.py) and [TE-CUP-SEC](code/build_tecupsec_causal_fusion.py) past-only fusion scripts. The [historical protocol description](docs/DA-TGT-README-20260909.txt) records the earlier data preparation. Older dated directories under `he_tgt/` are research implementations; they are not the current inference entry and may require separately retained artifacts. Their output files have been removed from the current branch.

The main comparison is between complete detection pipelines. The no-head graph study inherits the recorded learning rates and epoch counts; history trees remain fixed across graph seeds. Previously inspected temporal tails support retrospective evaluation under the stated partitions.
