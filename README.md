# HE-TGT: History-Enhanced Type Graph Transformer

Code for **HE-TGT: A History-Enhanced Type Graph Transformer for Anomaly Detection and Reference-Node Ranking in Industrial Cyber–Physical Systems**. Source release: 8 October 2026.

The manuscript source retains the filename `DA-TGT.tex`, but its current method is HE-TGT. The implementation is the locked **universal class-pair pipeline**, not the later experimental `HistoryJoint` model and not the dataset-specific retained detection policies.

## Which code implements the current paper?

| Role | File |
|---|---|
| Inference entry used by the experiments | [`he_tgt/unified_detection_20260929/predict_pair.py`](he_tgt/unified_detection_20260929/predict_pair.py) |
| All-class-pair fitting and probability refinement | [`pair_model.py`](he_tgt/unified_detection_20260929/pair_model.py) |
| Multiclass history fitting and feature selection | [`unified.py`](he_tgt/unified_detection_20260929/unified.py) |
| Typed graph encoder, `DATGTJoint` | [`backbone.py`](he_tgt/detection_history_final_20260929/retained_runtime/backbone.py) |
| Strictly past history construction | [`history.py`](he_tgt/detection_history_final_20260929/retained_runtime/history.py) |
| Matched graph training | [`fair_control.py`](he_tgt/experiments_20260928/fair_control.py) |
| Main results, ablations and reference lookup | [`evaluate_revision.py`](he_tgt/he_tgt_revision_20260929/evaluate_revision.py), [`metrics.json`](he_tgt/he_tgt_revision_20260929/metrics.json) |
| Portable inference entry with an explicit asset location | [`run_he_tgt.py`](he_tgt/run_he_tgt.py) |

The original source files are copied byte-for-byte; [`SOURCE_SHA256.json`](he_tgt/SOURCE_SHA256.json) records their hashes. The new launcher only redirects artifact paths. `current_model.load_backbone` is reused by HE-TGT; `current_model.predict` implements an older dataset-specific policy and **is not the HE-TGT entry point**. The old `code/single_hgan_joint_experiment.py` alone is not the complete current model.

## Method and evidence

HE-TGT uses a current typed graph, at most eight strictly earlier observations from the same directed endpoint pair within 30 seconds, a multiclass boosted-tree history classifier, and universal class-pair refinement. Equal-time observations cannot enter one another's histories. The locked rule uses history weight 0.25, pair-tree weight 1.0, and refinement strength 0.5 on both datasets. The graph branch supplies the node scores; the history stages change classification probabilities only.

| Dataset | Archived DA-TGT macro F1 | HE-TGT macro F1 | History-only macro F1 | HE-TGT focus recall | Node-head / lookup Top-1 |
|---|---:|---:|---:|---:|---:|
| IGCPS | 78.31% | 88.38% | **92.18%** | NM: 41.58% | 90.16% / 93.97% |
| TE-CUP-SEC | 80.77% | 84.65% | **91.11%** | Attack6: 24.31% | 82.01% / 84.67% |

These results are retrospective on previously inspected temporal tails. Five seeds vary graph fitting; one fixed tree bank is reused per dataset. History-only is stronger in macro F1 on both datasets, and the class-probability lookup is stronger than the learned node head. Reference sets are fixed within eligible classes and do not independently identify causal initiators. The `history_only` entries in the generic metrics file retain incidental graph-ranking diagnostics; those fields are not a history-only node head.

The compact evidence snapshot retains the complete historical evaluator output. The manuscript's component-removal table uses the graph backbone, graph/history blend, pair-only refinement and complete model, plus archived DA-TGT. This release does not change the manuscript or select a new model.

Additional experiments are clearly separated from main results:

- [`node_head_ablation_20260930`](he_tgt/node_head_ablation_20260930/RESULTS.md): validation-only node-head removal and fixed-pipeline diagnostics used in the response letter.
- [`information_matched_full_20260930`](he_tgt/information_matched_full_20260930/RESULTS.md): later history-branch transfer and equal final-policy-budget sensitivity. The alternative backbone does not uniformly lose to HE-TGT. These results were not substituted into the main manuscript.

Historical graph-only CPU timings do not measure the complete HE-TGT pipeline. This release makes no HE-TGT Jetson, full-system latency, causal-path recovery, or independent-event generalization claim.

## Run and reproduce

See [`he_tgt/REPRODUCING.md`](he_tgt/REPRODUCING.md) for dependencies, artifact layout, inference and verification. This is a **source and compact evidence release**, not a pretrained-model or raw-data release. Raw/fused data, checkpoints, tree banks and large prediction arrays are not included. Full reproduction requires those separately retained artifacts and the recorded preprocessing. Missing artifacts cannot be replaced by the JSON summaries.

```bash
python -m pip install -r he_tgt/requirements.txt
python he_tgt/verify_release.py
python he_tgt/run_he_tgt.py --help
```

## Earlier DA-TGT resources

The existing [`code/`](code/), [`reports/`](reports/) and [`protocol_manifest_20260831.json`](protocol_manifest_20260831.json) are retained historical DA-TGT resources. In particular, their sample-level reports are **not new HE-TGT predictions**. See the preserved [DA-TGT README](docs/DA-TGT-README-20260909.txt) for that protocol.

Past-only fusion remains in [`build_igcps_causal_fusion.py`](code/build_igcps_causal_fusion.py) and [`build_tecupsec_causal_fusion.py`](code/build_tecupsec_causal_fusion.py). TE-CUP-SEC source links remain in the historical README. No manuscript files, raw captures or fused CSV files are distributed here.
