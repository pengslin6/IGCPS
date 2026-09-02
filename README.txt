DA-TGT: Current Audited Code and Root-Localization Reports
===========================================================

Paper title: DA-TGT: A Delay-Aware Typed Graph Transformer for Joint Anomaly
Detection and Root-Cause Localization in Industrial Cyber-Physical Systems

DA-TGT is a Delay-Aware Typed Graph Transformer for joint anomaly detection
and root-cause localization in industrial cyber-physical systems. This
repository contains the implementation used by the revised DA-TGT manuscript,
two final root-localization reports, and a plain-text statistical audit. It
intentionally contains no figures, fused CSV files, raw captures, model
checkpoints, or manuscript files.

Current model boundary
----------------------
- One shared DA-TGT encoder for detection and root-cause localization.
- One current fused graph per sample (K=1).
- Typed network-endpoint and physical-process nodes.
- Two topology-constrained local blocks and one global Transformer.
- No Temporal Shift Module (TSM).
- No dynamic or cross-layer edge-modulation module.
- No model ensemble, teacher/student distillation, or causal-path recovery.

Repository contents
-------------------
- code/: causal fusion, model, calibration, controlled baselines, audits,
  report export, and manuscript-figure generation source code.
- code/single_hgan_joint_experiment.py: main DA-TGT training and evaluation
  entry point. The current class is DATGTJoint; the legacy class alias and
  checkpoint filenames are retained for compatibility with stored artifacts.
- code/hgan_causal_multiseed.py and code/paired_baseline_multiseed.py:
  seed-matched DA-TGT and controlled-baseline reruns used by the repeated-seed
  audit. The legacy filename is retained for checkpoint compatibility.
- code/paired_significance_analysis.py: paired confidence intervals,
  two-sided paired t tests with Holm correction, and Wilcoxon sensitivity
  checks for the six reported metrics.
- code/make_convergence_figure.py: plots retained final-refit joint-training
  loss trajectories from explicitly supplied history tables.
- protocol_manifest_20260831.json: frozen seeds, caps, hashes, baseline
  mappings, comparator rule, and retrospective-test status for the audit.
- reports/igcps_traceback_report.txt: final IGCPS sample-level report.
- reports/te_cup_sec_traceback_report.txt: final TE-CUP-SEC sample-level report.
- reports/paired_statistical_audit.txt: complete ten-model five-seed rankings,
  conservative paired tests, comparator mappings, and interpretation limits.

Repeated-seed audit boundary
----------------------------
The audit uses seeds 11, 22, 33, 44, and 55 and changes initialization only.
DA-TGT and all nine controlled adapters were completed on both datasets: 50
model--seed runs per dataset and 100 in total. The displayed paired comparison
selects the baseline with the highest observed five-seed mean for each metric.
On IGCPS, DA-TGT ranks first on Top-1, MRR, and NDCG@5 but second, fifth, and
second on accuracy, macro F1, and macro AUC. On TE-CUP-SEC, DA-TGT ranks first
in all six repeated means, although its accuracy gain is not statistically
significant. With five pairs, the exact two-sided Wilcoxon test cannot fall
below 0.0625. These are retrospective reproducibility estimates because the
temporal test tails had already been inspected during the broader revision
cycle; they are not preregistered or untouched confirmatory results.

All baseline encoders use the same supervised class target and the same form
of supervised reference-node scorer in the common harness. Their localization
columns therefore measure adapter behavior under this controlled task; they do
not imply that every source paper originally proposed exact-node localization.

TE-CUP-SEC data sources
-----------------------
The datasets are not copied into this repository.

Network-layer traffic data:
https://pan.baidu.com/wap/init?surl=VT1x56k2RN9tKlXdk5nq4Q
Extraction code: S2SL

Physical-layer process data:
https://github.com/jiw09005/TE-CUP-SEC-datasets

The final fusion implementation is code/build_tecupsec_causal_fusion.py. It
aggregates endpoint-preserving traffic features in completed one-second windows
and aligns them with the latest already available process observation. No
future interpolation is used.

Report protocol
---------------
The reports are exported by code/export_traceback_reports.py from the locked
checkpoints used by the revised manuscript. The IGCPS report applies the
validation-locked folded 12-coefficient detection-head adjustment. The
TE-CUP-SEC report uses the final unadjusted head and a deterministic test cap of
4,000 samples per class (seed 44). TE-CUP-SEC Attacks 1-3 do not have exact
affected nodes represented among the released 53 process variables, so those
samples are explicitly excluded from exact-node localization metrics.

The TXT reports contain supervised node rankings. They do not claim to recover
ground-truth causal propagation paths.

Environment
-----------
Install the packages listed in code/requirements.txt. Fusion also requires
tshark to read the network captures. The full report exporter additionally
requires the locally generated causal-fusion CSV files, final checkpoints, and
validation lock described by its command-line arguments and source constants.
