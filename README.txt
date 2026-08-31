DA-TGT: Current Audited Code and Root-Localization Reports
===========================================================

Paper title: DA-TGT: A Delay-Aware Typed Graph Transformer for Joint Anomaly
Detection and Root-Cause Localization in Industrial Cyber-Physical Systems

DA-TGT is a Delay-Aware Typed Graph Transformer for joint anomaly detection
and root-cause localization in industrial cyber-physical systems. This
repository contains the implementation used by the revised DA-TGT manuscript
and two final root-localization reports. It intentionally contains no
figures, fused CSV files, raw captures, model checkpoints, or manuscript files.

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
- protocol_manifest_20260831.json: frozen seeds, caps, hashes, baseline
  mappings, comparator rule, and retrospective-test status for the audit.
- reports/igcps_traceback_report.txt: final IGCPS sample-level report.
- reports/te_cup_sec_traceback_report.txt: final TE-CUP-SEC sample-level report.

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
