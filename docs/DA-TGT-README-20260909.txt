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
- Two topology-constrained local aggregation blocks (not local GAT attention)
  followed by two layers of one shared, four-head global Transformer.
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
- protocol_manifest_20260831.json: updated evidence/protocol record (2026-09-09),
  with the legacy filename retained. This is not a retrospective preregistration.
- code/run_locked_protocol.py and code/protocols/: explicit recorded settings
  for main-model and nine-adapter refits; --dry-run prints the commands only.
- reports/igcps_traceback_report.txt: final IGCPS sample-level report.
- reports/te_cup_sec_traceback_report.txt: final TE-CUP-SEC sample-level report.
- reports/paired_statistical_audit.txt: complete ten-model five-seed rankings,
  conservative paired tests, comparator mappings, and interpretation limits.

Repeated-seed audit boundary
----------------------------
The audit uses seeds 11, 22, 33, 44, and 55; the seed changes initialization,
training order, and stochastic regularization, while the protocol stays fixed.
DA-TGT and all nine controlled adapters were completed on both datasets: 50
model--seed runs per dataset and 100 in total. The displayed paired comparison
selects the baseline with the highest observed five-seed mean for each metric.
Following the corrected NS supervision and refits, DA-TGT ranks first on IGCPS
Top-1, MRR, and NDCG@5 but third, seventh, and fourth on accuracy, macro F1,
and macro AUC. On TE-CUP-SEC, DA-TGT ranks first
in all six repeated means, although its accuracy gain is not statistically
significant. With five pairs, the exact two-sided Wilcoxon test cannot fall
below 0.0625. These are retrospective reproducibility estimates because the
temporal test tails had already been inspected during the broader revision
cycle; they are not preregistered or untouched confirmatory results.

All configurations use the same class/reference targets and input schema, but
their optimization procedures are not equivalent. Baseline node heads train
for three epochs on frozen detector embeddings. DA-TGT jointly trains its
encoder and a graph-context node head. TE recency strength is two for DA-TGT
and one for the adapters; the IGCPS probability adjustment is DA-TGT-specific.
The ranking gaps therefore do not isolate encoder superiority. Adapter results
do not imply that every source paper originally proposed node localization.
DT-GNN and STCI are local adapter labels inspired by DTFL and ICAD; they are
not the published method names. Exploratory graph-attention variants are not
included in either manuscript comparison table.

Five-seed DA-TGT means (percent; Acc / macro F1 / macro AUC / Top-1 / MRR / NDCG@5):
IGCPS:      93.42 / 78.31 / 99.18 / 86.68 / 93.34 / 95.30
TE-CUP-SEC: 82.56 / 80.77 / 97.40 / 83.09 / 89.63 / 91.82
The development-fitted class-probability reference lookup outperforms the
learned node head on all three localization means in both datasets. Root labels
are scenario-derived references, not independently observed initiating nodes.

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
The two reports use fixed seed 11 from the current five-seed model records,
not the old seed-42 model and not an average or a best-seed selection. They
are exported by code/export_traceback_reports.py, which checks all six metrics
against the saved run and verifies the corrected IGCPS prediction arrays.
The IGCPS checkpoint was refitted with NS labels referring only to the high-
pressure and sub-high-pressure switches, excluding the medium-pressure heater.
The IGCPS report applies the
validation-locked conditional probability adjustment with 12 coefficients.
Its regularization is selected by an early/late validation split, then the
adjustment is refitted on the complete validation partition. The retained code
applies it after the neural forward pass; it is not merged into the classifier
weight matrix. The TE-CUP-SEC report uses the final unadjusted head and a deterministic test cap of
4,000 samples per class (seed 44). TE-CUP-SEC Attacks 1-3 do not have exact
affected nodes represented among the released 53 process variables, so those
samples are explicitly excluded from exact-node localization metrics.

The TXT reports contain supervised node rankings. They do not claim to recover
ground-truth causal propagation paths.

Correction of the original propagation visualization:
The old plot_causal_propagation_per_class routine averaged node scores within
each anomaly class, selected high-scoring nodes, and connected them in descending
score order. Its red edges were constructed by the plotting code; they were not
read from path labels or chosen by the pairwise propagation predictor. These
score-order links must not be interpreted as recovered causal propagation.
The current reports contain sample-level ranked scores instead. Missing path
annotations prevent independent validation, not candidate-path generation in
principle. Path targets generated by the same graph rules are not independent
ground truth. The archived source filenames are retained for compatibility.

Report seed-11 metrics (same metric order as above):
IGCPS:      92.639327 / 72.353160 / 99.427419 / 82.336957 / 91.168478 / 94.117595
TE-CUP-SEC: 83.188406 / 80.896581 / 97.442284 / 80.894188 / 87.479980 / 90.544769
Each report header records the checkpoint path/hash and reference policy.
The manifest hashes code and report text as UTF-8 with LF-normalized line endings
so that Git checkout settings do not invalidate verification. Checkpoint and
fused-input hashes still refer to the original file bytes.

Environment
-----------
Install the packages listed in code/requirements.txt. Fusion also requires
tshark to read the network captures. The full report exporter additionally
requires the locally generated causal-fusion CSV files, final checkpoints, and
validation lock described by its command-line arguments and source constants.

Reproduction (run from the repository root)
-----------------------------------------
python code/run_locked_protocol.py --dataset IGCPS --root PATH_TO_LOCAL_DATA --task all --dry-run
python code/run_locked_protocol.py --dataset TE-CUP-SEC --root PATH_TO_LOCAL_DATA --task all --dry-run
Omit --dry-run to perform the fixed refits. This does not conduct a new model
search or create an untouched test set. Existing result directories are not
overwritten by this launcher. Exact numerics may depend on runtime versions.
python code/export_traceback_reports.py --root PATH_TO_LOCAL_DATA --seed 11 --outdir LOCAL_REPORTS
python -m unittest discover -s code -p test_export_traceback_reports.py -v

The older single-run audit/figure helpers remain for archive compatibility;
their dated defaults are not the source of the current manuscript results.
single_hgan_shared_gat_experiment.py is retained because the calibration helper
imports it; its exploratory model is not the current model or a paper baseline.
