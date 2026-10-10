# HE-TGT manuscript revision — 29 September 2026

This snapshot supports the clean manuscript `ESWA-round2/DA-TGT.tex`, marked manuscript, and point-by-point response. The retained filename does not identify the current method: its current name is **HE-TGT**.

## Scientific scope

HE-TGT is the locked universal class-pair pipeline in `unified_detection_20260929/pair_refinement`, not the subsequently explored end-to-end HistoryJoint model. The fixed decision uses 0.25 multiclass-history weight, 0.5 pair-refinement weight, and tree-only pair predictions. It uses the matched jointly trained graph backbone, not the archived original DA-TGT graph checkpoints.

History is at most eight strictly earlier rows within 30 seconds from the same directed endpoint pair. Equal-time queries are evaluated before insertion. Input dimensions and fitted parameters differ by dataset; history rules, tree settings, and probability coefficients are shared. Graph learning rates, epochs, and retained recency weights differ by dataset as reported in the manuscript.

The experiment is retrospective. Previous temporal-tail inspection is not undone by configuration locks. No common candidate passed every stronger-incumbent protection criterion; the retained configuration followed the recorded diagnostic fallback. The current evaluation script performs no new model fitting or selection.

### Results that constrain interpretation

- HE-TGT improves macro F1 over archived DA-TGT: IGCPS 78.31% → 88.38%; TE-CUP-SEC 80.77% → 84.65%.
- Focus-class recall improves: NM 27.89% → 41.58%; Attack6 16.06% → 24.31%. These are multiclass-recognition recalls, not binary anomaly detection recalls.
- Normal FPR changes from 2.71% → 3.09% on IGCPS and 35.53% → 17.07% on TE-CUP-SEC.
- **History-only detection is stronger than full HE-TGT** in macro F1: 92.18% and 91.11%. The manuscript does not claim fusion superiority.
- The final-probability reference lookup exceeds the learned node head in all three aggregate ranking means on both datasets. Reference labels are fixed within every eligible class and do not independently identify causal initiators.
- Five seeds vary graph fitting only. One fixed history/pair-tree bank is reused per dataset. History-only scores therefore have no graph-seed standard deviation.
- Earlier graph-only timing and label-scarcity routines are historical and separate from the current no-head inference entry.

## Contents and reproduction

`evaluate_revision.py` verifies final-fit hashes, labels and raw row indices, reconstructs all ten locked HE-TGT probability arrays exactly, and evaluates six fixed configurations: archived DA-TGT, matched graph, history-only, graph/history blend, graph plus pair-only refinement, and full HE-TGT. `metrics.json` contains per-seed and aggregate results; `verification.json` records exact replay and source hashes.

Important machine-readable convention: the generic evaluator receives graph scores even for the history-only detection variant. Any graph-ranking fields under that variant are incidental retained-graph diagnostics, **not** an independently learned history-only node head. The ablation table uses detection fields only.

`figures.py` produces architecture, confusion, reliability, representative node-ranking, and matched-backbone convergence figures. `node_examples.json` records the deterministic first test window in each IGCPS anomaly class, seed 11. Blue node bars identify acceptable references, not verified causal initiators.

From the original workspace root, using its existing scientific Python environment:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
python -X utf8 he_tgt_revision_20260929/evaluate_revision.py
python -X utf8 he_tgt_revision_20260929/figures.py
python -X utf8 he_tgt_revision_20260929/audit_revision.py
```

The compact archive preserves the relative directory layout of relevant sources, settings, and summaries. It intentionally omits raw/fused data, graph caches, large saved arrays, fitted tree banks, and checkpoints. Exact replay also requires the separately retained local assets:

- `experiments_20260928/data_cache/<dataset>/{development,test}.pt` and `metadata.json`;
- `experiments_20260928/archived_analysis/<dataset>/replayed__seed<seed>.npz`;
- `detection_history_final_20260929/<dataset>/test.npz` and `seed<seed>.predictions.npz`;
- `unified_detection_20260929/pair_refinement/<dataset>/final_{pairs,prior}.joblib` and `final_seed<seed>.npz`;
- all files referenced by the saved `final_fit_lock.json` hashes;
- graph fitting histories in `experiments_20260928/fair_control/<dataset>`.

The frozen historical locks retain original absolute paths. Moving the archive to a new machine requires an explicit path map to the separately retained assets; do not rewrite the locks and pretend their historical hashes were unchanged. This is a provenance/evaluation package, not a standalone raw-data or model release. No public repository upload is claimed.

## Manuscript and response checks

`audit_revision.py` checks all 23 reviewer comments against the pre-edit response backup, acknowledgment-first responses, verbatim manuscript quotations, red location macros, baseline numerical preservation, new main-table values, abstract word count, resolved references, and PDF text consistency. It renders QA pages and records SHA-256 hashes. Page citations refer to **printed clean-manuscript page numbers**, excluding the unnumbered highlights page.

The marked manuscript is generated against the unchanged `HGAN-Trace-revise.tex`; revised tables, mathematical blocks, prose, titles, and captions are colored red where changed. Current model references are HE-TGT. Historical DA-TGT names and original reviewer wording are deliberately preserved.
