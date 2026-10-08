# Reproducing the locked HE-TGT pipeline

## Scope and environment

The copied dated directories are historical sources and evidence. Their relative layout and bytes are preserved. Their original absolute paths and statements about then-unpublished code are historical provenance, not current release status. `SOURCE_SHA256.json` checks the copied files. The manuscript audit in the older ZIP is not a validation of subsequently edited manuscript files.

The recorded experiment environment is Python 3.12.7, PyTorch 2.7.0, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1 and scikit-learn 1.5.1. See `experiments_20260928/environment.json`. Pickled scikit-learn artifacts should be loaded with the matching versions. `requirements.txt` records known versions and lists the additional helper dependencies. The current release does not guarantee reproducibility on untested package versions or accelerators.

## Required external artifacts

Set `ARTIFACT_ROOT` to the original experiment workspace or an explicitly mapped copy with this layout:

```text
ARTIFACT_ROOT/
  detection_history_final_20260929/
    retained_runtime/assets/<dataset>/
      metadata.json
      adjacency.pt
      backbone_seed11.pt  (also 22, 33, 44, 55)
    <dataset>/
      preprocessing.joblib
      test.npz
      seed11.predictions.npz  (also 22, 33, 44, 55)
  unified_detection_20260929/pair_refinement/
    final_fit_lock.json
    <dataset>/
      final_pairs.joblib
      final_prior.joblib
      final_seed11.npz  (also 22, 33, 44, 55)
  experiments_20260928/data_cache/<dataset>/
    test.pt
    development.pt
    metadata.json
```

Datasets are `IGCPS` and `TE-CUP-SEC`. Inference needs the graph assets, fitted pair/prior banks, lock and supplied input arrays. Verification additionally needs test caches and retained predictions. The original full evaluator also needs archived DA-TGT predictions and the artifacts named in its source and historical locks.

The graph tensor must follow the recorded node inventory, feature layout and fitting-only transforms. The history NPZ must contain aligned `static` and `history` arrays generated with the fitted preprocessing and `history.make_history`. Inference accepts no class or reference labels. Supplying already prepared arrays does not verify raw capture synchronization; that audit is a separate data-preparation step.

```bash
python he_tgt/run_he_tgt.py --artifact-root ARTIFACT_ROOT --dataset IGCPS --seed 11 --graph GRAPH.pt --history HISTORY.npz --output predictions.npz
```

`run_he_tgt.py` uses the published source and the separately supplied artifacts. It leaves the historical lock files unchanged, checks the manuscript's fixed coefficients, and refuses to overwrite an output. It does not invoke the dataset-specific `current_model.predict` policies.

## Verify the release

```bash
# No model assets: source hashes, syntax and compact metric means.
python he_tgt/verify_release.py

# With retained assets: 128 evenly spaced test rows for every dataset/seed,
# graph+history inference against saved outputs, and full saved-array macro F1.
python he_tgt/verify_release.py --artifact-root ARTIFACT_ROOT --output verification.json
```

The runtime check uses both datasets and all five graph seeds. It requires exact predicted classes and probabilities/node scores within 1e-6. It also reconstructs the full saved-array macro F1. It is an inference replay, not retraining, full raw-data regeneration or independent-event validation. `publication_verification.json` records the checks performed for this release.

## Training and main evaluation provenance

1. `experiments_20260928/common.py` prepares the chronological graph data; the original fused CSV paths and hashes are recorded in its `DATA` mapping. `code/` at repository root holds the fusion implementations.
2. `experiments_20260928/fair_control.py` fits the matched graph models. Dataset-specific selected graph settings are retained under `fair_control/<dataset>/selection_lock.json`.
3. `unified_detection_20260929/unified.py` constructs history features and the multiclass tree branch. `pair_model.py` fits/refines all unordered class pairs. Read `PROTOCOL.md`, `PAIR_PROTOCOL.md` and `pair_refinement/selection_lock.json` before attempting to reproduce the historical selection.
4. `he_tgt_revision_20260929/evaluate_revision.py` reconstructs the six fixed configurations, ranking lookup and subgroup results using saved artifacts; its `metrics.json` contains the full published evidence snapshot.

The old research scripts preserve workspace-specific paths and hash locks; they are not advertised as one-command training from a fresh clone. Relocation requires an explicit data/artifact path map. Never overwrite a historical lock merely to make a rerun pass. A new fit should get a new output directory and new provenance.

The later `information_matched_full_20260930` experiment changes the history coefficient to 0.50 only in its equal-budget sensitivity analysis. The current manuscript model and launcher remain at 0.25.
