# Reproducing the current no-node-head HE-TGT

## Versions and source scope

The implementation uses the no-head graph fitted on 9 October 2026, the fixed multiclass-history bank and universal class-pair refinement. The graph exposes class logits only. The initializer retains the evaluated parameter-initialization order before removing the node scorer; no node-scoring parameters remain in the fitted model.

`SOURCE_SHA256.json` records the published Python sources. Older dated files are preserved research code, not the current main entry. Their generated JSON summaries, histories, caches and sample-level reports are no longer distributed. `code/protocols/*.json` are executable configurations for historical baseline reproduction, not observation data.

The verified software versions are Python 3.12.7, PyTorch 2.7.0, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1 and scikit-learn 1.5.1. Additional packages are listed in `requirements.txt`. Serialized tree artifacts require compatible versions. New device/library combinations require their own numerical verification.

## External artifact layout

```text
ARTIFACT_ROOT/
  experiments_20260928/
    data_cache/<dataset>/
      metadata.json
      adjacency.pt
      development.pt
      test.pt
    fair_control/<dataset>/
      selection_lock.json
      final__DA-TGT__seed11.pt
      final__DA-TGT__seed11.predictions.npz
      ... (also seeds 22, 33, 44, 55)
  detection_history_final_20260929/<dataset>/
    test.npz
  unified_detection_20260929/pair_refinement/
    final_fit_lock.json
    <dataset>/
      final_prior.joblib
      final_pairs.joblib
      final_seed11.npz
      ... (also seeds 22, 33, 44, 55)

CHECKPOINT_ROOT/<dataset>/
  no_head_seed11.pt
  ... (also seeds 22, 33, 44, 55)
  without_head_seed11.predictions.npz
  ... (saved outputs are needed only for replay verification)
```

Datasets are `IGCPS` and `TE-CUP-SEC`. Inference needs metadata, adjacency, the requested no-head checkpoint and history banks, the history policy lock, and input arrays. The reference table is fitted from eligible development labels during initialization. Alternatively, supply a previously exported development-fitted table with `--reference-table`; this avoids loading the development cache at inference initialization. Never fit this table from test/query labels.

Input graphs must use the fitted feature transforms, node order and topology. The history NPZ contains aligned `static` and `history` arrays constructed by `history.make_history`. Query labels are not inference inputs. Passing prepared arrays does not regenerate or validate raw synchronization.

## Inference

```bash
python he_tgt/run_he_tgt.py --artifact-root ARTIFACT_ROOT --checkpoint-root CHECKPOINT_ROOT --dataset IGCPS --seed 11 --graph GRAPH.pt --history HISTORY.npz --output predictions.npz
```

`--device` selects a PyTorch device; `auto` selects an available supported accelerator or the default host device. The fixed tree classifiers use their retained library implementation. The output includes final probabilities/classes, graph probabilities, lookup scores and reference order. It contains no learned node-head scores. Existing output files are not overwritten.

## Source and numerical verification

```bash
python he_tgt/verify_release.py
python he_tgt/verify_release.py --artifact-root ARTIFACT_ROOT --checkpoint-root CHECKPOINT_ROOT --output external-verification.json
```

The optional replay checks 128 evenly spaced retained test rows per dataset and graph seed against the saved no-head outputs, requires identical predicted classes, and checks probabilities and lookup scores within absolute tolerance 1e-6. It also recomputes full saved-array macro F1. This is numerical replay, not retraining or an independent dataset experiment. Keep generated reports outside version control.

## Refit and evaluate the deletion study

Use a **new** `RUN_ROOT` and the external former-model/reference artifacts listed above:

```bash
python he_tgt/no_head_20261009/experiment.py prepare --artifact-root ARTIFACT_ROOT --output-root RUN_ROOT
python he_tgt/no_head_20261009/experiment.py train_all --artifact-root ARTIFACT_ROOT --output-root RUN_ROOT
python he_tgt/no_head_20261009/experiment.py evaluate --artifact-root ARTIFACT_ROOT --output-root RUN_ROOT
```

Training uses only development graph caches and inherited settings: IGCPS 0.002/17 epochs; TE-CUP-SEC 0.0007/16 epochs. The script verifies the no-head forward against the corresponding decoupled classifier and locks all ten fitted checkpoints before test evaluation. Historical history banks are held fixed, as in the reported study. This command does not claim to rerun all baseline searches or refit every pipeline component.

History construction/fitting is implemented in `unified_detection_20260929/unified.py` and `pair_model.py`. Raw graph preparation is in `experiments_20260928/common.py`; older research preparation scripts require an explicit source-data path map. Read the accompanying protocols before reproducing those stages. Do not overwrite original locks to make a rerun pass; create fresh outputs.

Fitting duration, graph operation counts and inference latency are different quantities. A benchmark must separately state its prepared-input scope, included history processing, warm-up, repeats, precision, threads and actual device. Numerical accuracy results do not establish a runtime measurement.
