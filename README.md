# HE-TGT

Implementation of **HE-TGT: A History-Enhanced Type Graph Transformer for Anomaly Detection and Reference-Node Ranking in Industrial Cyber–Physical Systems**.

This repository provides the **complete current HE-TGT implementation and nine comparison-model adapters**, including data fusion, graph/history preparation, fitting, repeated-seed evaluation, component removals, diagnostic plots and Jetson inference. The current branch contains one version of the method. Superseded main-model implementations and revision-specific comparison scripts have been removed from this branch.

## Repository contents

| Location | Purpose |
|---|---|
| `data_processing/` | Past-only cyber–physical fusion for both datasets |
| `he_tgt/prepare.py` | Chronological partitions, graph tensors, development-fitted transforms and history arrays |
| `he_tgt/model.py` | Typed graph Transformer and class prediction |
| `he_tgt/history.py` | Strictly earlier observations from the same directed endpoint pair |
| `he_tgt/history_models.py` | Multiclass history classifier, class-pair classifiers and probability fusion |
| `he_tgt/train.py` | Fit the history bank and graph models |
| `he_tgt/infer.py` | Anomaly detection and reference-node ranking |
| `he_tgt/evaluate.py` | Repeated-seed evaluation and fixed component removals |
| `he_tgt/diagnostics.py` | Class-wise tables, count confusion matrices and reliability plots |
| `he_tgt/export_jetson.py`, `he_tgt/edge.py` | Export fitted models and run on Jetson |
| `baselines/` | Comparison-model definitions, training and evaluation |
| `analysis/paired_statistics.py` | Paired detection statistics with Holm correction |
| `configs/` | Fixed dataset-specific experiment settings |
| `tests/` | Behavioral and training checks using generated inputs |

The comparison models are **GCN-AE, GAT-AE, VGAE, GraphSAGE-AE, IIoT-GNN, EE-GCN, STGaAN, STCI and DT-GNN**. These are the study's architecture adapters for the common task. Definitions are in `baselines/components.py`, construction/evaluation in `baselines/protocol.py`, and fitting in `baselines/training.py` and `baselines/run.py`.

Raw/fused datasets, trained weights, fitted classifier banks, prediction arrays and generated reports are **not distributed in this branch**. Supply source data to fit models, or supply compatible retained artifacts. Earlier Git history is unchanged; the manuscript's pinned source revision remains available there.

## Method

HE-TGT combines a typed graph Transformer with a multiclass history classifier and class-pair refinement. History contains at most eight **strictly earlier** observations from the same directed endpoint pair within 30 seconds. Simultaneous observations cannot enter one another's histories. The development-selected coefficients are fixed: graph/history weights `0.75/0.25`, pair-tree weight `1.0`, and pair-refinement strength `0.5`.

Final class probabilities are multiplied by a **development-fitted class-to-reference table**. Test/query labels are never inputs to this lookup. The reference targets are scenario-associated node sets; ranking measures agreement with those sets.

## Install the research environment

Use Python 3.10 or later. The retained experiment environment used Python 3.12, PyTorch 2.7.0 and the scientific-library versions in `requirements.txt`. Install a suitable PyTorch build for your environment, then the remaining dependencies:

```bash
git clone https://github.com/pengslin6/IGCPS.git
cd IGCPS
python -m pip install torch==2.7.0
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For **Jetson Nano inference**, use the Jetson instructions below. These research dependency pins are not an installation recipe for the Nano's JetPack environment.

## Prepare the data

```bash
python data_processing/build_igcps.py --help
python data_processing/build_tecupsec.py --help
```

IGCPS fusion consumes a physical observation CSV, network feature CSV and corresponding packet capture; TShark is required. TE-CUP-SEC fusion consumes process recordings and the packet-feature cache. Supply paths through the command-line options and write fused tables outside tracked source directories, for example in `data/fused/`.

Starting from the paper's fused tables:

```bash
python -m he_tgt.prepare --dataset IGCPS --csv data/fused/sr_com_causal_v2.csv --output data/prepared
python -m he_tgt.prepare --dataset TE-CUP-SEC --csv data/fused/tecupsec_information_physical_causal_v3.csv --output data/prepared
```

The default checks the source hashes in `configs/`. Use `--allow-new-data` for an explicitly new-data run; that run does not reproduce the paper's source recording. Each output dataset directory must be new.

Class-wise chronological partitions are 60% training, 20% validation and 20% test. Final fitting uses the first 80% as development data. IGCPS has no class cap. TE-CUP-SEC caps training/development at 1,000, validation at 500 and test at 4,000 windows per class, selecting evenly through each chronological class block; smaller classes retain all windows.

Prepared outputs include graph caches, adjacency, node/class metadata, split indices, preprocessing and aligned history arrays. History fitting uses development context. Test history may include already-observed test inputs, always strictly earlier than the query. Query labels and future observations are not history features.

## Fit and evaluate HE-TGT

```bash
python -m he_tgt.train --dataset IGCPS --data data/prepared --output runs/he_tgt --seeds 11 22 33 44 55
python -m he_tgt.train --dataset TE-CUP-SEC --data data/prepared --output runs/he_tgt --seeds 11 22 33 44 55

python -m he_tgt.evaluate --data data/prepared/IGCPS --model runs/he_tgt/IGCPS --output runs/eval_igcps
python -m he_tgt.evaluate --data data/prepared/TE-CUP-SEC --model runs/he_tgt/TE-CUP-SEC --output runs/eval_tecupsec
```

The history bank is fitted once and shared across graph seeds. Graph learning rate/epochs are `0.002/17` for IGCPS and `0.0007/16` for TE-CUP-SEC. The graph initializer preserves the random-stream sequence used for the retained checkpoints. `--device` accepts a PyTorch device such as `cuda`; `auto` selects an available accelerator. Existing run directories are not overwritten. Fitting reads development arrays only; evaluation reports mean and sample standard deviation across fits.

Fixed component removals reuse fitted models:

```bash
python -m he_tgt.evaluate --data data/prepared/IGCPS --model runs/he_tgt/IGCPS --output runs/components_igcps --variants full without_pair pair_only graph_only
```

`pair_only` removes the multiclass history blend and retains the history-based pair trees. It does not remove all history.

Generate class-wise summaries and diagnostic plots from the five saved prediction files:

```bash
python -m he_tgt.diagnostics --metadata runs/he_tgt/IGCPS/metadata.json --predictions runs/eval_igcps/full_seed11.npz runs/eval_igcps/full_seed22.npz runs/eval_igcps/full_seed33.npz runs/eval_igcps/full_seed44.npz runs/eval_igcps/full_seed55.npz --output runs/diagnostics_igcps
```

Confusion matrices sum **prediction instances across graph fits**. With five seeds, each true-class row totals five times its per-fit query count, not five times as many independent observations. Class-wise metrics are percentages; reliability diagrams use pooled prediction bins.

## Run the comparison models

```bash
python -m baselines.run --dataset IGCPS --csv data/fused/sr_com_causal_v2.csv --output runs/baselines_igcps
python -m baselines.run --dataset TE-CUP-SEC --csv data/fused/tecupsec_information_physical_causal_v3.csv --output runs/baselines_tecupsec
```

All nine models and five seeds run by default. For a subset, add `--models GCN-AE VGAE --seeds 11`. Per-model epoch settings are fixed in `configs/`, not selected on test results. These adapters process current-state graphs; the main comparison concerns complete fitted detection pipelines. Their shared reference-scoring adapter is retained separately from HE-TGT's probability-based reference lookup.

`analysis/paired_statistics.py` compares the seed-level `runs.csv` files. Supply an explicit baseline mapping for accuracy, macro F1 (`f1`) and AUC; see `python analysis/paired_statistics.py --help`. It reports paired tests and Holm correction across these three detection metrics.

## Run on NVIDIA Jetson Nano B01

### 1. Prepare a matching JetPack/PyTorch environment

Use a JetPack release supporting the original Nano and a corresponding NVIDIA PyTorch build. Check the installed release with `cat /etc/nv_tegra_release`. Consult [NVIDIA's installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) and [L4T PyTorch container compatibility table](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

For example, **JetPack 4.6.1 / L4T R32.7.1** has the NVIDIA container `r32.7.1-pth1.10-py3`. On a device with that matching release, run from the repository directory:

```bash
sudo docker run --rm -it --runtime nvidia \
  -v "$PWD":/workspace/IGCPS -w /workspace/IGCPS \
  nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3 bash

python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -m he_tgt.edge --smoke-test --device cuda
```

Select a compatible wheel/container for other releases; do not replace the NVIDIA build with the research-environment install command. The `edge.py` path requires only **NumPy and PyTorch**, with Python 3.6-compatible syntax. The smoke test checks graph execution with random weights and synthetic input; it is not a trained-model result.

### 2. Export fitted artifacts

In the research environment, export one fitted seed and optionally a prepared query batch:

```bash
python -m he_tgt.export_jetson --model runs/he_tgt/IGCPS --seed 11 --output artifacts/jetson_igcps --graph data/prepared/IGCPS/test.pt --history data/prepared/IGCPS/history_test.npz --queries 128
```

The bundle contains `model.json`, `graph.npz` and `trees.npz`; optional queries are saved as `input.npz`. Transfer the repository and the locally generated bundle to Jetson. Generated artifacts remain excluded from Git.

Export preserves the graph weights and converts numeric history-tree parameters to NumPy arrays, avoiding serialized scikit-learn model loading across different library environments. The fixed pair-tree weight is `1.0`, so the pair stage uses its fitted tree probabilities directly. This path does not use ONNX, TensorRT conversion or FP16 quantization.

### 3. Run fitted inference

Inside the matching Jetson environment, from the repository directory:

```bash
python3 -m he_tgt.edge --bundle artifacts/jetson_igcps --input artifacts/jetson_igcps/input.npz --output artifacts/jetson_igcps/predictions.npz --device cuda --batch-size 1
```

Outputs include class probabilities, predictions, graph probabilities, reference-node scores and descending reference order. For new observations, supply aligned float32 `graph` arrays of shape `[queries, nodes, input_dim]` and `history` arrays of shape `[queries, 4 * raw_feature_width + 3]`. Use the fitted transforms, node order, topology and strictly-past history construction. This command consumes prepared inputs; continuous acquisition and site-specific alarm delivery belong to the surrounding application.

Check exported predictions against the research pipeline after changing the device or library stack. The export path has been numerically checked against retained predictions in the development environment. Execution and performance should be verified on the target Jetson; these instructions report no new hardware benchmark measurements.
