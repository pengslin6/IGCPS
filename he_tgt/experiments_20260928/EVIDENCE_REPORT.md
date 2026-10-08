# ESWA empirical audit — 28 September 2026

All results below are generated from saved machine-readable outputs. Means and sample standard deviations describe five retained/fitted random seeds, not independent physical incidents. Confidence-stratum membership and classification-correctness membership can differ across seeds. All test-tail analyses remain retrospective.

## 1. Complete label statistics

Each eligible class has exactly one reference-node set, modal-set share 100%, and entropy zero bits in the full recording and every evaluated partition. IGCPS PM/PS/SS use the configured endpoint 10.0.6.118, not the changing source endpoint in each observation. Normal and TE-CUP-SEC Attack1–Attack3 have no eligible node target. Counts are fused observation windows.

### IGCPS

Data SHA-256: `6c8aaea31e4820fee7230ba35e04ce3907edaa5f5a56051cd5ec427d1f5998df`.

| Class | Full | Fit | Validation | Development | Test | Distinct sets | Mode / entropy | Observed source endpoints |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Normal | 2912 | 1747 | 582 | 2329 | 583 | 0 | Not eligible | 15 |
| NS | 238 | 142 | 48 | 190 | 48 | 1 | 100% / 0 bits | 10 |
| NM | 188 | 112 | 38 | 150 | 38 | 1 | 100% / 0 bits | 9 |
| PM | 664 | 398 | 133 | 531 | 133 | 1 | 100% / 0 bits | 12 |
| PS | 97 | 58 | 19 | 77 | 20 | 1 | 100% / 0 bits | 8 |
| SS | 642 | 385 | 128 | 513 | 129 | 1 | 100% / 0 bits | 11 |

Fit/validation preprocessing uses the first 60% of each class for numeric scaling. Final development/test preprocessing uses the first 80%. TE-CUP-SEC cap counts are shown here; uncapped counts are retained in `data_cache/IGCPS/label_statistics.csv`.

| Class | Full-data reference set |
| --- | --- |
| Normal | None |
| NS | Phy_Switch status of high pressure switch; Phy_Switch status of sub-high pressure switch |
| NM | Phy_Switch status of medium pressure skid heater; Phy_Inlet temperature of medium pressure skid heater; Phy_Outlet temperature of medium pressure skid heater; Phy_Outlet temperature of medium pressure skid sensor; Phy_Outlet pressure of medium pressure skid sensor |
| PM | IP_10.0.6.118 |
| PS | IP_10.0.6.118 |
| SS | IP_10.0.6.118 |

### TE-CUP-SEC

Data SHA-256: `092e56fb31a684c16826202c209bb8827af4933e9e7edf046d9888afb55c4c6f`.

| Class | Full | Fit | Validation | Development | Test | Distinct sets | Mode / entropy | Observed source endpoints |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Normal | 238751 | 1000 | 500 | 1000 | 4000 | 0 | Not eligible | 1 |
| Attack1 | 86161 | 1000 | 500 | 1000 | 4000 | 0 | Not eligible | 1 |
| Attack2 | 89581 | 1000 | 500 | 1000 | 4000 | 0 | Not eligible | 1 |
| Attack3 | 84450 | 1000 | 500 | 1000 | 4000 | 0 | Not eligible | 4 |
| Attack4 | 84606 | 1000 | 500 | 1000 | 4000 | 1 | 100% / 0 bits | 1 |
| Attack5 | 86377 | 1000 | 500 | 1000 | 4000 | 1 | 100% / 0 bits | 1 |
| Attack6 | 102568 | 1000 | 500 | 1000 | 4000 | 1 | 100% / 0 bits | 2 |
| Attack7 | 10793 | 1000 | 500 | 1000 | 2159 | 1 | 100% / 0 bits | 1 |
| Attack8 | 13077 | 1000 | 500 | 1000 | 2616 | 1 | 100% / 0 bits | 6 |

Fit/validation preprocessing uses the first 60% of each class for numeric scaling. Final development/test preprocessing uses the first 80%. TE-CUP-SEC cap counts are shown here; uncapped counts are retained in `data_cache/TE-CUP-SEC/label_statistics.csv`.

| Class | Full-data reference set |
| --- | --- |
| Normal | None |
| Attack1 | None |
| Attack2 | None |
| Attack3 | None |
| Attack4 | Phy_xmv3; Phy_xmv4 |
| Attack5 | Phy_xmv10 |
| Attack6 | Phy_xmv11 |
| Attack7 | Phy_xmeas15 |
| Attack8 | IP_192.168.6.2 |

## 2. Replay verification and paired subgroup analysis

All ten retained checkpoints reproduced all six archived aggregate metrics (tolerance 0.0001 percentage points). The IGCPS archived prediction arrays also match. Lookup frequencies are fitted only on eligible development labels; test-time lookup uses predicted class probabilities, not true classes. IGCPS uses its retained adjustment for both lookup and subgroup assignment. Ranking uses the retained evaluator; tie-aware Top-1 is separately recorded.

### IGCPS

#### Top-1 (%)

| Subgroup | Eligible windows | Head | Lookup | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| all | 368 | 86.68 ± 3.17 | 89.46 ± 3.53 | -2.77 ± 2.30 |
| class_correct | 311–338 | 95.96 ± 1.55 | 100.00 ± 0.00 | -4.04 ± 1.55 |
| class_wrong | 30–57 | 24.37 ± 9.46 | 19.00 ± 12.04 | 5.37 ± 6.63 |
| class:NS | 48 | 65.83 ± 17.28 | 76.25 ± 16.11 | -10.42 ± 17.37 |
| class:NM | 38 | 14.21 ± 15.19 | 27.89 ± 16.29 | -13.68 ± 8.61 |
| class:PM | 133 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| class:PS | 20 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| class:SS | 129 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| confidence:0.0-0.6 | 3–11 | 37.62 ± 13.08 | 65.42 ± 13.55 | -27.80 ± 19.25 |
| confidence:0.6-0.8 | 11–43 | 60.75 ± 27.01 | 72.92 ± 20.78 | -12.17 ± 21.13 |
| confidence:0.8-0.9 | 9–25 | 55.54 ± 13.19 | 58.73 ± 12.46 | -3.19 ± 13.67 |
| confidence:0.9-1.0 | 290–343 | 91.18 ± 4.10 | 92.65 ± 3.78 | -1.47 ± 1.12 |
| NM_to_PM | 17–33 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

#### MRR (%)

| Subgroup | Eligible windows | Head | Lookup | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| all | 368 | 93.34 ± 1.59 | 94.59 ± 1.75 | -1.25 ± 1.10 |
| class_correct | 311–338 | 97.98 ± 0.78 | 100.00 ± 0.00 | -2.02 ± 0.78 |
| class_wrong | 30–57 | 62.18 ± 4.73 | 58.32 ± 6.80 | 3.86 ± 3.31 |
| class:NS | 48 | 82.92 ± 8.64 | 88.12 ± 8.06 | -5.21 ± 8.68 |
| class:NM | 38 | 57.11 ± 7.59 | 62.63 ± 8.76 | -5.53 ± 6.34 |
| class:PM | 133 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| class:PS | 20 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| class:SS | 129 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| confidence:0.0-0.6 | 3–11 | 68.81 ± 6.54 | 82.71 ± 6.77 | -13.90 ± 9.63 |
| confidence:0.6-0.8 | 11–43 | 80.37 ± 13.51 | 86.46 ± 10.39 | -6.09 ± 10.57 |
| confidence:0.8-0.9 | 9–25 | 77.77 ± 6.59 | 79.37 ± 6.23 | -1.59 ± 6.83 |
| confidence:0.9-1.0 | 290–343 | 95.59 ± 2.05 | 96.18 ± 1.96 | -0.59 ± 0.40 |
| NM_to_PM | 17–33 | 50.00 ± 0.00 | 48.25 ± 3.09 | 1.75 ± 3.09 |

#### NDCG@5 (%)

| Subgroup | Eligible windows | Head | Lookup | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| all | 368 | 95.30 ± 0.90 | 96.32 ± 1.15 | -1.02 ± 0.79 |
| class_correct | 311–338 | 98.36 ± 0.62 | 100.00 ± 0.00 | -1.64 ± 0.62 |
| class_wrong | 30–57 | 74.70 ± 3.07 | 71.44 ± 5.60 | 3.26 ± 3.25 |
| class:NS | 48 | 88.46 ± 5.42 | 92.72 ± 4.94 | -4.26 ± 5.67 |
| class:NM | 38 | 69.03 ± 3.58 | 73.53 ± 6.94 | -4.50 ± 5.21 |
| class:PM | 133 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| class:PS | 20 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| class:SS | 129 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| confidence:0.0-0.6 | 3–11 | 79.91 ± 4.21 | 88.98 ± 4.11 | -9.07 ± 6.45 |
| confidence:0.6-0.8 | 11–43 | 84.57 ± 6.95 | 91.30 ± 6.48 | -6.72 ± 7.42 |
| confidence:0.8-0.9 | 9–25 | 83.60 ± 4.69 | 86.93 ± 3.81 | -3.33 ± 6.08 |
| confidence:0.9-1.0 | 290–343 | 96.97 ± 1.28 | 97.32 ± 1.40 | -0.34 ± 0.31 |
| NM_to_PM | 17–33 | 66.08 ± 0.00 | 63.40 ± 4.75 | 2.69 ± 4.75 |

#### Tie sensitivity

| Subgroup | Head expected Top-1 | Lookup expected Top-1 |
| --- | --- | --- |
| all | 86.68 ± 3.17 | 89.46 ± 3.53 |
| class_correct | 95.96 ± 1.55 | 100.00 ± 0.00 |
| class_wrong | 24.37 ± 9.46 | 19.00 ± 12.04 |
| class:NS | 65.83 ± 17.28 | 76.25 ± 16.11 |
| class:NM | 14.21 ± 15.19 | 27.89 ± 16.29 |
| class:PM | 100.00 ± 0.00 | 100.00 ± 0.00 |
| class:PS | 100.00 ± 0.00 | 100.00 ± 0.00 |
| class:SS | 100.00 ± 0.00 | 100.00 ± 0.00 |
| confidence:0.0-0.6 | 37.62 ± 13.08 | 65.42 ± 13.55 |
| confidence:0.6-0.8 | 60.75 ± 27.01 | 72.92 ± 20.78 |
| confidence:0.8-0.9 | 55.54 ± 13.19 | 58.73 ± 12.46 |
| confidence:0.9-1.0 | 91.18 ± 4.10 | 92.65 ± 3.78 |
| NM_to_PM | 0.00 ± 0.00 | 0.00 ± 0.00 |

### TE-CUP-SEC

#### Top-1 (%)

| Subgroup | Eligible windows | Head | Lookup | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| all | 16,775 | 83.09 ± 1.83 | 83.70 ± 1.62 | -0.61 ± 1.55 |
| class_correct | 12,364–12,710 | 97.34 ± 1.32 | 100.00 ± 0.00 | -2.66 ± 1.32 |
| class_wrong | 4,065–4,411 | 41.69 ± 4.46 | 36.42 ± 4.86 | 5.27 ± 4.84 |
| class:Attack4 | 4,000 | 99.99 ± 0.01 | 100.00 ± 0.00 | -0.01 ± 0.01 |
| class:Attack5 | 4,000 | 93.28 ± 4.03 | 95.36 ± 2.98 | -2.08 ± 2.22 |
| class:Attack6 | 4,000 | 59.88 ± 12.53 | 54.79 ± 13.44 | 5.08 ± 5.32 |
| class:Attack7 | 2,159 | 57.57 ± 7.57 | 67.80 ± 14.13 | -10.23 ± 8.83 |
| class:Attack8 | 2,616 | 98.23 ± 0.14 | 98.26 ± 0.19 | -0.03 ± 0.16 |
| confidence:0.0-0.6 | 2,740–4,048 | 66.87 ± 2.26 | 66.43 ± 4.04 | 0.44 ± 4.80 |
| confidence:0.6-0.8 | 187–608 | 55.99 ± 15.66 | 54.70 ± 27.57 | 1.29 ± 27.09 |
| confidence:0.8-0.9 | 102–312 | 26.16 ± 10.23 | 42.62 ± 22.14 | -16.46 ± 20.47 |
| confidence:0.9-1.0 | 12,017–13,470 | 89.09 ± 2.98 | 89.56 ± 3.00 | -0.47 ± 0.49 |

#### MRR (%)

| Subgroup | Eligible windows | Head | Lookup | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| all | 16,775 | 89.63 ± 2.10 | 91.53 ± 0.55 | -1.90 ± 1.88 |
| class_correct | 12,364–12,710 | 98.65 ± 0.66 | 100.00 ± 0.00 | -1.35 ± 0.66 |
| class_wrong | 4,065–4,411 | 63.48 ± 6.12 | 66.96 ± 1.40 | -3.48 ± 5.76 |
| class:Attack4 | 4,000 | 100.00 ± 0.01 | 100.00 ± 0.00 | -0.01 ± 0.01 |
| class:Attack5 | 4,000 | 96.21 ± 2.08 | 97.44 ± 1.53 | -1.23 ± 1.24 |
| class:Attack6 | 4,000 | 73.29 ± 11.50 | 77.32 ± 6.79 | -4.03 ± 5.81 |
| class:Attack7 | 2,159 | 77.39 ± 4.89 | 82.37 ± 8.43 | -4.99 ± 4.97 |
| class:Attack8 | 2,616 | 98.81 ± 0.10 | 98.86 ± 0.15 | -0.04 ± 0.16 |
| confidence:0.0-0.6 | 2,740–4,048 | 81.71 ± 1.74 | 81.89 ± 2.51 | -0.19 ± 3.00 |
| confidence:0.6-0.8 | 187–608 | 76.62 ± 8.73 | 76.90 ± 13.68 | -0.28 ± 14.17 |
| confidence:0.8-0.9 | 102–312 | 60.88 ± 3.07 | 71.03 ± 10.83 | -10.16 ± 10.55 |
| confidence:0.9-1.0 | 12,017–13,470 | 92.63 ± 2.97 | 94.77 ± 1.51 | -2.13 ± 1.54 |

#### NDCG@5 (%)

| Subgroup | Eligible windows | Head | Lookup | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| all | 16,775 | 91.82 ± 2.23 | 93.73 ± 0.39 | -1.91 ± 2.04 |
| class_correct | 12,364–12,710 | 99.00 ± 0.49 | 100.00 ± 0.00 | -1.00 ± 0.49 |
| class_wrong | 4,065–4,411 | 71.05 ± 7.07 | 75.55 ± 0.99 | -4.50 ± 6.70 |
| class:Attack4 | 4,000 | 100.00 ± 0.00 | 100.00 ± 0.00 | -0.00 ± 0.00 |
| class:Attack5 | 4,000 | 97.18 ± 1.55 | 98.10 ± 1.13 | -0.91 ± 0.93 |
| class:Attack6 | 4,000 | 78.18 ± 10.96 | 83.25 ± 5.01 | -5.07 ± 6.85 |
| class:Attack7 | 2,159 | 83.24 ± 3.67 | 86.92 ± 6.30 | -3.68 ± 3.71 |
| class:Attack8 | 2,616 | 99.08 ± 0.11 | 99.14 ± 0.11 | -0.06 ± 0.15 |
| confidence:0.0-0.6 | 2,740–4,048 | 86.40 ± 1.31 | 86.57 ± 1.88 | -0.17 ± 2.24 |
| confidence:0.6-0.8 | 187–608 | 82.68 ± 6.50 | 82.93 ± 10.09 | -0.25 ± 10.49 |
| confidence:0.8-0.9 | 102–312 | 70.99 ± 2.18 | 78.60 ± 7.98 | -7.61 ± 7.86 |
| confidence:0.9-1.0 | 12,017–13,470 | 93.94 ± 3.06 | 96.14 ± 1.12 | -2.20 ± 2.12 |

#### Tie sensitivity

| Subgroup | Head expected Top-1 | Lookup expected Top-1 |
| --- | --- | --- |
| all | 83.09 ± 1.83 | 83.70 ± 1.62 |
| class_correct | 97.34 ± 1.32 | 100.00 ± 0.00 |
| class_wrong | 41.69 ± 4.46 | 36.42 ± 4.86 |
| class:Attack4 | 99.99 ± 0.01 | 100.00 ± 0.00 |
| class:Attack5 | 93.28 ± 4.03 | 95.36 ± 2.98 |
| class:Attack6 | 59.88 ± 12.53 | 54.79 ± 13.44 |
| class:Attack7 | 57.57 ± 7.57 | 67.80 ± 14.13 |
| class:Attack8 | 98.23 ± 0.14 | 98.26 ± 0.19 |
| confidence:0.0-0.6 | 66.87 ± 2.26 | 66.43 ± 4.04 |
| confidence:0.6-0.8 | 55.99 ± 15.66 | 54.70 ± 27.57 |
| confidence:0.8-0.9 | 26.16 ± 10.23 | 42.62 ± 22.14 |
| confidence:0.9-1.0 | 89.09 ± 2.98 | 89.56 ± 3.00 |

## 3. Joint-training control

The pre-run protocol is in [PROTOCOL.md](PROTOCOL.md). Both native encoders and readouts are trained end to end using the same inputs, splits, node targets, eligibility masks, loss coefficients, optimizer, batch order, training perturbation, validation budget, and final seeds. Native dropout and parameter counts remain different. This compares the retained DA-TGT architecture with the local DT-GNN adapter under matched optimization; it is not a faithful reproduction of the source method's original application, a parameter-matched test, or a pure node-head ablation.

Each model receives two learning-rate candidates (0.0007 and 0.002), 18 epochs per candidate, and validation seed 42. The locked selection criterion is the harmonic mean of validation macro F1 and node Top-1, with AUC, earlier epoch, and lower learning rate as tie-breakers. Five seeds (11, 22, 33, 44, 55) are then fitted on the development partition. No test result changes the selected configuration. Previously inspected test tails and retrospective comparator selection still prevent confirmatory claims.

Unlike the archived pipeline, both newly fitted models omit post-forward probability adjustment. The common node objective excludes unannotated TE attacks even from the negative term; only true Normal windows receive normal-node BCE. These are new control fits, not replacement values for the archived main tables.

### IGCPS

| Model | Selected learning rate | Selected epoch | Trainable parameters |
| --- | --- | --- | --- |
| DA-TGT | 0.002 | 17 | 152362 |
| DT-GNN | 0.002 | 15 | 61000 |

| Model | accuracy | f1 | auc | top1 | mrr | ndcg5 |
| --- | --- | --- | --- | --- | --- | --- |
| DA-TGT | 93.04 ± 0.55 | 80.61 ± 2.57 | 98.92 ± 0.40 | 90.16 ± 2.77 | 95.08 ± 1.39 | 96.59 ± 0.87 |
| DT-GNN | 92.37 ± 1.14 | 77.90 ± 5.09 | 99.17 ± 0.40 | 40.92 ± 7.05 | 55.48 ± 3.42 | 56.66 ± 2.35 |

| Metric | Paired DA-TGT − DT-GNN (pp) | Seed-based t interval (95%) |
| --- | --- | --- |
| accuracy | 0.67 ± 1.57 | [-1.27, 2.62] |
| f1 | 2.71 ± 5.97 | [-4.71, 10.13] |
| auc | -0.26 ± 0.56 | [-0.95, 0.44] |
| top1 | 49.24 ± 5.27 | [42.69, 55.78] |
| mrr | 39.60 ± 2.74 | [36.19, 43.00] |
| ndcg5 | 39.93 ± 2.20 | [37.20, 42.66] |

The intervals summarize variation across five paired training seeds on the same retained test rows. They do not quantify uncertainty across new incidents/plants and are not confirmatory superiority tests.

| Model | Seed | Accuracy | Macro F1 | Top-1 | Fit seconds |
| --- | --- | --- | --- | --- | --- |
| DA-TGT | 11 | 93.17 | 81.07 | 90.22 | 143.47 |
| DA-TGT | 22 | 92.74 | 79.90 | 92.39 | 143.01 |
| DA-TGT | 33 | 92.32 | 77.11 | 85.60 | 143.22 |
| DA-TGT | 44 | 93.17 | 84.26 | 92.39 | 146.44 |
| DA-TGT | 55 | 93.80 | 80.71 | 90.22 | 144.46 |
| DT-GNN | 11 | 92.01 | 76.47 | 47.01 | 66.33 |
| DT-GNN | 22 | 91.59 | 77.86 | 39.40 | 97.32 |
| DT-GNN | 33 | 93.90 | 83.26 | 29.35 | 97.32 |
| DT-GNN | 44 | 93.17 | 81.68 | 45.11 | 98.94 |
| DT-GNN | 55 | 91.17 | 70.24 | 43.75 | 98.84 |

#### Same-pipeline lookup and tie sensitivity

The lookup below uses each newly trained model's own class probabilities and the same development-fitted reference table. It is distinct from the archived calibrated lookup above. No extra training or model selection is involved.

| New joint model | Head Top-1 | Own-probability lookup Top-1 | Head expected Top-1 under ties | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| DA-TGT | 90.16 ± 2.77 | 91.30 ± 2.15 | 90.16 ± 2.77 | -1.14 ± 0.78 |
| DT-GNN | 40.92 ± 7.05 | 92.45 ± 2.58 | 41.52 ± 7.03 | -51.52 ± 8.12 |

#### New-control class-specific detection

| Model | Class | Test windows | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- |
| DA-TGT | Normal | 583 | 95.37 ± 1.35 | 97.32 ± 0.65 |
| DA-TGT | NS | 48 | 92.08 ± 11.45 | 95.57 ± 6.48 |
| DA-TGT | NM | 38 | 25.79 ± 10.09 | 28.89 ± 9.90 |
| DA-TGT | PM | 133 | 97.59 ± 5.38 | 87.81 ± 2.91 |
| DA-TGT | PS | 20 | 80.00 ± 14.14 | 75.44 ± 14.96 |
| DA-TGT | SS | 129 | 100.00 ± 0.00 | 98.64 ± 1.24 |
| DT-GNN | Normal | 583 | 95.81 ± 0.60 | 97.00 ± 0.50 |
| DT-GNN | NS | 48 | 94.17 ± 9.13 | 96.80 ± 5.08 |
| DT-GNN | NM | 38 | 34.21 ± 17.65 | 36.04 ± 15.60 |
| DT-GNN | PM | 133 | 93.23 ± 4.78 | 86.25 ± 3.14 |
| DT-GNN | PS | 20 | 43.00 ± 10.37 | 55.11 ± 11.12 |
| DT-GNN | SS | 129 | 100.00 ± 0.00 | 96.21 ± 1.06 |

### TE-CUP-SEC

| Model | Selected learning rate | Selected epoch | Trainable parameters |
| --- | --- | --- | --- |
| DA-TGT | 0.0007 | 16 | 152509 |
| DT-GNN | 0.002 | 16 | 60555 |

| Model | accuracy | f1 | auc | top1 | mrr | ndcg5 |
| --- | --- | --- | --- | --- | --- | --- |
| DA-TGT | 83.15 ± 1.66 | 80.62 ± 1.18 | 97.39 ± 0.16 | 82.01 ± 2.38 | 89.08 ± 2.54 | 91.35 ± 2.55 |
| DT-GNN | 81.60 ± 1.05 | 78.23 ± 0.89 | 97.00 ± 0.35 | 63.83 ± 1.57 | 73.50 ± 1.44 | 76.02 ± 1.63 |

| Metric | Paired DA-TGT − DT-GNN (pp) | Seed-based t interval (95%) |
| --- | --- | --- |
| accuracy | 1.55 ± 1.80 | [-0.69, 3.79] |
| f1 | 2.39 ± 1.75 | [0.22, 4.56] |
| auc | 0.39 ± 0.48 | [-0.21, 0.98] |
| top1 | 18.17 ± 1.64 | [16.14, 20.21] |
| mrr | 15.59 ± 2.44 | [12.56, 18.62] |
| ndcg5 | 15.33 ± 2.94 | [11.68, 18.98] |

The intervals summarize variation across five paired training seeds on the same retained test rows. They do not quantify uncertainty across new incidents/plants and are not confirmatory superiority tests.

| Model | Seed | Accuracy | Macro F1 | Top-1 | Fit seconds |
| --- | --- | --- | --- | --- | --- |
| DA-TGT | 11 | 82.72 | 80.13 | 79.95 | 441.74 |
| DA-TGT | 22 | 83.34 | 80.50 | 82.17 | 443.21 |
| DA-TGT | 33 | 85.06 | 81.63 | 79.34 | 445.29 |
| DA-TGT | 44 | 84.03 | 81.86 | 83.62 | 446.65 |
| DA-TGT | 55 | 80.62 | 78.97 | 84.95 | 445.03 |
| DT-GNN | 11 | 80.10 | 77.06 | 63.76 | 218.15 |
| DT-GNN | 22 | 82.51 | 78.42 | 62.21 | 221.07 |
| DT-GNN | 33 | 81.21 | 77.85 | 62.63 | 221.67 |
| DT-GNN | 44 | 82.66 | 78.32 | 64.41 | 221.93 |
| DT-GNN | 55 | 81.52 | 79.49 | 66.17 | 182.70 |

#### Same-pipeline lookup and tie sensitivity

The lookup below uses each newly trained model's own class probabilities and the same development-fitted reference table. It is distinct from the archived calibrated lookup above. No extra training or model selection is involved.

| New joint model | Head Top-1 | Own-probability lookup Top-1 | Head expected Top-1 under ties | Paired head − lookup (pp) |
| --- | --- | --- | --- | --- |
| DA-TGT | 82.01 ± 2.38 | 83.02 ± 1.62 | 82.01 ± 2.38 | -1.02 ± 2.90 |
| DT-GNN | 63.83 ± 1.57 | 78.97 ± 1.12 | 66.17 ± 1.56 | -15.13 ± 1.07 |

#### New-control class-specific detection

| Model | Class | Test windows | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- |
| DA-TGT | Normal | 4000 | 74.30 ± 11.47 | 63.49 ± 7.11 |
| DA-TGT | Attack1 | 4000 | 100.00 ± 0.00 | 99.95 ± 0.06 |
| DA-TGT | Attack2 | 4000 | 100.00 ± 0.00 | 99.88 ± 0.26 |
| DA-TGT | Attack3 | 4000 | 100.00 ± 0.00 | 100.00 ± 0.01 |
| DA-TGT | Attack4 | 4000 | 100.00 ± 0.01 | 99.93 ± 0.06 |
| DA-TGT | Attack5 | 4000 | 95.62 ± 2.21 | 97.50 ± 1.32 |
| DA-TGT | Attack6 | 4000 | 10.79 ± 5.97 | 15.27 ± 7.50 |
| DA-TGT | Attack7 | 2159 | 67.54 ± 8.91 | 76.29 ± 4.36 |
| DA-TGT | Attack8 | 2616 | 98.16 ± 0.14 | 73.26 ± 3.59 |
| DT-GNN | Normal | 4000 | 77.14 ± 10.36 | 62.58 ± 5.32 |
| DT-GNN | Attack1 | 4000 | 100.00 ± 0.00 | 99.97 ± 0.07 |
| DT-GNN | Attack2 | 4000 | 99.92 ± 0.10 | 99.95 ± 0.04 |
| DT-GNN | Attack3 | 4000 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| DT-GNN | Attack4 | 4000 | 99.98 ± 0.01 | 99.86 ± 0.06 |
| DT-GNN | Attack5 | 4000 | 93.06 ± 2.30 | 91.66 ± 6.87 |
| DT-GNN | Attack6 | 4000 | 4.50 ± 5.81 | 6.32 ± 7.11 |
| DT-GNN | Attack7 | 2159 | 55.16 ± 3.24 | 69.13 ± 1.62 |
| DT-GNN | Attack8 | 2616 | 98.23 ± 0.31 | 74.59 ± 6.14 |

## 4. Evidence scope and artifacts

No new initiating-node or propagation-path annotations were created. No Jetson or end-to-end latency measurement was run. No architecture redesign was performed. The DT-GNN adapter was made jointly trainable for this control only, preserving its original evaluation-time outputs before fitting.

- `data_cache/<dataset>/`: exhaustive reference frequencies, split statistics and indices, cached tensors, hashes and node identities.
- `archived_analysis/<dataset>/`: replay verification, predictions, raw/adjusted detection metrics, confusion matrices, per-seed and aggregate subgroup results.
- `fair_control/<dataset>/`: locked selections, validation histories, implementation checks, final checkpoints/predictions, per-class metrics and paired summaries.
- `code_snapshot/`: unchanged copies of the retained experimental implementation.
- `analysis_progress.log`, `fair_progress.log`, and `parallel_*.log`: execution logs. The serial run was stopped after completed IGCPS fits were saved; unfinished fits were restarted with the unchanged fitting routine in independent processes. Completed results were reused, and no result-driven protocol change was made.

## 5. Reproduction

Run `analyze_archived.py --stage all`, then `fair_control.py`, then `verify_experiment_outputs.py` and `build_evidence_report.py` from this directory using the recorded PyTorch 2.7.0 CPU environment. As an equivalent scheduling option, use `parallel_fair_control.py --dataset IGCPS` and `parallel_fair_control.py --dataset TE-CUP-SEC`; never run the serial and parallel coordinators on the same dataset simultaneously. The parallel coordinator calls the unchanged serial fitting routine and reuses its final aggregation. The original CSV/checkpoint locations are declared in `common.py`; source hashes are checked before preparation. Existing completed caches/runs are reused. Do not edit the locked protocol or training code when resuming an existing fair-control run.

Status: ALL THREE EMPIRICAL TASKS COMPLETE.
