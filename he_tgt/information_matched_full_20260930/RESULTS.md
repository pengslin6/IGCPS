# Optional history-branch transfer to the DT-GNN graph backbone

**Role of this study.** The history classifier and pair refinement are components of HE-TGT. Original DT-GNN and the other published comparison adapters are not required to include them. This experiment transplants the HE-TGT history branch onto one alternative graph backbone to probe whether HE-TGT's graph architecture is necessary for its detection gains. It is an additional component-transfer diagnostic, not a mandatory fairness condition for comparing the complete methods. The manuscript's existing jointly trained graph-only comparison already addresses the reviewer's request for at least one optimization-matched localization comparison.

## Scope

The existing matched graph experiment supplied five development-fitted graph models per architecture and dataset (seeds 11, 22, 33, 44, 55). Both architectures received the same graph input and joint-training protocol, including the two-learning-rate/18-epoch validation budget. This study then applied the **same development-fitted past-only history classifier, class-pair classifiers, and probability rule** to both graph probability outputs. Historical features and downstream model fits were shared rather than separately optimized for one architecture. Both datasets used the same fixed HE-TGT policy. All comparisons are retrospective on the retained test rows.

The published HE-TGT probabilities were reproduced exactly from the matched HE-TGT graph predictions plus the locked history and pair models. The saved HE-TGT fixed-policy results are consequently the current manuscript's main results, not a different reimplementation.

## Diagnostic A: current fixed HE-TGT policy applied to both architectures

All values are percentages and five-seed means; the exact seed-level results and SDs are in [results.json](results.json).

| Dataset | Full pipeline | Macro F1 | Normal FPR | NM / Attack6 recall | Direct node Top-1 | Class lookup Top-1 |
|---|---|---:|---:|---:|---:|---:|
| IGCPS | HE-TGT | 88.38 | 3.09 | 41.58 | 90.16 | 93.97 |
| IGCPS | DT-GNN + same history pipeline | 87.92 | 2.95 | 47.89 | 40.92 | 94.62 |
| TE-CUP-SEC | HE-TGT | 84.65 | 17.07 | 24.31 | 82.00 | 84.67 |
| TE-CUP-SEC | DT-GNN + same history pipeline | 86.62 | 19.08 | 25.74 | 63.83 | 87.64 |

HE-TGT's macro-F1 difference relative to the matched DT-GNN pipeline is **+0.46 points on IGCPS** and **−1.97 points on TE-CUP-SEC**. The paired-seed differences do not consistently favor either architecture. On TE, HE-TGT has a lower Normal FPR, while DT-GNN has higher macro F1 and Attack6 recall. On both datasets, HE-TGT's direct learned node ranking is substantially higher than DT-GNN's direct node ranking; however, the DT-GNN full pipeline has the higher class-lookup Top-1 under the fixed reference-label mapping. This comparison does not establish that HE-TGT is uniformly superior in detection or reference ranking.

The fixed probability policy was originally developed with HE-TGT graph outputs. Giving DT-GNN that identical rule matches the deployed information and inference path, but does not give the two graph architectures independently optimized historical policies.

## Diagnostic B: equal final-policy selection budget

Each graph architecture was therefore separately evaluated under the **same 108 pre-fitted validation candidates**, the same selection criterion, and one selected policy shared across the two datasets. Both happened to select the same policy: `network_with_physical_changes`, pair minimum leaf size 60, pair-tree weight 1.0, history weight 0.50, pair-refinement strength 0.50. Selected pair classifiers were then fitted on the same development rows before test evaluation. This changes the current manuscript's history weight from 0.25 to 0.50, so these are **sensitivity results**, not revised HE-TGT main-table values.

| Dataset | Equal-budget full pipeline | Macro F1 | Normal FPR | NM / Attack6 recall | Direct node Top-1 | Class lookup Top-1 |
|---|---|---:|---:|---:|---:|---:|
| IGCPS | HE-TGT graph + selected history | 90.85 | 1.89 | 50.00 | 90.16 | 94.84 |
| IGCPS | DT-GNN graph + selected history | 90.87 | 1.99 | 53.68 | 40.92 | 95.22 |
| TE-CUP-SEC | HE-TGT graph + selected history | 86.94 | 17.80 | 24.08 | 82.00 | 88.66 |
| TE-CUP-SEC | DT-GNN graph + selected history | 87.05 | 19.06 | 25.65 | 63.83 | 88.85 |

Under the symmetric 108-candidate final-policy budget, macro F1 is essentially tied on IGCPS (−0.02 points for HE-TGT) and is 0.11 points lower for HE-TGT on TE. HE-TGT retains its higher direct node-head Top-1 and lower TE Normal FPR. The historical prior and candidate feature family were inherited from earlier HE-TGT development. Thus the final 108-option search is symmetric, but the full earlier research trajectory was not; this sensitivity analysis does not prove complete equality of all development effort.

## Interpretation and audit

- The archived nine-adapter table compares complete implemented methods, including HE-TGT's distinctive history branch. That is a valid system-level comparison when its scope is stated clearly. This transfer experiment addresses the narrower question of whether a different graph backbone could also use the HE-TGT history branch.
- Direct node-head Top-1 and class-lookup Top-1 answer different questions. The lookup uses final history-enhanced class probabilities and a development-fitted class-reference table; the direct node heads use graph embeddings. The reference labels remain class-associated, so neither score verifies an initiating node.
- The history branch improves either graph backbone, but this diagnostic does not demonstrate a general **detection** advantage of the HE-TGT graph architecture once DT-GNN receives the same history branch. It does not invalidate the observed performance of HE-TGT as a complete method against the original adapters. The direct reference-node head remains stronger than the DT-GNN head under the matched graph training comparison.
- Five graph seeds share the same fitted historical classifiers and test rows. There was earlier test-tail inspection and no independent event-held-out confirmation; the equal-budget selection is another retrospective sensitivity analysis.
- [verification.json](verification.json) checks 20 saved fixed-policy prediction files for row identity, probability validity, macro F1, Normal FPR, focus-class recall, and lookup Top-1. It also confirms exact reproduction of the existing five-seed HE-TGT probabilities. No manuscript or response files were changed by this experiment.
