# Full-pipeline information-matched comparison

This is a retrospective comparison on the existing IGCPS and TE-CUP-SEC partitions. The two architectures are the jointly trained HE-TGT graph backbone and the retained DT-GNN adapter from `experiments_20260928/fair_control`. Their graph fits already received the same graph inputs, supervision masks and weights, two-learning-rate/18-epoch validation budget, development refit rows, and five seeds (11, 22, 33, 44, 55). Native encoders, parameter counts, and readouts remain different.

## Primary locked-policy comparison

For each dataset and seed, take the saved final graph probabilities and node scores from the matched graph experiment. Apply the **same** development-fitted past-only multiclass history classifier and class-pair models to both architectures, with the current HE-TGT policy: history weight 0.25, pair-tree weight 1, pair refinement strength 0.5. This gives DT-GNN the identical history information, history fitting rows, tree parameters, and inference rule. Fit no new model or calibrator and adjust no threshold. Assess macro F1, accuracy, Normal FPR, anomaly recall, NM/Attack6 recall and F1, direct node ranking, and class-probability lookup ranking. All ranking metrics use the identical eligible-row evaluation and the development-fitted class-reference table.

The fixed policy was developed with HE-TGT graph outputs, so it may favor HE-TGT. The primary result is an information-matched **fixed-policy** comparison, not a fully symmetric historical hyperparameter search.

## Equal-policy-budget sensitivity analysis

As a separate sensitivity analysis, give each graph architecture the same 108 validation candidates: 4 pre-fitted pair feature/minimum-leaf settings × 3 pair-tree weights × 3 history weights × 3 pair-refinement strengths. Use the same fixed seven-leaf/60-minimum historical prior, the same pre-fitted pair models, and the same validation rows. Require one common policy across both datasets for each graph architecture. Select by highest smaller-dataset macro-F1 gain over that architecture's own graph stage, then highest mean gain, then lower mean Normal FPR, then lexical configuration name. Lock selections and source hashes **before** reading final test predictions. These policies form a new, symmetric comparison; they are not the published HE-TGT policy.

The retained test tails have been inspected during earlier research. Paired graph seeds and overlapping windows are not independent incidents. Do not present seed intervals as prospective confirmation. Report unsuccessful and unfavorable results. Do not overwrite manuscript tables or source artifacts automatically.
