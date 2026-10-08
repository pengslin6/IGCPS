# Reviewer-requested empirical audit, 28 September 2026

This protocol is fixed before running the new jointly trained comparison. The
existing temporal test tails have been inspected previously. All new results
remain retrospective, and no claim of a pristine confirmatory test is made.

## Data and reference-label audit

Use the two fusion CSV files identified by the prior experiment manifest and
verify their SHA-256 hashes. Preserve the current reference-label policy and the
classwise chronological 60/20/20 selection split and 80/20 final refit split.
Retain the documented TE class caps (1000 fitting, 500 validation, 4000 test).
Report full-recording and partition-specific label-set frequencies, distinct
sets, modal-set shares, conditional set entropy, and endpoint variability.
Counts refer to fused observation windows, not independently annotated events.

## Archived model subgroup analysis

Replay the five retained checkpoints (11, 22, 33, 44, 55) and verify their metrics
against the retained reports before using predictions. Compare the learned head
with the development-fitted class-probability lookup on exactly the same rows.
Report each class, correctly/incorrectly classified eligible rows, NM-to-PM
errors, and fixed confidence bands [0, .6), [.6, .8), [.8, .9), [.9, 1].
Use the published ranking rule and also record tie-aware Top-1 sensitivity.
Summarize paired seed differences without treating repeated windows or seeds as
independent physical incidents. Audit raw and adjusted IGCPS class predictions.

## Joint-training comparison

Compare the retained DA-TGT architecture with the existing DT-GNN adapter.
DT-GNN is chosen before this run because it is the strongest retained IGCPS
adapter in accuracy, AUC, and all three ranking metrics. Use the same comparator
on both datasets. This choice uses prior results and is retrospective.

Keep the native encoders and readouts. DT-GNN's existing attention node head is
connected to the encoder throughout optimization. Its batched implementation
must reproduce the original single-graph forward outputs with dropout disabled.
This is a comparison under matched optimization, not identical architectures,
identical parameter counts, or a reproduction of the original DTFL application.

For both architectures use the same features, topology, labels, masks, class
weights, recency weights, batches (64), optimizer (AdamW), weight decay (0.0005),
gradient clipping (1), and cosine schedule (18-epoch horizon). Apply input noise
with standard deviation 0.02 to both. Preserve native architectural dropout.
The loss is class-weighted, recency-weighted cross entropy plus 0.30 times the
same positively weighted node BCE. Add 0.03 times normal-row node BCE inside the
node objective. Unannotated attacks are excluded from node supervision, including
the negative term. Both architectures train end to end, without pretraining,
post-forward calibration, ensembling, or a frozen-encoder stage.

Give each architecture the same validation budget: learning rates 0.0007 and
0.002, 18 epochs each, selection seed 42. Select learning rate and epoch by the
harmonic mean of validation macro F1 and reference Top-1, then validation AUC,
earlier epoch, and lower learning rate to resolve ties. Save the selection lock
before final fitting. Refit both architectures on the same development rows for
five paired seeds (11, 22, 33, 44, 55), then evaluate the retained temporal tails.
Report means, sample SDs, and paired differences. Save histories, checkpoints,
predictions, counts, code/data hashes, and actual elapsed times. Do not change
the protocol in response to test scores or replace the original paper tables.

The common node mask corrects the old loss's treatment of unannotated attacks.
The new uncalibrated DA-TGT control therefore needs its own refits and must be
reported separately from the archived calibrated pipeline.
