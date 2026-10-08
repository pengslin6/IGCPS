# Matched node-head removal validation

This is an adaptive retrospective validation experiment. No test arrays or labels are accessed. The decision is limited to whether the learned node head should be retained in the current graph branch; it does not establish independent causal localization.

Two graph models are trained from scratch on the existing chronological training partition and evaluated on validation only. Both use the same type adapters, two local graph blocks, two Transformer layers, classifier, batch size 64, AdamW, weight decay 0.0005, dropout 0.25, input noise 0.02, gradient clipping 1, learning rate 0.0007, 18-epoch cosine schedule, and seeds 11/22/33 on both datasets.

- `with_head`: the current joint graph model. The supervised node loss has weight 0.30. Eligible anomalous references and true-Normal negatives are supervised; attacks without reference annotations are excluded from node loss. Detection uses the detached node-logit softmax for its source-weighted summary.
- `without_head`: deletes the node scorer and all node supervision. Detection uses an independent learned attention pool with the same representation width. It has no direct node-ranking output.

Both variants are compared by detection macro F1, accuracy, Normal FPR, anomaly recall, and NM/Attack6 recall/F1. For a common ranking comparison, predicted class probabilities from each model are mapped through the same training-fitted class-to-reference table; true validation classes and validation reference labels are never inputs to the lookup. The with-head model additionally reports its direct node-head Top-1/MRR/NDCG@5.

Decision rule fixed before results: remove the node head only if `without_head` is non-inferior on both datasets for mean macro F1 (margin -0.25 pp), Normal FPR (no more than +0.50 pp), anomaly recall (no more than -0.50 pp), and focus recall (no more than -2.0 pp), while mean lookup Top-1 is no more than 0.10 pp lower. All three seed-level macro-F1 differences must exceed -1.0 pp. Otherwise retain the node head pending a better design. Because lookup labels are class-associated and fixed within eligible classes, higher lookup ranking does not prove event-specific root-cause identification.
