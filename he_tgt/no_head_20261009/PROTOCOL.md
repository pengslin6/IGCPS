# Fixed-setting no-node-head refits

This is the training/evaluation implementation of the 9 October 2026 deletion study, relocated for external artifact paths. It fits two datasets with graph seeds 11, 22, 33, 44, and 55 and replays the ten former-model checkpoints. It does not refit the fixed multiclass history or class-pair banks or the nine detection baselines.

The graph objective is class cross-entropy weighted by the retained class and recency weights. The node scorer is deleted, and independent detection attention pooling replaces node-score-weighted pooling. Graph settings are inherited: IGCPS learning rate 0.002 for 17 epochs; TE-CUP-SEC 0.0007 for 16 epochs; batch size 64; AdamW weight decay 0.0005; cosine T_max 18; minimum learning rate 0.1 times the initial rate; dropout 0.25; input noise 0.02; gradient clipping 1. Both the supervision and pooling change in this comparison.

History probability weight is 0.25, pair-tree weight 1.0, and refinement strength 0.5. Both variants map final class probabilities through the same development-fitted reference table. Reference labels are class-associated investigation targets. No query label is an inference input.

The temporal tails were inspected previously. Five seeds vary graph fitting with fixed history banks. This is a retrospective deletion study, not independent-event or cross-plant validation. Complete all graph fits and save their hashes before evaluating the test caches. The script records fitting duration only; this is not inference latency. Device profiling is a separate operation.

Raw data, graph/history caches, fitted models, predictions and result summaries are external artifacts and are excluded from this source repository. Use a new output root when reproducing the study. Existing original input locks and experimental records must remain unchanged.
