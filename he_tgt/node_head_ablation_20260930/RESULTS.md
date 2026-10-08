# Matched node-head removal validation

Decision: retain the node head under the predeclared rule.

Validation only; three matched seeds; no test access. Lookup ranking is class-associated reference ranking, not causal localization.

## IGCPS

|Variant|Macro F1|Lookup Top1|Normal FPR|Anomaly recall|Focus recall|
|---|---:|---:|---:|---:|---:|
|with_head|93.885±4.965|98.543±1.860|0.745±0.358|99.454±0.000|85.965±17.913|
|without_head|96.716±0.594|98.634±1.366|0.515±0.595|99.454±0.000|86.842±13.158|

With-head direct node Top1: 92.259±2.208.
Without-minus-with differences: {"accuracy": 0.5625879043600435, "f1": 2.831366821938417, "normal_fpr": -0.22909507445589916, "anomaly_recall": 0.0, "focus_recall": 0.8771929824561511, "focus_f1": 0.8502470011248562, "top1": 0.09107468123863782, "mrr": 0.0455373406193047, "ndcg5": 0.030888907584113667}
Checks: {"f1": true, "normal_fpr": true, "anomaly_recall": true, "focus_recall": true, "lookup_top1": true, "seed_f1": false}

## TE-CUP-SEC

|Variant|Macro F1|Lookup Top1|Normal FPR|Anomaly recall|Focus recall|
|---|---:|---:|---:|---:|---:|
|with_head|83.735±0.840|91.880±0.183|55.267±20.195|90.717±3.049|42.933±15.300|
|without_head|82.970±1.759|91.360±2.140|51.267±34.207|89.683±6.295|40.800±30.673|

With-head direct node Top1: 91.267±2.080.
Without-minus-with differences: {"accuracy": 0.007407407407413302, "f1": -0.7654576982866246, "normal_fpr": -4.0, "anomaly_recall": -1.0333333333333314, "focus_recall": -2.133333333333333, "focus_f1": -4.556934807072658, "top1": -0.519999999999996, "mrr": -0.18933333333332314, "ndcg5": -0.13629026742975725}
Checks: {"f1": false, "normal_fpr": true, "anomaly_recall": false, "focus_recall": false, "lookup_top1": false, "seed_f1": false}

## Current full-pipeline diagnostic

As a secondary diagnostic, the matched graph probabilities were passed through the existing fixed HE-TGT history blend and class-pair refinement. This diagnostic was not part of the predeclared deletion gate and was not used to retune any coefficient.

|Dataset|Variant|Macro F1|Lookup Top1|Normal FPR|Focus recall|
|---|---|---:|---:|---:|---:|
|IGCPS|with head|94.565|98.452|0.115|85.088|
|IGCPS|without head|96.184|98.452|0.115|85.088|
|TE-CUP-SEC|with head|87.031|93.520|30.800|45.200|
|TE-CUP-SEC|without head|86.361|92.947|30.067|42.667|

The full-pipeline diagnostic reaches the same conservative decision. Removing the head improves IGCPS macro F1, but on TE-CUP-SEC it reduces macro F1 by 0.670 points, lookup Top1 by 0.573 points, anomaly recall by 0.825 points, and Attack6 recall by 2.533 points. Therefore the current evidence does not support removing the node head from the shared two-dataset model.

## Interpretation

Retaining the head does not establish that its direct ranking is superior to the class-probability lookup. The direct head remains an explicit supervised reference-ranking output and supplies detached source weights to the graph detector. The lookup remains a stronger aggregate ranking control in the existing manuscript. These labels are class-associated reference sets and do not independently verify causal initiating nodes.

Independent verification recomputed 24 validation prediction files (12 graph outputs and 12 full-pipeline diagnostics), checked sample order and probability validity, reproduced classification/lookup/direct-node ranking metrics, verified that the no-head state dictionaries contain no `root_scorer`, and confirmed that the run did not access test arrays.
