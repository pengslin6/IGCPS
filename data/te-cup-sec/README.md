# TE-CUP-SEC Full-Data Results

This directory contains the latest full TE-CUP-SEC auxiliary experiment outputs for HGAN-Trace.

Run configuration:

- Input file: `combined.csv` (not stored in this repository because of file size)
- Records: 801,418
- Classes: Normal + Attack1--Attack8
- Graph nodes: 55 total, including 53 TE process variables and 2 synthetic network placeholders
- Split: 480,847 train / 160,284 validation / 160,287 held-out test
- Command: `python hgan_trace_detection_traceback.py --csv combined.csv --epochs 2 --lr 0.002 --outdir trace_TECUPSEC_full_nocap_20260626_001 --ablation --seed 42`

Files:

- `model_comparison_classification.csv`: detection metrics
- `model_comparison_traceback.csv`: traceback metrics
- `model_comparison_full.csv`: combined detection, traceback, and resource metrics
- `model_size_comparison.csv`: model size and resource metrics

Key results:

- DT-GNN gives the best non-ablated detection F1: 82.43%.
- HGAN-Trace full model gives 84.27% accuracy, 75.44% F1, 25.79% RCA, and 59.09% MRR.
- w/o DW-Sep and w/o Temporal Shift both reach 100.00% RCA/MRR/NDCG@5 in the ablation study.
