# HGAN-Trace Code

This directory contains the code released with the HGAN-Trace report.

- `cyber_physical_fusion.py`: cyber-physical data alignment and fault-aware adaptive interpolation code.
- `hgan_trace_detection_traceback.py`: anomaly detection, root-cause traceback, evaluation, and visualization code.

Typical workflow:

```bash
python cyber_physical_fusion.py
python hgan_trace_detection_traceback.py --csv sr_com.csv --ablation --outdir traceback_results
```

Full TE-CUP-SEC auxiliary run:

```bash
python hgan_trace_detection_traceback.py --csv combined.csv --epochs 2 --lr 0.002 --outdir trace_TECUPSEC_full_nocap_20260626_001 --ablation --seed 42
```

The scripts expect the required CSV datasets to be available locally.
