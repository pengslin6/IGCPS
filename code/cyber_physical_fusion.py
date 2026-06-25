"""
Cyber-physical data alignment with fault-aware adaptive interpolation.

For each network record at time t_n, interpolates physical features to that exact timestamp:
- Continuous features: cubic spline in smooth regions, linear in transient regions
- Discrete features (switch status): piecewise-constant (zero-order hold)
- Labels: nearest-second lookup
"""
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

GRADIENT_THRESHOLD = 3.0

DISCRETE_KEYWORDS = ['switch', 'controller status']


def is_discrete_col(col_name):
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in DISCRETE_KEYWORDS)


def adaptive_interpolate(timestamps, values, target_timestamps, tau=GRADIENT_THRESHOLD):
    """Fault-aware adaptive interpolation for a single continuous feature."""
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    target_ts = np.asarray(target_timestamps, dtype=np.float64)

    if len(ts) < 3:
        f = interp1d(ts, vals, kind='linear', fill_value='extrapolate')
        return f(target_ts)

    gradients = np.abs(np.diff(vals))
    grad_mean = np.mean(gradients)
    grad_std = np.std(gradients) + 1e-8

    transient_indices = set()
    for idx in np.where(gradients > grad_mean + tau * grad_std)[0]:
        transient_indices.update(range(max(0, idx - 1), min(len(vals), idx + 3)))

    cs = CubicSpline(ts, vals, bc_type='natural')
    interpolated = cs(target_ts)

    if transient_indices:
        f_linear = interp1d(ts, vals, kind='linear', fill_value='extrapolate')
        for i, t in enumerate(target_ts):
            idx = np.searchsorted(ts, t) - 1
            idx = max(0, min(idx, len(vals) - 2))
            if idx in transient_indices or (idx + 1) in transient_indices:
                interpolated[i] = f_linear(t)

    return interpolated


def piecewise_constant(timestamps, values, target_timestamps):
    """Zero-order hold for discrete features."""
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values)
    target_ts = np.asarray(target_timestamps, dtype=np.float64)

    indices = np.searchsorted(ts, target_ts, side='right') - 1
    indices = np.clip(indices, 0, len(vals) - 1)
    return vals[indices]


def align_data():
    phy_file = 'np.csv'
    net_file = 'sr_features.csv'
    output_file = 'sr_com.csv'

    print("=" * 60)
    print("Fault-aware adaptive interpolation alignment")
    print("=" * 60)

    # Load files
    print(f"\n1. Loading files...")
    phy_df = pd.read_csv(phy_file)
    net_df = pd.read_csv(net_file)
    print(f"   Physical: {len(phy_df)} rows, Network: {len(net_df)} rows")

    label_col = phy_df.columns[-1]
    phy_feature_cols = [c for c in phy_df.columns if c not in ['Time', label_col]]
    net_feature_cols = [c for c in net_df.columns if c != 'Time']

    # Parse timestamps
    print(f"\n2. Parsing timestamps...")
    phy_df['Time'] = pd.to_datetime(phy_df['Time'], errors='coerce')
    net_df['Time'] = pd.to_datetime(net_df['Time'], errors='coerce')

    phy_df = phy_df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
    net_df = net_df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)

    print(f"   Physical valid: {len(phy_df)}, Network valid: {len(net_df)}")
    print(f"   Physical range: {phy_df['Time'].min()} to {phy_df['Time'].max()}")
    print(f"   Network range:  {net_df['Time'].min()} to {net_df['Time'].max()}")

    # Overlapping interval
    overlap_start = max(phy_df['Time'].min(), net_df['Time'].min())
    overlap_end = min(phy_df['Time'].max(), net_df['Time'].max())

    if overlap_start >= overlap_end:
        print("   ERROR: No time overlap between files")
        return False

    print(f"   Overlap: {overlap_start} to {overlap_end}")

    # Filter to overlap
    net_df = net_df[(net_df['Time'] >= overlap_start) & (net_df['Time'] <= overlap_end)].reset_index(drop=True)
    phy_df_overlap = phy_df[(phy_df['Time'] >= overlap_start) & (phy_df['Time'] <= overlap_end)].reset_index(drop=True)

    print(f"   After overlap filter: Physical={len(phy_df_overlap)}, Network={len(net_df)}")

    if len(phy_df_overlap) < 2 or len(net_df) == 0:
        print("   ERROR: Insufficient data in overlap")
        return False

    # Convert physical timestamps to numeric (seconds since epoch)
    phy_ts = phy_df_overlap['Time'].astype(np.int64) / 1e9
    net_ts = net_df['Time'].astype(np.int64) / 1e9

    # Interpolate physical features to network timestamps
    print(f"\n3. Interpolating physical features to network timestamps...")
    continuous_cols = [c for c in phy_feature_cols if not is_discrete_col(c)]
    discrete_cols = [c for c in phy_feature_cols if is_discrete_col(c)]
    print(f"   Continuous features ({len(continuous_cols)}): {continuous_cols[:3]}...")
    print(f"   Discrete features ({len(discrete_cols)}): {discrete_cols}")

    interpolated_phy = {}

    for col in continuous_cols:
        vals = phy_df_overlap[col].astype(float).values
        interpolated_phy[col] = adaptive_interpolate(phy_ts.values, vals, net_ts.values)

    for col in discrete_cols:
        vals = phy_df_overlap[col].values
        interpolated_phy[col] = piecewise_constant(phy_ts.values, vals, net_ts.values)

    # Labels via nearest-second lookup
    print(f"\n4. Assigning labels via nearest-second lookup...")
    phy_time_sec = phy_df_overlap['Time'].dt.floor('s')
    net_time_sec = net_df['Time'].dt.floor('s')

    label_by_sec = {}
    for _, row in phy_df_overlap.iterrows():
        sec = row['Time'].floor('s')
        if sec not in label_by_sec:
            label_by_sec[sec] = row[label_col]

    labels = []
    for t_sec in net_time_sec:
        if t_sec in label_by_sec:
            labels.append(label_by_sec[t_sec])
        else:
            # Find nearest physical second
            diffs = abs(phy_time_sec - t_sec)
            nearest_idx = diffs.idxmin()
            labels.append(phy_df_overlap.loc[nearest_idx, label_col])

    # Build output DataFrame
    print(f"\n5. Building output...")
    result = pd.DataFrame()
    result['Time'] = net_df['Time']

    for col in net_feature_cols:
        result[col] = net_df[col].values

    for col in phy_feature_cols:
        result[f'phy_{col}'] = interpolated_phy[col]

    result['phy_label'] = labels

    result.sort_values('Time', inplace=True)
    result.reset_index(drop=True, inplace=True)
    result.to_csv(output_file, index=False)

    print(f"\n   Output: {output_file}")
    print(f"   Shape: {result.shape[0]} rows x {result.shape[1]} cols")
    print(f"   Structure: Time + Network({len(net_feature_cols)}) + Physical({len(phy_feature_cols)}) + Label")

    # Label distribution
    print(f"\n6. Label distribution:")
    for lbl, cnt in result['phy_label'].value_counts().items():
        print(f"   {lbl}: {cnt} ({cnt/len(result)*100:.1f}%)")

    print(f"\n   Done!")
    return True


if __name__ == "__main__":
    align_data()
