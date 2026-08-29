"""Build a leakage-free, causally aligned IGCPS cyber-physical table.

The legacy pipeline computes flow statistics over an entire bidirectional flow
and copies the final values back to every packet. It also interpolates physical
features with future samples while assigning labels by a separate second-level
lookup. This script fixes both issues without overwriting the legacy data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\pcsys\Desktop\公开数据集\cps\sr")
SCRIPT_ROOT = Path(__file__).resolve().parent

DEFAULT_PHYSICAL = DATA_ROOT / "np.csv"
DEFAULT_NETWORK = DATA_ROOT / "sr_features.csv"
DEFAULT_PCAP = DATA_ROOT / "attack_1.pcapng"
DEFAULT_LEGACY = SCRIPT_ROOT / "sr_com_new.csv"
DEFAULT_OUTPUT = SCRIPT_ROOT / "sr_com_causal_v2.csv"
DEFAULT_AUDIT = SCRIPT_ROOT / "sr_com_causal_v2_audit.json"
DEFAULT_TSHARK = Path(r"C:\Program Files\Wireshark\tshark.exe")

FLOW_COLUMNS = (
    "Flow Duration",
    "Flow Count per IP Pair",
    "Number of Packets from Source",
    "Number of Packets from Destination",
    "Packet Length Mean",
    "Packet length Std",
    "Inter Arrival Time Mean",
    "Inter Arrival Time Std",
)

FLOW_ID_COLUMNS = (
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol Type",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_times(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert("Asia/Shanghai")


def flow_key(row: pd.Series) -> tuple[str, str, str, str, str]:
    addr1, addr2 = str(row["Source IP"]), str(row["Destination IP"])
    port1, port2 = str(row["Source Port"]), str(row["Destination Port"])
    return (
        min(addr1, addr2),
        max(addr1, addr2),
        min(port1, port2),
        max(port1, port2),
        str(row["Protocol Type"]),
    )


@dataclass
class OnlineMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.m2, 0.0) / self.count) if self.count else 0.0


@dataclass
class FlowState:
    first_time: float
    last_time: float
    first_source: str
    first_destination: str
    packets: int = 0
    source_packets: int = 0
    destination_packets: int = 0
    lengths: OnlineMoments | None = None
    inter_arrivals: OnlineMoments | None = None

    def __post_init__(self) -> None:
        self.lengths = OnlineMoments()
        self.inter_arrivals = OnlineMoments()


def read_pcap_frames(tshark: Path, pcap: Path) -> pd.DataFrame:
    command = [
        str(tshark),
        "-r",
        str(pcap),
        "-T",
        "fields",
        "-E",
        "separator=\t",
        "-e",
        "frame.time_epoch",
        "-e",
        "frame.len",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    records: list[tuple[float, float]] = []
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"Malformed tshark row {line_number}: {line!r}")
        records.append((float(fields[0]), float(fields[1])))
    return pd.DataFrame(records, columns=["epoch_seconds", "frame_length"])


def causalize_network(network: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    if len(network) != len(frames):
        raise ValueError(
            f"Packet count mismatch: CSV={len(network)}, PCAP={len(frames)}"
        )

    network = network.copy()
    csv_epoch = network["Time"].astype("int64").to_numpy(dtype=np.float64) / 1e9
    frame_epoch = frames["epoch_seconds"].to_numpy(dtype=np.float64)
    max_clock_error = float(np.max(np.abs(csv_epoch - frame_epoch)))
    if max_clock_error > 0.01:
        raise ValueError(
            f"CSV/PCAP row order is not reliable; max clock error={max_clock_error:.6f}s"
        )

    values = {column: np.zeros(len(network), dtype=np.float64) for column in FLOW_COLUMNS}
    states: dict[tuple[str, str, str, str, str], FlowState] = {}

    for position, (_, row) in enumerate(network.iterrows()):
        timestamp = frame_epoch[position]
        key = flow_key(row)
        state = states.get(key)
        if state is None:
            state = FlowState(
                first_time=timestamp,
                last_time=timestamp,
                first_source=str(row["Source IP"]),
                first_destination=str(row["Destination IP"]),
            )
            states[key] = state

        if state.packets:
            state.inter_arrivals.update(max(timestamp - state.last_time, 0.0))
        state.last_time = timestamp
        state.packets += 1
        if str(row["Source IP"]) == state.first_source:
            state.source_packets += 1
        else:
            state.destination_packets += 1
        state.lengths.update(float(frames.iloc[position]["frame_length"]))

        values["Flow Duration"][position] = max(timestamp - state.first_time, 0.0)
        values["Flow Count per IP Pair"][position] = state.packets
        values["Number of Packets from Source"][position] = state.source_packets
        values["Number of Packets from Destination"][position] = state.destination_packets
        values["Packet Length Mean"][position] = state.lengths.mean
        values["Packet length Std"][position] = state.lengths.std
        values["Inter Arrival Time Mean"][position] = state.inter_arrivals.mean
        values["Inter Arrival Time Std"][position] = state.inter_arrivals.std

    for column, column_values in values.items():
        network[column] = column_values
    network.attrs["max_csv_pcap_clock_error_seconds"] = max_clock_error
    network.attrs["flow_count"] = len(states)
    return network


def aggregate_causal_windows(
    network: pd.DataFrame, window_seconds: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate packets by endpoint-preserving, right-closed causal windows."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    missing = [column for column in FLOW_ID_COLUMNS if column not in network.columns]
    if missing:
        raise ValueError(f"Missing endpoint columns: {missing}")

    frame = network.sort_values("Time", kind="stable").copy()
    event_ns = frame["Time"].astype("int64").to_numpy(dtype=np.int64)
    step_ns = int(round(window_seconds * 1e9))
    endpoint_ns = ((event_ns + step_ns - 1) // step_ns) * step_ns
    frame["_window_time"] = pd.to_datetime(endpoint_ns, utc=True).tz_convert(
        "Asia/Shanghai"
    )
    frame["_cyber_wait_s"] = (endpoint_ns - event_ns) / 1e9

    excluded = {"Time", "_window_time", "_cyber_wait_s", *FLOW_ID_COLUMNS}
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    keys = ["_window_time", *FLOW_ID_COLUMNS]
    grouped = frame.groupby(keys, sort=False, dropna=False, observed=True)
    moments = grouped[numeric_columns].agg(["mean", "std", "first", "last"])
    moments = moments.fillna(0.0)

    output = moments.index.to_frame(index=False).rename(
        columns={"_window_time": "Time"}
    )
    for column in numeric_columns:
        output[f"cyber_{column}_mean"] = moments[(column, "mean")].to_numpy()
        output[f"cyber_{column}_std"] = moments[(column, "std")].to_numpy()
        output[f"cyber_{column}_last"] = moments[(column, "last")].to_numpy()
        output[f"cyber_{column}_delta"] = (
            moments[(column, "last")] - moments[(column, "first")]
        ).to_numpy()

    output["cyber_event_count"] = grouped.size().to_numpy(dtype=np.float64)
    output["cyber_wait_mean_s"] = grouped["_cyber_wait_s"].mean().to_numpy()
    output["cyber_wait_max_s"] = grouped["_cyber_wait_s"].max().to_numpy()
    output["causal_window_s"] = float(window_seconds)
    output = output.sort_values("Time", kind="stable").reset_index(drop=True)

    diagnostics = {
        "method": "right-closed fixed window grouped by source, destination, ports, and protocol",
        "window_seconds": float(window_seconds),
        "input_packet_rows": int(len(network)),
        "output_endpoint_window_rows": int(len(output)),
        "numeric_source_features": numeric_columns,
        "cyber_wait_seconds": {
            "mean": float(output["cyber_wait_mean_s"].mean()),
            "p95_max": float(output["cyber_wait_max_s"].quantile(0.95)),
            "max": float(output["cyber_wait_max_s"].max()),
        },
    }
    return output, diagnostics


def align_causally(
    network: pd.DataFrame,
    physical: pd.DataFrame,
    tolerance_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_column = "label" if "label" in physical.columns else physical.columns[-1]
    physical_columns = [
        column for column in physical.columns if column not in {"Time", label_column}
    ]
    network_columns = [column for column in network.columns if column != "Time"]

    physical = physical.sort_values("Time").drop_duplicates("Time", keep="last").copy()
    network = network.sort_values("Time", kind="stable").copy()
    input_network_rows = len(network)
    in_overlap = network["Time"].between(
        physical["Time"].min(), physical["Time"].max(), inclusive="both"
    )
    outside_overlap = int((~in_overlap).sum())
    network = network.loc[in_overlap].copy()
    physical["_physical_time"] = physical["Time"]

    aligned = pd.merge_asof(
        network,
        physical[["Time", "_physical_time", *physical_columns, label_column]],
        on="Time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        allow_exact_matches=True,
    )
    valid = aligned[label_column].notna()
    dropped = int((~valid).sum())
    aligned = aligned.loc[valid].copy()
    age_ms = (
        (aligned["Time"] - aligned["_physical_time"]).dt.total_seconds() * 1000.0
    )

    result = pd.DataFrame({"Time": aligned["Time"]})
    for column in network_columns:
        result[column] = aligned[column].to_numpy()
    for column in physical_columns:
        result[f"phy_{column}"] = aligned[column].to_numpy()
    result["phy_label"] = aligned[label_column].to_numpy()

    diagnostics = {
        "input_network_rows": int(input_network_rows),
        "network_rows_in_modal_overlap": int(len(network)),
        "dropped_outside_modal_overlap": outside_overlap,
        "output_rows": int(len(result)),
        "dropped_without_recent_physical_sample": dropped,
        "tolerance_seconds": float(tolerance_seconds),
        "alignment_age_ms": {
            "min": float(age_ms.min()),
            "median": float(age_ms.median()),
            "p95": float(age_ms.quantile(0.95)),
            "max": float(age_ms.max()),
        },
        "label_distribution": {
            str(label): int(count)
            for label, count in result["phy_label"].value_counts().items()
        },
    }
    return result, diagnostics


def compare_with_legacy(
    legacy_path: Path,
    result: pd.DataFrame,
    causal_network: pd.DataFrame,
) -> dict[str, Any]:
    if not legacy_path.exists():
        return {"available": False}

    def add_occurrence(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values("Time", kind="stable").copy()
        frame["_time_occurrence"] = frame.groupby("Time", sort=False).cumcount()
        return frame

    legacy = pd.read_csv(legacy_path)
    legacy["Time"] = parse_times(legacy["Time"])
    legacy = add_occurrence(legacy)
    current = add_occurrence(result[["Time", "phy_label"]])
    labels = legacy[["Time", "_time_occurrence", "phy_label"]].merge(
        current,
        on=["Time", "_time_occurrence"],
        suffixes=("_legacy", "_causal"),
        how="inner",
    )
    mismatches = labels["phy_label_legacy"] != labels["phy_label_causal"]

    causal_network = add_occurrence(causal_network)
    old_network_columns = legacy[["Time", "_time_occurrence", *FLOW_COLUMNS]]
    new_network_columns = causal_network[
        ["Time", "_time_occurrence", *FLOW_COLUMNS]
    ]
    flows = old_network_columns.merge(
        new_network_columns,
        on=["Time", "_time_occurrence"],
        suffixes=("_legacy", "_causal"),
        how="inner",
    )
    changed_flow_rows: dict[str, int] = {}
    for column in FLOW_COLUMNS:
        old = pd.to_numeric(flows[f"{column}_legacy"], errors="coerce").to_numpy()
        new = pd.to_numeric(flows[f"{column}_causal"], errors="coerce").to_numpy()
        changed_flow_rows[column] = int(
            np.sum(~np.isclose(old, new, rtol=1e-9, atol=1e-12, equal_nan=True))
        )

    return {
        "available": True,
        "matched_rows": int(len(labels)),
        "label_mismatch_rows": int(mismatches.sum()),
        "label_mismatch_rate": float(mismatches.mean()) if len(labels) else None,
        "legacy_flow_rows_changed": changed_flow_rows,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    for required in (args.physical, args.network, args.pcap, args.tshark):
        if not required.exists():
            raise FileNotFoundError(required)

    physical = pd.read_csv(args.physical)
    network = pd.read_csv(args.network)
    physical["Time"] = parse_times(physical["Time"])
    network["Time"] = parse_times(network["Time"])
    physical = physical.dropna(subset=["Time"]).reset_index(drop=True)
    network = network.dropna(subset=["Time"]).reset_index(drop=True)

    frames = read_pcap_frames(args.tshark, args.pcap)
    causal_network = causalize_network(network, frames)
    event_result, event_alignment = align_causally(
        causal_network,
        physical,
        tolerance_seconds=args.tolerance_seconds,
    )
    windowed_network, windowing = aggregate_causal_windows(
        causal_network, args.window_seconds
    )
    result, alignment = align_causally(
        windowed_network, physical, tolerance_seconds=args.tolerance_seconds
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    physical_deltas = (
        physical.sort_values("Time")["Time"].diff().dt.total_seconds().dropna()
    )
    audit = {
        "method": (
            "causal cumulative flow statistics, endpoint-preserving fixed windows, "
            "and backward physical ZOH"
        ),
        "causal": True,
        "inputs": {
            "physical": str(args.physical),
            "network": str(args.network),
            "pcap": str(args.pcap),
            "physical_sha256": file_sha256(args.physical),
            "network_sha256": file_sha256(args.network),
            "pcap_sha256": file_sha256(args.pcap),
        },
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "physical_rows": int(len(physical)),
        "pcap_rows": int(len(frames)),
        "flow_count": int(causal_network.attrs["flow_count"]),
        "max_csv_pcap_clock_error_seconds": float(
            causal_network.attrs["max_csv_pcap_clock_error_seconds"]
        ),
        "physical_sampling_seconds": {
            "median": float(physical_deltas.median()),
            "p95": float(physical_deltas.quantile(0.95)),
            "max": float(physical_deltas.max()),
        },
        "windowing": windowing,
        "packet_clock_alignment": event_alignment,
        "fused_window_alignment": alignment,
        "legacy_comparison": compare_with_legacy(
            args.legacy, event_result, causal_network
        ),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--pcap", type=Path, default=DEFAULT_PCAP)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--tshark", type=Path, default=DEFAULT_TSHARK)
    parser.add_argument("--tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    report = build(parse_args())
    print(json.dumps(report, indent=2))
