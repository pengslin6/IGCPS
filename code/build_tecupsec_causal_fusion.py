"""Build a leakage-audited TE-CUP-SEC information--physical table.

The released process CSVs and locally downloaded packet captures use the same
wall clock, but several process files retain only minute-resolution timestamps.
Existing network caches summarize packets in one-second bins and timestamp a
row at the *start* of its bin.  This builder repairs both issues:

* reconstruct the documented 1 Hz process clock without interpolation;
* timestamp each network bin at its right edge, when all bin data are available;
* align only to the latest process observation at or before that edge; and
* omit scenario, source-file, and absolute-epoch fields from model inputs.

Raw source files and all previous fused outputs are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\BaiduNetdiskDownload")
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "fused_outputs"
DEFAULT_OUTPUT = SCRIPT_ROOT / "tecupsec_information_physical_causal_v3.csv"
DEFAULT_AUDIT = SCRIPT_ROOT / "tecupsec_information_physical_causal_v3_audit.json"

SCENARIOS = ["normal", *[f"attack{index}" for index in range(1, 9)]]
XMEAS_COLUMNS = [f"xmeas{index}" for index in range(1, 42)]
XMV_COLUMNS = [f"xmv{index}" for index in range(1, 13)]
PROCESS_COLUMNS = [*XMEAS_COLUMNS, *XMV_COLUMNS]

NETWORK_FEATURE_COLUMNS = [
    "packet_count",
    "total_packet_len",
    "avg_packet_len",
    "total_ip_len",
    "tcp_count",
    "udp_count",
    "icmp_count",
    "other_proto_count",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "unique_src_ip_count",
    "unique_dst_ip_count",
    "unique_src_port_count",
    "unique_dst_port_count",
    "duration_ms",
    "avg_inter_arrival_ms",
]

ATTACKS: dict[int, dict[str, str]] = {
    0: {
        "type": "Normal operation",
        "affected_variable": "none",
    },
    1: {
        "type": "Malicious command injection",
        "affected_variable": "Production setpoint",
    },
    2: {
        "type": "Malicious command injection",
        "affected_variable": "Stripper level setpoint",
    },
    3: {
        "type": "Malicious command injection",
        "affected_variable": "yA setpoint and yAC setpoint",
    },
    4: {
        "type": "Physical process disturbance",
        "affected_variable": "A/C feed ratio",
    },
    5: {
        "type": "Physical process disturbance",
        "affected_variable": "Reactor cooling-water outlet valve",
    },
    6: {
        "type": "Physical process disturbance",
        "affected_variable": "Separator cooling-water outlet valve",
    },
    7: {
        "type": "False data injection",
        "affected_variable": "Stripper level measurement",
    },
    8: {
        "type": "Denial of service",
        "affected_variable": "Separator PLC communication",
    },
}

CAPTURE_MAGIC = {
    bytes.fromhex("d4c3b2a1"): "pcap-le",
    bytes.fromhex("a1b2c3d4"): "pcap-be",
    bytes.fromhex("4d3cb2a1"): "pcap-ns-le",
    bytes.fromhex("a1b23c4d"): "pcap-ns-be",
    bytes.fromhex("0a0d0d0a"): "pcapng",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scenario_id(name: str) -> int:
    return 0 if name == "normal" else int(name.removeprefix("attack"))


def process_path(data_root: Path, scenario: str) -> Path:
    return data_root / scenario / f"{scenario}.csv"


def reconstruct_process_clock(
    recorded: pd.Series,
) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    parsed = pd.to_datetime(recorded, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Invalid process timestamps: {int(parsed.isna().sum())}")
    if not parsed.is_monotonic_increasing:
        raise ValueError("Process timestamps are not monotonic")

    seconds_are_zero = bool((parsed.dt.second == 0).all())
    minute_resolution = seconds_are_zero and parsed.nunique() < len(parsed) / 2
    row_offsets = pd.to_timedelta(np.arange(len(parsed), dtype=np.int64), unit="s")

    if minute_resolution:
        recorded_minutes = parsed.dt.floor("min")
        first_minute = recorded_minutes.iloc[0]
        first_count = int((recorded_minutes == first_minute).sum())
        if not 1 <= first_count <= 60:
            raise ValueError(f"Unexpected first-minute row count: {first_count}")
        start = first_minute + pd.Timedelta(seconds=60 - first_count)
        reconstructed = pd.DatetimeIndex(start + row_offsets)
        minute_matches = reconstructed.floor("min") == pd.DatetimeIndex(
            recorded_minutes
        )
        if not bool(np.all(minute_matches)):
            mismatch = int(np.sum(~minute_matches))
            raise ValueError(
                f"Minute-resolution reconstruction disagrees on {mismatch} rows"
            )
        method = "minute bucket occupancy plus zero-based 1 Hz row order"
    else:
        recorded_ns = parsed.astype("int64").to_numpy(dtype=np.int64)
        offset_ns = np.arange(len(parsed), dtype=np.int64) * 1_000_000_000
        start_ns = int(np.median(recorded_ns - offset_ns))
        start_ns = int(round(start_ns / 1_000_000_000)) * 1_000_000_000
        reconstructed = pd.to_datetime(start_ns + offset_ns)
        start = reconstructed[0]
        method = "robust 1 Hz sequence fitted to second-resolution timestamps"

    residual_seconds = (
        parsed.reset_index(drop=True) - pd.Series(reconstructed)
    ).dt.total_seconds()
    if not minute_resolution and float(residual_seconds.abs().max()) > 5.0:
        raise ValueError(
            "Second-resolution timestamp repair exceeded the 5 s audit bound"
        )

    report = {
        "method": method,
        "minute_resolution_source": minute_resolution,
        "recorded_start": parsed.iloc[0].isoformat(),
        "recorded_end": parsed.iloc[-1].isoformat(),
        "reconstructed_start": pd.Timestamp(reconstructed[0]).isoformat(),
        "reconstructed_end": pd.Timestamp(reconstructed[-1]).isoformat(),
        "rows": int(len(parsed)),
        "recorded_unique_timestamps": int(parsed.nunique()),
        "recorded_duplicate_rows": int(parsed.duplicated().sum()),
        "recorded_minus_reconstructed_seconds": {
            "min": float(residual_seconds.min()),
            "median": float(residual_seconds.median()),
            "p95_absolute": float(residual_seconds.abs().quantile(0.95)),
            "max": float(residual_seconds.max()),
            "max_absolute": float(residual_seconds.abs().max()),
        },
    }
    return reconstructed, report


def capture_inventory(directory: Path) -> dict[str, Any]:
    captures = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() != ".csv"
    )
    formats: dict[str, int] = {}
    invalid: list[str] = []
    starts: list[pd.Timestamp] = []
    manifest_rows: list[str] = []
    stamp_pattern = re.compile(r"_(\d{14})_\d+$")

    for path in captures:
        with path.open("rb") as stream:
            magic = stream.read(4)
        capture_format = CAPTURE_MAGIC.get(magic, "unknown")
        formats[capture_format] = formats.get(capture_format, 0) + 1
        if capture_format == "unknown":
            invalid.append(path.name)
        match = stamp_pattern.search(path.name)
        if match:
            starts.append(pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S"))
        manifest_rows.append(f"{path.name}\t{path.stat().st_size}\t{magic.hex()}")

    intervals = pd.Series(starts).sort_values().diff().dt.total_seconds().dropna()
    manifest_sha = hashlib.sha256(
        "\n".join(manifest_rows).encode("utf-8")
    ).hexdigest()
    return {
        "capture_files": len(captures),
        "total_bytes": int(sum(path.stat().st_size for path in captures)),
        "formats": formats,
        "invalid_capture_files": invalid,
        "filename_manifest_sha256": manifest_sha,
        "filename_clock": {
            "parsed_starts": len(starts),
            "first": starts[0].isoformat() if starts else None,
            "last": starts[-1].isoformat() if starts else None,
            "interval_seconds": {
                "min": float(intervals.min()) if len(intervals) else None,
                "median": float(intervals.median()) if len(intervals) else None,
                "max": float(intervals.max()) if len(intervals) else None,
                "outside_1790_to_1810": int(
                    ((intervals < 1790) | (intervals > 1810)).sum()
                )
                if len(intervals)
                else 0,
            },
        },
    }


def load_process(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"time", "label", *PROCESS_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing process columns {missing}")

    reconstructed, clock_report = reconstruct_process_clock(frame["time"])
    output = pd.DataFrame({"Time": reconstructed})
    for column in PROCESS_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"{path.name}: {int(values.isna().sum())} invalid values in {column}"
            )
        output[f"phy_{column}"] = values.to_numpy(dtype=np.float64)
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{path.name}: invalid labels")
    output["phy_label"] = labels.to_numpy(dtype=np.int16)
    return output, {
        "file": str(path),
        "sha256": sha256_file(path),
        "clock": clock_report,
        "label_counts": {
            str(label): int(count)
            for label, count in output["phy_label"].value_counts().sort_index().items()
        },
    }


def load_network_cache(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"Time", "src_ip", "dst_ip", *NETWORK_FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing network columns {missing}")

    window_start = pd.to_datetime(frame["Time"], errors="coerce")
    if window_start.isna().any():
        raise ValueError(f"{path.name}: invalid network timestamps")
    if not window_start.is_monotonic_increasing:
        raise ValueError(f"{path.name}: network timestamps are not monotonic")
    if window_start.duplicated().any():
        raise ValueError(f"{path.name}: duplicate one-second network bins")

    # A complete [t, t+1) bin becomes observable at t+1.
    output = pd.DataFrame({"Time": window_start + pd.Timedelta(seconds=1)})
    output["Source IP"] = frame["src_ip"].fillna("NO_IP").astype(str)
    output["Destination IP"] = frame["dst_ip"].fillna("NO_IP").astype(str)
    for column in NETWORK_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"{path.name}: {int(values.isna().sum())} invalid values in {column}"
            )
        output[f"cyber_{column}"] = values.to_numpy(dtype=np.float64)
    output["cyber_window_s"] = 1.0

    endpoint_missing = (
        (output["Source IP"] == "NO_IP")
        | (output["Destination IP"] == "NO_IP")
        | (output["Source IP"] == "")
        | (output["Destination IP"] == "")
    )
    input_rows = len(output)
    output = output.loc[~endpoint_missing].reset_index(drop=True)
    return output, {
        "file": str(path),
        "sha256": sha256_file(path),
        "cached_rows": int(input_rows),
        "rows": int(len(output)),
        "cached_window_start": window_start.iloc[0].isoformat(),
        "cached_window_end": window_start.iloc[-1].isoformat(),
        "causal_availability_start": output["Time"].iloc[0].isoformat(),
        "causal_availability_end": output["Time"].iloc[-1].isoformat(),
        "dropped_rows_without_ip_pair": int(endpoint_missing.sum()),
    }


def align_scenario(
    network: pd.DataFrame,
    process: pd.DataFrame,
    tolerance_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    process = process.sort_values("Time", kind="stable").copy()
    network = network.sort_values("Time", kind="stable").copy()
    process["_process_time"] = process["Time"]

    overlap = network["Time"].between(
        process["Time"].min(), process["Time"].max(), inclusive="both"
    )
    outside = int((~overlap).sum())
    network = network.loc[overlap].copy()
    process_features = [column for column in process if column.startswith("phy_")]

    aligned = pd.merge_asof(
        network,
        process[["Time", "_process_time", *process_features]],
        on="Time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        allow_exact_matches=True,
    )
    valid = aligned["_process_time"].notna()
    dropped_stale = int((~valid).sum())
    aligned = aligned.loc[valid].copy()
    age = (aligned["Time"] - aligned["_process_time"]).dt.total_seconds()
    if bool((age < 0).any()):
        raise AssertionError("Future process observation entered causal alignment")

    output = pd.DataFrame({"Time": aligned["Time"]})
    output["Source IP"] = aligned["Source IP"].to_numpy()
    output["Destination IP"] = aligned["Destination IP"].to_numpy()
    for column in network.columns:
        if column not in {"Time", "Source IP", "Destination IP"}:
            output[column] = aligned[column].to_numpy()
    output["alignment_age_s"] = age.to_numpy(dtype=np.float64)
    for column in PROCESS_COLUMNS:
        output[f"phy_{column}"] = aligned[f"phy_{column}"].to_numpy()
    output["phy_label"] = aligned["phy_label"].to_numpy(dtype=np.int16)

    forbidden = {"scenario", "source_file", "ts_ms"}
    present_forbidden = sorted(forbidden & set(output.columns))
    if present_forbidden:
        raise AssertionError(f"Leakage-prone fields retained: {present_forbidden}")
    if output.isna().any().any():
        bad = output.columns[output.isna().any()].tolist()
        raise ValueError(f"Aligned output contains missing values: {bad}")

    return output, {
        "network_rows": int(len(overlap)),
        "network_rows_outside_process_clock": outside,
        "dropped_without_recent_process_observation": dropped_stale,
        "output_rows": int(len(output)),
        "alignment_direction": "backward/as-of",
        "tolerance_seconds": float(tolerance_seconds),
        "future_observation_rows": int((age < 0).sum()),
        "process_observation_age_seconds": {
            "min": float(age.min()),
            "median": float(age.median()),
            "p95": float(age.quantile(0.95)),
            "max": float(age.max()),
        },
        "label_counts": {
            str(label): int(count)
            for label, count in output["phy_label"].value_counts().sort_index().items()
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    required_paths: list[Path] = []
    for scenario in SCENARIOS:
        required_paths.extend(
            [
                process_path(args.data_root, scenario),
                args.cache_root / f"{scenario}_features.csv",
            ]
        )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TE-CUP-SEC inputs: {missing}")

    scenario_inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    scenario_reports: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        process, process_report = load_process(
            process_path(args.data_root, scenario)
        )
        network, network_report = load_network_cache(
            args.cache_root / f"{scenario}_features.csv"
        )
        scenario_inputs[scenario] = (process, network)
        scenario_reports[scenario] = {
            "scenario_id": scenario_id(scenario),
            "process": process_report,
            "network_cache": network_report,
            "raw_capture_inventory": capture_inventory(args.data_root / scenario),
        }

    ordered = sorted(
        SCENARIOS,
        key=lambda name: scenario_inputs[name][0]["Time"].iloc[0],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.unlink(missing_ok=True)

    wrote_header = False
    output_rows = 0
    output_columns: list[str] | None = None
    row_ranges: dict[str, list[int]] = {}
    try:
        for scenario in ordered:
            process, network = scenario_inputs.pop(scenario)
            fused, alignment_report = align_scenario(
                network, process, tolerance_seconds=args.tolerance_seconds
            )
            if output_columns is None:
                output_columns = fused.columns.tolist()
            elif fused.columns.tolist() != output_columns:
                raise AssertionError(f"{scenario}: output schema changed")
            fused.to_csv(
                temporary,
                mode="a",
                header=not wrote_header,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%S",
            )
            wrote_header = True
            row_ranges[scenario] = [output_rows, output_rows + len(fused)]
            output_rows += len(fused)
            scenario_reports[scenario]["alignment"] = alignment_report
            print(
                f"[TE-CUP-SEC] {scenario}: {len(fused):,} fused rows; "
                f"total={output_rows:,}",
                flush=True,
            )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, args.output)

    total_capture_bytes = sum(
        report["raw_capture_inventory"]["total_bytes"]
        for report in scenario_reports.values()
    )
    total_capture_files = sum(
        report["raw_capture_inventory"]["capture_files"]
        for report in scenario_reports.values()
    )
    audit = {
        "dataset": "TE-CUP-SEC",
        "method": (
            "right-edge one-second network windows plus reconstructed 1 Hz "
            "process observations and backward/as-of causal alignment"
        ),
        "causal": True,
        "output": {
            "file": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": output_rows,
            "columns": len(output_columns or []),
            "scenario_order": ordered,
            "scenario_row_ranges_half_open": row_ranges,
        },
        "raw_network_data": {
            "root": str(args.data_root),
            "capture_files": total_capture_files,
            "total_bytes": total_capture_bytes,
        },
        "fusion_policy": {
            "network_window_seconds": 1.0,
            "network_timestamp_semantics": (
                "a cached bin labelled t summarizes [t,t+1) and is emitted at t+1"
            ),
            "process_sampling_seconds": 1.0,
            "process_interpolation": "none",
            "cross_modal_alignment": "latest process observation at or before network availability",
            "future_fill": "none",
            "model_input_layers": {
                "information": "network endpoints and causal traffic statistics",
                "physical": "41 XMEAS and 12 XMV process variables",
            },
            "excluded_leakage_fields": ["scenario", "source_file", "ts_ms"],
        },
        "scenario_reports": [scenario_reports[name] for name in ordered],
        "attack_semantics": ATTACKS,
        "limitations": [
            "The capture files do not provide an independent hardware clock-sync trace.",
            "One-second aggregation bounds online network feature latency by one second.",
            "Capture integrity is checked by magic bytes and a filename/size manifest; full 32+ GB capture hashes are not recomputed.",
        ],
    }
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--tolerance-seconds", type=float, default=1.5)
    return parser.parse_args()


if __name__ == "__main__":
    report = build(parse_args())
    print(json.dumps(report["output"], ensure_ascii=False, indent=2))
