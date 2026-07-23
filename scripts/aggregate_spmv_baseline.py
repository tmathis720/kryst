#!/usr/bin/env python3
"""Combine JSONL SpMV runs and calculate strong-scaling efficiency."""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["source_artifact"] = str(path)
                    records.append(record)

    rank_one = {}
    for record in records:
        if record.get("scope") != "distributed" or record.get("ranks") != 1:
            continue
        key = (
            record["case"],
            record["scalar"],
            record["implementation"],
            record.get("threads_per_rank", 1),
        )
        rank_one[key] = record["nanoseconds_per_spmv"]

    for record in records:
        if record.get("scope") != "distributed":
            continue
        key = (
            record["case"],
            record["scalar"],
            record["implementation"],
            record.get("threads_per_rank", 1),
        )
        baseline = rank_one.get(key)
        if baseline is not None:
            record["strong_scaling_efficiency"] = baseline / (
                record["ranks"] * record["nanoseconds_per_spmv"]
            )

    def version(command):
        try:
            return subprocess.check_output(command, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "rustc": version(["rustc", "--version"]),
            "cargo": version(["cargo", "--version"]),
        },
        "metric_definitions": {
            "effective_bandwidth_gb_s": "modeled bytes per SpMV / max-rank wall nanoseconds",
            "strong_scaling_efficiency": "T(1) / (ranks * T(ranks)) for matching case/scalar/thread configuration",
            "load_imbalance": "maximum work or time divided by arithmetic mean",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
