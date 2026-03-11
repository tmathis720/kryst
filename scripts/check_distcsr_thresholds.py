#!/usr/bin/env python3
"""Validate DistCSR benchmark artifact route-selection guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail CI when DistCSR benchmark route/fallback thresholds regress."
    )
    parser.add_argument("--artifact", required=True, help="Path to benchmark artifact json")
    parser.add_argument("--thresholds", required=True, help="Path to threshold config json")
    return parser.parse_args()


def as_object(value: object, ctx: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{ctx} must be an object")
    return value


def as_cases(artifact: dict[str, object]) -> list[dict[str, object]]:
    raw_cases = artifact.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("artifact.cases must be an array")
    return [as_object(case, "artifact.cases[]") for case in raw_cases]


def load_json(path: Path) -> dict[str, object]:
    try:
        return as_object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json in {path}: {exc}") from exc


def selection_rate(cases: list[dict[str, object]], expected_route: str) -> float:
    if not cases:
        return 0.0
    selected = 0
    for case in cases:
        details = as_object(case.get("details", {}), "artifact.cases[].details")
        route = details.get("pc_dist_selected_route")
        if route == expected_route:
            selected += 1
    return selected / len(cases)


def fallback_total(case: dict[str, object]) -> int:
    details = as_object(case.get("details", {}), "artifact.cases[].details")
    value = details.get("fallback_total", 0)
    if not isinstance(value, int):
        raise ValueError(
            f"artifact case {case.get('id', '<unknown>')} has non-integer fallback_total"
        )
    return value


def main() -> int:
    args = parse_args()
    artifact = load_json(Path(args.artifact))
    thresholds = load_json(Path(args.thresholds))

    if artifact.get("schema_version") != 1:
        raise ValueError("artifact schema_version must be 1")
    if thresholds.get("schema_version") != 1:
        raise ValueError("thresholds schema_version must be 1")

    cases = [c for c in as_cases(artifact) if c.get("status") != "skipped"]
    if not cases:
        print("DISTCSR THRESHOLD CHECK FAILED: no non-skipped benchmark cases in artifact")
        return 1

    native_cfg = as_object(thresholds.get("native_route", {}), "thresholds.native_route")
    fallback_cfg = as_object(thresholds.get("fallback", {}), "thresholds.fallback")

    target_route = native_cfg.get("target_selected", "distributed_native")
    if not isinstance(target_route, str):
        raise ValueError("thresholds.native_route.target_selected must be a string")

    selection_min = native_cfg.get("selection_rate_min", 1.0)
    if not isinstance(selection_min, (int, float)):
        raise ValueError("thresholds.native_route.selection_rate_min must be numeric")

    failures: list[str] = []
    route_hits = selection_rate(cases, target_route)
    if route_hits < float(selection_min):
        failures.append(
            "native route selection rate regressed: "
            f"{route_hits:.2%} selected {target_route}, minimum required {float(selection_min):.2%}"
        )

    required_cases = native_cfg.get("per_case_required", [])
    if required_cases:
        if not isinstance(required_cases, list) or not all(
            isinstance(v, str) for v in required_cases
        ):
            raise ValueError("thresholds.native_route.per_case_required must be an array of case ids")
        by_id = {str(case.get("id")): case for case in cases}
        for case_id in required_cases:
            case = by_id.get(case_id)
            if case is None:
                failures.append(f"required case missing from artifact: {case_id}")
                continue
            details = as_object(case.get("details", {}), "artifact.cases[].details")
            selected_route = details.get("pc_dist_selected_route")
            if selected_route != target_route:
                failures.append(
                    f"case {case_id} selected route '{selected_route}' instead of '{target_route}'"
                )

    freq_max = fallback_cfg.get("frequency_max", 0.0)
    if not isinstance(freq_max, (int, float)):
        raise ValueError("thresholds.fallback.frequency_max must be numeric")

    fallback_cases = [case for case in cases if fallback_total(case) > 0]
    fallback_rate = len(fallback_cases) / len(cases)
    if fallback_rate > float(freq_max):
        regressed = ", ".join(str(case.get("id")) for case in fallback_cases)
        failures.append(
            "fallback frequency regressed: "
            f"{fallback_rate:.2%} of cases reported fallback_total>0 (max {float(freq_max):.2%}); "
            f"cases: {regressed}"
        )

    per_case_max = fallback_cfg.get("per_case_max_total", {})
    if per_case_max:
        if not isinstance(per_case_max, dict):
            raise ValueError("thresholds.fallback.per_case_max_total must be an object")
        by_id = {str(case.get("id")): case for case in cases}
        for case_id, max_total in per_case_max.items():
            if not isinstance(max_total, int):
                raise ValueError(
                    "thresholds.fallback.per_case_max_total values must be integer counts"
                )
            case = by_id.get(case_id)
            if case is None:
                failures.append(f"threshold configured for missing case: {case_id}")
                continue
            observed = fallback_total(case)
            if observed > max_total:
                failures.append(
                    f"case {case_id} fallback_total regression: observed {observed}, max {max_total}"
                )

    if failures:
        print("DISTCSR THRESHOLD CHECK FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(
        "DISTCSR THRESHOLD CHECK PASSED: "
        f"selection={route_hits:.2%}, fallback_frequency={fallback_rate:.2%}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as err:
        print(f"DISTCSR THRESHOLD CHECK ERROR: {err}")
        raise SystemExit(2)
