#!/usr/bin/env python3
"""Validate DistCSR benchmark artifact rollout/exit guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail CI when DistCSR benchmark rollout thresholds regress."
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


def get_number(value: object, ctx: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{ctx} must be numeric")
    return float(value)


def resolve_number(
    details: dict[str, object], keys: list[str], case_id: str, label: str
) -> float | None:
    for key in keys:
        value = details.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if keys:
        joined = ", ".join(keys)
        raise ValueError(f"artifact case {case_id} missing numeric {label} field(s): {joined}")
    return None


def resolve_mapping(value: object, ctx: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{ctx} must be an object")
    out: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{ctx} keys must be strings")
        out[key] = get_number(raw, f"{ctx}.{key}")
    return out


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
    perf_cfg = as_object(thresholds.get("performance", {}), "thresholds.performance")
    conv_cfg = as_object(thresholds.get("convergence", {}), "thresholds.convergence")
    fallback_diag_cfg = as_object(
        thresholds.get("fallback_diagnostics", {}), "thresholds.fallback_diagnostics"
    )

    target_route = native_cfg.get("target_selected", "distributed_native")
    if not isinstance(target_route, str):
        raise ValueError("thresholds.native_route.target_selected must be a string")

    selection_min = get_number(
        native_cfg.get("selection_rate_min", 1.0),
        "thresholds.native_route.selection_rate_min",
    )

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

    freq_max = get_number(
        fallback_cfg.get("frequency_max", 0.0),
        "thresholds.fallback.frequency_max",
    )

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

    speedup_min = perf_cfg.get("speedup_min")
    speedup_values: list[float] = []
    if speedup_min is not None:
        speedup_min = get_number(speedup_min, "thresholds.performance.speedup_min")
        speedup_case_min = resolve_mapping(
            perf_cfg.get("per_case_speedup_min", {}),
            "thresholds.performance.per_case_speedup_min",
        )
        current_keys = perf_cfg.get("current_time_keys", ["solve_ms", "total_solve_ms"])
        baseline_keys = perf_cfg.get(
            "baseline_time_keys", ["baseline_solve_ms", "baseline_total_solve_ms"]
        )
        if not isinstance(current_keys, list) or not all(isinstance(k, str) for k in current_keys):
            raise ValueError("thresholds.performance.current_time_keys must be an array of strings")
        if not isinstance(baseline_keys, list) or not all(isinstance(k, str) for k in baseline_keys):
            raise ValueError("thresholds.performance.baseline_time_keys must be an array of strings")

        by_id = {str(case.get("id")): case for case in cases}
        for case in cases:
            case_id = str(case.get("id"))
            details = as_object(case.get("details", {}), "artifact.cases[].details")
            current = resolve_number(details, current_keys, case_id, "current solve time")
            baseline = resolve_number(details, baseline_keys, case_id, "baseline solve time")
            if current is None or baseline is None or current <= 0 or baseline <= 0:
                failures.append(
                    f"case {case_id} has invalid timing values for speedup check (baseline={baseline}, current={current})"
                )
                continue
            speedup = baseline / current
            speedup_values.append(speedup)
            required = speedup_case_min.get(case_id, speedup_min)
            if speedup < required:
                failures.append(
                    f"case {case_id} speedup regression: observed {speedup:.3f}x, required >= {required:.3f}x"
                )

        for case_id in speedup_case_min:
            if case_id not in by_id:
                failures.append(f"threshold configured for missing case: {case_id}")

    if conv_cfg:
        iter_growth_max = get_number(
            conv_cfg.get("iteration_growth_max", 1.10),
            "thresholds.convergence.iteration_growth_max",
        )
        resid_growth_max = get_number(
            conv_cfg.get("residual_growth_max", 1.50),
            "thresholds.convergence.residual_growth_max",
        )
        iteration_keys = conv_cfg.get("iteration_keys", ["iterations"])
        baseline_iteration_keys = conv_cfg.get("baseline_iteration_keys", ["baseline_iterations"])
        residual_keys = conv_cfg.get("final_residual_keys", ["final_residual"])
        baseline_residual_keys = conv_cfg.get(
            "baseline_final_residual_keys", ["baseline_final_residual"]
        )
        for cfg_name, cfg_value in [
            ("iteration_keys", iteration_keys),
            ("baseline_iteration_keys", baseline_iteration_keys),
            ("final_residual_keys", residual_keys),
            ("baseline_final_residual_keys", baseline_residual_keys),
        ]:
            if not isinstance(cfg_value, list) or not all(isinstance(k, str) for k in cfg_value):
                raise ValueError(f"thresholds.convergence.{cfg_name} must be an array of strings")

        for case in cases:
            case_id = str(case.get("id"))
            details = as_object(case.get("details", {}), "artifact.cases[].details")
            iterations = resolve_number(details, iteration_keys, case_id, "iteration count")
            baseline_iterations = resolve_number(
                details, baseline_iteration_keys, case_id, "baseline iteration count"
            )
            final_residual = resolve_number(details, residual_keys, case_id, "final residual")
            baseline_final_residual = resolve_number(
                details, baseline_residual_keys, case_id, "baseline final residual"
            )

            if iterations is None or baseline_iterations is None or baseline_iterations <= 0:
                failures.append(
                    f"case {case_id} has invalid iteration values for convergence check"
                )
            elif iterations / baseline_iterations > iter_growth_max:
                failures.append(
                    f"case {case_id} convergence regression (iterations): observed {iterations:.2f}, "
                    f"baseline {baseline_iterations:.2f}, max growth {iter_growth_max:.3f}"
                )

            safe_baseline_resid = max(float(baseline_final_residual or 0.0), 1e-30)
            safe_resid = max(float(final_residual or 0.0), 1e-30)
            if safe_resid / safe_baseline_resid > resid_growth_max:
                failures.append(
                    f"case {case_id} convergence regression (final residual): observed {safe_resid:.3e}, "
                    f"baseline {safe_baseline_resid:.3e}, max growth {resid_growth_max:.3f}"
                )

    if fallback_diag_cfg:
        require_when_fallback = fallback_diag_cfg.get("require_when_fallback", True)
        if not isinstance(require_when_fallback, bool):
            raise ValueError("thresholds.fallback_diagnostics.require_when_fallback must be boolean")
        required_fields = fallback_diag_cfg.get(
            "required_fields",
            [
                "pc_dist_fallback_chain",
                "pc_dist_fallback_reason",
                "pc_dist_fallback_counters",
                "solver_view_snapshot",
            ],
        )
        if not isinstance(required_fields, list) or not all(
            isinstance(field, str) for field in required_fields
        ):
            raise ValueError("thresholds.fallback_diagnostics.required_fields must be an array of strings")

        for case in fallback_cases:
            case_id = str(case.get("id"))
            details = as_object(case.get("details", {}), "artifact.cases[].details")
            if not require_when_fallback and fallback_total(case) <= 0:
                continue
            for field in required_fields:
                value = details.get(field)
                if value is None:
                    failures.append(
                        f"case {case_id} missing fallback diagnostics field '{field}'"
                    )
                elif isinstance(value, str) and not value.strip():
                    failures.append(
                        f"case {case_id} has empty fallback diagnostics field '{field}'"
                    )

    if failures:
        print("DISTCSR THRESHOLD CHECK FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(
        "DISTCSR THRESHOLD CHECK PASSED: "
        f"selection={route_hits:.2%}, fallback_frequency={fallback_rate:.2%}"
        + (
            ""
            if not speedup_values
            else f", median_speedup={sorted(speedup_values)[len(speedup_values) // 2]:.3f}x"
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as err:
        print(f"DISTCSR THRESHOLD CHECK ERROR: {err}")
        raise SystemExit(2)
