#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _num_partitions(payload: Dict[str, Any]):
    n = _safe_get(payload, ["cut_result", "num_partitions"], None)
    if isinstance(n, int):
        return n
    labels = payload.get("partition_labels")
    if isinstance(labels, list) and labels:
        try:
            return len(set(int(x) for x in labels))
        except Exception:
            return None
    return None


def _overall_eval(payload: Dict[str, Any]):
    ev = payload.get("overall_evaluation")
    if not isinstance(ev, dict):
        return ("-", "-")
    metric = str(ev.get("metric", "-"))
    passed = ev.get("pass")
    if passed is None:
        return (metric, "-")
    return (metric, "PASS" if bool(passed) else "FAIL")


def _format_row(cols, widths):
    return "  ".join(c.ljust(w) for c, w in zip(cols, widths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="results/qi_runs/*.json")
    args = parser.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        print(f"No files matched: {args.glob}")
        return 1

    rows: List[List[str]] = []
    for p in paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        n = payload.get("num_qubits")
        backend = payload.get("backend")
        parts = _num_partitions(payload)
        basis = payload.get("measurement_basis", "-")
        metric, verdict = _overall_eval(payload)
        rows.append(
            [
                str(n) if n is not None else "-",
                str(parts) if parts is not None else "-",
                str(backend) if backend is not None else "-",
                str(basis),
                str(metric),
                str(verdict),
                p.as_posix(),
            ]
        )

    header = ["n", "parts", "backend", "basis", "metric", "eval", "file"]
    widths = [max(len(r[i]) for r in ([header] + rows)) for i in range(len(header))]

    print(_format_row(header, widths))
    print(_format_row(["-" * w for w in widths], widths))
    for r in rows:
        print(_format_row(r, widths))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
