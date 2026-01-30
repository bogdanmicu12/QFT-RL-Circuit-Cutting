from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from qft_builder import QFTCircuitBuilder
from circuit_cutting import CircuitCuttingExecutor


def _normalize_bitstring(bitstring, num_bits):
    bs = bitstring.replace(" ", "")
    if len(bs) < num_bits:
        bs = bs.zfill(num_bits)
    return bs


def _normalize_counts_keys(counts, num_bits):
    if not counts:
        return {}
    out= {}
    for k, v in counts.items():
        out[_normalize_bitstring(k, num_bits)] = out.get(_normalize_bitstring(k, num_bits), 0) + int(v)
    return out


def _prepare_for_counts(circuit, *, measurement_basis):
    if measurement_basis not in {"z", "x"}:
        raise ValueError("measurement_basis must be one of: 'z', 'x'")

    qc = circuit.copy()
    qc = qc.remove_final_measurements(inplace=False)
    if measurement_basis == "x":
        qc.h(range(qc.num_qubits))
    qc.measure_all()
    return qc


def _get_qi_backend(backend_name):
    try:
        from qiskit_quantuminspire.qi_provider import QIProvider
    except ImportError as exc:
        raise SystemExit(
            "qiskit-quantuminspire is required. Install with: pip install qiskit-quantuminspire"
        ) from exc

    provider = QIProvider()

    backends = provider.backends()
    for b in backends:
        if getattr(b, "name", None) == backend_name:
            return b
    backend_name_lower = backend_name.lower()
    for b in backends:
        name = getattr(b, "name", "")
        if backend_name_lower in name.lower():
            return b

    available = sorted(getattr(b, "name", "") for b in backends)
    raise SystemExit(
        f"Backend '{backend_name}' not found. Available: {available}\n"
        "Tip: ensure you've run `qi login`, then try again."
    )


def _is_qi_auth_error(exc):
    msg = f"{exc}"
    return (
        "AuthorisationError" in exc.__class__.__name__
        or "token refresh" in msg.lower()
        or "oauth/token" in msg.lower()
        or "403" in msg
        or "forbidden" in msg.lower()
    )


def _default_partition_labels(num_qubits):
    split = num_qubits // 2
    return [0] * split + [1] * (num_qubits - split)


def _contiguous_partition_labels(num_qubits, num_partitions):
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if num_partitions == 1:
        return [0] * num_qubits
    base = num_qubits // num_partitions
    extra = num_qubits % num_partitions
    labels = []
    for pid in range(num_partitions):
        size = base + (1 if pid < extra else 0)
        labels.extend([pid] * size)
    return labels


def _cuts_from_partition_labels(partition_labels):
    cuts= []
    for i in range(1, len(partition_labels)):
        if partition_labels[i] != partition_labels[i - 1]:
            cuts.append((0, i))
    return cuts


def _counts_to_probs(counts):
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _z_expectations_from_counts(counts, num_qubits):
    probs = _counts_to_probs(_normalize_counts_keys(counts, num_qubits))
    if not probs:
        return [0.0] * num_qubits

    z = [0.0] * num_qubits
    for bitstring, p in probs.items():
        bs = _normalize_bitstring(bitstring, num_qubits)
        for i in range(num_qubits):
            z[i] += (1.0 if bs[i] == "0" else -1.0) * p
    return z


def _wilson_ci(k, n, z = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * ((phat * (1 - phat) / n + (z * z) / (4 * n * n)) ** 0.5)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def _success_metrics_from_counts(
    counts, *, expected, num_qubits
):
    counts_n = _normalize_counts_keys(counts, num_qubits)
    shots = int(sum(counts_n.values()))
    exp = _normalize_bitstring(expected, num_qubits)
    k = int(counts_n.get(exp, 0))
    p = (k / shots) if shots > 0 else 0.0
    ci_lo, ci_hi = _wilson_ci(k, shots)
    baseline = 1.0 / (2**num_qubits) if num_qubits > 0 else 0.0
    advantage = (p - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0
    return {
        "expected_bitstring": exp,
        "k_success": k,
        "shots": shots,
        "p_success": p,
        "p_success_wilson95": [ci_lo, ci_hi],
        "baseline_uniform": baseline,
        "advantage_over_uniform": advantage,
    }


def _z_expectations_ideal(qc_with_measure):
    qc = qc_with_measure.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    n = qc.num_qubits

    z = [0.0] * n
    for basis_index, p in enumerate(probs):
        for q in range(n):
            bit = (basis_index >> q) & 1
            z_q = 1.0 if bit == 0 else -1.0
            z[n - 1 - q] += z_q * float(p)
    return z


def _max_abs_diff(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _apply_basis_state_prep(qc, input_bitstring):
    n = qc.num_qubits
    bs = _normalize_bitstring(input_bitstring, n)
    for pos, ch in enumerate(bs):
        if ch == "1":
            qc.x(n - 1 - pos)


def _build_proof_circuit(
    *,
    num_qubits,
    proof_circuit,
    input_bitstring,
):
    if proof_circuit not in {"qft", "qft_iqft"}:
        raise ValueError("proof_circuit must be one of: 'qft', 'qft_iqft'")

    qc = QuantumCircuit(num_qubits)
    _apply_basis_state_prep(qc, input_bitstring)

    builder = QFTCircuitBuilder(num_qubits=num_qubits)
    qft = builder.build_qft_circuit()
    qc.compose(qft, inplace=True)

    if proof_circuit == "qft_iqft":
        qc.compose(qft.inverse(), inplace=True)

    qc.name = f"proof_{proof_circuit}_{num_qubits}q"
    return qc


def _bit_from_full_input(full_input_bs, full_num_qubits, orig_qubit):
    bs = _normalize_bitstring(full_input_bs, full_num_qubits)
    pos = (full_num_qubits - 1 - orig_qubit)
    return bs[pos]


def _expected_subcircuit_bitstring(
    *,
    full_input_bs,
    full_num_qubits,
    sub_qubit_map,
    sub_num_qubits,
):
    out = ["0"] * sub_num_qubits
    for orig_q, new_q in sub_qubit_map.items():
        bit = _bit_from_full_input(full_input_bs, full_num_qubits, int(orig_q))
        out[sub_num_qubits - 1 - int(new_q)] = bit
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="Run cut QFT partitions on Quantum Inspire")
    parser.add_argument("--backend", type=str, default="Tuna-9")
    parser.add_argument("--num-qubits", type=int, default=8)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default="./results/qi_runs")

    parser.add_argument(
        "--proof-circuit",
        choices=["qft", "qft_iqft"],
        default="qft",
    )
    parser.add_argument(
        "--input-bitstring",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--random-input",
        action="store_true",
    )
    parser.add_argument("--input-seed", type=int, default=0)

    # Cutting/partition controls
    parser.add_argument("--num-partitions", type=int, default=2)
    parser.add_argument("--max-partition-qubits", type=int, default=9)
    # Evaluation controls 
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--measurement-basis",
        choices=["z", "x"],
        default="z",
    )
    parser.add_argument(
        "--eval-metric",
        choices=["z", "success"],
        default="z",
    )
    parser.add_argument(
        "--expected-bitstring",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--min-success-prob",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--success-use-ci",
        action="store_true",
    )
    parser.add_argument(
        "--max-abs-z-diff",
        type=float,
        default=0.15,
    )

    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--max-partitions", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search-max-num-qubits", type=int, default=45)

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--max-num-qubits", type=int, default=18)
    parser.add_argument("--min-num-qubits", type=int, default=2)
    parser.add_argument("--sweep-shots", type=int, default=None)

    args = parser.parse_args()

    rng = random.Random(int(args.input_seed))

    def _choose_input_bitstring(num_qubits):
        if args.input_bitstring is not None:
            bs = _normalize_bitstring(str(args.input_bitstring), num_qubits)
            if len(bs) != num_qubits:
                raise SystemExit(f"input-bitstring must be length {num_qubits} (or shorter; will be zero-filled)")
            return bs
        if bool(args.random_input):
            return "".join("1" if rng.random() < 0.5 else "0" for _ in range(num_qubits))
        return "0" * num_qubits

    def run_one(num_qubits, shots):
        input_bs = _choose_input_bitstring(num_qubits)
        full_circuit = _build_proof_circuit(
            num_qubits=num_qubits,
            proof_circuit=str(args.proof_circuit),
            input_bitstring=input_bs,
        )

        executor = CircuitCuttingExecutor(max_subcircuit_qubits=max(2, min(args.max_partition_qubits, num_qubits)))
        if args.num_partitions == 2:
            partition_labels = _default_partition_labels(num_qubits)
        else:
            partition_labels = _contiguous_partition_labels(num_qubits, args.num_partitions)
        cuts = _cuts_from_partition_labels(partition_labels)

        max_part_size = 0
        for pid in set(partition_labels):
            max_part_size = max(max_part_size, sum(1 for x in partition_labels if x == pid))
        if max_part_size > int(args.max_partition_qubits):
            raise SystemExit(
                f"Partition too large: {max_part_size} qubits > max-partition-qubits={args.max_partition_qubits}. "
                f"Increase num-partitions or reduce num-qubits."
            )

        cut_result = executor.apply_wire_cuts(full_circuit, cuts=cuts, partition_labels=partition_labels)
        subcircuits_info = cut_result["subcircuits"]

        print(
            f"Built {args.proof_circuit.upper()}({num_qubits}) with input {input_bs} and produced {len(subcircuits_info)} subcircuits"
        )
        for info in subcircuits_info:
            qc_info = info["circuit"]
            print(f"  - {qc_info.name}: {qc_info.num_qubits} qubits, depth={qc_info.depth()}, ops={qc_info.size()}")

        if args.dry_run:
            return {
                "timestamp": datetime.now().isoformat(),
                "backend": None,
                "num_qubits": num_qubits,
                "shots_per_subcircuit": shots,
                "partition_labels": partition_labels,
                "cuts": cuts,
                "cut_result": {
                    "num_partitions": cut_result.get("num_partitions"),
                    "crossing_edges": cut_result.get("crossing_edges"),
                },
                "subcircuit_results": [],
                "reconstructed": None,
                "dry_run": True,
            }

        backend = _get_qi_backend(args.backend)
        print(f"\nUsing QI backend: {backend.name}")

        subcircuit_results = []
        for idx, info in enumerate(subcircuits_info):
            qc_raw: QuantumCircuit = info["circuit"]
            qc = _prepare_for_counts(qc_raw, measurement_basis=str(args.measurement_basis))
            qc.name = f"cut_part_{idx}_{qc_raw.num_qubits}q"
            qc_compiled = transpile(qc, backend)

            counts_runs = []
            job_ids = []
            sys_msgs_all = []
            for rep in range(int(args.repeats)):
                print(f"Submitting {qc.name} (shots={shots}) [rep {rep + 1}/{args.repeats}]...")
                try:
                    job = backend.run(qc_compiled, shots=shots)
                    result = job.result()
                except Exception as exc:
                    if _is_qi_auth_error(exc):
                        raise SystemExit(
                            "Quantum Inspire authorization failed while submitting/running the job.\n"
                            "Fix: run `qi logout` then `qi login` in a terminal, and retry.\n"
                            "If this persists, your QI account may not have access to this backend.\n"
                            f"Details: {exc}"
                        )
                    raise

                counts = _normalize_counts_keys(result.get_counts(qc_compiled), qc.num_qubits)
                counts_runs.append(counts)
                sys_msgs_all.append(getattr(result, "system_messages", None))
                job_ids.append(getattr(job, "job_id", lambda: None)())
                print(f"  done: {len(counts)} outcomes")

            counts_primary = counts_runs[0] if counts_runs else {}

            eval_metrics = None
            if args.eval:
                if str(args.eval_metric) == "z":
                    z_hw = _z_expectations_from_counts(counts_primary, qc.num_qubits)
                    z_ideal = _z_expectations_ideal(qc)
                    max_abs = _max_abs_diff(z_hw, z_ideal)
                    eval_metrics = {
                        "metric": "z",
                        "measurement_basis": str(args.measurement_basis),
                        "z_hw": z_hw,
                        "z_ideal": z_ideal,
                        "max_abs_z_diff": max_abs,
                        "pass": bool(max_abs <= float(args.max_abs_z_diff)),
                        "criterion": {"max_abs_z_diff": float(args.max_abs_z_diff)},
                    }
                else:
                    expected = args.expected_bitstring
                    if expected is None:
                        if str(args.proof_circuit) == "qft_iqft":
                            expected = _expected_subcircuit_bitstring(
                                full_input_bs=input_bs,
                                full_num_qubits=num_qubits,
                                sub_qubit_map=dict(info.get("qubit_map") or {}),
                                sub_num_qubits=qc.num_qubits,
                            )
                        else:
                            raise SystemExit(
                                "For proof-circuit qft (no inverse), expected output is not a single bitstring. "
                                "Provide expected-bitstring or use proof-circuit qft_iqft."
                            )
                    m = _success_metrics_from_counts(
                        counts_primary, expected=str(expected), num_qubits=qc.num_qubits
                    )
                    p = float(m["p_success"])
                    ci_lo = float(m["p_success_wilson95"][0])
                    threshold = float(args.min_success_prob)
                    passed = (ci_lo >= threshold) if bool(args.success_use_ci) else (p >= threshold)
                    eval_metrics = {
                        "metric": "success",
                        "measurement_basis": str(args.measurement_basis),
                        "pass": bool(passed),
                        "criterion": {
                            "min_success_prob": threshold,
                            "use_wilson95_lower_bound": bool(args.success_use_ci),
                        },
                        **m,
                    }

            subcircuit_results.append(
                {
                    "name": qc.name,
                    "partition_id": info.get("partition_id"),
                    "qubit_range": info.get("qubit_range"),
                    "num_qubits": qc.num_qubits,
                    "shots": shots,
                    "counts": counts_primary,
                    "counts_runs": counts_runs,
                    "system_messages": sys_msgs_all,
                    "job_ids": job_ids,
                    "evaluation": eval_metrics,
                }
            )

        reconstructed = executor.reconstruct_results(subcircuit_results, cut_result)

        overall_eval = None
        if args.eval:
            per_part = [sr.get("evaluation") for sr in subcircuit_results if sr.get("evaluation")]
            if str(args.eval_metric) == "z":
                max_abs_all = max((e.get("max_abs_z_diff", 0.0) for e in per_part), default=0.0)
                overall_eval = {
                    "metric": "z",
                    "measurement_basis": str(args.measurement_basis),
                    "max_abs_z_diff_across_partitions": max_abs_all,
                    "pass": bool(max_abs_all <= float(args.max_abs_z_diff) and all(e.get("pass") for e in per_part)),
                    "criterion": {"max_abs_z_diff": float(args.max_abs_z_diff)},
                }
            else:
                p_success_all = [float(e.get("p_success", 0.0)) for e in per_part]
                ci_low_all = [float((e.get("p_success_wilson95") or [0.0])[0]) for e in per_part]
                threshold = float(args.min_success_prob)
                passed = (
                    (min(ci_low_all) >= threshold) if bool(args.success_use_ci) else (min(p_success_all) >= threshold)
                ) if p_success_all else False
                overall_eval = {
                    "metric": "success",
                    "measurement_basis": str(args.measurement_basis),
                    "min_p_success_across_partitions": min(p_success_all) if p_success_all else 0.0,
                    "mean_p_success_across_partitions": (sum(p_success_all) / len(p_success_all)) if p_success_all else 0.0,
                    "min_p_success_wilson95_lower_across_partitions": min(ci_low_all) if ci_low_all else 0.0,
                    "pass": bool(passed),
                    "criterion": {
                        "min_success_prob": threshold,
                        "use_wilson95_lower_bound": bool(args.success_use_ci),
                    },
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "backend": backend.name,
            "num_qubits": num_qubits,
            "shots_per_subcircuit": shots,
            "measurement_basis": str(args.measurement_basis),
            "proof_circuit": str(args.proof_circuit),
            "input_bitstring": input_bs,
            "partition_labels": partition_labels,
            "cuts": cuts,
            "cut_result": {
                "num_partitions": cut_result.get("num_partitions"),
                "crossing_edges": cut_result.get("crossing_edges"),
            },
            "subcircuit_results": subcircuit_results,
            "reconstructed": reconstructed,
            "overall_evaluation": overall_eval,
            "disclaimer": (
                "Subcircuits were executed on Quantum Inspire. "
                "Reconstruction uses the project's current naive method and is not full circuit-cutting."
            ),
        }

    out_dir = Path(args.output)

    if args.search:
        search_summary= []
        best = None

        max_n = int(args.search_max_num_qubits)
        for n in range(2, max_n + 1):
            for p in range(1, int(args.max_partitions) + 1):
                labels = _contiguous_partition_labels(n, p)
                max_part = 0
                for pid in set(labels):
                    max_part = max(max_part, sum(1 for x in labels if x == pid))
                if max_part > int(args.max_partition_qubits):
                    continue

                args.num_qubits = n
                args.num_partitions = p
                print(f"SEARCH: trying n={n}, partitions={p} (max_part={max_part})")
                try:
                    payload = run_one(n, int(args.shots))
                    out_path = out_dir / (
                        f"qi_cut_qft_{n}q_{args.backend}_{p}parts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    _save_json(out_path, payload)

                    ok = bool((payload.get("overall_evaluation") or {}).get("pass", True))
                    rec = {
                        "num_qubits": n,
                        "num_partitions": p,
                        "ok": ok,
                        "file": str(out_path),
                    }
                    search_summary.append(rec)
                    if ok:
                        if best is None or n > best["num_qubits"] or (n == best["num_qubits"] and p < best["num_partitions"]):
                            best = rec
                except Exception as exc:
                    search_summary.append({"num_qubits": n, "num_partitions": p, "ok": False, "error": repr(exc)})

        summary_path = out_dir / f"qi_cut_qft_search_{args.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        _save_json(summary_path, {"backend": args.backend, "results": search_summary, "best": best})
        print(f"\nSearch summary saved to: {summary_path}")
        if best:
            print(f"Best passing scenario: QFT({best['num_qubits']}) with {best['num_partitions']} partitions")
        else:
            print("No passing scenario found under current evaluation criterion.")
        return 0

    if args.grid:
        grid_summary = []
        best= None

        for p in range(2, int(args.max_partitions) + 1):
            args.num_partitions = p
            n = p * int(args.max_partition_qubits)
            print(f"GRID: trying total QFT({n}) with {p} partitions (~{args.max_partition_qubits}q each)")
            try:
                payload = run_one(n, int(args.shots))
                out_path = out_dir / f"qi_cut_qft_{n}q_{args.backend}_{p}parts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                _save_json(out_path, payload)
                ok = bool((payload.get("overall_evaluation") or {}).get("pass", True))
                rec = {"num_qubits": n, "num_partitions": p, "ok": ok, "file": str(out_path)}
                grid_summary.append(rec)
                if ok:
                    if best is None or n > best["num_qubits"]:
                        best = rec
            except Exception as exc:
                grid_summary.append({"num_qubits": n, "num_partitions": p, "ok": False, "error": repr(exc)})

        summary_path = out_dir / f"qi_cut_qft_grid_{args.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        _save_json(summary_path, {"backend": args.backend, "results": grid_summary, "best": best})
        print(f"\nGrid summary saved to: {summary_path}")
        if best:
            print(f"Best passing scenario: QFT({best['num_qubits']}) with {best['num_partitions']} partitions")
        else:
            print("No passing scenario found under current evaluation criterion.")
        return 0

    if args.sweep:
        shots = int(args.sweep_shots) if args.sweep_shots is not None else int(args.shots)
        sweep_summary = []
        max_ok = None

        for n in range(int(args.min_num_qubits), int(args.max_num_qubits) + 1):
            print(f"SWEEP: trying QFT({n}) on {args.backend} (shots={shots})")
            try:
                payload = run_one(n, shots)
                out_path = out_dir / f"qi_cut_qft_{n}q_{args.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                _save_json(out_path, payload)
                sweep_summary.append({"n": n, "ok": True, "file": str(out_path)})
                max_ok = n
            except Exception as exc:
                sweep_summary.append({"n": n, "ok": False, "error": repr(exc)})
                print(f"FAILED at n={n}: {exc}")
                break

        summary_path = out_dir / f"qi_cut_qft_sweep_{args.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        _save_json(summary_path, {"backend": args.backend, "max_ok": max_ok, "results": sweep_summary})
        print(f"\nSweep summary saved to: {summary_path}")
        if max_ok is not None:
            print(f"Max size that completed: QFT({max_ok}) (with current 2-way partitioning)")
        else:
            print("No successful runs in sweep (check login/backend access).")
        return 0

    payload = run_one(int(args.num_qubits), int(args.shots))
    out_path = out_dir / f"qi_cut_qft_{args.num_qubits}q_{args.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _save_json(out_path, payload)
    print(f"\nSaved run data to: {out_path}")
    print("Note: this demonstrates hardware-grounded execution of the partitions.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
