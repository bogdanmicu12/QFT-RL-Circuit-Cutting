from __future__ import annotations

from math import pi
import math

from qiskit import QuantumCircuit, transpile
from qiskit.providers.exceptions import JobTimeoutError
from qiskit.exceptions import QiskitError

from qi_backend import get_tuna9_backend


def oracle_mark_state(qc: QuantumCircuit, marked: str) -> None:
    n = len(marked)

    for i, bit in enumerate(marked):
        if bit == "0":
            qc.x(i)

    qc.mcp(pi, list(range(n - 1)), n - 1)

    for i, bit in enumerate(marked):
        if bit == "0":
            qc.x(i)


def diffuser(qc: QuantumCircuit) -> None:
    n = qc.num_qubits
    qc.h(range(n))
    qc.x(range(n))

    qc.mcp(pi, list(range(n - 1)), n - 1)

    qc.x(range(n))
    qc.h(range(n))


def make_grover_circuit(n: int, marked: str, iters: int) -> QuantumCircuit:
    if len(marked) != n or any(c not in "01" for c in marked):
        raise ValueError("marked must be a length-n bitstring (e.g. '1011').")

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for _ in range(iters):
        oracle_mark_state(qc, marked)
        diffuser(qc)

    qc.measure(range(n), range(n))
    return qc


def _optimal_grover_iters(n: int) -> int:
    return max(1, int((pi / 4) * (2 ** (n / 2))))


def _expected_success_prob(n: int, iters: int) -> float:
    """Textbook success probability for 1 marked item for Grover."""
    N = 2**n
    theta = math.asin(1 / math.sqrt(N))
    return math.sin((2 * iters + 1) * theta) ** 2


def _choose_working_params(
    backend,
    marked_9: str,
    *,
    n_max: int = 9,
    min_n: int = 3,
    max_transpiled_size: int = 1800,
    max_transpiled_depth: int = 1400,
    optimization_level: int = 1,
    min_expected_success: float = 0.15,
):
    """
    The logic here is basically pick the largest n that likely runs and gives meaningful Grover amplification.

	Tuna-9 has practical limits on compiled program size and multi-controlled phase gates explode quickly with n, 
    so we start with the textbook iteration count and do down until the transpiled circuit fits within the hardware limits.
    
	I defined an *idealized* (literature) success probability above `min_expected_success`. If 9 qubits cannot meet this under 
    the hardware limits, we go down to the largest n that can.
    """
    best_fallback = None
    for n in range(n_max, min_n - 1, -1):
        marked = marked_9[:n]
        if len(marked) != n:
            marked = (marked_9 + ("1" * n))[:n]

        iters = _optimal_grover_iters(n)
        best_for_n = None
        while iters >= 1:
            qc = make_grover_circuit(n, marked, iters)
            tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
            if tqc.size() <= max_transpiled_size and tqc.depth() <= max_transpiled_depth:
                p = _expected_success_prob(n, iters)
                if best_fallback is None:
                    best_fallback = (n, marked, iters, tqc, p)
                if best_for_n is None or p > best_for_n[-1]:
                    best_for_n = (n, marked, iters, tqc, p)
            iters -= 1

        if best_for_n is not None and best_for_n[-1] >= min_expected_success:
            return best_for_n

    if best_fallback is not None:
        return best_fallback

    raise RuntimeError(
        "Could not find a Grover circuit setting that fits the backend size/depth limits"
    )


def _candidate_params(
    backend,
    marked_9: str,
    *,
    n_max: int = 9,
    min_n: int = 3,
    max_transpiled_size: int = 1800,
    max_transpiled_depth: int = 800,
    optimization_level: int = 1,
    min_expected_success: float = 0.80,
):
    """Return candidate configs ordered by descending n. It picks for each n the best (highest ideal success probability) 
    circuit that fits the size/depth constraints, then returns those meeting `min_expected_success`
    """
    candidates = []
    for n in range(n_max, min_n - 1, -1):
        marked = marked_9[:n]
        if len(marked) != n:
            marked = (marked_9 + ("1" * n))[:n]

        best_for_n = None
        iters = _optimal_grover_iters(n)
        while iters >= 1:
            qc = make_grover_circuit(n, marked, iters)
            tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
            if tqc.size() <= max_transpiled_size and tqc.depth() <= max_transpiled_depth:
                p = _expected_success_prob(n, iters)
                if best_for_n is None or p > best_for_n[-1]:
                    best_for_n = (n, marked, iters, tqc, p)
            iters -= 1

        if best_for_n is not None and best_for_n[-1] >= min_expected_success:
            candidates.append(best_for_n)

    return candidates


def _run_and_summarize(backend, *, tqc: QuantumCircuit, marked: str, shots: int):
    job = backend.run(tqc, shots=shots)
    print(f"Submitted Grover job id: {job.job_id()}")

    try:
        result = job.result(timeout=1800)
    except JobTimeoutError:
        print("Timed out waiting for Grover result.")
        print(f"Job id: {job.job_id()}")
        print(f"Status: {job.status()}")
        return None

    if hasattr(result, "system_messages"):
        print("System messages:")
        print(result.system_messages)

    try:
        counts = result.get_counts()
    except QiskitError as e:
        print("Could not read counts because the experiment failed.")
        print(str(e))
        return None

    success_key = marked[::-1]
    success_ct = counts.get(success_key, 0)
    success_p = success_ct / shots

    n = len(marked)
    baseline = 1 / (2**n)

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("Grover top counts (bitstring: shots):")
    print(top)
    print(f"Expected marked (Qiskit order): {success_key}")
    print(f"Observed success: {success_ct}/{shots} = {success_p:.4f} (baseline≈{baseline:.5f})")
    return {
        "n": n,
        "success_key": success_key,
        "success_p": success_p,
        "baseline": baseline,
        "top": top,
    }


def main() -> None:
    backend = get_tuna9_backend()

    shots = 4000
    marked_9 = "101101011"

    candidates = _candidate_params(
        backend,
        marked_9,
        n_max=9,
        min_n=3,
        max_transpiled_size=1800,
        max_transpiled_depth=800,
        optimization_level=1,
        min_expected_success=0.80,
    )

    if not candidates:
        print("No candidate Grover circuits met the size/depth/expected-success constraints.")
        print("Falling back to the largest circuit that fits size/depth limits.")
        n, marked, iters, tqc, expected_p = _choose_working_params(
            backend,
            marked_9,
            max_transpiled_size=1800,
            max_transpiled_depth=800,
            optimization_level=1,
            min_expected_success=0.0,
        )
        print(f"Using n={n}, iters={iters}, marked={marked}")
        print(f"Ideal success p≈{expected_p:.3f}")
        print(f"Transpiled depth={tqc.depth()}, size={tqc.size()} instructions")
        _run_and_summarize(backend, tqc=tqc, marked=marked, shots=shots)
        return

    success_ratio_target = 2.0
    best = None
    for n, marked, iters, tqc, expected_p in candidates:
        baseline = 1 / (2**n)
        print(f"\nTrying n={n}, iters={iters}, marked={marked}")
        print(f"Ideal success p≈{expected_p:.3f} (baseline≈{baseline:.5f})")
        print(f"Transpiled depth={tqc.depth()}, size={tqc.size()} instructions")

        out = _run_and_summarize(backend, tqc=tqc, marked=marked, shots=shots)
        if out is None:
            continue

        ratio = out["success_p"] / out["baseline"] if out["baseline"] else 0.0
        if best is None or ratio > best["ratio"]:
            best = {"n": n, "iters": iters, "marked": marked, "ratio": ratio}

        if ratio >= success_ratio_target:
            print(f"\nSelected n={n} (first meeting success ratio ≥ {success_ratio_target}× baseline).")
            return

    if best is not None:
        print(
            f"\nNo run reached {success_ratio_target}× baseline; best was n={best['n']} "
            f"(iters={best['iters']}) at {best['ratio']:.2f}× baseline."
        )


if __name__ == "__main__":
    main()