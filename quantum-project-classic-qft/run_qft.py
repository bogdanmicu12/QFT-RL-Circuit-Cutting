from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.providers.exceptions import JobTimeoutError
from qiskit.exceptions import QiskitError
from qiskit.synthesis.qft import synth_qft_full

from qi_backend import get_tuna9_backend


def _prepare_basis_state(qc: QuantumCircuit, x: int) -> None:
    for i in range(qc.num_qubits):
        if (x >> i) & 1:
            qc.x(i)


def make_qft_roundtrip_circuit(n: int, x: int, *, approximation_degree: int = 0) -> QuantumCircuit:
    """QFT then inverse-QFT; should measure back the input |x> ideally.

    This is a more conclusive check, as measuring right after a standalone QFT often looks nearly uniform because the information is in
    relative phases.
    """
    if x < 0 or x >= 2**n:
        raise ValueError("x must satisfy 0 <= x < 2**n")

    qc = QuantumCircuit(n, n)
    _prepare_basis_state(qc, x)
    qc.compose(
        synth_qft_full(
            n, do_swaps=True, approximation_degree=approximation_degree, inverse=False
        ),
        range(n),
        inplace=True,
    )
    qc.compose(
        synth_qft_full(
            n, do_swaps=True, approximation_degree=approximation_degree, inverse=True
        ),
        range(n),
        inplace=True,
    )
    qc.measure(range(n), range(n))
    return qc


def _run_roundtrip(
    backend,
    *,
    n: int,
    x: int,
    shots: int,
    approximation_degree: int,
    timeout_s: int = 1800,
):
    qc = make_qft_roundtrip_circuit(n, x, approximation_degree=approximation_degree)
    tqc = transpile(qc, backend=backend, optimization_level=1)
    print(
        f"Transpiled depth={tqc.depth()}, size={tqc.size()} instructions "
        f"(approximation_degree={approximation_degree})"
    )

    job = backend.run(tqc, shots=shots)
    print(f"Submitted QFT job id: {job.job_id()}")

    try:
        result = job.result(timeout=timeout_s)
    except JobTimeoutError:
        print("Timed out waiting for QFT result.")
        print(f"Job id: {job.job_id()}")
        print(f"Status: {job.status()}")
        return None

    try:
        counts = result.get_counts()
    except QiskitError as e:
        print("Could not read counts because the experiment failed.")
        print(str(e))
        if hasattr(result, "system_messages"):
            print("System messages:")
            print(result.system_messages)
        return None

    expected_key = format(x, f"0{n}b")
    success_ct = counts.get(expected_key, 0)
    success_p = success_ct / shots
    baseline = 1 / (2**n)
    ratio = (success_p / baseline) if baseline else 0.0

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("QFT roundtrip top counts (bitstring: shots):")
    print(top)
    print(f"Expected output (Qiskit order): {expected_key}")
    print(
        f"Observed success: {success_ct}/{shots} = {success_p:.4f} "
        f"(baseline≈{baseline:.5f}, ratio≈{ratio:.1f}x)"
    )

    top_key, top_ct = top[0]
    is_top = (top_key == expected_key)
    return {
        "n": n,
        "approximation_degree": approximation_degree,
        "success_p": success_p,
        "baseline": baseline,
        "ratio": ratio,
        "is_top": is_top,
        "top_key": top_key,
        "top_ct": top_ct,
    }


def main() -> None:
    backend = get_tuna9_backend()

    shots = 4000
    x = 1
    min_success_p = 0.10

    best = None
    for n in range(9, 2, -1):
        if x >= 2**n:
            continue
        for approximation_degree in (0, 1, 2, 3, 4):
            print(
                f"\nTrying QFT roundtrip with n={n}, x={x}, approximation_degree={approximation_degree}"
            )
            out = _run_roundtrip(
                backend,
                n=n,
                x=x,
                shots=shots,
                approximation_degree=approximation_degree,
            )
            if out is None:
                continue
            if best is None or out["ratio"] > best["ratio"]:
                best = out
            if out["is_top"] and out["success_p"] >= min_success_p:
                print(
                    f"\nSelected n={n} (expected output is top and success ≥ {min_success_p:.0%})."
                )
                return

    if best is not None:
        print(
            f"\nNo run met the conclusive criteria (top outcome and success ≥ {min_success_p:.0%}). "
            f"Best ratio seen was n={best['n']} (approx={best['approximation_degree']}) at {best['ratio']:.1f}x baseline."
        )


if __name__ == "__main__":
    main()
