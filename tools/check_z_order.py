from __future__ import annotations

import sys
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_cut_qft_quantuminspire as r


def main():
    sim = AerSimulator()

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    z_ideal = r._z_expectations_ideal(qc)
    counts = sim.run(transpile(qc, sim), shots=10000).result().get_counts()
    z_sample = r._z_expectations_from_counts(counts, 1)
    print("1q H ideal", z_ideal, "sample", z_sample)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.measure_all()
    z2_ideal = r._z_expectations_ideal(qc2)
    counts2 = sim.run(transpile(qc2, sim), shots=20000).result().get_counts()
    z2_sample = r._z_expectations_from_counts(counts2, 2)
    print("2q H on q0 ideal", z2_ideal, "sample", z2_sample)

    qc3 = QuantumCircuit(2)
    qc3.x(1)
    qc3.measure_all()
    z3_ideal = r._z_expectations_ideal(qc3)
    counts3 = sim.run(transpile(qc3, sim), shots=20000).result().get_counts()
    z3_sample = r._z_expectations_from_counts(counts3, 2)
    print("2q X on q1 ideal", z3_ideal, "sample", z3_sample, "example_counts", list(counts3.items())[:3])


if __name__ == "__main__":
    main()
