import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
import networkx as nx
import matplotlib.pyplot as plt

def compute_qft_phase_weights(num_qubits):
    phase_weights = np.zeros(num_qubits)
    for j in range(num_qubits):
        phase_sum = 0.0
        for k in range(j+1, num_qubits):
            phase_sum += 1.0/(2.0**(k-j))
        phase_weights[j] = (2.0*np.pi) * phase_sum
    if np.max(phase_weights) > 0:
        phase_weights = phase_weights / np.max(phase_weights)
    return phase_weights


class QFTCircuitBuilder:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = None
        
    def build_qft_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
    
        for j in range(self.num_qubits):
            qc.h(j)
            
            for k in range(j+1, self.num_qubits):
                angle = np.pi/(2**(k-j))
                qc.cp(angle, k, j)
        
        for i in range(self.num_qubits // 2):
            qc.swap(i, self.num_qubits-i-1)
        
        self.circuit = qc
        return qc


if __name__ == "__main__":
    print("Building QFT circuit for 8 qubits...")
    builder = QFTCircuitBuilder(num_qubits=8)

    circuit = builder.build_qft_circuit()
    print(f"\nCircuit depth: {circuit.depth()}")
    print(f"Circuit size: {circuit.size()}")
    print("\nCircuit:")
    print(circuit.draw(output='text'))
