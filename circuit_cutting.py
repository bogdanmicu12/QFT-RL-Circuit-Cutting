from qiskit import QuantumCircuit

class CircuitCuttingExecutor:
    def __init__(self, max_subcircuit_qubits = 4):
        self.max_subcircuit_qubits = max_subcircuit_qubits
        self._cache = {}
    
    
    def apply_wire_cuts(
        self, 
        circuit, 
        cuts,
        partition_labels = None
    ):
        if partition_labels is None:
            partition_labels = self._create_partition_from_cuts(
                circuit.num_qubits, cuts
            )
        
        partitions = {}
        for qubit_idx, label in enumerate(partition_labels):
            if label not in partitions:
                partitions[label] = []
            partitions[label].append(qubit_idx)
        
        subcircuits = []
        for partition_id, qubit_list in sorted(partitions.items()):
            subcircuit_info = self._extract_subcircuit(
                circuit, qubit_list, partition_id, cuts
            )
            subcircuits.append(subcircuit_info)
        
        crossing_edges = self._count_crossing_edges(circuit, partition_labels)
        
        return {
            "subcircuits": subcircuits,
            "num_partitions": len(partitions),
            "partition_labels": partition_labels,
            "cuts": cuts,
            "crossing_edges": crossing_edges
        }
    
    def _create_partition_from_cuts(self, num_qubits, cuts):
        partition_labels = [0] * num_qubits
        cut_qubits = set(q for _, q in cuts)
        qubits_per_partition = max(1, num_qubits // (len(cuts) + 1))
        
        for i in range(num_qubits):
            partition_labels[i] = min(i // qubits_per_partition, len(cuts))
        
        return partition_labels
    
    def _extract_subcircuit(self, circuit, qubit_indices,
                            partition_id):
        qubit_map = {orig: new for new, orig in enumerate(sorted(qubit_indices))}
        n_qubits = len(qubit_indices)
        subcircuit = QuantumCircuit(n_qubits, name=f"subcircuit_{partition_id}")
        
        for instruction in circuit.data:
            gate_qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            if all(q in qubit_map for q in gate_qubits):
                new_qubits = [qubit_map[q] for q in gate_qubits]
                subcircuit.append(instruction.operation, 
                                [subcircuit.qubits[q] for q in new_qubits])
        
        return {
            "circuit": subcircuit,
            "partition_id": partition_id,
            "qubit_range": sorted(qubit_indices),
            "qubit_map": qubit_map,
            "num_qubits": n_qubits
        }
    
    def _count_crossing_edges(self, circuit, partition_labels):
        crossing = 0
        for instruction in circuit.data:
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            if len(qubits) >= 2:
                partitions = set(partition_labels[q] for q in qubits)
                if len(partitions) > 1:
                    crossing += 1
        return crossing
    
    def calculate_cutting_overhead(self, num_cuts):
        return 4 ** num_cuts
    
    def reconstruct_results(self, subcircuit_results, cut_result):
        partition_labels = cut_result["partition_labels"]
        reconstructed_counts = {}
        
        for subresult in subcircuit_results:
            counts = subresult["counts"]
            total = sum(counts.values())
            for bitstring, count in counts.items():
                prob = count / total
                if bitstring not in reconstructed_counts:
                    reconstructed_counts[bitstring] = 0
                reconstructed_counts[bitstring] += prob
        
        total = sum(reconstructed_counts.values())
        if total > 0:
            reconstructed_probs = {k: v/total for k, v in reconstructed_counts.items()}
        else:
            reconstructed_probs = {}
        
        return {
            "counts": reconstructed_counts,
            "probabilities": reconstructed_probs,
            "num_cuts": len(cut_result["cuts"]),
            "num_partitions": cut_result["num_partitions"]
        }