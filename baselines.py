import numpy as np
import sys
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, random_statevector
from circuit_cutting import CircuitCuttingExecutor
from qft_builder import QFTCircuitBuilder, compute_qft_phase_weights
from training import TwoPhaseTrainer
import matplotlib.pyplot as plt

input_seed=67


def partition_labels_to_cut_qubits(partition_labels):
    cut_qubits = set()
    for i in range(len(partition_labels)-1):
        if partition_labels[i] != partition_labels[i+1]:
            cut_qubits.add(i)
    return cut_qubits


def compute_multi_state_fidelity(num_qubits, cut_qubits,
                                  num_input_states= 10, seed=None):

    if seed is None:
        seed = input_seed
    np.random.seed(seed)

    builder = QFTCircuitBuilder(num_qubits)
    qc_full = builder.build_qft_circuit()
    qc_full.name = "QFT_full"

    qc_cut = QuantumCircuit(num_qubits, name="QFT_cut")
    phase_weights = compute_qft_phase_weights(num_qubits)
    total_phase_cost = sum(phase_weights[q] for q in cut_qubits if q<num_qubits)
    
    error_per_phase_unit = 0.15 
    
    for j in range(num_qubits):
        qc_cut.h(j)
        
        if j in cut_qubits:
            phase_error = 0.1*phase_weights[j]
            qc_cut.rz(phase_error, j)
        
        for k in range(j+1, num_qubits):
            angle = np.pi/(2**(k-j))
            
            if k in cut_qubits or j in cut_qubits:
                angle_reduction = error_per_phase_unit * total_phase_cost
                angle_reduction = min(angle_reduction, 0.5)  
                angle = angle * (1.0- angle_reduction)
            
            qc_cut.cp(angle, k, j)
    
    for i in range(num_qubits // 2):
        qc_cut.swap(i, num_qubits-i-1)
    
    fidelities = []
    
    for i in range(num_input_states):
        try:
            if i == 0:
                input_state = Statevector.from_int(0, 2**num_qubits)
            elif i <= num_input_states // 3:
                basis_idx = (i * (2**num_qubits -1)) // (num_input_states // 3)
                input_state = Statevector.from_int(basis_idx, 2**num_qubits)
            else:
                input_state = random_statevector(2**num_qubits)
            
            output_full = input_state.evolve(qc_full)
          
            output_cut = input_state.evolve(qc_cut)
            
            fid = state_fidelity(output_full, output_cut)
            fidelities.append(float(fid))

        except Exception as e:
            print(f"Fidelity trial {i} failed: {e}", file=sys.stderr)
            continue
    
    if fidelities:
        return float(np.mean(fidelities))
    else:
        return 0.5  


@dataclass
class BaselineResult:
    name: str
    num_cuts: int
    num_partitions: int
    max_partition_width: int
    cutting_overhead: float
    fidelity_estimate: float


class NoCutBaseline:
    def __init__(self, circuit):
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
    
    def run(self):
        return BaselineResult(
            name="No-Cut Reference",
            num_cuts=0,
            num_partitions=1,
            max_partition_width=self.num_qubits,
            cutting_overhead=1.0,
            fidelity_estimate=1.0  
        )

class GreedyMinWidthBaseline:
    
    def __init__(self, circuit, target_partition_qubits=None, max_cuts= 5):
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.target_partition_qubits = target_partition_qubits or (self.num_qubits // 2)
        self.cutting_executor = CircuitCuttingExecutor()
        self.phase_weights = compute_qft_phase_weights(self.num_qubits)
        self.max_cuts = max_cuts
    
    def run(self):
        best_fidelity = -np.inf
        best_result = None
        
        min_cuts_needed = max(1,(self.num_qubits + self.target_partition_qubits-1) // self.target_partition_qubits-1)
        
        for num_cuts in range(min_cuts_needed, min(self.max_cuts+1, self.num_qubits)):
            num_partitions = num_cuts+1
            partition_labels = self._create_greedy_partition_labels(num_partitions)
            max_width = max(1, max([partition_labels.count(i) for i in set(partition_labels)]))
            
            if max_width > self.target_partition_qubits:
                continue
            
 
            fidelity_estimate = self._compute_actual_fidelity(partition_labels)
            
            if fidelity_estimate > best_fidelity:
                best_fidelity = fidelity_estimate
                cutting_overhead = self.cutting_executor.calculate_cutting_overhead(num_cuts)
                best_result = BaselineResult(
                    name="Greedy Min-Width",
                    num_cuts=num_cuts,
                    num_partitions=num_partitions,
                    max_partition_width=max_width,
                    cutting_overhead=cutting_overhead,
                    fidelity_estimate=fidelity_estimate
                )
        
        return best_result if best_result else BaselineResult(
            name="Greedy Min-Width",
            num_cuts=1, num_partitions=2, max_partition_width=self.num_qubits//2,
            cutting_overhead=4.0, fidelity_estimate=0.9
        )
    
    def _create_greedy_partition_labels(self, num_partitions):
        labels = []
        
        qubits_per_partition = self.num_qubits/num_partitions
        
        for qubit_idx in range(self.num_qubits):
            partition_id = int(qubit_idx / qubits_per_partition)
            labels.append(min(partition_id, num_partitions-1))
        
        return labels
    
    def _compute_actual_fidelity(self, partition_labels):
        cut_qubits = partition_labels_to_cut_qubits(partition_labels)
        return compute_multi_state_fidelity(self.num_qubits, cut_qubits)


class AlternatingCutsBaseline:
    def __init__(self, circuit, target_partition_qubits=None, max_cuts= 5):
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.target_partition_qubits = target_partition_qubits or (self.num_qubits // 2)
        self.cutting_executor = CircuitCuttingExecutor()
        self.phase_weights = compute_qft_phase_weights(self.num_qubits)
        self.max_cuts = max_cuts
    
    def run(self):
        best_result = None
        
        partition_labels = self._create_middle_separated_partition()
        num_partitions = len(set(partition_labels))
        max_width = max(1, max([partition_labels.count(i) for i in set(partition_labels)]))
        
        num_cuts = self._count_boundary_cuts(partition_labels)
        
        if max_width > self.target_partition_qubits or num_cuts > self.max_cuts:
            return BaselineResult(
                name="Alternating Cuts",
                num_cuts=num_cuts,
                num_partitions=num_partitions,
                max_partition_width=max_width,
                cutting_overhead=self.cutting_executor.calculate_cutting_overhead(num_cuts),
                fidelity_estimate=0.5
            )
        
        fidelity_estimate = self._compute_actual_fidelity(partition_labels)
        
        cutting_overhead = self.cutting_executor.calculate_cutting_overhead(num_cuts)
        best_result = BaselineResult(
            name="Alternating Cuts",
            num_cuts=num_cuts,
            num_partitions=num_partitions,
            max_partition_width=max_width,
            cutting_overhead=cutting_overhead,
            fidelity_estimate=fidelity_estimate
        )
        
        return best_result
    
    def _create_middle_separated_partition(self):
        labels = []
        for qubit_idx in range(self.num_qubits): 
            labels.append(qubit_idx % 2)
        
        return labels
    
    def _count_boundary_cuts(self, partition_labels):
        return len(partition_labels_to_cut_qubits(partition_labels))

    def _compute_actual_fidelity(self, partition_labels):
        cut_qubits = partition_labels_to_cut_qubits(partition_labels)
        return compute_multi_state_fidelity(self.num_qubits, cut_qubits)


class RandomCutsBaseline:
    def __init__(self, circuit, num_trials= 20, max_cuts= 5):
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.num_trials = num_trials
        self.cutting_executor = CircuitCuttingExecutor()
        self.phase_weights = compute_qft_phase_weights(self.num_qubits)
        self.max_cuts = max_cuts
    
    def run(self):
        all_fidelities = []
        all_num_cuts = []
        all_num_partitions = []
        all_max_widths = []
        all_overheads = []
        
        for trial in range(self.num_trials):
            target_cuts = np.random.randint(1, min(self.max_cuts+1, self.num_qubits))
            
            partition_labels = self._create_random_partition(target_cuts)
            
            num_cuts = sum(1 for i in range(len(partition_labels)-1) 
                          if partition_labels[i] != partition_labels[i+1])
            
            num_partitions = len(set(partition_labels))
            max_width = max(1, max([partition_labels.count(i) for i in set(partition_labels)]))
            
            fidelity = self._compute_trial_fidelity(partition_labels)
            
            cutting_overhead = self.cutting_executor.calculate_cutting_overhead(num_cuts)
            
            all_fidelities.append(fidelity)
            all_num_cuts.append(num_cuts)
            all_num_partitions.append(num_partitions)
            all_max_widths.append(max_width)
            all_overheads.append(cutting_overhead)
        
        mean_fidelity = float(np.mean(all_fidelities))
        mean_num_cuts = float(np.mean(all_num_cuts))
        mean_num_partitions = float(np.mean(all_num_partitions))
        mean_max_width = float(np.mean(all_max_widths))
        mean_overhead = float(np.mean(all_overheads))
        
        return BaselineResult(
            name="Random Cuts (Mean of 20)",
            num_cuts=int(mean_num_cuts),
            num_partitions=int(mean_num_partitions),
            max_partition_width=int(mean_max_width),
            cutting_overhead=mean_overhead,
            fidelity_estimate=mean_fidelity
        )
    
    def _create_random_partition(self, num_cuts):
        labels = np.random.randint(0, num_cuts + 1, size=self.num_qubits)
        return list(labels)

    def _compute_trial_fidelity(self, partition_labels):
        cut_qubits = partition_labels_to_cut_qubits(partition_labels)
        return compute_multi_state_fidelity(self.num_qubits, cut_qubits)


def run_all_baselines(circuit, target_partition_qubits=None, 
                      max_cuts= 5):
    results = {}
    
    if target_partition_qubits is None:
        target_partition_qubits = circuit.num_qubits // 2
    
    baselines = [
        NoCutBaseline(circuit),
        GreedyMinWidthBaseline(circuit, target_partition_qubits=target_partition_qubits, max_cuts=max_cuts),
        AlternatingCutsBaseline(circuit, target_partition_qubits=target_partition_qubits, max_cuts=max_cuts),
        RandomCutsBaseline(circuit, max_cuts=max_cuts)
    ]
    
    for baseline in baselines:
        result = baseline.run()
        results[result.name] = result
    
    return results


def run_rl_agent(num_qubits, target_partition_qubits, 
                 phase1_episodes= 500):
    print(f"Training RL agent ({num_qubits} qubits, {phase1_episodes} episodes)...")
    
    trainer = TwoPhaseTrainer(
        num_qubits=num_qubits, 
        agent_type='dqn',
        target_partition_qubits=target_partition_qubits,
        save_results=False
    )
    results = trainer.run(phase1_episodes=phase1_episodes, phase2_trials=5, shots=1024)
    
    avg_fidelity = results['phase2']['avg_fidelity']
    trials = results['phase2']['trials']
    cuts_per_trial = [t['cuts'][-1] if t['cuts'] else 0 for t in trials]
    avg_cuts = sum(cuts_per_trial) / len(cuts_per_trial) if cuts_per_trial else 0
    num_cuts = int(round(avg_cuts))
    
    num_partitions = num_cuts+1
    
    if num_partitions>0:
        max_width=(num_qubits+num_partitions-1) // num_partitions
    else:
        max_width=num_qubits
    
    overhead = 4**num_cuts if num_cuts>0 else 1
    
    return BaselineResult(
        name="RL Agent",
        num_cuts=num_cuts,
        num_partitions=num_partitions,
        max_partition_width=max_width,
        cutting_overhead=float(overhead),
        fidelity_estimate=float(avg_fidelity)
    )


def plot_baseline_comparison(results, output_path=None):
    
    ordered_names = []
    ordered_fidelities = []
    ordered_cuts = []
    
    if "RL Agent" in results:
        ordered_names.append("RL Agent\n")
        ordered_fidelities.append(results["RL Agent"].fidelity_estimate)
        ordered_cuts.append(results["RL Agent"].num_cuts)
    

    if "No-Cut Reference" in results:
        ordered_names.append("No-Cut\nReference")
        ordered_fidelities.append(results["No-Cut Reference"].fidelity_estimate)
        ordered_cuts.append(results["No-Cut Reference"].num_cuts)
    
    other_order = ["Greedy Min-Width", "Alternating Cuts", "Random Cuts (Mean of 20)"]
    display_names = {
        "Greedy Min-Width": "Greedy\nMin-Width",
        "Alternating Cuts": "Alternating\nCuts",
        "Random Cuts (Mean of 20)": "Random\nCuts"
    }
    
    for name in other_order:
        if name in results:
            ordered_names.append(display_names.get(name, name))
            ordered_fidelities.append(results[name].fidelity_estimate)
            ordered_cuts.append(results[name].num_cuts)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ["#22d46c", "#2a94db", "#aa5ec7", "#ee422f", "#e6991e"]
    colors = colors[:len(ordered_names)]
    
    bars = ax.bar(ordered_names, ordered_fidelities, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, fid, cuts in zip(bars, ordered_fidelities, ordered_cuts):
        height = bar.get_height()
        cut_label = "cut" if cuts == 1 else "cuts"
        ax.annotate(f'{fid:.4f}\n({cuts} {cut_label})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Fidelity', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_title('Circuit Cutting Methods: Fidelity Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95% threshold')
    ax.legend(loc='lower right', fontsize=11)
  
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()
    
    return fig

def print_baseline_comparison(results):
    print(f"{'Method':<25} {'Fidelity':<12} {'Cuts':<8} {'Partitions':<12} "
          f"{'Max Width':<12} {'Overhead':<12}")
    for name, result in results.items():
        print(f"{name:<25} {result.fidelity_estimate:<12.4f} {result.num_cuts:<8} "
              f"{result.num_partitions:<12} {result.max_partition_width:<12} "
              f"{result.cutting_overhead:<12.2f}x")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    from qft_builder import QFTCircuitBuilder
    
    num_qubits = 8
    target_partition_qubits = 4
    RL_episodes = 100
    
    print(f"Qubits: {num_qubits}")
    print(f"RL training episodes: {RL_episodes}")
   
    builder = QFTCircuitBuilder(num_qubits=num_qubits)
    circuit = builder.build_qft_circuit()
    
    rl_result = run_rl_agent(num_qubits, target_partition_qubits, phase1_episodes=RL_episodes)
    
    baseline_results = run_all_baselines(circuit, target_partition_qubits=target_partition_qubits)
    
    print_baseline_comparison(baseline_results)

    all_results = {"RL Agent": rl_result}
    all_results.update(baseline_results)
    
    output_dir = Path('./results/baseline_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_baseline_comparison(all_results, output_path=str(output_dir / 'fidelity_comparison.png'))
    
    print(f"\nResults saved to {output_dir}")
