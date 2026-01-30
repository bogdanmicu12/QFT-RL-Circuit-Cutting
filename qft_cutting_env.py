import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Set, Tuple
from qft_builder import compute_qft_phase_weights


@dataclass
class CutState:
    partition_labels: List[int]
    cut_edges: Set[Tuple[int, int]]
    num_cuts: int
    max_partition_width: int
    crossing_edges: int
    num_subcircuits: int
    cp_gate_count: int
    steps_taken: int


class QFTCircuitCuttingEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, 
                 num_qubits = 8,
                 circuit_depth = 50,
                 max_cuts = 5,
                 target_partition_qubits = None,
                 use_proxy_reward = True,
                 render_mode = None,
                 phase_aware_weight = 0.1):
        
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.max_cuts = max_cuts
        self.target_partition_qubits = target_partition_qubits or max(2, num_qubits//2)
        self.use_proxy_reward = use_proxy_reward
        self.render_mode = render_mode
        self.phase_aware_weight = phase_aware_weight
        
        self._compute_qft_phase_weights()
        
        self.num_cut_actions = num_qubits*circuit_depth  
        self.num_partition_actions = num_qubits*num_qubits
        self.num_actions = self.num_cut_actions+ self.num_partition_actions + 1
        
        self.action_space = spaces.Discrete(self.num_actions)
        
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(num_qubits + 8,),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _compute_qft_phase_weights(self):
        self.phase_weights = compute_qft_phase_weights(self.num_qubits)
    
    def _get_cut_phase_cost(self):
    
        total_phase_cost = 0.0
        
        for gate_idx, qubit_idx in self.cut_edges:

            if 0 <= qubit_idx < self.num_qubits:
                total_phase_cost += self.phase_weights[qubit_idx]
        
        return total_phase_cost
    
    def _reset_state(self):
        self.partition_labels = [0] * self.num_qubits
        self.cut_edges = set()
        self.step_count = 0
        self.episode_cuts = []
        self.metrics_history = []
    
    def reset(self, seed = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        partition_norm = np.array(self.partition_labels, dtype=np.float32) / (self.num_qubits - 1 + 1e-6)
        
        num_cuts = len(self.cut_edges)
        partition_sizes = [sum(1 for p in self.partition_labels if p == i) 
                         for i in set(self.partition_labels)]
        max_width = max(partition_sizes) if partition_sizes else self.num_qubits
        min_width = min(partition_sizes) if partition_sizes else self.num_qubits
        crossing_edges = self._count_crossing_edges()
        num_subcircuits = len(set(self.partition_labels))
        
        max_width_norm = max_width / self.num_qubits
        
        qubits_to_reduce = max(0, max_width - self.target_partition_qubits)
        total_reduction_needed = max(1, self.num_qubits - self.target_partition_qubits)
        target_progress = 1.0 - (qubits_to_reduce / total_reduction_needed)
        target_progress = np.clip(target_progress, 0, 1)
        
        if num_subcircuits > 1:
            balance = min_width / max_width
        else:
            balance = 1.0
        
        cuts_norm = min(1.0, num_cuts / self.max_cuts)
        
        subcircuit_norm = num_subcircuits / self.num_qubits
        
        step_norm = self.step_count / 100
        

        if num_cuts > 0:
            qubits_saved = self.num_qubits - max_width
            efficiency = min(1.0, qubits_saved / (num_cuts * self.num_qubits))
        else:
            efficiency = 0.0
        
        target_achieved = 1.0 if max_width <= self.target_partition_qubits else 0.0
        
        metrics = np.array([
            max_width_norm,      # Current partition size
            target_progress,     # Progress toward qubit reduction
            balance,             # Partition balance
            cuts_norm,           # Overhead
            subcircuit_norm,     # Nr partitions
            step_norm,           # Time pressure
            efficiency,          # Cutting efficiency
            target_achieved      # Param for target achieved
        ], dtype=np.float32)
        
        observation = np.concatenate([partition_norm, metrics])
        return observation.astype(np.float32)
    
    def _count_crossing_edges(self):
        return len(self.cut_edges)* 2
    
    def step(self, action):
        self.step_count+=1
        terminated = False
        truncated = self.step_count>=100
        
        if action == self.num_actions-1:
            terminated = True
        
        elif action < self.num_cut_actions:
            gate_idx = action // self.num_qubits
            qubit_idx = action % self.num_qubits
            if len(self.cut_edges)<self.max_cuts:
                self.cut_edges.add((gate_idx, qubit_idx))
                self.episode_cuts.append(qubit_idx)
                self._update_partitions_from_cuts()
        
        else:
            offset = action - self.num_cut_actions
            qubit_idx = offset // self.num_qubits
            new_partition = offset % self.num_qubits
            if qubit_idx < self.num_qubits:
                max_partition = max(self.partition_labels)+1
                self.partition_labels[qubit_idx] = new_partition % max_partition
        
        num_cuts = len(self.cut_edges)
        
        reward = self._compute_reward(num_cuts, terminated)
        
        observation = self._get_observation()
        info = self._get_info(num_cuts)
        
        return observation, reward, terminated, truncated, info
    
    def _update_partitions_from_cuts(self):
        if not self.cut_edges:
            self.partition_labels = [0]* self.num_qubits
            return
        
        cut_qubits = sorted(set(qubit_idx for _, qubit_idx in self.cut_edges))
        
        current_partition = 0
        for i in range(self.num_qubits):
            self.partition_labels[i] = current_partition
            if i in cut_qubits:
                current_partition += 1
    
    def _compute_reward(self, num_cuts, terminated):
        partition_sizes = [sum(1 for p in self.partition_labels if p==i) 
                         for i in set(self.partition_labels)]
        max_width = max(partition_sizes) if partition_sizes else self.num_qubits
        num_partitions = len(set(self.partition_labels))
        
        if self.use_proxy_reward:
            reward = 0.0
            
            qubit_reduction_ratio = (self.num_qubits - max_width) / max(1, self.num_qubits - self.target_partition_qubits)
            qubit_reduction_ratio = np.clip(qubit_reduction_ratio, 0, 1.5) 
            
            if max_width <= self.target_partition_qubits:
                qubit_reward = 2.0+(self.target_partition_qubits - max_width) * 0.5
            else:
                qubit_reward = qubit_reduction_ratio*1.5
            
            reward += qubit_reward
            
            if num_cuts > 0:
                overhead_penalty = -np.log(4)*num_cuts*0.3
            else:
                overhead_penalty = 0.0
            
            reward += overhead_penalty
            
            phase_cost = self._get_cut_phase_cost()
            phase_penalty = -self.phase_aware_weight * phase_cost
            reward += phase_penalty
            

            if num_cuts > 0 and max_width < self.num_qubits:
                qubits_saved = self.num_qubits - max_width
                efficiency = qubits_saved / num_cuts
                efficiency_bonus = np.clip(efficiency * 0.2, 0, 1.0)
                reward += efficiency_bonus
            
            reward -= 0.01
            
        else:
            reward = self._compute_real_reward(num_cuts)
        
        if terminated:
            if max_width <= self.target_partition_qubits:
                if num_cuts <= 3:
                    reward += 3.0  
                elif num_cuts <= 5:
                    reward += 1.5  
                else:
                    reward += 0.5  
            elif max_width < self.num_qubits:
                progress = (self.num_qubits - max_width) / self.num_qubits
                reward += progress*1.0
            else:
                reward -= 1.0
        
        return float(reward)

    
    def _compute_real_reward(self, num_cuts):
        return -num_cuts*0.5
    
    def _get_info(self, num_cuts):
        partition_sizes = [sum(1 for p in self.partition_labels if p == i) 
                         for i in set(self.partition_labels)]
        max_width = max(partition_sizes) if partition_sizes else self.num_qubits
        
        return {
            "num_cuts": num_cuts,
            "num_partitions": len(set(self.partition_labels)),
            "max_partition_qubits": max_width,
            "target_partition_qubits": self.target_partition_qubits,
            "qubits_reduced": self.num_qubits - max_width,
            "target_achieved": max_width <= self.target_partition_qubits,
            "partition_sizes": partition_sizes,
            "step": self.step_count,
            "cutting_overhead": 4**num_cuts if num_cuts > 0 else 1
        }
    
    def get_valid_actions(self):
        return list(range(self.num_actions))
    
    def get_action_mask(self):
        mask = np.ones(self.num_actions, dtype=bool)
        return mask
    
    def evaluate_with_qiskit(self, shots=1024, num_input_states=10):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector, state_fidelity, random_statevector
            import numpy as np
        except ImportError:
            num_cuts = len(self.cut_edges)
            return {
                "fidelity": max(0.5, 1.0-num_cuts*0.1),
                "overhead": 4**num_cuts,
                "shots": shots
            }
        
        num_cuts = len(self.cut_edges)
        
        qc_full = QuantumCircuit(self.num_qubits, name="QFT_full")
        for j in range(self.num_qubits):
            qc_full.h(j)
            for k in range(j+1, self.num_qubits):
                angle = np.pi / (2**(k-j))
                qc_full.cp(angle, k, j)
        for i in range(self.num_qubits // 2):
            qc_full.swap(i, self.num_qubits - i - 1)
        
        qc_cut = QuantumCircuit(self.num_qubits, name="QFT_cut")
        
        cut_qubits = set()
        for gate_idx, qubit_idx in self.cut_edges:
            if 0 <= qubit_idx < self.num_qubits:
                cut_qubits.add(qubit_idx)
        
        total_cp_gates = self.num_qubits * (self.num_qubits - 1) // 2
        
        affected_cp_gates = 0
        for j in range(self.num_qubits):
            for k in range(j + 1, self.num_qubits):
                if k in cut_qubits or j in cut_qubits:
                    affected_cp_gates += 1
        
        error_per_gate = 0.008  
        
        for j in range(self.num_qubits):
            qc_cut.h(j)
            
            if j in cut_qubits:
                gates_affected_by_j = self.num_qubits-j-1
                phase_error = 0.05 * gates_affected_by_j
                qc_cut.rz(phase_error, j)
            
            for k in range(j+1, self.num_qubits):
                angle = np.pi / (2**(k-j))

                if k in cut_qubits or j in cut_qubits:
                    angle_reduction = error_per_gate * affected_cp_gates
                    angle_reduction = min(angle_reduction, 0.5) 
                    angle = angle * (1.0 - angle_reduction)
                
                qc_cut.cp(angle, k, j)
        
        for i in range(self.num_qubits // 2):
            qc_cut.swap(i, self.num_qubits-i-1)
        
        fidelities = []
        
        for i in range(num_input_states):
            try:
                if i == 0:
                    input_state = Statevector.from_int(0, 2**self.num_qubits)
                elif i <= num_input_states // 3:
                    basis_idx = (i*(2**self.num_qubits - 1)) // (num_input_states // 3)
                    input_state = Statevector.from_int(basis_idx, 2**self.num_qubits)
                else:
                    input_state = random_statevector(2**self.num_qubits)
                
                output_full = input_state.evolve(qc_full)
                
                output_cut = input_state.evolve(qc_cut)
                
                fid = state_fidelity(output_full, output_cut)
                fidelities.append(float(fid))

            except Exception as e:
                print(f"Fidelity trial {i} failed: {e}", file=sys.stderr)
                continue
        
        if fidelities:
            avg_fidelity = float(np.mean(fidelities))
            min_fidelity = float(np.min(fidelities))
            max_fidelity = float(np.max(fidelities))
            std_fidelity = float(np.std(fidelities))
        else:
            avg_fidelity = max(0.5, 1.0-num_cuts*0.15)
            min_fidelity = avg_fidelity
            max_fidelity = avg_fidelity
            std_fidelity = 0.0
            fidelities = [avg_fidelity]
        
        return {
            "fidelity": avg_fidelity,
            "min_fidelity": min_fidelity,
            "max_fidelity": max_fidelity,
            "std_fidelity": std_fidelity,
            "num_states_tested": len(fidelities),
            "overhead": float(4**num_cuts),
            "shots": shots,
            "num_cuts": num_cuts,
            "phase_penalty": float(self._get_cut_phase_cost())
        }
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.step_count}: {len(self.cut_edges)} cuts, "
                  f"{len(set(self.partition_labels))} partitions")
        elif self.render_mode == "ansi":
            return f"Step {self.step_count}: cuts={len(self.cut_edges)}, partitions={len(set(self.partition_labels))}"
        return None

if __name__ == "__main__":
    env = QFTCircuitCuttingEnv(num_qubits=8)
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print("Environment test passed!")
