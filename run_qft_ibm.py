from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
import torch
from circuit_cutting import CircuitCuttingExecutor
from qft_builder import QFTCircuitBuilder
import json

from qiskit.circuit import ClassicalRegister

from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Batch
from collections import Counter

from qft_cutting_env import QFTCircuitCuttingEnv
from rl_agent import DQNAgent, PPOAgent

def _get_cuts_from_model(num_qubits: int, 
                         model_path: str,
                         agent_type: str = 'dqn',
                         max_steps: int = 50):
    """
    Load a trained RL model and use it to generate circuit cuts.
    
    This uses the EXACT SAME inference logic as training.py Phase2Evaluator
    to ensure cuts match the model's learned behavior.
    
    Args:
        num_qubits: Number of qubits
        model_path: Path to trained model checkpoint
        agent_type: 'dqn' or 'ppo'
        max_steps: Maximum steps to run the agent
        
    Returns:
        List of cuts as (gate_idx, qubit_idx) tuples
    """
    env = QFTCircuitCuttingEnv(
        num_qubits=num_qubits,
        circuit_depth=50,
        target_partition_qubits=num_qubits // 2
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(state_dim, action_dim, device='cpu')
    elif agent_type.lower() == 'ppo':
        agent = PPOAgent(state_dim, action_dim, device='cpu')
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load_checkpoint(model_path)
    print(f"[+] Loaded {agent_type.upper()} model from {Path(model_path).name}")
    
    state, info = env.reset()
    step_count = 0
    
    for step in range(max_steps):
        if agent_type.lower() == 'dqn':
            action_mask = env.get_action_mask() if hasattr(env, 'get_action_mask') else None
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(agent.device) if action_mask is not None else None
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor, mask_tensor)
                action = q_values.max(1)[1].item()
        else:
            action, _, _ = agent.select_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        step_count = step
        
        if terminated or truncated:
            break
        
        state = next_state
    
    cuts = list(env.cut_edges)
    num_cuts = len(cuts)
    
    print(f"[+] Model Generated Cuts: {cuts} (number of cuts: {num_cuts})")
    
    return cuts

def get_subcircuits(full_circuit: QuantumCircuit):
    models_dir = Path("./results/models/")

    model_files = sorted(models_dir.glob("phase1_best_episode_*.pt"))
    model_path = str(model_files[-1])
    print(f"[+] Auto-loaded best model: {model_files[-1].name}")
    
    cuts = _get_cuts_from_model(
        num_qubits=num_qubits,
        model_path=model_path,
        agent_type="dqn",
        max_steps=50
    )
    
    max_partition_qubits = 4 
    executor = CircuitCuttingExecutor(max_subcircuit_qubits=max(2, min(max_partition_qubits, num_qubits)))
    partition_labels = executor._create_partition_from_cuts(num_qubits, cuts)
    print(f"[+] ML Partitioner generated partition labels: {partition_labels}")

    max_part_size = 0
    for pid in set(partition_labels):
        max_part_size = max(max_part_size, sum(1 for x in partition_labels if x == pid))
    if max_part_size > int(max_partition_qubits):
        raise SystemExit(
            f"Partition too large: {max_part_size} qubits > --max-partition-qubits={max_partition_qubits}. "
            f"Increase --num-partitions or reduce --num-qubits."
        )

    cut_result = executor.apply_wire_cuts(full_circuit, cuts=cuts, partition_labels=partition_labels)
    return cut_result["subcircuits"]

num_qubits = 8
qft = QFTCircuitBuilder(num_qubits).build_qft_circuit()
subcircuits = get_subcircuits(qft)

for idx, subcircuit in enumerate(subcircuits):
    qubit_count = subcircuit["num_qubits"]
    subcircuit["circuit"].add_register(ClassicalRegister(qubit_count))
    subcircuit["circuit"].measure(range(qubit_count), range(qubit_count))

subcircuits = [subcircuit["circuit"] for subcircuit in subcircuits]
# subcircuits = [qft]

service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
 .
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuits = pm.run(subcircuits)

sampler = Sampler(mode=backend)

# job = sampler.run(isa_circuits, shots=1024)
job = service.job("d5savgoubqnc73c4gfv0")
 
print(f">>> Job ID: {job.job_id()}")

results = job.result()
all_results = {}

for idx, res in enumerate(results):
    data_attr = list(res.data.keys())[0]
    bitstrings_arrays = getattr(res.data, data_attr).array

    bitstrings = [''.join(map(str, b)) for b in bitstrings_arrays]
    counts_dict = dict(Counter(bitstrings))
    shots = len(bitstrings)
    
    all_results[f"circuit_{idx}"] = {
        "num_qubits": len(bitstrings_arrays[0]),
        "counts": counts_dict,
        "probabilities": {k: v/shots for k, v in counts_dict.items()}
    }

with open("qft_results_cut_8.json", "w") as f:
    json.dump(all_results, f, indent=4)