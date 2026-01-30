import json
import matplotlib.pyplot as plt
from pathlib import Path
from training import TwoPhaseTrainer

def train_agent(num_qubits):
    print(f"Training {num_qubits}-qubit agent...")
    trainer = TwoPhaseTrainer(num_qubits=num_qubits, agent_type='dqn', 
                             target_partition_qubits=num_qubits//2,
                             save_results=False) 
    results = trainer.run(phase1_episodes=100, phase2_trials=5, shots=1024)
    
    trials = results['phase2']['trials']
    fidelities_per_trial = [t['real_fidelity'] for t in trials]
    cuts_per_trial = [t['cuts'][-1] if t['cuts'] else 0 for t in trials]
    overhead_per_trial = [t['overhead'][-1] if t['overhead'] else 1 for t in trials]
    
    avg_fidelity = results['phase2']['avg_fidelity']
    avg_cuts = sum(cuts_per_trial) / len(cuts_per_trial)
    avg_overhead = sum(overhead_per_trial) / len(overhead_per_trial)
    

    print(f"{num_qubits} qubits completed:")
    print(f"Avg Fidelity: {avg_fidelity:.10f}")
    print(f"Avg Cuts: {avg_cuts:.2f}")
    print(f"Avg Overhead: {avg_overhead:.2f}x")
    print(f"Fidelities per trial: {[f'{f:.8f}' for f in fidelities_per_trial]}")
    print(f"Cuts per trial: {cuts_per_trial}")
    
    return {
        'num_qubits': num_qubits, 
        'fidelity': avg_fidelity,
        'avg_cuts': avg_cuts,
        'avg_overhead': avg_overhead,
        'fidelities_per_trial': fidelities_per_trial,
        'cuts_per_trial': cuts_per_trial
    }

if __name__ == "__main__":
    qubit_sizes = [10, 12, 14, 16]
    
    print(f"\nQubit sizes to test: {qubit_sizes}")
    
    results = []
    for num_qubits in qubit_sizes:
        result = train_agent(num_qubits)
        results.append(result)
    
    qubits = [r['num_qubits'] for r in results]
    fidelities = [r['fidelity'] for r in results]
    cuts = [r['avg_cuts'] for r in results]
    overheads = [r['avg_overhead'] for r in results]
    
    output_dir = Path('./results/scalability')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'scalability_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{'Qubits':<10} {'Fidelity':<18} {'Cuts':<10} {'Overhead':<12} {'Status':<20}")
    for r in results:
        status = ""
        if r['avg_cuts']==0:
            status = "NO CUTS!"
        elif r['avg_cuts']<1:
            status = "Few cuts"
        elif r['fidelity']>0.99 and r['avg_cuts']>=1:
            status = "Excellent"
        elif r['fidelity']>0.95:
            status = "Very Good"
        elif r['fidelity']>0.90:
            status = "Good"
        else:
            status = "Low fidelity"
        
        print(f"{r['num_qubits']:<10} {r['fidelity']:<18.10f} {r['avg_cuts']:<10.2f} "
              f"{r['avg_overhead']:<12.2f}x {status:<20}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(qubits, fidelities, 'o-', linewidth=2, markersize=10, label='RL Agent')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Qubits', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title('RL Agent Scalability: Fidelity vs Qubit Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    for x, y in zip(qubits, fidelities):
        plt.annotate(f'{y:.6f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.savefig(output_dir /'scalability_plot.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to {output_dir}")
    plt.show()
