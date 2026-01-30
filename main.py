import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from qft_builder import QFTCircuitBuilder
from baselines import run_all_baselines, print_baseline_comparison
from training import TwoPhaseTrainer
from evaluation import ComprehensiveEvaluator
from qft_cutting_env import QFTCircuitCuttingEnv


def print_circuit_cutting_visualization(circuit, num_qubits):
    
    print(f"\nFull QFT: ({num_qubits} qubits)")
    print(f"Depth: {circuit.depth()} | Gates: {circuit.size()}\n")
    print(circuit)
    
    env = QFTCircuitCuttingEnv(num_qubits=num_qubits, phase_aware_weight=0.1)
    phase_weights = env.phase_weights
    
    print(f"\nQubit Phase Importance (normalized 0-1):")
    for j in range(num_qubits):
        weight = phase_weights[j]
        
        importance = "CRITICAL" if weight>0.9 else "HIGH" if weight>0.7 else "MEDIUM" if weight>0.4 else "LOW"
        print(f"  q{j}: {weight:.3f} {importance}")


def main():
    parser = argparse.ArgumentParser(
        description='RL-based QFT Circuit Cutting Optimizer'
    )
    
    #circuit
    parser.add_argument('--num-qubits', type=int, default=8)
    parser.add_argument('--circuit-depth', type=int, default=50)

    #training 
    parser.add_argument('--agent-type', type=str, default='dqn', choices=['dqn', 'ppo'])
    parser.add_argument('--phase1-episodes', type=int, default=500)
    parser.add_argument('--phase2-trials', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--target-partition-qubits', type=int, default=None)
    
    #evaluation 
    parser.add_argument('--shots', type=int, default=1024)
    
    #pipeline 
    parser.add_argument('--run-baselines', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    
    #output
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--log', action='store_true')
    
    args = parser.parse_args()
    
    print(f"Qubits: {args.num_qubits}")
    print(f"Target partition qubits: {args.target_partition_qubits or args.num_qubits // 2}")
    print(f"Agent: {args.agent_type.upper()}")
    print(f"Phase 1 episodes: {args.phase1_episodes}")
    print(f"Phase 2 trials: {args.phase2_trials}")
    print(f"Output: {args.output_dir}")

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'results': {}
    }
    
    try:
        builder = QFTCircuitBuilder(args.num_qubits)
        circuit = builder.build_qft_circuit()
        print(f"QFT circuit: {args.num_qubits} qubits")
        
        print_circuit_cutting_visualization(circuit, args.num_qubits)
        
        if args.run_baselines and not args.skip_training:
            target_partition_qubits = args.target_partition_qubits or (args.num_qubits//2)
            baseline_results = run_all_baselines(circuit, target_partition_qubits=target_partition_qubits)
            print_baseline_comparison(baseline_results)
            
            baseline_data = {}
            for name, result in baseline_results.items():
                baseline_data[name] = {
                    'num_cuts': result.num_cuts,
                    'num_partitions': result.num_partitions,
                    'max_partition_width': result.max_partition_width,
                    'cutting_overhead': float(result.cutting_overhead),
                    'fidelity_estimate': float(result.fidelity_estimate),
                }
            
            results['results']['baselines'] = baseline_data
        else:
            baseline_results = {}
        
        if not args.skip_training:
            
            trainer = TwoPhaseTrainer(
                num_qubits=args.num_qubits,
                agent_type=args.agent_type,
                output_dir=args.output_dir,
                target_partition_qubits=args.target_partition_qubits or (args.num_qubits//2)
            )
            
            training_results = trainer.run(
                phase1_episodes=args.phase1_episodes,
                phase2_trials=args.phase2_trials,
                phase1_batch_size=args.batch_size,
                shots=args.shots
            )
            
            results['results']['training'] = {
                'phase1_summary': training_results['phase1'].get('summary', {}),
                'phase2_summary': {
                    'avg_fidelity': float(training_results['phase2'].get('avg_fidelity', 0)),
                    'max_fidelity': float(training_results['phase2'].get('max_fidelity', 0))
                }
            }
        else:
            training_results = {}
        
        evaluator = ComprehensiveEvaluator(output_dir=str(output_path/"evaluation"))
        eval_results = evaluator.evaluate(training_results, baseline_results, save_plots=True)
        
    
        results_file = output_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
