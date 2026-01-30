import numpy as np
import json
import pickle
from pathlib import Path
import torch
from datetime import datetime

from qft_cutting_env import QFTCircuitCuttingEnv
from rl_agent import DQNAgent


class ProgressTracker:
    
    def __init__(self, log_dir = None):
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        self.loss_history = []
    
    def record_episode(self, episode, reward, length, metrics):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_metrics.append(metrics)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            
            recent_metrics = self.episode_metrics[-100:]
            avg_max_width = np.mean([m.get('max_partition_qubits', 0) for m in recent_metrics])
            target_achieved = sum(1 for m in recent_metrics if m.get('target_achieved', False))
            avg_cuts = np.mean([m.get('num_cuts', 0) for m in recent_metrics])
            
            print(f"Episode {episode+1}: Avg Reward={avg_reward:.3f}, Avg Length={avg_length:.1f}")
            print(f"Avg Max Width={avg_max_width:.1f}, Target Achieved={target_achieved}%, Avg Cuts={avg_cuts:.2f}")
    
    def save_progress(self, path):
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_metrics': self.episode_metrics,
            'loss_history': self.loss_history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_summary(self):
        if not self.episode_rewards:
            return {}
        
        partition_widths = [m.get('max_partition_qubits', 0) for m in self.episode_metrics]
        qubits_reduced = [m.get('qubits_reduced', 0) for m in self.episode_metrics]
        target_achieved = sum(1 for m in self.episode_metrics if m.get('target_achieved', False))
        cuts_list = [m.get('num_cuts', 0) for m in self.episode_metrics]
        
        return {
            'total_episodes': int(len(self.episode_rewards)),
            'avg_reward': float(np.mean(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'avg_length': float(np.mean(self.episode_lengths)),
            'avg_loss': float(np.mean(self.loss_history)) if self.loss_history else 0.0,
            'avg_max_partition_qubits': float(np.mean(partition_widths)),
            'min_max_partition_qubits': int(np.min(partition_widths)),
            'avg_qubits_reduced': float(np.mean(qubits_reduced)),
            'target_achieved_count': int(target_achieved),
            'target_achieved_pct': float(100 * target_achieved / len(self.episode_metrics)) if self.episode_metrics else 0.0,
            'avg_cuts': float(np.mean(cuts_list)),
        }


class Phase1Trainer:
    def __init__(self, env, agent_type = 'dqn',
                 learning_rate = 1e-4, device = 'cpu', checkpoint_dir = './results/models',
                 save_checkpoints = True):
        self.env = env
        self.agent_type = agent_type
        self.device = device
        self.learning_rate = learning_rate
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        if save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ProgressTracker()
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        if agent_type.lower() == 'dqn':
            self.agent = DQNAgent(state_dim, action_dim, learning_rate=learning_rate, device=device)
        elif agent_type.lower() == 'ppo':
            raise ValueError(
                "PPO is not implemented in Phase1Trainer (no rollout collection/update). Use agent_type='dqn'."
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def train(self, num_episodes = 1000, batch_size = 32):
        
        print(f"Training for {num_episodes} episodes with proxy reward...")
        
        best_avg_reward = -np.inf
        best_model_path = None
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
    
                action_mask = self.env.get_action_mask()
                action = self.agent.select_action(state, action_mask)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                self.agent.store_transition(state, action, reward, next_state, done)
                loss_val = self.agent.train_step(batch_size)
                if loss_val is not None:
                    self.tracker.loss_history.append(loss_val)

                state = next_state

            self.tracker.record_episode(episode, episode_reward, episode_length, info)

            if episode > 100:
                avg_reward = np.mean(self.tracker.episode_rewards[-100:])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    if self.save_checkpoints:
                        best_model_path = self.checkpoint_dir / f"phase1_best_episode_{episode}.pt"
                        self.agent.save_checkpoint(str(best_model_path))
        
        summary = self.tracker.get_summary()
        print(f"\nPhase 1 Complete: {summary}")
        
        return {
            'agent': self.agent,
            'best_model_path': best_model_path,
            'summary': summary
        }


class Phase2Evaluator:
    def __init__(self, env, agent, agent_type = 'dqn'):
        self.env = env
        self.agent = agent
        self.agent_type = agent_type
    
    def evaluate(self, num_trials = 5, shots = 1024):
        print(f"Phase 2")
        print(f"Evaluating with {num_trials} trials and {shots} shots...")
        
        results = []
        
        for trial in range(num_trials):
            state, info = self.env.reset()
            trial_data = {
                'cuts': [],
                'fidelities': [],
                'overhead': [],
                'metrics': []
            }
            
            done = False
            while not done:
                if self.agent_type.lower() == 'dqn':
                    action_mask = self.env.get_action_mask()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.agent.device)
                    with torch.no_grad():
                        q_values = self.agent.policy_net(state_tensor, mask_tensor)
                        action = q_values.max(1)[1].item()
                else:
                    action, _, _ = self.agent.select_action(state)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                trial_data['cuts'].append(info.get('num_cuts', 0))
                trial_data['overhead'].append(info.get('cutting_overhead', 1.0))
                
                state = next_state
            
            real_metrics = self.env.evaluate_with_qiskit(shots=shots)
            trial_data['real_fidelity'] = real_metrics.get('fidelity', 0.0)
            
            cut_qubits = set()
            for gate_idx, qubit_idx in self.env.cut_edges:
                if 0 <= qubit_idx < self.env.num_qubits:
                    cut_qubits.add(qubit_idx)
            cut_qubits_list = sorted(cut_qubits)
            
            results.append(trial_data)
            print(f"Trial {trial+1}: Fidelity={real_metrics.get('fidelity', 0):.3f}, "
                  f"Cuts={trial_data['cuts'][-1] if trial_data['cuts'] else 0}, "
                  f"CutQubits={cut_qubits_list}")
        
        return {
            'trials': results,
            'avg_fidelity': np.mean([r['real_fidelity'] for r in results]),
            'max_fidelity': np.max([r['real_fidelity'] for r in results])
        }


class TwoPhaseTrainer:
    
    def __init__(self, num_qubits=8, agent_type='dqn',
                 device='cpu', output_dir='./results', 
                 target_partition_qubits=None, save_results=True):
        self.num_qubits = num_qubits
        self.agent_type = agent_type
        self.device = device
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        
        self.models_dir = self.output_dir / 'models'
        self.results_dir = self.output_dir / 'results'
        self.evaluation_dir = self.output_dir / 'evaluation'
        
        if save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for d in [self.models_dir, self.results_dir, self.evaluation_dir]:
                d.mkdir(parents=True, exist_ok=True)
        
        self.target_partition_qubits = target_partition_qubits or max(2, num_qubits // 2)
        self.env = QFTCircuitCuttingEnv(
            num_qubits=num_qubits,
            target_partition_qubits=self.target_partition_qubits,
            use_proxy_reward=True,
            phase_aware_weight=0.5  
        )
    
    def run(self, phase1_episodes= 1000, phase2_trials= 5,
            phase1_batch_size= 32, shots = 1024):
        print(f"No. Qubits: {self.num_qubits}, Agent: {self.agent_type.upper()}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trainer1 = Phase1Trainer(self.env, self.agent_type, device=self.device, 
                                 checkpoint_dir=str(self.models_dir), save_checkpoints=self.save_results)
        phase1_result = trainer1.train(num_episodes=phase1_episodes, batch_size=phase1_batch_size)
     
        if self.save_results:
            phase1_path = self.results_dir / f"phase1_results_{timestamp}.json"
            with open(phase1_path, 'w') as f:
                json.dump(phase1_result['summary'], f, indent=2)
        
        evaluator = Phase2Evaluator(self.env, phase1_result['agent'], self.agent_type)
        phase2_result = evaluator.evaluate(num_trials=phase2_trials, shots=shots)
        
        if self.save_results:
            phase2_path = self.results_dir / f"phase2_results_{timestamp}.json"
            with open(phase2_path, 'w') as f:
                json.dump({
                    'avg_fidelity': float(phase2_result['avg_fidelity']),
                    'max_fidelity': float(phase2_result['max_fidelity']),
                    'num_trials': phase2_trials
                }, f, indent=2)
        
        return {
            'phase1': phase1_result,
            'phase2': phase2_result,
            'num_qubits': self.num_qubits,
            'timestamp': timestamp
        }


if __name__ == "__main__":
    trainer = TwoPhaseTrainer(num_qubits=8, agent_type='dqn')
    results = trainer.run(phase1_episodes=100, phase2_trials=2)
    print(f"\nTraining results saved to {trainer.output_dir}")
