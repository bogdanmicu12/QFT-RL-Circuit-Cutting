import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
    
    def forward(self, state, action_mask = None):
        features = self.feature_net(state)
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask.bool(), -1e9)
        
        return q_values


class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha= 0.6, beta= 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, priority = 1.0):
        self.buffer.append(Transition(state, action, reward, next_state, done))
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], np.array([])
        
        probs = np.array(self.priorities) / np.sum(self.priorities)
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                  p=probs, replace=False)

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        samples = [self.buffer[i] for i in indices]
        return samples, weights
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim, 
                 learning_rate = 1e-4, gamma = 0.99,
                 epsilon_start = 1.0, epsilon_end = 0.01,
                 epsilon_decay = 0.995, use_prioritized = False,
                 device = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.use_prioritized = use_prioritized
        
        self.policy_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if use_prioritized:
            self.memory = PrioritizedReplayBuffer()
        else:
            self.memory = ReplayBuffer()
        
        self.steps = 0
        self.loss_history = []
    
    def select_action(self, state, action_mask = None):
        if np.random.random() < self.epsilon:
            valid_actions = np.where(action_mask)[0] if action_mask is not None else np.arange(self.action_dim)
            return np.random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device) if action_mask is not None else None
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor, mask_tensor)
            action = q_values.max(1)[1].item()
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size = 32):
        if len(self.memory) < batch_size:
            return None
        
        if self.use_prioritized:
            batch, weights = self.memory.sample(batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = self.memory.sample(batch_size)
            weights = torch.ones(len(batch)).to(self.device)
        
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = (weights * torch.nn.functional.smooth_l1_loss(q_values, targets, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.loss_history.append(loss.item())
        return loss.item()
    
    def save_checkpoint(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.feature_net(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, state, action = None):
        action_logits, value = self(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_logprobs, entropy, value.squeeze(-1)


class PPOBuffer:
    def __init__(self, state_dim, max_size = 4000):
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
        self.length = 0
    
    def store(self, state, action, reward, value, log_prob, done):
        idx = self.length
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        self.length += 1
    
    def compute_returns_and_advantages(self, gamma = 0.99, gae_lambda = 0.95):
        advantages = np.zeros_like(self.rewards)
        gae = 0
        
        for t in reversed(range(self.length)):
            if t == self.length - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
        
        returns = advantages + self.values[:self.length]
        return advantages, returns


class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 learning_rate = 3e-4, gamma = 0.99,
                 gae_lambda = 0.95, clip_ratio = 0.2,
                 entropy_coef = 0.01, device = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.device = device
        
        self.net = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action_and_value(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, buffer, num_epochs = 4, batch_size = 32):
        advantages, returns = buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(buffer.length)
            
            for start_idx in range(0, buffer.length, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                states = torch.FloatTensor(buffer.states[batch_indices]).to(self.device)
                actions = torch.LongTensor(buffer.actions[batch_indices]).to(self.device)
                old_log_probs = torch.FloatTensor(buffer.log_probs[batch_indices]).to(self.device)
                returns_batch = torch.FloatTensor(returns[batch_indices]).to(self.device)
                advantages_batch = torch.FloatTensor(advantages[batch_indices]).to(self.device)
                
                new_action, new_log_probs, entropy, value = self.net.get_action_and_value(states, actions)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (value - returns_batch).pow(2).mean()
                
                total_loss = policy_loss + value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                
                self.loss_history.append(total_loss.item())

