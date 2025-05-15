import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    """Neural network for Deep Q-Learning"""
    def __init__(self, n_states, n_actions, n_hidden=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(n_states).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.net(x)
    
class TorchDQNLearning:
    """Deep Q-Network implementation with experience replay and target network"""
    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Parse DQN-specific arguments
        self.hidden_dims = [int(x) for x in getattr(args, 'hidden_dims', [128, 64])]
        
        # Initialize policy and target networks
        self.policy_net = DQNNetwork(n_states, n_actions, self.hidden_dims).to(self.device)
        self.target_net = DQNNetwork(n_states, n_actions, self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is used for evaluation only
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        
        # Training parameters
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.lr = args.learning_rate
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.grad_clip = args.grad_clip
        
        # Experience replay
        from helpers import TorchExperienceReplay
        self.replay_buffer = TorchExperienceReplay(args.max_buffer_size, args)
        self.batch_size = args.batch_size
        self.min_experiences = args.min_buffer_size
        
        # Target network update frequency
        self.target_update_freq = args.update_freq
        self.steps = 0
        
        # Normalization
        from helpers import TorchRunningNormalizer
        self.normalizer = TorchRunningNormalizer(n_states)
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Normalize state
        normalized_state = torch.from_numpy(self.normalizer.normalize(state)).float().to(self.device)
        
        # Get action values
        with torch.no_grad():
            q_values = self.policy_net(normalized_state)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """Update the DQN using a batch of experiences"""
        # Clip reward for stability
        reward = np.clip(reward / 50.0, -2.0, 2.0)
        
        # Add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Don't update until we have enough experiences
        if len(self.replay_buffer.buffer) < self.min_experiences:
            return {}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors and normalize states
        normalized_states = torch.from_numpy(self.normalizer.normalize(states)).float().to(self.device)
        normalized_next_states = torch.from_numpy(self.normalizer.normalize(next_states)).float().to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.policy_net(normalized_states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values with target network
        with torch.no_grad():
            # Get actions from policy network
            policy_actions = self.policy_net(normalized_next_states).argmax(dim=1)
            # Get Q-values from target network
            target_q_values = self.target_net(normalized_next_states).gather(1, policy_actions.unsqueeze(1)).squeeze(1)
            # Apply Bellman equation
            target_values = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q_values
        
        # Compute TD errors for prioritized replay update
        td_errors = (target_values - current_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Compute loss detaching the targets
        loss = F.mse_loss(current_q_values, target_values.detach())
        
        # Optimize the model
        self.policy_net.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        with torch.no_grad():
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon and learning rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Adjust learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_min, param_group['lr'] * self.lr_decay)
            self.lr = param_group['lr']
        
        # Return metrics for logging
        with torch.no_grad():
            mean_q_value = current_q_values.mean().item()
        
        return {
            'td_errors': td_errors.tolist(),
            'mean_td_error': np.mean(np.abs(td_errors)),
            'loss': loss.item(),
            'mean_q_value': mean_q_value
        }
    
class TorchDQNGDLearning(TorchDQNLearning):
    """
    Deep Q-Network with Residual Gradient (GD) update.
    Computes the full gradient of TD loss using autograd.
    """
    def __init__(self, n_states, n_actions, args):
        super().__init__(n_states, n_actions, args)

    def update(self, state, action, reward, next_state, done):
        reward = np.clip(reward / 50.0, -2.0, 2.0)
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer.buffer) < self.min_experiences:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)

        # Normalize states
        norm_states = torch.from_numpy(self.normalizer.normalize(states)).float().to(self.device)
        norm_next_states = torch.from_numpy(self.normalizer.normalize(next_states)).float().to(self.device)

        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(norm_states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values with target network (Double DQN approach)
        with torch.no_grad():
            # Get actions from policy network
            policy_actions = self.policy_net(norm_next_states).argmax(dim=1)
            # Get Q-values from target network
            target_q_values = self.policy_net(norm_next_states).gather(1, policy_actions.unsqueeze(1)).squeeze(1)
            # Apply Bellman equation
            target_values = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q_values
        
        # Compute TD errors for prioritized replay update
        td_errors = (target_values - current_q_values).detach()
        loss = F.mse_loss(current_q_values, target_values) # No detach here

        self.policy_net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        with torch.no_grad():
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad

        # Update target net
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Epsilon and LR decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_min, param_group['lr'] * self.lr_decay)
            self.lr = param_group['lr']

        with torch.no_grad():
            mean_q_value = current_q_values.mean().item()

        return {
            'td_errors': td_errors.tolist(),
            'mean_td_error': np.mean(np.abs(td_errors.detach().cpu().numpy())),
            'loss': loss.item(),
            'mean_q_value': mean_q_value
        }

    
class TorchDQNGDSTDMLearning(TorchDQNLearning):
    """
    Deep Q-Network implementation with GDS-TDM gradient update technique
    
    This combines:
    1. DQN's neural network function approximation and experience replay
    2. GDS-TDM's gradient direction sign technique for more stable updates
    """
    def __init__(self, n_states, n_actions, args):
        super().__init__(n_states, n_actions, args)
        # Inherits all DQN components including networks and experience replay
        
    def update(self, state, action, reward, next_state, done):
        """Update the DQN using a batch of experiences with GDS-TDM technique"""
        # Clip reward for stability
        reward = np.clip(reward / 50.0, -2.0, 2.0)
        
        # Add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Don't update until we have enough experiences
        if len(self.replay_buffer.buffer) < self.min_experiences:
            return {}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors and normalize states
        normalized_states = torch.from_numpy(self.normalizer.normalize(states)).float().to(self.device)
        normalized_next_states = torch.from_numpy(self.normalizer.normalize(next_states)).float().to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Track gradients for current state
        self.policy_net.zero_grad()
        current_q_values = self.policy_net(normalized_states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values with target network
        with torch.no_grad():
            # Get actions from policy network
            policy_actions = self.policy_net(normalized_next_states).argmax(dim=1)
            # Get Q-values from target network
            target_q_values = self.policy_net(normalized_next_states).gather(1, policy_actions.unsqueeze(1)).squeeze(1)
            # Apply Bellman equation
            target_values = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q_values
        
        # Compute TD errors for prioritized replay update
        td_errors = target_values - current_q_values
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # === Begin GDS-TDM specific implementation ===
        
        # === Step 1: Get TD gradients ===
        self.policy_net.zero_grad()
        td_loss = F.mse_loss(current_q_values, target_values.detach())  # TD uses detached target
        td_loss.backward(retain_graph=True)
        td_grads = {name: p.grad.clone() for name, p in self.policy_net.named_parameters() if p.grad is not None}

        # === Step 2: Get GD gradients ===
        self.policy_net.zero_grad()
        gd_loss = F.mse_loss(current_q_values, target_values)  # full autograd for GD
        gd_loss.backward()
        gd_grads = {name: p.grad.clone() for name, p in self.policy_net.named_parameters() if p.grad is not None}

        # === Step 3: GDS-TDM update ===
        for name, param in self.policy_net.named_parameters():
            if name in td_grads and name in gd_grads:
                grad_td = td_grads[name]
                grad_gd = gd_grads[name]
                param.grad = torch.sign(grad_gd) * torch.abs(grad_td)

        # Clip, step, sync target net
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        with torch.no_grad():
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad
        
        # === End GDS-TDM specific implementation ===
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon and learning rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Adjust learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_min, param_group['lr'] * self.lr_decay)
            self.lr = param_group['lr']
        
        # Return metrics for logging
        with torch.no_grad():
            mean_q_value = current_q_values.mean().item()
        
        return {
            'td_errors': td_errors.tolist(),
            'mean_td_error': np.mean(np.abs(td_errors.detach().cpu().numpy())),
            'loss': td_loss.item(),
            'mean_q_value': mean_q_value
        }
        
    def compute_td_error(self, state, action, reward, next_state, done):
        """Compute TD error for a single transition"""
        try:
            with torch.no_grad():
                # Process states
                normalized_state = torch.from_numpy(self.normalizer.normalize(state)).float().unsqueeze(0).to(self.device)
                normalized_next_state = torch.from_numpy(self.normalizer.normalize(next_state)).float().unsqueeze(0).to(self.device)
                
                # Get current Q value
                q_values = self.policy_net(normalized_state)
                current_q = q_values[0, action].item()
                
                # Get max next Q value
                next_action = self.policy_net(normalized_next_state).argmax(dim=1).item()
                next_q = self.target_net(normalized_next_state)[0, next_action].item()
                
                # Calculate target and TD error
                target = reward + (1 - float(done)) * self.gamma * next_q
                td_error = target - current_q
                
                return td_error
        except Exception as e:
            print(f"TD error calculation failed: {e}")
            return 0.0
    