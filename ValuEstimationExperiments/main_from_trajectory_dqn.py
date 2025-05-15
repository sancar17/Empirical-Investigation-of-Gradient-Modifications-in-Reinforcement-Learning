import torch
import numpy as np
import gymnasium as gym
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
from collections import deque
import cv2
import wandb
import random
import argparse
from typing import Dict, List, Tuple, Any, Union
import copy
from ValuEstimationExperiments.main_from_trajectory_q_learning import  TorchLinearTrajectoryQLearning
    
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.net(x)

class BaseDQNTrajectory:
    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions

        self.policy_net = DQNNetwork(n_states, n_actions).to(self.device)
        self.target_net = DQNNetwork(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.lr = args.learning_rate
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.grad_clip = args.grad_clip
        self.update_freq = args.update_freq
        self.steps = 0

        self.normalizer = TorchRunningNormalizer(n_states)
        self.trajectory_buffer = TrajectoryBuffer(args.num_trajectories)
        self.batch_size = args.batch_size

        self.current_trajectory_episodes = 0
        self.trajectory_shift_threshold = args.trajectory_shift_threshold
        self.max_episodes_per_trajectory = args.max_episodes_per_trajectory
        self.td_error_window = deque(maxlen=100)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        norm_state = torch.from_numpy(self.normalizer.normalize(state)).float().to(self.device)
        with torch.no_grad():
            return self.policy_net(norm_state).argmax().item()

    def evaluate_td_error(self, mean_td_error):
        self.td_error_window.append(mean_td_error)
        if len(self.td_error_window) < 10:
            return mean_td_error, False
        avg_td_error = np.mean(self.td_error_window)
        should_shift = ((avg_td_error < self.trajectory_shift_threshold or
                         self.current_trajectory_episodes >= self.max_episodes_per_trajectory) and
                        self.trajectory_buffer.current_trajectory_index < len(self.trajectory_buffer.trajectories) - 1)
        return avg_td_error, should_shift

    def shift_to_next_trajectory(self):
        next_index = self.trajectory_buffer.current_trajectory_index + 1
        if next_index < len(self.trajectory_buffer.trajectories):
            success = self.trajectory_buffer.shift_to_trajectory(next_index)
            if success:
                self.td_error_window.clear()
                self.current_trajectory_episodes = 0
                return True
        return False

class DQNTrajectoryTD(BaseDQNTrajectory):
    def update(self):
        if len(self.trajectory_buffer) < 1:
            return {}
        states, actions, rewards, next_states, dones = self.trajectory_buffer.sample(len(self.trajectory_buffer))
        rewards = np.clip(rewards / 50.0, -2.0, 2.0)

        norm_states = torch.from_numpy(self.normalizer.normalize(states)).float().to(self.device)
        norm_next_states = torch.from_numpy(self.normalizer.normalize(next_states)).float().to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(norm_states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(norm_next_states).max(dim=1)[0]
            targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        td_errors = (targets - q_values).detach().cpu().numpy()

        loss = F.mse_loss(q_values, targets.detach())
        
        self.policy_net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        with torch.no_grad():
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            'td_errors': td_errors.tolist(),
            'mean_td_error': np.mean(np.abs(td_errors)),
            'loss': loss.item()
        }


class DQNTrajectoryGD(BaseDQNTrajectory):
    def update(self):
        if len(self.trajectory_buffer) < 1:
            return {}
        states, actions, rewards, next_states, dones = self.trajectory_buffer.sample(len(self.trajectory_buffer))
        rewards = np.clip(rewards / 50.0, -2.0, 2.0)

        norm_states = torch.from_numpy(self.normalizer.normalize(states)).float().to(self.device)
        norm_next_states = torch.from_numpy(self.normalizer.normalize(next_states)).float().to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(norm_states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_actions = self.policy_net(norm_next_states).argmax(dim=1)
        next_q_values = self.policy_net(norm_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        td_errors = targets - q_values
        loss = F.mse_loss(q_values, targets)

        self.policy_net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        with torch.no_grad():
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            'td_errors': td_errors.detach().cpu().numpy().tolist(),
            'mean_td_error': td_errors.abs().mean().item(),
            'loss': loss.item()
        }

class DQNTrajectoryGDS(BaseDQNTrajectory):
    def update(self):
        if len(self.trajectory_buffer) < 1:
            return {}

        # Sample full trajectory buffer
        states, actions, rewards, next_states, dones = self.trajectory_buffer.sample(len(self.trajectory_buffer))
        rewards = np.clip(rewards / 50.0, -2.0, 2.0)

        # Prepare tensors
        norm_states = torch.from_numpy(self.normalizer.normalize(states)).float().to(self.device)
        norm_next_states = torch.from_numpy(self.normalizer.normalize(next_states)).float().to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q(s,a)
        q_values = self.policy_net(norm_states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Select next actions using policy net
        next_actions = self.policy_net(norm_next_states).argmax(dim=1)
        next_q_values = self.policy_net(norm_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Target (no detach)
        targets = rewards_tensor + (1.0 - dones_tensor) * self.gamma * next_q_values

        # TD errors
        td_errors = targets - q_values

        # === Step 1: Get TD gradients ===
        self.policy_net.zero_grad()
        td_loss = F.mse_loss(q_values, targets.detach(), reduction='sum')  # TD uses detached target
        td_loss.backward(retain_graph=True)
        td_grads = {name: p.grad.clone() for name, p in self.policy_net.named_parameters() if p.grad is not None}

        # === Step 2: Get GD gradients ===
        self.policy_net.zero_grad()
        gd_loss = F.mse_loss(q_values, targets, reduction='sum')  # full autograd for GD
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

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            'td_errors': td_errors.detach().cpu().numpy().tolist(),
            'mean_td_error': td_errors.abs().mean().item(),
            'loss': td_loss.item()
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Trajectory-Based RL Training Arguments')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    
    # Method selection - added GD option
    parser.add_argument('--method', type=str, default='linear', 
                        choices=['linear', 'gds-tdm', 'gd'])
    parser.add_argument('--feature_extractor', type=str, default='polynomial', 
                        choices=['polynomial'])
    
    # Learning parameters
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.999)
    parser.add_argument('--min_epsilon', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--update_freq', type=int, default=50)
    parser.add_argument('--min_episodes', type=int, default=5000, 
                   help='Minimum number of episodes before starting early stopping checks')
    parser.add_argument('--validate_freq', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=1000)
    
    # Feature extractor parameters
    parser.add_argument('--poly_degree', type=int, default=3)
    
    # Gradient clipping and early stopping
    parser.add_argument('--grad_clip', type=float, default=0.7)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--early_stopping_threshold', type=float, default=0)
    
    # Save and logging
    parser.add_argument('--save_name', default="trajectory_learning", type=str)
    parser.add_argument('--wandb_name', type=str, default="trajectory_learning")
    parser.add_argument('--wandb_project', type=str, default='Tabular Q Learning Lunar Lander"')
    parser.add_argument('--use_wandb', default=True, action='store_true')
    
    # Trajectory-specific arguments
    parser.add_argument('--pretrained_model', type=str, default="./results/degree3_t2/20250203_011519/best_model.pt", 
                       help='Path to pretrained model to collect trajectories')
    parser.add_argument('--num_trajectories', type=int, default=5, 
                       help='Number of trajectories to collect')
    parser.add_argument('--trajectory_shift_threshold', type=float, default=0.02, 
                       help='Success rate threshold to shift to next trajectory')
    parser.add_argument('--max_episodes_per_trajectory', type=int, default=10000, 
                       help='Maximum number of episodes to spend on each trajectory')
    
    # Episodes and seed
    parser.add_argument('--episodes', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)

    # Parse command line arguments first
    args = parser.parse_args()

    # Handle config file if provided
    if args.config:
        config_path = args.config
        print(f"Loading configuration from: {config_path}")
        
        # Verify file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
            
        # Load and parse JSON
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse config file. Invalid JSON: {str(e)}")
            
        if not isinstance(config_dict, dict):
            raise ValueError(f"Config file must contain a JSON object, got {type(config_dict)}")
            
        # Get valid argument names from parser
        valid_args = set(vars(args).keys())
        
        # Track unknown parameters
        unknown_params = []
        
        # Update args with config file values
        for key, value in config_dict.items():
            if key == 'config':  # Skip config path itself
                continue
            if key in valid_args:
                setattr(args, key, value)
                print(f"Loaded config: {key} = {value}")
            else:
                unknown_params.append(key)
                
        # Warn about unknown parameters
        if unknown_params:
            print("\nWarning: Found unknown parameters in config file:")
            for param in unknown_params:
                print(f"- {param}")
            print("\nValid parameters are:")
            for param in sorted(valid_args):
                print(f"- {param}")
            
    return args

def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_learning_curve(rewards: List[float], save_path: str, method_name: str):
    """Plot learning curve with individual points and min-max range"""
    plt.figure(figsize=(12, 6))
    rewards_array = np.array(rewards)
    
    # Plot individual episode rewards
    plt.plot(rewards_array, '.', color='lightgray', alpha=0.3, label='Individual Episodes')
    
    # Calculate and plot moving statistics
    window_size = 100
    if len(rewards_array) >= window_size:
        min_vals = []
        max_vals = []
        avg_vals = []
        
        for i in range(len(rewards_array) - window_size + 1):
            window = rewards_array[i:i+window_size]
            min_vals.append(np.min(window))
            max_vals.append(np.max(window))
            avg_vals.append(np.mean(window))
        
        x = range(window_size-1, len(rewards_array))
        plt.plot(x, avg_vals, label='100-Episode Moving Average', color='blue', linewidth=2)
        plt.fill_between(x, min_vals, max_vals, alpha=0.2, color='blue',
                        label='Min-Max Range over 100 episodes')
    
    plt.title(f'Learning Curve - {method_name}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def save_video(frames: List[np.ndarray], path: str):
    """Save frames as MP4 video"""
    if not frames:
        return
    
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, 30, (width, height))
    
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

class TorchRunningNormalizer:
    """Improved GPU-accelerated running normalizer with proper batch processing"""
    def __init__(self, shape: Tuple[int, ...], clip_range: Tuple[float, float] = (-1, 1)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = torch.zeros(shape).to(self.device)
        self.var = torch.ones(shape).to(self.device)
        self.count = 0
        self.eps = 1e-8
        self.clip_range = clip_range

    def update(self, x: Union[np.ndarray, torch.Tensor]):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=True)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_size
        
        # Update mean using parallel updates formula
        new_mean = self.mean + (delta * batch_size / total_count)
        
        # Update variance using Welford's online algorithm
        if self.count > 0:
            m2_a = self.var * self.count
            m2_b = batch_var * batch_size
            delta_squared = delta ** 2
            m2_c = delta_squared * self.count * batch_size / total_count
            new_var = (m2_a + m2_b + m2_c) / total_count
        else:
            new_var = batch_var
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        std = torch.sqrt(self.var + self.eps)
        normalized = (x - self.mean) / std
        clipped = torch.clamp(normalized, self.clip_range[0], self.clip_range[1])
        return clipped.cpu().numpy()

class EarlyStoppingMonitor:
    """Monitor validation performance for early stopping"""
    def __init__(self, patience: int):
        self.patience = patience
        self.best_value = float('-inf')
        self.counter = 0
        
    def update(self, value: float) -> bool:
        """
        Update monitor with new validation value
        Returns True if should stop training
        """
        if value > self.best_value:
            self.best_value = value
            self.counter = 0
            return False
        
        self.counter += 1
        return self.counter >= self.patience

class TorchFeatureExtractor:
    """GPU-accelerated polynomial feature extractor"""
    def __init__(self, n_states, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.feature_type = args.feature_extractor
        self.normalizer = TorchRunningNormalizer(n_states)
        
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(degree=args.poly_degree, include_bias=True)
        self.n_features = self.poly.fit_transform(np.zeros((1, n_states))).shape[1]
        print(f"Using polynomial features with {self.n_features} features")
    
    def extract(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        normalized = torch.from_numpy(self.normalizer.normalize(state)).float().to(self.device)
        
        poly_features = self.poly.transform(normalized.cpu().numpy().reshape(1, -1))[0]
        return torch.from_numpy(poly_features).float().to(self.device)

    def process_batch_features(self, states):
        features_list = []
        for state in states:
            features = self.extract(state)
            features_list.append(features)
        return torch.stack(features_list)

class Trajectory:
    """Store a single trajectory of experience with seed information"""
    def __init__(self, initial_state, seed):
        self.experiences = []
        self.initial_state = initial_state  # Keep for reference
        self.seed = seed  # Store the seed that generated this trajectory
        self.total_reward = 0
        
    def add(self, state, action, reward, next_state, done):
        self.experiences.append((state, action, reward, next_state, done))
        self.total_reward += reward
        
    def __len__(self):
        return len(self.experiences)

class TrajectoryBuffer:
    """Store and manage multiple trajectories for training"""
    def __init__(self, capacity):
        self.trajectories = []
        self.capacity = capacity
        self.current_trajectory_index = 0
        self.buffer = deque(maxlen=capacity*1000)  # Conservative buffer size
        
    def add_trajectory(self, trajectory):
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(trajectory)
            
    def shift_to_trajectory(self, index):
        """Replace buffer with experiences from the specified trajectory"""
        if 0 <= index < len(self.trajectories):
            self.current_trajectory_index = index
            # Clear current buffer
            self.buffer.clear()
            # Add experiences from the selected trajectory
            for exp in self.trajectories[index].experiences:
                self.buffer.append(exp)
            return True
        return False
            
    def __len__(self):
        return len(self.buffer)
        
    def sample(self, batch_size):
        """Return all experiences from the current trajectory"""
        # Ignore batch_size parameter and return all experiences
        batch = list(self.buffer)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
    
    def clear(self):
        self.trajectories = []
        self.current = None
        self.current_trajectory_index = -1
    
def collect_trajectories(pretrained_model_path, num_trajectories, args):
    """Collect trajectories using a pretrained model and store seeds"""
    print(f"Collecting {num_trajectories} trajectories using pretrained model: {pretrained_model_path}")
    
    # Load pretrained model
    checkpoint = torch.load(pretrained_model_path)
    
    # Create environment
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    # Create agent based on method
    agent = TorchLinearTrajectoryQLearning(n_states, n_actions, args)
    
    # Load weights and other parameters from the pretrained model
    if 'weights' in checkpoint:
        agent.weights = checkpoint['weights']
        agent.target_weights = checkpoint['weights'].clone()
    elif isinstance(checkpoint, torch.Tensor):
        agent.weights = checkpoint
        agent.target_weights = checkpoint.clone()
    
    # Set epsilon to a low value for mostly optimal behavior
    agent.epsilon = 0.05
    
    trajectories = []
    
    # Collect trajectories
    for i in range(num_trajectories):
        # Use a different seed for each trajectory
        trajectory_seed = args.seed + i*100
        state, info = env.reset(seed=trajectory_seed)
        
        # Create trajectory with initial state and seed
        trajectory = Trajectory(initial_state=state.copy(), seed=trajectory_seed)
        
        done = False
        truncated = False
        
        frames = []  # For saving video

        while not (done or truncated):
            frame = env.render()  # Render frame for video
            frames.append(frame)
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            trajectory.add(state.copy(), action, reward, next_state.copy(), done)

            state = next_state.copy()

        # Save trajectory video
        video_path = os.path.join("traj_results", f"trajectory_{i}.mp4")
        save_video(frames, video_path)
        
        print(f"Trajectory {i+1}/{num_trajectories} collected with {len(trajectory)} steps and total reward {trajectory.total_reward:.2f}")
        trajectories.append(trajectory)
    
    env.close()
    return trajectories

def validate_on_trajectories(agent, trajectories):
    """Validate agent on all collected trajectories using seeds"""
    results = {}
    
    for i, trajectory in enumerate(trajectories):
        # Use the stored seed to reset the environment to the same initial state
        env = gym.make('LunarLander-v2')
        state, _ = env.reset(seed=trajectory.seed)
        
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
        
        results[f'trajectory_{i}'] = {
            'reward': total_reward,
            'success': total_reward > 200
        }
        
        env.close()
    
    # Add aggregate statistics
    results['mean_reward'] = np.mean([results[f'trajectory_{i}']['reward'] for i in range(len(trajectories))])
    results['success_rate'] = np.mean([results[f'trajectory_{i}']['success'] for i in range(len(trajectories))])
    
    return results

# Modified train_and_evaluate to log all trajectory training into the same wandb session/log scope
def train_and_evaluate(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('traj_results', args.save_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"trajectory_learning_{timestamp}",
            config=vars(args)
        )

    trajectories = collect_trajectories(args.pretrained_model, args.num_trajectories, args)
    env = gym.make('LunarLander-v2')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if args.method == "dqn":
        agent_cls = DQNTrajectoryTD
    elif args.method == "dqn-gds-tdm":
        agent_cls = DQNTrajectoryGDS
    elif args.method == "dqn-gd":
        agent_cls = DQNTrajectoryGD
    else:
        print("Invalid method")
        exit()

    all_training_history = {}
    global_episode = 0

    agent = agent_cls(n_states, n_actions, args)

    for traj_index, trajectory in enumerate(trajectories):
        print(f"\n=== Starting training on trajectory {traj_index} ===")

        # Clear the buffer, add new trajectory, shift
        agent.trajectory_buffer.clear()
        agent.trajectory_buffer.add_trajectory(trajectory)
        agent.trajectory_buffer.shift_to_trajectory(0)

        training_history = {
            'td_errors': [],
            'mean_td_errors': [],
            'lr': [],
            'eps': [],
            'trajectory_index': traj_index
        }

        for episode in range(args.episodes):
            print(traj_index, episode)
            agent.current_trajectory_episodes += 1
            update_info = agent.update()
            mean_td_error = update_info.get("mean_td_error", 0)

            training_history['td_errors'].append(update_info.get('td_errors', []))
            training_history['mean_td_errors'].append(mean_td_error)
            training_history['lr'].append(agent.lr)
            training_history['eps'].append(agent.epsilon)

            if args.use_wandb:
                wandb.log({
                    'training/episode': global_episode,
                    'training/mean_td_error': mean_td_error,
                    'training/lr': agent.lr,
                    'training/epsilon': agent.epsilon,
                    'training/current_trajectory': traj_index,
                }, step=global_episode)

            global_episode += 1

        all_training_history[f"trajectory_{traj_index}"] = training_history

        # Save model after each trajectory
        save_dict = {
            'policy_state_dict': agent.policy_net.state_dict(),
            'target_state_dict': agent.target_net.state_dict(),
            'epsilon': agent.epsilon,
            'lr': agent.lr,
            'normalizer_mean': agent.normalizer.mean,
            'normalizer_var': agent.normalizer.var,
            'normalizer_count': agent.normalizer.count
        }
        torch.save(save_dict, os.path.join(save_dir, f"model_trajectory_{traj_index}.pt"))

    with open(os.path.join(save_dir, 'full_training_history.json'), 'w') as f:
        json.dump(all_training_history, f, cls=NumpyEncoder, indent=2)

    print("\nTraining completed for all trajectories.")
    return all_training_history

# Helper class for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):  # Handle numpy boolean type
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def main():
    args = parse_args()
    train_and_evaluate(args)

if __name__ == "__main__":
    main()