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
    
import gc

def parse_args():
    parser = argparse.ArgumentParser(description='Trajectory-Based RL Training Arguments')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    
    # Method selection - added GD option
    parser.add_argument('--method', type=str, default='linear', 
                        choices=['linear', 'gds-tdm', 'gd'],
                        help='Learning method: linear (TD), gds-tdm (mixed), or gd (gradient descent only)')
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

class TorchLinearTrajectoryQLearning:
    """Linear Q-Learning that trains on individual trajectories"""
    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.feature_extractor = TorchFeatureExtractor(n_states, args)
        self.n_features = self.feature_extractor.n_features
        
        # Initialize weights
        self.weights = torch.randn(n_actions, self.n_features).to(self.device) * 0.1
        self.target_weights = self.weights.clone()
        
        # Training parameters from args
        self.lr = args.learning_rate
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.grad_clip = args.grad_clip
        
        # Trajectory buffer and update frequency
        self.trajectory_buffer = TrajectoryBuffer(args.num_trajectories)
        self.batch_size = args.batch_size
        self.target_update_freq = args.update_freq
        self.steps = 0
        
        # TD error tracking for trajectory shifting
        self.trajectory_shift_threshold = args.trajectory_shift_threshold
        self.td_error_window = deque(maxlen=100)
        
        # Maximum episodes per trajectory
        self.max_episodes_per_trajectory = args.max_episodes_per_trajectory
        self.current_trajectory_episodes = 0
    
    def get_q_values(self, state, weights=None):
        if weights is None:
            weights = self.weights
        features = self.feature_extractor.extract(state)
        features = torch.tensor(features, device=self.device) if isinstance(features, np.ndarray) else features
        return torch.mv(weights, features).detach().cpu().numpy()
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_q_values(state))
    
    def update(self, state=None, action=None, reward=None, next_state=None, done=None):
        """Efficient vectorized update from trajectory buffer using the entire current trajectory"""
        if len(self.trajectory_buffer) < 1:
            return {}
         
        current_batch_size = len(self.trajectory_buffer)
        states, actions, rewards, next_states, dones = self.trajectory_buffer.sample(current_batch_size)
         
        # Normalize rewards for stability
        rewards = np.clip(rewards / 50.0, -2.0, 2.0)
         
        # Extract features
        current_features = self.feature_extractor.process_batch_features(states)  # shape: [B, F]
        next_features = self.feature_extractor.process_batch_features(next_states)  # shape: [B, F]
         
        # Q-values
        current_q_values = torch.matmul(self.weights, current_features.T)  # shape: [A, B]
        next_q_values = torch.matmul(self.target_weights, next_features.T)  # shape: [A, B]
         
        # Get max Q for next states
        max_next_q = next_q_values.max(dim=0)[0]  # shape: [B]
        targets = torch.from_numpy(rewards).to(self.device) + \
                (1 - torch.from_numpy(dones).float().to(self.device)) * self.gamma * max_next_q
         
        # Gather Q(s,a)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)  # shape: [B]
        q_sa = current_q_values.gather(0, actions_tensor.view(1, -1)).squeeze(0)  # shape: [B]
         
        # TD error
        td_errors = targets - q_sa  # shape: [B]
         
        # Compute gradients
        grads = td_errors.view(-1, 1) * current_features  # shape: [B, F]
         
        # Clip gradients
        grad_norms = torch.norm(grads, dim=1)
        clip_mask = grad_norms > self.grad_clip
        grads[clip_mask] *= (self.grad_clip / grad_norms[clip_mask]).view(-1, 1)
         
        # Initialize update matrix
        weight_update = torch.zeros_like(self.weights)  # shape: [A, F]
         
        # Aggregate updates per action
        lr_tensor = torch.tensor(self.lr, dtype=self.weights.dtype, device=self.device)
        grads = grads.to(self.weights.dtype)
        weight_update.index_add_(0, actions_tensor, lr_tensor * grads)
         
        # Apply weight update
        self.weights += weight_update
         
        self.steps += 1
        self.target_weights = self.weights.clone()
         
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
         
        # Logging
        update_info = {
            'td_errors': td_errors.detach().cpu().numpy().tolist(),
            'grad_norms': grad_norms.detach().cpu().numpy().tolist(),
            'mean_td_error': td_errors.abs().mean().item(),
            'mean_grad_norm': grad_norms.mean().item(),
            'current_trajectory_index': self.trajectory_buffer.current_trajectory_index,
            'batch_size': current_batch_size
        }

        return update_info


    def evaluate_td_error(self, mean_td_error):
        """Track TD error and determine if we should shift to next trajectory"""
        self.td_error_window.append(mean_td_error)
        
        if len(self.td_error_window) < 10:  # Need at least some data to make a decision
            return mean_td_error, False
            
        avg_td_error = np.mean(self.td_error_window)
        
        # Check if either condition is met:
        # 1. TD error  below threshold
        # 2. Maximum episodes per trajectory has been reached
        should_shift = ((avg_td_error < self.trajectory_shift_threshold or 
                         self.current_trajectory_episodes >= self.max_episodes_per_trajectory) and
                        self.trajectory_buffer.current_trajectory_index < len(self.trajectory_buffer.trajectories) - 1)
                        
        return avg_td_error, should_shift
    
    def evaluate_success_rate(self, is_success):
        """Track success rate and determine if we should shift to next trajectory"""
        self.success_window.append(1 if is_success else 0)
        
        if len(self.success_window) < 100:
            return 0, False
            
        success_rate = 100 * sum(self.success_window) / len(self.success_window)
        should_shift = (success_rate >= self.trajectory_shift_threshold and 
                        self.trajectory_buffer.current_trajectory_index < len(self.trajectory_buffer.trajectories) - 1)
                        
        return success_rate, should_shift
    
    def shift_to_next_trajectory(self):
        """Shift to the next trajectory if available"""
        next_index = self.trajectory_buffer.current_trajectory_index + 1
        if next_index < len(self.trajectory_buffer.trajectories):
            success = self.trajectory_buffer.shift_to_trajectory(next_index)
            if success:
                # Reset TD error window and episode counter for new trajectory
                self.td_error_window.clear()
                self.current_trajectory_episodes = 0
                return True
        return False

class TorchLinearTrajectoryQLearningGD:
    """Linear Q-Learning that trains on individual trajectories using only Gradient Descent updates"""
    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.feature_extractor = TorchFeatureExtractor(n_states, args)
        self.n_features = self.feature_extractor.n_features
        
        # Initialize weights
        self.weights = torch.randn(n_actions, self.n_features).to(self.device) * 0.1
        self.target_weights = self.weights.clone()
        
        # Training parameters from args
        self.lr = args.learning_rate
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.grad_clip = args.grad_clip
        
        # Trajectory buffer and update frequency
        self.trajectory_buffer = TrajectoryBuffer(args.num_trajectories)
        self.batch_size = args.batch_size
        self.target_update_freq = args.update_freq
        self.steps = 0
        
        # TD error tracking for trajectory shifting
        self.trajectory_shift_threshold = args.trajectory_shift_threshold
        self.td_error_window = deque(maxlen=100)
        
        # Maximum episodes per trajectory
        self.max_episodes_per_trajectory = args.max_episodes_per_trajectory
        self.current_trajectory_episodes = 0
    
    def get_q_values(self, state, weights=None):
        if weights is None:
            weights = self.weights
        features = self.feature_extractor.extract(state)
        features = torch.tensor(features, device=self.device) if isinstance(features, np.ndarray) else features
        return torch.mv(weights, features).detach().cpu().numpy()
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_q_values(state))
    
    def update(self, state=None, action=None, reward=None, next_state=None, done=None):
        """GD update using autograd for the entire trajectory"""
        if len(self.trajectory_buffer) < 1:
            return {}

        current_batch_size = len(self.trajectory_buffer)
        states, actions, rewards, next_states, dones = self.trajectory_buffer.sample(current_batch_size)

        # Normalize rewards
        rewards = np.clip(rewards / 50.0, -2.0, 2.0)

        # Extract features
        current_features = self.feature_extractor.process_batch_features(states)  # shape: [B, F]
        next_features = self.feature_extractor.process_batch_features(next_states)  # shape: [B, F]

        # Convert to tensors
        current_features = current_features.to(self.device)
        next_features = next_features.to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)

        # Make weights require grad
        self.weights.requires_grad_(True)

        # Q-values
        current_q_values = torch.matmul(self.weights, current_features.T)  # [A, B]
        next_q_values = torch.matmul(self.weights, next_features.T)        # [A, B]

        # Compute TD target
        max_next_q = next_q_values.max(dim=0)[0]                           # [B]
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q        # [B]

        # Get Q(s,a)
        q_sa = current_q_values.gather(0, actions_tensor.unsqueeze(0)).squeeze(0)  # [B]
        td_errors = targets - q_sa

        # Compute loss
        loss = 0.5 * td_errors.pow(2).mean()

        # Backprop
        loss.backward()

        # Clip gradients
        grad_norms = self.weights.grad.norm(dim=1)
        clip_mask = grad_norms > self.grad_clip
        if clip_mask.any():
            self.weights.grad[clip_mask] *= (self.grad_clip / grad_norms[clip_mask].clamp(min=1e-8)).unsqueeze(1)

        # Gradient step
        with torch.no_grad():
            self.weights -= self.lr * self.weights.grad
            self.weights.grad.zero_()

        self.weights.grad = None  
        del loss 
        torch.cuda.empty_cache() 

        # Update target
        self.steps += 1
        self.target_weights = self.weights.clone().detach()

        # Learning rate decay
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Logging
        update_info = {
            'td_errors': td_errors.detach().cpu().numpy().tolist(),
            'mean_td_error': td_errors.abs().mean().item(),
            'grad_norms': grad_norms.detach().cpu().numpy().tolist(),
            'mean_grad_norm': grad_norms.mean().item(),
            'batch_size': current_batch_size,
            'current_trajectory_index': self.trajectory_buffer.current_trajectory_index,
        }

        return update_info



    def evaluate_td_error(self, mean_td_error):
        """Track TD error and determine if we should shift to next trajectory"""
        self.td_error_window.append(mean_td_error)
        
        if len(self.td_error_window) < 10:  # Need at least some data to make a decision
            return mean_td_error, False
            
        avg_td_error = np.mean(self.td_error_window)
        
        # Check if either condition is met:
        # 1. TD error is below threshold
        # 2. Maximum episodes per trajectory has been reached
        should_shift = ((avg_td_error < self.trajectory_shift_threshold or 
                         self.current_trajectory_episodes >= self.max_episodes_per_trajectory) and
                        self.trajectory_buffer.current_trajectory_index < len(self.trajectory_buffer.trajectories) - 1)
                        
        return avg_td_error, should_shift
    
    def shift_to_next_trajectory(self):
        """Shift to the next trajectory if available"""
        next_index = self.trajectory_buffer.current_trajectory_index + 1
        if next_index < len(self.trajectory_buffer.trajectories):
            success = self.trajectory_buffer.shift_to_trajectory(next_index)
            if success:
                # Reset TD error window and episode counter for new trajectory
                self.td_error_window.clear()
                self.current_trajectory_episodes = 0
                return True
        return False
    
class TorchLinearTrajectoryQLearningGDSTDM:
    """Linear Q-Learning that trains on individual trajectories"""
    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.feature_extractor = TorchFeatureExtractor(n_states, args)
        self.n_features = self.feature_extractor.n_features
        
        # Initialize weights (removed momentum)
        self.weights = torch.randn(n_actions, self.n_features).to(self.device) * 0.1
        self.target_weights = self.weights.clone()
        
        # Training parameters from args
        self.lr = args.learning_rate
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.grad_clip = args.grad_clip
        
        # Trajectory buffer and update frequency
        self.trajectory_buffer = TrajectoryBuffer(args.num_trajectories)
        self.batch_size = args.batch_size
        self.target_update_freq = args.update_freq
        self.steps = 0
        
        # TD error tracking for trajectory shifting
        self.trajectory_shift_threshold = args.trajectory_shift_threshold
        self.td_error_window = deque(maxlen=100)
        
        # Maximum episodes per trajectory
        self.max_episodes_per_trajectory = args.max_episodes_per_trajectory
        self.current_trajectory_episodes = 0
    
    def get_q_values(self, state, weights=None):
        if weights is None:
            weights = self.weights
        features = self.feature_extractor.extract(state)
        features = torch.tensor(features, device=self.device) if isinstance(features, np.ndarray) else features
        return torch.mv(weights, features).detach().cpu().numpy()
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_q_values(state))


    def update(self, state=None, action=None, reward=None, next_state=None, done=None):
        if len(self.trajectory_buffer) < 1:
            return {}

        current_batch_size = len(self.trajectory_buffer)
        states, actions, rewards, next_states, dones = self.trajectory_buffer.sample(current_batch_size)

        # Normalize rewards
        rewards = np.clip(rewards / 50.0, -2.0, 2.0)

        # Extract features
        current_features = self.feature_extractor.process_batch_features(states).to(self.device)
        next_features = self.feature_extractor.process_batch_features(next_states).to(self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)

        # === AUTOGRAD BLOCK ===
        with torch.enable_grad():
            # Fresh autograd-tracked weights
            weights = self.weights.detach().clone().requires_grad_(True)

            current_q_values = weights @ current_features.T
            next_q_values = weights @ next_features.T

            max_next_q = next_q_values.max(dim=0)[0]
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q
            q_sa = current_q_values.gather(0, actions_tensor.view(1, -1)).squeeze(0)
            td_errors = targets - q_sa

            loss = 0.5 * td_errors.pow(2).mean()
            loss.backward()
            grad_gd = weights.grad.detach()

        # === NO GRADIENT TRACKING FROM HERE ===
        with torch.no_grad():
            u_td = td_errors.view(-1, 1) * current_features
            u_gd = grad_gd[actions_tensor]

            gds_updates = torch.sign(u_gd) * torch.abs(u_td)

            grad_norms = torch.norm(gds_updates, dim=1)
            clip_mask = grad_norms > self.grad_clip
            gds_updates[clip_mask] *= (self.grad_clip / grad_norms[clip_mask].clamp(min=1e-8)).view(-1, 1)

            weight_update = torch.zeros_like(self.weights)
            lr_tensor = torch.tensor(self.lr, dtype=self.weights.dtype, device=self.device)
            gds_updates = gds_updates.to(self.weights.dtype)
            weight_update.index_add_(0, actions_tensor, lr_tensor * gds_updates)

            self.weights -= weight_update

        # Clean up autograd objects
        del loss, weights, current_q_values, next_q_values, q_sa
        torch.cuda.empty_cache()
        gc.collect()

        # Sync target
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_weights = self.weights.clone().detach()

        # Decay
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            'td_errors': td_errors.detach().cpu().numpy().tolist(),
            'mean_td_error': td_errors.abs().mean().item(),
            'grad_norms': grad_norms.detach().cpu().numpy().tolist(),
            'mean_grad_norm': grad_norms.mean().item(),
            'mean_gds_tdm_updates': grad_norms.mean().item(),
            'current_trajectory_index': self.trajectory_buffer.current_trajectory_index,
            'batch_size': current_batch_size
        }



    def evaluate_td_error(self, mean_td_error):
        """Track TD error and determine if we should shift to next trajectory"""
        self.td_error_window.append(mean_td_error)
        
        if len(self.td_error_window) < 10:  # Need at least some data to make a decision
            return mean_td_error, False
            
        avg_td_error = np.mean(self.td_error_window)
        
        # Check if either condition is met:
        # 1. TD error is below threshold
        # 2. Maximum episodes per trajectory has been reached
        should_shift = ((avg_td_error < self.trajectory_shift_threshold or 
                         self.current_trajectory_episodes >= self.max_episodes_per_trajectory) and
                        self.trajectory_buffer.current_trajectory_index < len(self.trajectory_buffer.trajectories) - 1)
                        
        return avg_td_error, should_shift
    
    def evaluate_success_rate(self, is_success):
        """Track success rate and determine if we should shift to next trajectory"""
        self.success_window.append(1 if is_success else 0)
        
        if len(self.success_window) < 100:
            return 0, False
            
        success_rate = 100 * sum(self.success_window) / len(self.success_window)
        should_shift = (success_rate >= self.trajectory_shift_threshold and 
                        self.trajectory_buffer.current_trajectory_index < len(self.trajectory_buffer.trajectories) - 1)
                        
        return success_rate, should_shift
    
    def shift_to_next_trajectory(self):
        """Shift to the next trajectory if available"""
        next_index = self.trajectory_buffer.current_trajectory_index + 1
        if next_index < len(self.trajectory_buffer.trajectories):
            success = self.trajectory_buffer.shift_to_trajectory(next_index)
            if success:
                # Reset TD error window and episode counter for new trajectory
                self.td_error_window.clear()
                self.current_trajectory_episodes = 0
                return True
        return False
    
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
    if args.method == "linear":
        agent = TorchLinearTrajectoryQLearning(n_states, n_actions, args)
    elif args.method == "gds-tdm":
        agent = TorchLinearTrajectoryQLearningGDSTDM(n_states, n_actions, args)
    elif args.method == "gd":
        agent = TorchLinearTrajectoryQLearningGD(n_states, n_actions, args)
    else:
        print("Method should be linear, gds-tdm, or gd")
        exit()
    
    # Load weights and other parameters from the pretrained model
    if 'weights' in checkpoint:
        agent.weights = checkpoint['weights']
        agent.target_weights = checkpoint['weights'].clone()
    elif isinstance(checkpoint, torch.Tensor):
        agent.weights = checkpoint
        agent.target_weights = checkpoint.clone()
    
    # Load normalizer if available
    if 'normalizer_mean' in checkpoint:
        agent.feature_extractor.normalizer.mean = checkpoint['normalizer_mean']
        agent.feature_extractor.normalizer.var = checkpoint['normalizer_var']
        agent.feature_extractor.normalizer.count = checkpoint['normalizer_count']
    
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

def train_and_evaluate(args):
    """Main training loop with trajectory-based learning using TD error thresholds"""
    # Setup directories and save config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('traj_results', args.save_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seeds
    set_seed(args.seed)
    
    # Initialize WandB
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"trajectory_learning_{timestamp}",
            config=vars(args)
        )
    
    # Collect trajectories using pretrained model
    trajectories = collect_trajectories(args.pretrained_model, args.num_trajectories, args)
    
    # Create agent based on method
    env = gym.make('LunarLander-v2')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    if args.method == "linear":
        agent = TorchLinearTrajectoryQLearning(n_states, n_actions, args)
    elif args.method == "gds-tdm":
        agent = TorchLinearTrajectoryQLearningGDSTDM(n_states, n_actions, args)
    elif args.method == "gd":
        agent = TorchLinearTrajectoryQLearningGD(n_states, n_actions, args)
    else:
        print("Method should be linear, gds-tdm, or gd")
        exit()
    
    # Add trajectories to buffer
    for trajectory in trajectories:
        agent.trajectory_buffer.add_trajectory(trajectory)
    
    # Start with the first trajectory
    agent.trajectory_buffer.shift_to_trajectory(0)
    
    # Initialize early stopping
    early_stopping = EarlyStoppingMonitor(patience=args.early_stopping_patience)
    
    # Training metrics storage
    training_history = {
        'rewards': [], 
        'epsilons': [],
        'learning_rates': [],
        'success_rates': [],
        'validation_rewards': [],
        'validation_success_rates': [],
        'current_trajectory': [],
        'trajectory_shifts': [],
        'best_episode_reward': float('-inf'),
        'best_episode_num': 0,
        'td_errors': [],
        'trajectory_episodes': []  # Track episodes spent on each trajectory
    }
    
    # Per-trajectory validation metrics
    trajectory_validation = {f'trajectory_{i}_rewards': [] for i in range(len(trajectories))}
    trajectory_validation.update({f'trajectory_{i}_success': [] for i in range(len(trajectories))})
    
    best_val_reward = float('-inf')
    
    for episode in range(args.episodes):
        print(f"Episode {episode}, Trajectory {agent.trajectory_buffer.current_trajectory_index}, Episodes on current trajectory: {agent.current_trajectory_episodes}")
        
        agent.current_trajectory_episodes += 1

        # One full update over the current trajectory
        update_info = agent.update()
        mean_td_error = update_info.get('mean_td_error', 0)
        training_history['td_errors'].append(mean_td_error)
        
        # Evaluate trajectory shift condition
        avg_td_error, should_shift = agent.evaluate_td_error(mean_td_error)
         
        if should_shift and agent.current_trajectory_episodes>10000:
            shift_reason = "TD error threshold" if avg_td_error < agent.trajectory_shift_threshold else "max episodes"
            shifted = agent.shift_to_next_trajectory()
            if shifted:
                new_idx = agent.trajectory_buffer.current_trajectory_index
                current_trajectory_idx = new_idx - 1
                print(f"Shifted from trajectory {current_trajectory_idx} to {new_idx} (reason: {shift_reason}, avg TD error: {avg_td_error:.4f})")
                training_history['trajectory_shifts'].append((episode, current_trajectory_idx, new_idx, shift_reason))
                training_history['trajectory_episodes'].append(agent.current_trajectory_episodes)
                
                if args.use_wandb:
                    wandb.log({
                        'training/trajectory_shift': 1,
                        'training/new_trajectory_index': new_idx,
                        'training/shift_reason': shift_reason,
                        'training/episodes_on_trajectory': agent.current_trajectory_episodes
                    }, step=episode)
         
        # Handle the case for the final trajectory
        if not should_shift and agent.trajectory_buffer.current_trajectory_index == len(agent.trajectory_buffer.trajectories) - 1:
            # We're on the final trajectory
            if agent.current_trajectory_episodes >= agent.max_episodes_per_trajectory:
                print(f"Reached max episodes ({agent.max_episodes_per_trajectory}) on final trajectory. Stopping training.")
                break
         
        # Store episode metrics
        training_history['epsilons'].append(agent.epsilon)
        training_history['learning_rates'].append(agent.lr)
        training_history['current_trajectory'].append(agent.trajectory_buffer.current_trajectory_index)
         
        # Logging for each episode
        if args.use_wandb:
            log_data = {
                'training/episode': episode,
                'training/epsilon': agent.epsilon,
                'training/learning_rate': agent.lr,
                'training/avg_td_error': avg_td_error,
                'training/mean_td_error': mean_td_error,
                'training/current_trajectory': agent.trajectory_buffer.current_trajectory_index,
                'training/episodes_on_trajectory': agent.current_trajectory_episodes
            }
            
            # Add algorithm-specific metrics
            if args.method == "gd":
                log_data['training/mean_gd_updates'] = update_info.get('mean_gd_updates', 0)
            elif args.method == "gds-tdm":
                log_data['training/mean_td_updates'] = update_info.get('mean_td_updates', 0)
                log_data['training/mean_gd_updates'] = update_info.get('mean_gd_updates', 0)
                log_data['training/mean_gds_updates'] = update_info.get('mean_gds_updates', 0)
            
            wandb.log(log_data, step=episode)
         
        # Validation
        if (episode + 1) % args.validate_freq == 0:
            validation_results = validate_on_trajectories(agent, trajectories)
            
            # Log overall validation stats
            training_history['validation_rewards'].append(validation_results['mean_reward'])
            training_history['validation_success_rates'].append(validation_results['success_rate'])
            
            # Log per-trajectory validation stats
            for i in range(len(trajectories)):
                trajectory_key = f'trajectory_{i}'
                trajectory_validation[f'trajectory_{i}_rewards'].append(validation_results[trajectory_key]['reward'])
                trajectory_validation[f'trajectory_{i}_success'].append(validation_results[trajectory_key]['success'])
            
            if args.use_wandb:
                log_data = {
                    'validation/mean_reward': validation_results['mean_reward'],
                    'validation/success_rate': validation_results['success_rate']
                }
                
                # Add per-trajectory metrics to wandb
                for i in range(len(trajectories)):
                    log_data[f'validation/trajectory_{i}_reward'] = validation_results[f'trajectory_{i}']['reward']
                    log_data[f'validation/trajectory_{i}_success'] = int(validation_results[f'trajectory_{i}']['success'])
                
                wandb.log(log_data, step=episode)
            
            # Early stopping check - only after min_episodes
            if episode + 1 >= args.min_episodes:
                if early_stopping.update(validation_results['mean_reward']):
                    print(f"Early stopping triggered at episode {episode + 1}")
                    print(f"Best validation reward achieved: {best_val_reward:.2f}")
                    break
            
            # Save best model
            if validation_results['mean_reward'] > best_val_reward:
                best_val_reward = validation_results['mean_reward']
                model_save_path = os.path.join(save_dir, 'best_model.pt')
                
                save_dict = {
                    'weights': agent.weights,
                    'epsilon': agent.epsilon,
                    'lr': agent.lr,
                    'normalizer_mean': agent.feature_extractor.normalizer.mean,
                    'normalizer_var': agent.feature_extractor.normalizer.var,
                    'normalizer_count': agent.feature_extractor.normalizer.count
                }
                
                torch.save(save_dict, model_save_path)
        
        # Regular model saving
        if (episode + 1) % args.save_model_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'model_ep{episode+1}.pt')
            torch.save(agent.weights, checkpoint_path)
    
    # Save training history
    history_to_save = {
        'training_metrics': training_history,
        'trajectory_validation': trajectory_validation
    }
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        history_json = json.dumps(history_to_save, cls=NumpyEncoder, indent=2)
        f.write(history_json)
    
    # Generate learning curves
    plot_learning_curve(
        training_history['rewards'],
        os.path.join(save_dir, 'reward_curve.png'),
        f"{args.method.upper()} Trajectory Learning - Rewards"
    )
    
    if training_history['td_errors']:
        plot_learning_curve(
            training_history['td_errors'],
            os.path.join(save_dir, 'td_error_curve.png'),
            f"{args.method.upper()} Trajectory Learning - TD Errors"
        )
    
    # Generate trajectory-specific plots
    for i in range(len(trajectories)):
        plt.figure(figsize=(12, 6))
        rewards = trajectory_validation[f'trajectory_{i}_rewards']
        plt.plot(range(0, len(rewards) * args.validate_freq, args.validate_freq), rewards)
        plt.title(f"Validation Rewards for Trajectory {i}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'trajectory_{i}_rewards.png'))
        plt.close()
    
    # Final testing with best model
    print("\nRunning final test episodes...")
    test_env = gym.make('LunarLander-v2', render_mode='rgb_array')
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    agent.weights = checkpoint['weights']
    agent.feature_extractor.normalizer.mean = checkpoint['normalizer_mean']
    agent.feature_extractor.normalizer.var = checkpoint['normalizer_var']
    agent.feature_extractor.normalizer.count = checkpoint['normalizer_count']
    agent.epsilon = 0.05  # Use low epsilon for testing
    
    # Test on all trajectories
    test_results = {}
    for i, trajectory in enumerate(trajectories):
        # Reset with the trajectory's seed
        state, _ = test_env.reset(seed=trajectory.seed)
        
        frames = []
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            frames.append(test_env.render())
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = test_env.step(action)
            
            total_reward += reward
            state = next_state
        
        test_results[f'trajectory_{i}'] = {
            'reward': float(total_reward),
            'success': total_reward > 200
        }
        
        # Save video
        if frames:
            save_video(frames, os.path.join(save_dir, f'test_trajectory_{i}.mp4'))
    
    # Calculate test statistics
    test_statistics = {
        'mean_reward': float(np.mean([test_results[f'trajectory_{i}']['reward'] for i in range(len(trajectories))])),
        'success_rate': float(np.mean([test_results[f'trajectory_{i}']['success'] for i in range(len(trajectories))])),
        'per_trajectory': test_results
    }
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_statistics, f, cls=NumpyEncoder, indent=2)
    
    # Log final test results to wandb
    if args.use_wandb:
        wandb.log({f'test/{k}': v for k, v in test_statistics.items() if k != 'per_trajectory'})
        for i in range(len(trajectories)):
            wandb.log({
                f'test/trajectory_{i}_reward': test_results[f'trajectory_{i}']['reward'],
                f'test/trajectory_{i}_success': int(test_results[f'trajectory_{i}']['success'])
            })
        wandb.finish()
    
    # Print final results
    print("\nTraining Completed!")
    print(f"Method: {args.method.upper()}")
    print(f"Total episodes: {episode + 1}")
    print(f"Best validation reward: {best_val_reward:.2f}")
    print("\nTest Results:")
    print(f"Mean reward: {test_statistics['mean_reward']:.2f}")
    print(f"Success rate: {test_statistics['success_rate']:.2%}")
    
    # Print per-trajectory results
    print("\nPer-Trajectory Results:")
    for i in range(len(trajectories)):
        result = test_results[f'trajectory_{i}']
        print(f"Trajectory {i}: Reward = {result['reward']:.2f}, Success = {result['success']}")
    
    print(f"\nResults saved to {save_dir}")
    
    # Cleanup
    env.close()
    test_env.close()
    
    return {
        'training_history': training_history,
        'trajectory_validation': trajectory_validation,
        'best_validation_reward': best_val_reward,
        'test_statistics': test_statistics,
        'save_dir': save_dir
    }

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