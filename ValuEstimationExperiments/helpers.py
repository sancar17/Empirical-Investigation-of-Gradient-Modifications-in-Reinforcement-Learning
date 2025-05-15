import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
from collections import deque
import cv2
import wandb
import random
import argparse
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='RL Training Arguments')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    # Method selection
    parser.add_argument('--method', type=str, default='linear', choices=['tabular', 'linear', 'gds-tdm', 'dqn', 'sarsa', 'sarsa-gds-tdm', 'dqn-gds-tdm', 'dual'])
    parser.add_argument('--feature_extractor', type=str, default='polynomial', 
                        choices=['normalized', 'polynomial', 'rbf'])
    parser.add_argument('--repeat_num', type=int, default=3)
    
    # Learning parameters
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--epsilon_decay', type=float, default=0.999)
    parser.add_argument('--min_epsilon', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--update_freq', type=int, default=50)
    parser.add_argument('--min_buffer_size', type=int, default=1000)
    parser.add_argument('--max_buffer_size', type=int, default=10000)
    parser.add_argument('--episodes', type=int, default=100000)
    parser.add_argument('--min_episodes', type=int, default=2000, 
                   help='Minimum number of episodes before starting early stopping checks')
    parser.add_argument('--validate_freq', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=1000)
    
    # Feature extractor parameters
    parser.add_argument('--poly_degree', type=int, default=3)
    parser.add_argument('--rbf_centers', type=int, default=200)
    parser.add_argument('--rbf_width', type=float, default=0.1, help='Width parameter for RBF features')
    parser.add_argument('--num_bins', type=int, default=30)
    
    # Gradient clipping and early stopping
    parser.add_argument('--grad_clip', type=float, default=0.7)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--early_stopping_threshold', type=float, default=0)
    
    # Save and logging
    parser.add_argument('--save_name', default="repeat_3", type=str)
    parser.add_argument('--wandb_name', type=str, default="repeat_3")
    parser.add_argument('--wandb_project', type=str, default='Tabular Q Learning Lunar Lander"')
    parser.add_argument('--use_wandb', default=True, action='store_true')
    
    # Reproducibility
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
        
        new_mean = self.mean + (delta * batch_size / total_count)
        
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
    
class TorchExperienceReplay:
    def __init__(self, capacity, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = deque(maxlen=args.max_buffer_size)  # Use max instead of capacity
        self.priorities = deque(maxlen=args.max_buffer_size)  # Use max here too
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon = 0.01
    
    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((
            torch.FloatTensor(state).to(self.device),
            action,
            reward,
            torch.FloatTensor(next_state).to(self.device),
            done
        ))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        probs = torch.tensor(self.priorities).to(self.device)
        probs = probs / probs.sum()
        
        indices = torch.multinomial(probs, batch_size).cpu().numpy()
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (torch.stack(states).cpu().numpy(),
                np.array(actions),
                np.array(rewards),
                torch.stack(next_states).cpu().numpy(),
                np.array(dones),
                weights.cpu().numpy(),
                indices)
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha

def parse_args_sarsa():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--method", type=str, choices=["td", "gd", "gds_tdm", "hybrid", "hybrid_repeat"], default="gds_tdm")
    parser.add_argument("--save_name", type=str, default="sarsa_gds_tdm_repeat3")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--validate_freq", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--min_lr", type=float, default=0.0001)
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min_epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="sarsa-onpolicy")
    parser.add_argument("--wandb_name", type=str, default="sarsa_gds_tdm_repeat3")
    parser.add_argument("--feature_extractor", type=str, default="polynomial")
    parser.add_argument("--poly_degree", type=int, default=3)
    parser.add_argument("--repeat_num", type=int, default=3)
    return parser.parse_args()




