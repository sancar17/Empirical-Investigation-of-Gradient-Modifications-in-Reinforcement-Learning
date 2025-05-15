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
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Union
from helpers import set_seed, plot_learning_curve, save_video, EarlyStoppingMonitor, TorchRunningNormalizer, TorchExperienceReplay
from dqn import DQNNetwork, TorchDQNLearning, TorchDQNGDSTDMLearning, TorchDQNGDLearning
import torch.nn.functional as F

class HybridTDGDTrainer:
    """
    Implements hybrid training that starts with Temporal Difference (TD) learning 
    and switches to Gradient Descent (GD) learning after a specified number of episodes
    """
    
    def __init__(self, args):
        # Setup directories and save config
        self.args = args
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('results', args.save_name, self.timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Set all random seeds
        set_seed(args.seed)
        
        # Initialize WandB
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"hybrid_td_gd_{self.timestamp}",
                config=vars(args)
            )
        
        # Initialize environments
        self.env = gym.make('LunarLander-v2')
        self.val_env = ValidationEnv(num_episodes=20)
        
        # Get state and action dimensions
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        
        # Create initial TD agent (standard DQN)
        self.td_agent = TorchDQNLearning(self.n_states, self.n_actions, args)
        self.gd_agent = None  # Will be initialized later during the switch
        
        # Initialize early stopping
        self.early_stopping = EarlyStoppingMonitor(patience=args.early_stopping_patience)
        
        # Training metrics storage
        self.training_history = {
            'rewards': [], 
            'epsilons': [],
            'learning_rates': [],
            'success_rates': [],
            'validation_rewards': [],
            'validation_success_rates': [],
            'best_episode_reward': float('-inf'),
            'best_episode_num': 0,
            'errors': [],
            'mean_errors': [],
            'current_method': [],  # Track which method is being used at each episode
            'switch_episode': -1   # Will be set when the switch happens
        }
        
        self.best_val_reward = float('-inf')
        self.moving_avg_window = 100  # Window size for moving averages
        
        # Current active agent (starts with TD)
        self.current_agent = self.td_agent
        self.current_method = "td"
    
    def switch_to_gd(self, episode):
        """Switch from TD learning to Gradient Descent learning"""
        print(f"\n==== Switching from TD to Gradient Descent at episode {episode} ====\n")
        
        # Initialize GD agent with same parameters
        self.gd_agent = TorchDQNGDLearning(self.n_states, self.n_actions, self.args)
        
        # Copy policy and target network weights
        self.gd_agent.policy_net.load_state_dict(self.td_agent.policy_net.state_dict())
        self.gd_agent.target_net.load_state_dict(self.td_agent.target_net.state_dict())
        
        # Copy optimizer state
        self.gd_agent.optimizer = torch.optim.Adam(
            self.gd_agent.policy_net.parameters(), 
            lr=self.td_agent.lr
        )
        
        optimizer_state = deepcopy(self.td_agent.optimizer.state_dict())
        self.gd_agent.optimizer.load_state_dict(optimizer_state)
        
        # Copy experience replay buffer
        self.gd_agent.replay_buffer = deepcopy(self.td_agent.replay_buffer)
        
        # Copy normalizer
        self.gd_agent.normalizer.mean = self.td_agent.normalizer.mean.clone()
        self.gd_agent.normalizer.var = self.td_agent.normalizer.var.clone()
        self.gd_agent.normalizer.count = self.td_agent.normalizer.count
        
        # Copy learning parameters
        self.gd_agent.epsilon = self.td_agent.epsilon
        self.gd_agent.lr = self.td_agent.lr
        self.gd_agent.steps = self.td_agent.steps
        
        # Update current agent
        self.current_agent = self.gd_agent
        self.current_method = "gd"
        
        # Record the switch in history
        self.training_history['switch_episode'] = episode
        
        # Save TD agent before switch
        td_model_save_path = os.path.join(self.save_dir, 'td_agent_at_switch.pt')
        save_dict = {
            'policy_net': self.td_agent.policy_net.state_dict(),
            'target_net': self.td_agent.target_net.state_dict(),
            'optimizer': self.td_agent.optimizer.state_dict(),
            'epsilon': self.td_agent.epsilon,
            'lr': self.td_agent.lr,
            'normalizer_mean': self.td_agent.normalizer.mean,
            'normalizer_var': self.td_agent.normalizer.var,
            'normalizer_count': self.td_agent.normalizer.count
        }
        torch.save(save_dict, td_model_save_path)
        
        # Log the switch to wandb
        if self.args.use_wandb:
            wandb.log({
                'switch/episode': episode,
                'switch/td_epsilon': self.td_agent.epsilon,
                'switch/td_learning_rate': self.td_agent.lr
            })
    
    def train(self):
        """Main training loop with TD to GD switch"""
        
        for episode in range(self.args.episodes):
            # Check if it's time to switch from TD to GD
            if episode == self.args.switch_episode and self.current_method == "td":
                self.switch_to_gd(episode)
            
            state, _ = self.env.reset(seed=self.args.seed + episode)
            total_reward = 0
            done = False
            episode_errors = []
            episode_info = {}
            
            while not done:
                action = self.current_agent.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Update and collect metrics
                update_info = self.current_agent.update(state, action, reward, next_state, done)
                
                if update_info:
                    if 'td_errors' in update_info:
                        episode_errors.extend(update_info['td_errors'])
                    elif 'errors' in update_info:
                        episode_errors.extend(update_info['errors'])
                    episode_info.update(update_info)
                
                total_reward += reward
                state = next_state
                
                if truncated:
                    break
            
            # Store episode metrics
            self.training_history['rewards'].append(total_reward)
            self.training_history['epsilons'].append(self.current_agent.epsilon)
            self.training_history['learning_rates'].append(self.current_agent.lr)
            self.training_history['success_rates'].append(float(total_reward > 200))
            self.training_history['current_method'].append(self.current_method)
            
            if episode_errors:
                mean_error = np.mean(np.abs(episode_errors))
                self.training_history['errors'].append(mean_error)
                self.training_history['mean_errors'].append(
                    np.mean(self.training_history['errors'][-self.moving_avg_window:])
                )
            
            # Track best episode
            if total_reward > self.training_history['best_episode_reward']:
                self.training_history['best_episode_reward'] = total_reward
                self.training_history['best_episode_num'] = episode
            
            # Calculate running success rate
            recent_success_rate = np.mean(self.training_history['success_rates'][-100:]) if len(self.training_history['success_rates']) > 0 else 0
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Method: {self.current_method}, Reward: {total_reward:.1f}, "
                      f"Success Rate: {recent_success_rate:.2%}, Epsilon: {self.current_agent.epsilon:.4f}")
            
            # Logging for each episode
            if self.args.use_wandb:
                metrics = {
                    'training/episode': episode,
                    'training/reward': total_reward,
                    'training/epsilon': self.current_agent.epsilon,
                    'training/learning_rate': self.current_agent.lr,
                    'training/success_rate': recent_success_rate,
                    'training/method': self.current_method,
                    'training/mean_error': self.training_history['mean_errors'][-1] if self.training_history['mean_errors'] else 0
                }
                
                # Add Q-value metrics if available
                if 'mean_q_value' in episode_info:
                    metrics['training/mean_q_value'] = episode_info['mean_q_value']
                
                wandb.log(metrics)
            
            # Validation
            if (episode + 1) % self.args.validate_freq == 0:
                val_results = validate_agent(self.current_agent, self.val_env)
                self.training_history['validation_rewards'].append(val_results['mean_reward'])
                self.training_history['validation_success_rates'].append(val_results['success_rate'])
                
                if self.args.use_wandb:
                    wandb.log({
                        'validation/reward': val_results['mean_reward'],
                        'validation/reward_std': val_results['std_reward'],
                        'validation/success_rate': val_results['success_rate'],
                        'validation/method': self.current_method
                    })
                
                # Early stopping check - only after min_episodes
                if episode + 1 >= self.args.min_episodes:
                    if self.early_stopping.update(val_results['mean_reward']):
                        print(f"Early stopping triggered at episode {episode + 1}")
                        print(f"Best validation reward achieved: {self.best_val_reward:.2f}")
                        break
                
                # Save best model
                if val_results['mean_reward'] > self.best_val_reward:
                    self.best_val_reward = val_results['mean_reward']
                    model_save_path = os.path.join(self.save_dir, 'best_model.pt')
                    
                    save_dict = {
                        'policy_net': self.current_agent.policy_net.state_dict(),
                        'target_net': self.current_agent.target_net.state_dict(),
                        'optimizer': self.current_agent.optimizer.state_dict(),
                        'epsilon': self.current_agent.epsilon,
                        'lr': self.current_agent.lr,
                        'normalizer_mean': self.current_agent.normalizer.mean,
                        'normalizer_var': self.current_agent.normalizer.var,
                        'normalizer_count': self.current_agent.normalizer.count,
                        'method': self.current_method
                    }
                    
                    torch.save(save_dict, model_save_path)
            
            # Regular model saving
            if (episode + 1) % self.args.save_model_freq == 0:
                checkpoint_path = os.path.join(self.save_dir, f'model_ep{episode+1}_{self.current_method}.pt')
                torch.save({
                    'policy_net': self.current_agent.policy_net.state_dict(),
                    'target_net': self.current_agent.target_net.state_dict(),
                    'optimizer': self.current_agent.optimizer.state_dict(),
                }, checkpoint_path)
        
        # Save training history
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            # Convert numpy values to regular floats for JSON serialization
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.float32, np.float64, np.int32, np.int64)):
                    serializable_history[key] = [float(v) for v in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)
        
        # Generate learning curves
        plot_learning_curve(
            self.training_history['rewards'],
            os.path.join(self.save_dir, 'reward_curve.png'),
            f"Hybrid TD → GD - Rewards (Switch at ep {self.training_history['switch_episode']})"
        )
        
        if self.training_history['errors']:
            plot_learning_curve(
                self.training_history['errors'],
                os.path.join(self.save_dir, 'error_curve.png'),
                f"Hybrid TD → GD - Errors (Switch at ep {self.training_history['switch_episode']})"
            )
        
        # Plot method switch visualization
        self.plot_method_switch()
        
        # Final testing
        self.run_final_testing()
        
        # Cleanup
        self.env.close()
        self.val_env.close()
    
    def plot_method_switch(self):
        """Create a visualization showing the performance before and after the method switch"""
        rewards = np.array(self.training_history['rewards'])
        methods = np.array(self.training_history['current_method'])
        switch_ep = self.training_history['switch_episode']
        
        plt.figure(figsize=(12, 6))
        
        # Determine window size for moving average
        window_size = min(100, len(rewards) // 10)
        window_size = max(window_size, 5)  # At least 5 episodes for smoothing
        
        # Calculate moving average
        smoothed_rewards = []
        for i in range(len(rewards)):
            window_start = max(0, i - window_size + 1)
            smoothed_rewards.append(np.mean(rewards[window_start:i+1]))
        
        # Plot TD period
        td_indices = np.where(methods == 'td')[0]
        if len(td_indices) > 0:
            plt.plot(td_indices, [smoothed_rewards[i] for i in td_indices], 
                    color='blue', linewidth=2, label='TD Learning')
        
        # Plot GD period
        gd_indices = np.where(methods == 'gd')[0]
        if len(gd_indices) > 0:
            plt.plot(gd_indices, [smoothed_rewards[i] for i in gd_indices], 
                    color='green', linewidth=2, label='Gradient Descent')
        
        # Mark the switch point
        if switch_ep >= 0:
            plt.axvline(x=switch_ep, color='red', linestyle='--', 
                       label=f'Switch at episode {switch_ep}')
        
        plt.title('Learning Curve - Method Switch Visualization')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'method_switch.png'))
        plt.close()
    
    def run_final_testing(self):
        """Run final test episodes with the best model"""
        print("\nRunning final test episodes...")
        test_env = gym.make('LunarLander-v2', render_mode='rgb_array')
        test_results = []
        test_success_count = 0
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.save_dir, 'best_model.pt'))
        
        # Set up the right agent type based on method in checkpoint
        if checkpoint['method'] == 'td':
            test_agent = self.td_agent
        else:
            if self.gd_agent is None:
                # If we never switched during training, create the GD agent now
                self.gd_agent = TorchDQNGDLearning(self.n_states, self.n_actions, self.args)
            test_agent = self.gd_agent
        
        # Load model parameters
        test_agent.policy_net.load_state_dict(checkpoint['policy_net'])
        test_agent.target_net.load_state_dict(checkpoint['target_net'])
        test_agent.normalizer.mean = checkpoint['normalizer_mean']
        test_agent.normalizer.var = checkpoint['normalizer_var']
        test_agent.normalizer.count = checkpoint['normalizer_count']
        test_agent.epsilon = 0.05  # Use low epsilon for testing
        
        for test_ep in range(20):
            state, _ = test_env.reset(seed=2000 + test_ep)
            frames = []
            total_reward = 0
            done = False
            
            while not done:
                frames.append(test_env.render())
                action = test_agent.select_action(state)
                next_state, reward, done, truncated, _ = test_env.step(action)
                total_reward += reward
                state = next_state
                
                if truncated:
                    break
            
            test_results.append(total_reward)
            if total_reward > 200:
                test_success_count += 1
            
            # Save video
            if frames and test_ep < 5:  # Save only first 5 videos to save space
                save_video(frames, os.path.join(self.save_dir, f'test_episode_{test_ep}.mp4'))
        
        # Calculate test statistics
        test_statistics = {
            'mean_reward': float(np.mean(test_results)),
            'std_reward': float(np.std(test_results)),
            'min_reward': float(np.min(test_results)),
            'max_reward': float(np.max(test_results)),
            'success_rate': float(test_success_count / 20),
            'best_method': checkpoint['method']
        }
        
        # Save test results
        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_statistics, f, indent=2)
        
        # Log final test results to wandb
        if self.args.use_wandb:
            wandb.log({f'test/{k}': v for k, v in test_statistics.items()})
            wandb.finish()
        
        # Print final results
        print("\nTraining Completed!")
        print(f"Best method: {test_statistics['best_method']}")
        print(f"Switch episode: {self.training_history['switch_episode']}")
        print(f"Mean reward: {test_statistics['mean_reward']:.2f} ± {test_statistics['std_reward']:.2f}")
        print(f"Success rate: {test_statistics['success_rate']:.2%}")
        print(f"Reward range: [{test_statistics['min_reward']:.1f}, {test_statistics['max_reward']:.1f}]")
        print(f"\nResults saved to {self.save_dir}")
        
        # Cleanup
        test_env.close()
        
        return test_statistics


class ValidationEnv:
    """Environment wrapper for validation with fixed seeds"""
    def __init__(self, num_episodes: int = 20, base_seed: int = 1000):
        self.num_episodes = num_episodes
        self.base_seed = base_seed
        self.validation_seeds = [base_seed + i for i in range(num_episodes)]
        self.current_episode = 0
        self.env = gym.make('LunarLander-v2')
    
    def reset(self):
        if self.current_episode >= self.num_episodes:
            self.current_episode = 0
        seed = self.validation_seeds[self.current_episode]
        self.current_episode += 1
        return self.env.reset(seed=seed)
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()


def validate_agent(agent, val_env, num_episodes=20):
    """Run validation episodes with fixed seeds"""
    rewards = []
    success_count = 0
    
    for _ in range(num_episodes):
        state, _ = val_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = val_env.step(action)
            total_reward += reward
            state = next_state
            if truncated:
                break
        
        rewards.append(total_reward)
        if total_reward > 200:
            success_count += 1
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': success_count / num_episodes
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid TD to Gradient Descent Training Arguments')
    
    # Method switching parameter
    parser.add_argument('--switch_episode', type=int, default=3000,
                       help='Episode to switch from TD to Gradient Descent learning')
    
    # DQN architecture parameters
    parser.add_argument('--hidden_dims', type=str, default='120,84',
                        help='Hidden layer dimensions for network, comma-separated')
    
    # Learning parameters
    parser.add_argument('--learning_rate', type=float, default=0.0025)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--lr_decay', type=float, default=0.9999)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.9999)
    parser.add_argument('--min_epsilon', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    
    # Experience replay parameters
    parser.add_argument('--max_buffer_size', type=int, default=10000)
    parser.add_argument('--min_buffer_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--prioritized_replay', action='store_true', default=True)
    parser.add_argument('--prioritized_alpha', type=float, default=0.6)
    parser.add_argument('--prioritized_beta', type=float, default=0.4)
    parser.add_argument('--prioritized_beta_growth', type=float, default=0.001)
    
    # Training configuration
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--min_episodes', type=int, default=1000, 
                   help='Minimum number of episodes before starting early stopping checks')
    parser.add_argument('--validate_freq', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=50)
    
    # Gradient clipping and early stopping
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=50)
    
    # Save and logging
    parser.add_argument('--save_name', default="dqn_t1_hybrid_training_td_gd", type=str)
    parser.add_argument('--wandb_name', type=str, default="dqn_t1_hybrid_training_td_gd")
    parser.add_argument('--wandb_project', type=str, default='Q Learning Experiments')
    parser.add_argument('--use_wandb', action='store_true', default=True)
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Process the hidden_dims argument
    if hasattr(args, 'hidden_dims') and isinstance(args.hidden_dims, str):
        args.hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    return args


def main():
    args = parse_args()
    
    # Create and train the hybrid trainer
    trainer = HybridTDGDTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()