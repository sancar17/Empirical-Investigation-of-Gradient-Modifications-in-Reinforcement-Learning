import torch
import numpy as np
from dqn import DQNNetwork, TorchDQNLearning, TorchDQNGDSTDMLearning
from main import ValidationEnv
import os
import wandb

class DQNRepeatingAgent(TorchDQNLearning):
    """DQN with action repetition support"""
    def __init__(self, n_states, n_actions, args):
        super().__init__(n_states, n_actions, args)
        # Set action repetition count
        self.repeat_num = getattr(args, 'repeat_num', 1)
        
    def repeat_action(self, env, state, action):
        """
        Execute an action in the environment repeat_num times, or until done.
        """
        total_reward = 0
        done = False
        truncated = False
        next_state = state
        
        # Repeat the action repeat_num times or until done
        for _ in range(self.repeat_num):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        return next_state, total_reward, done, truncated

class DQNGDSTDMRepeatingAgent(TorchDQNGDSTDMLearning):
    """DQN-GDS-TDM with action repetition support"""
    def __init__(self, n_states, n_actions, args):
        super().__init__(n_states, n_actions, args)
        # Set action repetition count
        self.repeat_num = getattr(args, 'repeat_num', 1)
        
    def repeat_action(self, env, state, action):
        """
        Execute an action in the environment repeat_num times, or until done.
        """
        total_reward = 0
        done = False
        truncated = False
        next_state = state
        
        # Repeat the action repeat_num times or until done
        for _ in range(self.repeat_num):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        return next_state, total_reward, done, truncated

class HybridDQNRepeatingTrainer:
    """
    Implements hybrid training that starts with DQN with action repetition (n=3)
    and switches to DQN without repetition (n=1) after a specified number of episodes
    """
    
    def __init__(self, args):
        # Import necessary modules
        import torch
        import numpy as np
        import gymnasium as gym
        import os
        from datetime import datetime
        import matplotlib.pyplot as plt
        import json
        import wandb
        import os
        from copy import deepcopy
        from helpers import set_seed, plot_learning_curve, save_video, EarlyStoppingMonitor
        
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
                name=args.wandb_name or f"hybrid_repeat_dqn_{self.timestamp}",
                config=vars(args)
            )
        
        # Initialize environments
        self.env = gym.make('LunarLander-v2')
        self.val_env = ValidationEnv(num_episodes=20)
        
        # Get state and action dimensions
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        
        # Create agents with different repeat settings
        # Make a copy of args for each agent
        repeat_args = deepcopy(args)
        repeat_args.repeat_num = 3  # Set action repetition to 3 for first phase
        
        no_repeat_args = deepcopy(args)
        no_repeat_args.repeat_num = 1  # Set action repetition to 1 for second phase
        
        # Create the agents based on method
        if args.method == 'dqn':
            self.repeat_agent = DQNRepeatingAgent(self.n_states, self.n_actions, repeat_args)
            self.agent_class = DQNRepeatingAgent
        elif args.method == 'dqn-gds-tdm':
            self.repeat_agent = DQNGDSTDMRepeatingAgent(self.n_states, self.n_actions, repeat_args)
            self.agent_class = DQNGDSTDMRepeatingAgent
        else:
            raise ValueError(f"Unsupported method: {args.method}. Use 'dqn' or 'dqn-gds-tdm'")
        
        self.no_repeat_agent = None  # Will be initialized later during the switch
        
        # Store args for creating second agent later
        self.no_repeat_args = no_repeat_args
        
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
            'td_errors': [],
            'mean_td_errors': [],
            'mean_q_values': [],
            'repeat_num': [],      # Track repeat_num at each episode
            'switch_episode': -1   # Will be set when the switch happens
        }
        
        self.best_val_reward = float('-inf')
        self.moving_avg_window = 100  # Window size for moving averages
        
        # Current active agent (starts with DQN with repetition)
        self.current_agent = self.repeat_agent
        self.current_repeat_num = 5
    
    def switch_to_no_repeat(self, episode):
        """Switch from DQN with repetition to DQN without repetition"""
        from copy import deepcopy
        
        print(f"\n==== Switching from {self.args.method} (repeat={self.current_repeat_num}) to {self.args.method} (repeat=1) at episode {episode} ====\n")
        
        # Initialize second agent with no repetition
        self.no_repeat_agent = self.agent_class(self.n_states, self.n_actions, self.no_repeat_args)
        
        # Explicitly set repeat_num attribute
        self.no_repeat_agent.repeat_num = 1
        
        # Copy trained network weights from first agent
        self.no_repeat_agent.policy_net.load_state_dict(self.repeat_agent.policy_net.state_dict())
        self.no_repeat_agent.target_net.load_state_dict(self.repeat_agent.target_net.state_dict())
        
        # Copy optimizer state
        self.no_repeat_agent.optimizer.load_state_dict(self.repeat_agent.optimizer.state_dict())
        
        # Copy experience replay buffer
        self.no_repeat_agent.replay_buffer = deepcopy(self.repeat_agent.replay_buffer)
        
        # Copy normalizer
        self.no_repeat_agent.normalizer.mean = self.repeat_agent.normalizer.mean.clone()
        self.no_repeat_agent.normalizer.var = self.repeat_agent.normalizer.var.clone()
        self.no_repeat_agent.normalizer.count = self.repeat_agent.normalizer.count
        
        # Copy learning parameters
        self.no_repeat_agent.epsilon = self.repeat_agent.epsilon
        self.no_repeat_agent.lr = self.repeat_agent.lr
        self.no_repeat_agent.steps = self.repeat_agent.steps
        
        # Update current agent
        self.current_agent = self.no_repeat_agent
        self.current_repeat_num = 1
        
        # Record the switch in history
        self.training_history['switch_episode'] = episode
        
        # Save repeating agent before switch
        repeating_model_save_path = os.path.join(self.save_dir, 'repeat_agent_at_switch.pt')
        save_dict = {
            'policy_net': self.repeat_agent.policy_net.state_dict(),
            'target_net': self.repeat_agent.target_net.state_dict(),
            'optimizer': self.repeat_agent.optimizer.state_dict(),
            'epsilon': self.repeat_agent.epsilon,
            'lr': self.repeat_agent.lr,
            'normalizer_mean': self.repeat_agent.normalizer.mean,
            'normalizer_var': self.repeat_agent.normalizer.var,
            'normalizer_count': self.repeat_agent.normalizer.count,
            'repeat_num': self.repeat_agent.repeat_num,
            'steps': self.repeat_agent.steps
        }
        torch.save(save_dict, repeating_model_save_path)
        
        # Log the switch to wandb
        if self.args.use_wandb:
            wandb.log({
                'switch/episode': episode,
                'switch/epsilon': self.repeat_agent.epsilon,
                'switch/learning_rate': self.repeat_agent.lr,
                'switch/from_repeat_num': self.repeat_agent.repeat_num,
                'switch/to_repeat_num': self.no_repeat_agent.repeat_num
            })
    
    def train(self):
        """Main training loop with switching from DQN with repetition to DQN without repetition"""
        import numpy as np
        import torch
        import wandb
        import matplotlib.pyplot as plt
        import os
        import json
        from helpers import plot_learning_curve, save_video
        
        for episode in range(self.args.episodes):
            # Check if it's time to switch
            if episode == self.args.switch_episode and self.current_repeat_num == 5:
                self.switch_to_no_repeat(episode)
            
            state, _ = self.env.reset(seed=self.args.seed + episode)
            total_reward = 0
            done = False
            truncated = False
            episode_td_errors = []
            episode_q_values = []
            episode_info = {}
            
            while not done and not truncated:
                # Select action
                action = self.current_agent.select_action(state)
                
                # Execute action with repetition based on current agent's setting
                next_state, reward, done, truncated = self.current_agent.repeat_action(self.env, state, action)
                
                # Update and collect TD errors and other metrics
                update_info = self.current_agent.update(state, action, reward, next_state, done)
                
                if update_info:
                    if 'td_errors' in update_info:
                        episode_td_errors.extend(update_info['td_errors'])
                    if 'mean_q_value' in update_info:
                        episode_q_values.append(update_info['mean_q_value'])
                    episode_info.update(update_info)
                
                total_reward += reward
                state = next_state
            
            # Store episode metrics
            self.training_history['rewards'].append(total_reward)
            self.training_history['epsilons'].append(self.current_agent.epsilon)
            self.training_history['learning_rates'].append(self.current_agent.lr)
            self.training_history['success_rates'].append(float(total_reward > 200))
            self.training_history['repeat_num'].append(self.current_repeat_num)
            
            if episode_td_errors:
                mean_td_error = np.mean(np.abs(episode_td_errors))
                self.training_history['td_errors'].append(mean_td_error)
                self.training_history['mean_td_errors'].append(
                    np.mean(self.training_history['td_errors'][-self.moving_avg_window:])
                )
            
            if episode_q_values:
                mean_q_value = np.mean(episode_q_values)
                self.training_history['mean_q_values'].append(mean_q_value)
            
            # Track best episode
            if total_reward > self.training_history['best_episode_reward']:
                self.training_history['best_episode_reward'] = total_reward
                self.training_history['best_episode_num'] = episode
            
            # Calculate running success rate
            recent_success_rate = np.mean(self.training_history['success_rates'][-100:]) if len(self.training_history['success_rates']) > 0 else 0
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}, {self.args.method} with Repeat: {self.current_repeat_num}, "
                      f"Reward: {total_reward:.1f}, Success Rate: {recent_success_rate:.2%}, "
                      f"Epsilon: {self.current_agent.epsilon:.4f}")
            
            # Logging for each episode
            if self.args.use_wandb:
                metrics = {
                    'training/episode': episode,
                    'training/reward': total_reward,
                    'training/epsilon': self.current_agent.epsilon,
                    'training/learning_rate': self.current_agent.lr,
                    'training/success_rate': recent_success_rate,
                    'training/repeat_num': self.current_repeat_num,
                    'training/mean_td_error': self.training_history['mean_td_errors'][-1] if self.training_history['mean_td_errors'] else 0
                }
                
                if self.training_history['mean_q_values']:
                    metrics['training/mean_q_value'] = self.training_history['mean_q_values'][-1]
                
                wandb.log(metrics)
            
            # Validation
            if (episode + 1) % self.args.validate_freq == 0:
                # For validation, use the current agent with its current repeat_num setting
                val_results = validate_agent(self.current_agent, self.val_env)
                self.training_history['validation_rewards'].append(val_results['mean_reward'])
                self.training_history['validation_success_rates'].append(val_results['success_rate'])
                
                if self.args.use_wandb:
                    wandb.log({
                        'validation/reward': val_results['mean_reward'],
                        'validation/reward_std': val_results['std_reward'],
                        'validation/success_rate': val_results['success_rate'],
                        'validation/repeat_num': self.current_repeat_num
                    })
                
                # Early stopping check - only after min_episodes
                if episode + 1 >= self.args.min_episodes:
                    if self.early_stopping.update(val_results['mean_reward']):
                        print(f"Early stopping triggered at episode {episode + 1}")
                        print(f"Best validation reward achieved: {self.best_val_reward:.2f}")
                        break
                
                # Save best model
                if val_results['mean_reward'] > self.best_val_reward and episode + 1 >= self.args.min_episodes:
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
                        'repeat_num': self.current_repeat_num,
                        'steps': self.current_agent.steps
                    }
                    
                    torch.save(save_dict, model_save_path)
            
            # Regular model saving
            if (episode + 1) % self.args.save_model_freq == 0:
                checkpoint_path = os.path.join(self.save_dir, f'model_ep{episode+1}_repeat{self.current_repeat_num}.pt')
                torch.save({
                    'policy_net': self.current_agent.policy_net.state_dict(),
                    'target_net': self.current_agent.target_net.state_dict(),
                    'repeat_num': self.current_repeat_num
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
            f"Hybrid {self.args.method} Learning - Rewards (Switch from repeat=3 to repeat=1 at ep {self.training_history['switch_episode']})"
        )
        
        if self.training_history['td_errors']:
            plot_learning_curve(
                self.training_history['td_errors'],
                os.path.join(self.save_dir, 'td_error_curve.png'),
                f"Hybrid {self.args.method} Learning - TD Errors (Switch from repeat=3 to repeat=1 at ep {self.training_history['switch_episode']})"
            )
        
        if self.training_history['mean_q_values']:
            plot_learning_curve(
                self.training_history['mean_q_values'],
                os.path.join(self.save_dir, 'q_value_curve.png'),
                f"Hybrid {self.args.method} Learning - Q Values (Switch from repeat=3 to repeat=1 at ep {self.training_history['switch_episode']})"
            )
        
        # Plot method switch visualization
        self.plot_method_switch()
        
        # Final testing
        self.run_final_testing()
        
        # Cleanup
        self.env.close()
        self.val_env.close()
    
    def plot_method_switch(self):
        """Create a visualization showing the performance before and after the action repetition switch"""
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        
        rewards = np.array(self.training_history['rewards'])
        repeat_nums = np.array(self.training_history['repeat_num'])
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
        
        # Plot DQN period with repetition=3
        repeat3_indices = np.where(repeat_nums == 5)[0]
        if len(repeat3_indices) > 0:
            plt.plot(repeat3_indices, [smoothed_rewards[i] for i in repeat3_indices], 
                    color='blue', linewidth=2, label=f'{self.args.method} (repeat=5)')
        
        # Plot DQN period with repetition=1
        repeat1_indices = np.where(repeat_nums == 1)[0]
        if len(repeat1_indices) > 0:
            plt.plot(repeat1_indices, [smoothed_rewards[i] for i in repeat1_indices], 
                    color='green', linewidth=2, label=f'{self.args.method} (repeat=1)')
        
        # Mark the switch point
        if switch_ep >= 0:
            plt.axvline(x=switch_ep, color='red', linestyle='--', 
                       label=f'Switch at episode {switch_ep}')
        
        plt.title(f'Learning Curve - {self.args.method} Action Repetition Switch Visualization')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'repetition_switch.png'))
        plt.close()
    
    def run_final_testing(self):
        """Run final test episodes with the best model"""
        import numpy as np
        import torch
        import gymnasium as gym
        import os
        import json
        import wandb
        from helpers import save_video
        
        print("\nRunning final test episodes...")
        test_env = gym.make('LunarLander-v2', render_mode='rgb_array')
        test_results = []
        test_success_count = 0
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.save_dir, 'best_model.pt'))
        
        # Check if repeat_num exists in the checkpoint, if not, use the current agent's repeat_num
        repeat_num = checkpoint.get('repeat_num', self.current_repeat_num)
        
        # Set up the right agent based on repeat_num in checkpoint
        if repeat_num == 3:
            test_agent = self.repeat_agent
        else:
            if self.no_repeat_agent is None:
                # If we never switched during training, create the no-repeat agent now
                self.no_repeat_agent = self.agent_class(self.n_states, self.n_actions, self.no_repeat_args)
                # Explicitly set repeat_num
                self.no_repeat_agent.repeat_num = 1
            test_agent = self.no_repeat_agent
        
        # Load model parameters
        test_agent.policy_net.load_state_dict(checkpoint['policy_net'])
        test_agent.target_net.load_state_dict(checkpoint['target_net'])
        test_agent.normalizer.mean = checkpoint['normalizer_mean']
        test_agent.normalizer.var = checkpoint['normalizer_var']
        test_agent.normalizer.count = checkpoint['normalizer_count']
        test_agent.epsilon = 0.05  # Use low epsilon for testing
        
        # Set repeat_num for the agent, whether it was in the checkpoint or not
        test_agent.repeat_num = repeat_num
        print(f"Testing with repeat_num={repeat_num}")
        
        for test_ep in range(20):
            state, _ = test_env.reset(seed=2000 + test_ep)
            frames = []
            total_reward = 0
            done = False
            truncated = False
            
            while not done and not truncated:
                frames.append(test_env.render())
                action = test_agent.select_action(state)
                
                # Execute action with the repeat_num from the best model
                next_state, reward, done, truncated = test_agent.repeat_action(test_env, state, action)
                
                total_reward += reward
                state = next_state
            
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
            'best_repeat_num': int(repeat_num)
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
        print(f"Best repeat_num: {test_statistics['best_repeat_num']}")
        print(f"Switch episode: {self.training_history['switch_episode']}")
        print(f"Mean reward: {test_statistics['mean_reward']:.2f} ± {test_statistics['std_reward']:.2f}")
        print(f"Success rate: {test_statistics['success_rate']:.2%}")
        print(f"Reward range: [{test_statistics['min_reward']:.1f}, {test_statistics['max_reward']:.1f}]")
        print(f"\nResults saved to {self.save_dir}")
        
        # Cleanup
        test_env.close()
        
        return test_statistics

def validate_agent(agent, val_env, num_episodes=20):
    """Run validation episodes with fixed seeds"""
    rewards = []
    success_count = 0
    
    for _ in range(num_episodes):
        state, _ = val_env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            action = agent.select_action(state)
            
            # Execute action with repetition based on agent's setting
            next_state, reward, done, truncated = agent.repeat_action(val_env, state, action)
            
            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        if total_reward > 200:
            success_count += 1
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': success_count / num_episodes
    }

def parse_args():
    """Parse command line arguments"""
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description='Hybrid DQN Training with Action Repetition Variation')
    
    # Config file argument
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    
    # Method parameter
    parser.add_argument('--method', type=str, default='dqn', choices=['dqn', 'dqn-gds-tdm'],
                       help='Which DQN method to use')
    
    # Method switching parameter
    parser.add_argument('--switch_episode', type=int, default=4000,
                       help='Episode to switch from DQN (repeat=3) to DQN (repeat=1)')
    
    # Network parameters for DQN
    parser.add_argument('--hidden_dims', type=str, default='120,84',
                       help='Comma-separated list of hidden layer dimensions')
    
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
    
    # Action repetition parameter (not used directly, hardcoded in the trainer)
    parser.add_argument('--repeat_num', type=int, default=3,
                      help='Action repetition count (only used as initial value)')
    
    # Training configuration
    parser.add_argument('--episodes', type=int, default=2000000)
    parser.add_argument('--min_episodes', type=int, default=5000, 
                   help='Minimum number of episodes before starting early stopping checks')
    parser.add_argument('--validate_freq', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=1000)
    
    # Gradient clipping and early stopping
    parser.add_argument('--grad_clip', type=float, default=0.7)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    
    # Save and logging
    parser.add_argument('--save_name', default="hybrid_repeat_dqn", type=str)
    parser.add_argument('--wandb_name', type=str, default="hybrid_repeat_dqn")
    parser.add_argument('--wandb_project', type=str, default='Q Learning Experiments')
    parser.add_argument('--use_wandb', action='store_true', default=True)
    
    # Random seed
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
    
    # Parse hidden_dims from string to list
    if isinstance(args.hidden_dims, str):
        args.hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    return args

def main():
    args = parse_args()
    
    # Create and train the hybrid trainer
    trainer = HybridDQNRepeatingTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()