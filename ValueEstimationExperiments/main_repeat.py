import torch
import numpy as np
import gymnasium as gym
import os
from datetime import datetime
import json
import wandb
import argparse
from helpers import set_seed, plot_learning_curve, save_video, parse_args, EarlyStoppingMonitor, TorchRunningNormalizer, TorchExperienceReplay
from dqn_repeat import TorchDQNGDSTDMLearning, TorchDQNLearning, TorchDQNGDLearning
from main import TorchGDLearning

class TorchLinearQLearning:
    """GPU-accelerated Linear Q-Learning with gradient clipping and action repetition"""
    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.feature_extractor = TorchFeatureExtractor(n_states, args)
        self.n_features = self.feature_extractor.n_features
        
        # Initialize weights and optimizer parameters
        self.weights = torch.randn(n_actions, self.n_features).to(self.device) * 0.1
        self.target_weights = self.weights.clone()
        self.velocity = torch.zeros_like(self.weights)
        
        # Training parameters from args
        self.lr = args.learning_rate
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.momentum = 0.95
        self.grad_clip = args.grad_clip
        
        # Action repetition parameter
        self.repeat_num = args.repeat_num if hasattr(args, 'repeat_num') else 1
        
        # Experience replay and update frequency
        # Experience replay parameters
        self.replay_buffer = TorchExperienceReplay(args.max_buffer_size, args)  # Use TorchExperienceReplay instead of deque
        self.batch_size = args.batch_size
        self.min_experiences = args.min_buffer_size
        self.target_update_freq = args.update_freq
        self.steps = 0
    
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
    
    def repeat_action(self, env, state, action):
        """Execute the same action for repeat_num steps or until done"""
        total_reward = 0
        done = False
        truncated = False
        next_state = state
        
        for _ in range(self.repeat_num):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        return next_state, total_reward, done, truncated
    
    def update(self, state, action, reward, next_state, done):
        reward = np.clip(reward / 50.0, -2.0, 2.0)
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer.buffer) < self.min_experiences:
            return {}
        
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size)
        
        current_features = self.feature_extractor.process_batch_features(states)
        current_q_values = torch.matmul(self.weights, current_features.t())
        next_features = self.feature_extractor.process_batch_features(next_states)
        next_q_values = torch.matmul(self.target_weights, next_features.t())
        
        max_next_q = next_q_values.max(dim=0)[0].cpu().numpy()
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        td_errors = []
        update_info = {'td_errors': [], 'grad_norms': []}
        
        for i, action_idx in enumerate(actions):
            td_error = targets[i] - current_q_values[action_idx, i].item()
            td_errors.append(td_error)
            
            gradient = td_error * current_features[i]
            
            # Gradient clipping
            grad_norm = torch.norm(gradient)
            if grad_norm > self.grad_clip:
                gradient = gradient * (self.grad_clip / grad_norm)
            
            update_info['td_errors'].append(td_error)
            update_info['grad_norms'].append(grad_norm.item())
            
            self.velocity[action_idx] = self.momentum * self.velocity[action_idx] + \
                                      self.lr * gradient
            self.weights[action_idx] += self.velocity[action_idx]
        
        self.replay_buffer.update_priorities(indices, td_errors)
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_weights = self.weights.clone()
        
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        update_info['mean_td_error'] = np.mean(np.abs(td_errors))
        update_info['mean_grad_norm'] = np.mean(update_info['grad_norms'])
        return update_info

import gc

class TorchGDSTDMLearning(TorchLinearQLearning):
    """GPU-accelerated GDS-TDM following paper equations exactly with action repetition"""

    def repeat_action(self, env, state, action):
        """Execute the same action for repeat_num steps or until done"""
        total_reward = 0
        done = False
        truncated = False
        next_state = state

        for _ in range(self.repeat_num):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        return next_state, total_reward, done, truncated

    def update(self, state, action, reward, next_state, done):
        reward = np.clip(reward / 50.0, -2.0, 2.0)
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer.buffer) < self.min_experiences:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones, _, indices = self.replay_buffer.sample(self.batch_size)

        # Feature extraction
        current_features = self.feature_extractor.process_batch_features(states).to(self.device)
        next_features = self.feature_extractor.process_batch_features(next_states).to(self.device)

        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # === AUTOGRAD BLOCK ===
        with torch.enable_grad():
            weights = self.weights.detach().clone().requires_grad_(True)

            current_q_values = weights @ current_features.T  # [A, B]
            next_q_values = weights @ next_features.T        # [A, B]

            max_next_q = next_q_values.max(dim=0)[0]
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q
            q_sa = current_q_values.gather(0, actions_tensor.view(1, -1)).squeeze(0)
            td_errors = targets - q_sa

            loss = 0.5 * td_errors.pow(2).mean()
            loss.backward()
            grad_gd = weights.grad.detach()

        # === NO-GRADIENT BLOCK ===
        with torch.no_grad():
            u_td = td_errors.view(-1, 1) * current_features
            u_gd = grad_gd[actions_tensor]

            u_gds = torch.sign(u_gd) * torch.abs(u_td)

            grad_norms = torch.norm(u_gds, dim=1)
            clip_mask = grad_norms > self.grad_clip
            u_gds[clip_mask] *= (self.grad_clip / grad_norms[clip_mask].clamp(min=1e-8)).view(-1, 1)

            # Momentum update and apply
            grads = torch.zeros_like(self.weights)
            for i in range(self.batch_size):
                a = actions_tensor[i]
                self.velocity[a] = self.momentum * self.velocity[a] + self.lr * u_gds[i]
                grads[a] += self.velocity[a]

            self.weights -= grads

        # Clean up computation graph and memory
        del loss, weights, current_q_values, next_q_values, q_sa
        torch.cuda.empty_cache()
        gc.collect()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Sync target
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_weights = self.weights.clone().detach()

        # Decay learning rate and epsilon
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Logging
        update_info = {
            'td_errors': td_errors.detach().cpu().numpy().tolist(),
            'td_updates': torch.norm(u_td, dim=1).detach().cpu().numpy().tolist(),
            'gd_updates': torch.norm(u_gd, dim=1).detach().cpu().numpy().tolist(),
            'gds_updates': grad_norms.detach().cpu().numpy().tolist(),
            'grad_norms': grad_norms.detach().cpu().numpy().tolist(),
        }

        for key in ['td_errors', 'grad_norms', 'td_updates', 'gd_updates', 'gds_updates']:
            update_info[f'mean_{key}'] = float(np.mean(np.abs(update_info[key])))

        return update_info

class TorchGDRepeatLearning(TorchGDLearning):
    """Residual Gradient Q-learning with action repetition"""

    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.feature_extractor = TorchFeatureExtractor(n_states, args)
        self.n_features = self.feature_extractor.n_features
        
        # Initialize weights and optimizer parameters
        self.weights = torch.randn(n_actions, self.n_features).to(self.device) * 0.1
        self.target_weights = self.weights.clone()
        self.velocity = torch.zeros_like(self.weights)
        
        # Training parameters from args
        self.lr = args.learning_rate
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.momentum = 0.95
        self.grad_clip = args.grad_clip
        
        # Action repetition parameter
        self.repeat_num = args.repeat_num if hasattr(args, 'repeat_num') else 1
        
        # Experience replay and update frequency
        self.replay_buffer = TorchExperienceReplay(args.max_buffer_size, args)  # Use TorchExperienceReplay instead of deque
        self.batch_size = args.batch_size
        self.min_experiences = args.min_buffer_size
        self.target_update_freq = args.update_freq
        self.steps = 0

    def repeat_action(self, env, state, action):
        """Execute the same action for repeat_num steps or until done"""
        total_reward = 0
        done = False
        truncated = False
        next_state = state

        for _ in range(self.repeat_num):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        return next_state, total_reward, done, truncated

    def update(self, state, action, reward, next_state, done):
        reward = np.clip(reward / 50.0, -2.0, 2.0)
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer.buffer) < self.min_experiences:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones, _, indices = \
            self.replay_buffer.sample(self.batch_size)

        # Feature extraction
        current_features = self.feature_extractor.process_batch_features(states).to(self.device)  # [B, F]
        next_features = self.feature_extractor.process_batch_features(next_states).to(self.device)  # [B, F]
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Enable autograd
        self.weights.requires_grad_(True)
        self.weights.grad = None

        # Q-values
        current_q_values = self.weights @ current_features.T  # [A, B]
        next_q_values = self.weights @ next_features.T        # [A, B]

        # Targets and TD error
        max_next_q = next_q_values.max(dim=0)[0]              # [B]
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q
        q_sa = current_q_values.gather(0, actions_tensor.view(1, -1)).squeeze(0)
        td_errors = targets - q_sa

        # Loss and autograd
        loss = 0.5 * td_errors.pow(2).mean()
        loss.backward()
        grad_gd = self.weights.grad.detach()  # [A, F]

        # Per-action gradient
        grad_gd_per_action = grad_gd[actions_tensor]  # [B, F]

        # Clip gradient
        grad_norms = torch.norm(grad_gd_per_action, dim=1)
        clip_mask = grad_norms > self.grad_clip
        grad_gd_per_action[clip_mask] *= (self.grad_clip / grad_norms[clip_mask].clamp(min=1e-8)).view(-1, 1)

        # Log metrics
        update_info = {
            'td_errors': td_errors.detach().cpu().numpy().tolist(),
            'gd_updates': torch.norm(grad_gd_per_action, dim=1).detach().cpu().numpy().tolist(),
            'grad_norms': grad_norms.detach().cpu().numpy().tolist()
        }

        # Apply momentum + update weights
        grads = torch.zeros_like(self.weights)
        for i in range(self.batch_size):
            a = actions_tensor[i]
            self.velocity[a] = self.momentum * self.velocity[a] + self.lr * grad_gd_per_action[i]
            grads[a] += self.velocity[a]

        with torch.no_grad():
            self.weights.copy_(self.weights - grads)
            self.weights.grad.zero_()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Target update
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_weights = self.weights.clone().detach()

        # Decay learning rate and epsilon
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Logging
        for key in ['td_errors', 'grad_norms', 'gd_updates']:
            update_info[f'mean_{key}'] = float(np.mean(np.abs(update_info[key])))

        return update_info

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
    """Run validation episodes with fixed seeds and action repetition support"""
    print("validating")
    rewards = []
    success_count = 0
    
    for _ in range(num_episodes):
        state, _ = val_env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            action = agent.select_action(state)
            
            # Execute action with repetition if applicable
            if hasattr(agent, 'repeat_action'):
                next_state, reward, done, truncated = agent.repeat_action(val_env, state, action)
            else:
                # Default behavior for agents without action repetition
                print("repeat action not implemented")
                exit()
                next_state, reward, done, truncated, _ = val_env.step(action)
            
            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        if total_reward > 200:
            success_count += 1
    
    print("validated")
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': success_count / num_episodes
    }

def train_and_evaluate(args: argparse.Namespace):
    """Main training loop with action repetition, enhanced TD error logging and evaluation"""
    # Setup directories and save config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', args.save_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set all random seeds
    set_seed(args.seed)
    
    # Initialize WandB
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{args.method}_{args.feature_extractor}_{timestamp}",
            config=vars(args)
        )
    
    # Initialize environments
    env = gym.make('LunarLander-v2')
    val_env = ValidationEnv(num_episodes=20)
    
    # Create agent
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = create_agent(args)
    
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
        'best_episode_reward': float('-inf'),
        'best_episode_num': 0,
        'td_errors': [],  # Store TD errors during training
        'validation_td_errors': [],  # Store TD errors during validation
        'mean_td_errors': [],  # Moving average of TD errors
        'mean_validation_td_errors': [],  # Moving average of validation TD errors
        'gd_updates': [],  # For GDS-TDM
        'td_updates': [],  # For GDS-TDM
        'gds_updates': []  # For GDS-TDM
    }
    
    best_val_reward = float('-inf')
    moving_avg_window = 100  # Window size for moving averages
    
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0
        done = False
        truncated = False
        episode_td_errors = []
        episode_info = {}
        
        while not done and not truncated:
            # Select action
            action = agent.select_action(state)
            
            # Execute action with repetition if applicable
            if hasattr(agent, 'repeat_action'):
                next_state, reward, done, truncated = agent.repeat_action(env, state, action)
            else:
                # Default behavior for agents without action repetition
                print("repeat action not implemented")
                exit()
                next_state, reward, done, truncated, _ = env.step(action)
            
            # Update and collect TD errors and other metrics
            update_info = agent.update(state, action, reward, next_state, done)
            
            if update_info:
                if 'td_errors' in update_info:
                    episode_td_errors.extend(update_info['td_errors'])
                episode_info.update(update_info)
            
            total_reward += reward
            state = next_state
        
        # Store episode metrics
        training_history['rewards'].append(total_reward)
        training_history['epsilons'].append(agent.epsilon)
        training_history['learning_rates'].append(agent.lr)
        training_history['success_rates'].append(float(total_reward > 200))
        
        if episode_td_errors:
            mean_td_error = np.mean(np.abs(episode_td_errors))
            training_history['td_errors'].append(mean_td_error)
            training_history['mean_td_errors'].append(
                np.mean(training_history['td_errors'][-moving_avg_window:])
            )
        
        # Track best episode
        if total_reward > training_history['best_episode_reward']:
            training_history['best_episode_reward'] = total_reward
            training_history['best_episode_num'] = episode
        
        # Calculate running success rate
        recent_success_rate = np.mean(training_history['success_rates'][-100:])
        
        # Logging for each episode
        if args.use_wandb:
            metrics = {
                'training/episode': episode,
                'training/reward': total_reward,
                'training/epsilon': agent.epsilon,
                'training/learning_rate': agent.lr,
                'training/success_rate': recent_success_rate,
                'training/mean_td_error': training_history['mean_td_errors'][-1] if training_history['mean_td_errors'] else 0
            }
            
            # Add GDS-TDM specific metrics
            if args.method == 'gds-tdm':
                for key in ['mean_td_updates', 'mean_gd_updates', 'mean_gds_updates']:
                    if key in episode_info:
                        metrics[f'training/{key}'] = episode_info[key]
            
            wandb.log(metrics)
        
        # Validation
        if (episode + 1) % args.validate_freq == 0:
            val_results = validate_agent(agent, val_env)
            training_history['validation_rewards'].append(val_results['mean_reward'])
            training_history['validation_success_rates'].append(val_results['success_rate'])
            
            if 'td_errors' in val_results:
                training_history['validation_td_errors'].append(np.mean(np.abs(val_results['td_errors'])))
                training_history['mean_validation_td_errors'].append(
                    np.mean(training_history['validation_td_errors'][-moving_avg_window:])
                )
            
            if args.use_wandb:
                wandb.log({
                    'validation/reward': val_results['mean_reward'],
                    'validation/reward_std': val_results['std_reward'],
                    'validation/success_rate': val_results['success_rate'],
                    'validation/mean_td_error': training_history['mean_validation_td_errors'][-1] 
                        if training_history['mean_validation_td_errors'] else 0
                })
            
            # Early stopping check - only after min_episodes
            if episode + 1 >= args.min_episodes:
                if early_stopping.update(val_results['mean_reward']):
                    print(f"Early stopping triggered at episode {episode + 1}")
                    print(f"Best validation reward achieved: {best_val_reward:.2f}")
                    break
            
            # Save best model
            if val_results['mean_reward'] > best_val_reward:
                best_val_reward = val_results['mean_reward']
                model_save_path = os.path.join(save_dir, 'best_model.pt')
                
                save_dict = {
                    'epsilon': agent.epsilon,
                    'lr': agent.lr
                }
                
                # Add type-specific model parameters
                if hasattr(agent, 'weights'):
                    save_dict.update({
                        'weights': agent.weights,
                        'normalizer_mean': agent.feature_extractor.normalizer.mean,
                        'normalizer_var': agent.feature_extractor.normalizer.var,
                        'normalizer_count': agent.feature_extractor.normalizer.count
                    })
                elif hasattr(agent, 'q_table'):
                    save_dict.update({
                        'q_table': agent.q_table,
                        'normalizer_mean': agent.normalizer.mean,
                        'normalizer_var': agent.normalizer.var,
                        'normalizer_count': agent.normalizer.count
                    })
                
                torch.save(save_dict, model_save_path)

        if episode==0:
                model_save_path = os.path.join(save_dir, 'best_model.pt')
                
                save_dict = {
                    'epsilon': agent.epsilon,
                    'lr': agent.lr
                }
                
                # Add type-specific model parameters
                if hasattr(agent, 'weights'):
                    save_dict.update({
                        'weights': agent.weights,
                        'normalizer_mean': agent.feature_extractor.normalizer.mean,
                        'normalizer_var': agent.feature_extractor.normalizer.var,
                        'normalizer_count': agent.feature_extractor.normalizer.count
                    })
                elif hasattr(agent, 'q_table'):
                    save_dict.update({
                        'q_table': agent.q_table,
                        'normalizer_mean': agent.normalizer.mean,
                        'normalizer_var': agent.normalizer.var,
                        'normalizer_count': agent.normalizer.count
                    })
                
                torch.save(save_dict, model_save_path)
            
        # Regular model saving
        if (episode + 1) % args.save_model_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'model_ep{episode+1}.pt')
            if hasattr(agent, 'weights'):
                torch.save(agent.weights, checkpoint_path)
            elif hasattr(agent, 'q_table'):
                torch.save(agent.q_table, checkpoint_path)
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Generate learning curves
    plot_learning_curve(
        training_history['rewards'],
        os.path.join(save_dir, 'reward_curve.png'),
        f"{args.method}_{args.feature_extractor} - Rewards"
    )
    
    if training_history['td_errors']:
        plot_learning_curve(
            training_history['td_errors'],
            os.path.join(save_dir, 'td_error_curve.png'),
            f"{args.method}_{args.feature_extractor} - TD Errors"
        )
    
    # Final testing with best model
    print("\nRunning final test episodes...")
    test_env = gym.make('LunarLander-v2', render_mode='rgb_array')
    test_results = []
    test_td_errors = []
    test_success_count = 0
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    if hasattr(agent, 'weights'):
        agent.weights = checkpoint['weights']
        agent.feature_extractor.normalizer.mean = checkpoint['normalizer_mean']
        agent.feature_extractor.normalizer.var = checkpoint['normalizer_var']
        agent.feature_extractor.normalizer.count = checkpoint['normalizer_count']
    elif hasattr(agent, 'q_table'):
        agent.q_table = checkpoint['q_table']
        agent.normalizer.mean = checkpoint['normalizer_mean']
        agent.normalizer.var = checkpoint['normalizer_var']
        agent.normalizer.count = checkpoint['normalizer_count']
    agent.epsilon = checkpoint['epsilon']
    agent.lr = checkpoint['lr']
    
    for test_ep in range(20):
        state, _ = test_env.reset(seed=2000 + test_ep)
        frames = []
        total_reward = 0
        episode_td_errors = []
        done = False
        truncated = False
        
        while not done and not truncated:
            frames.append(test_env.render())
            action = agent.select_action(state)
            
            # Execute action with repetition if applicable
            if hasattr(agent, 'repeat_action'):
                next_state, reward, done, truncated = agent.repeat_action(test_env, state, action)
            else:
                print("repeat action not implemented")
                exit()
                # Default behavior for agents without action repetition
                next_state, reward, done, truncated, _ = test_env.step(action)
            
            # Collect TD errors during testing
            if hasattr(agent, 'compute_td_error'):
                td_error = agent.compute_td_error(state, action, reward, next_state, done)
                episode_td_errors.append(td_error)
            
            total_reward += reward
            state = next_state
        
        test_results.append(total_reward)
        if episode_td_errors:
            test_td_errors.append(np.mean(np.abs(episode_td_errors)))
        if total_reward > 200:
            test_success_count += 1
        
        # Save video
        if frames:
            save_video(frames, os.path.join(save_dir, f'test_episode_{test_ep}.mp4'))
    
    # Calculate test statistics
    test_statistics = {
        'mean_reward': float(np.mean(test_results)),
        'std_reward': float(np.std(test_results)),
        'min_reward': float(np.min(test_results)),
        'max_reward': float(np.max(test_results)),
        'success_rate': float(test_success_count / 20),
        'mean_td_error': float(np.mean(test_td_errors)) if test_td_errors else 0,
        'std_td_error': float(np.std(test_td_errors)) if test_td_errors else 0
    }
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_statistics, f, indent=2)
    
    # Log final test results to wandb
    if args.use_wandb:
        wandb.log({f'test/{k}': v for k, v in test_statistics.items()})
        wandb.finish()
    
    # Print final results
    print("\nTraining Completed!")
    print(f"Total episodes: {episode + 1}")
    print(f"Best validation reward: {best_val_reward:.2f}")
    print("\nTest Results:")
    print(f"Mean reward: {test_statistics['mean_reward']:.2f} ± {test_statistics['std_reward']:.2f}")
    print(f"Success rate: {test_statistics['success_rate']:.2%}")
    print(f"Mean TD error: {test_statistics['mean_td_error']:.4f} ± {test_statistics['std_td_error']:.4f}")
    print(f"Reward range: [{test_statistics['min_reward']:.1f}, {test_statistics['max_reward']:.1f}]")
    print(f"\nResults saved to {save_dir}")
    
    # Cleanup
    env.close()
    val_env.close()
    test_env.close()
    
    return {
        'training_history': training_history,
        'best_validation_reward': best_val_reward,
        'test_statistics': test_statistics,
        'save_dir': save_dir
    }
           
class TorchFeatureExtractor:
    """GPU-accelerated feature extractor"""
    def __init__(self, n_states, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.feature_type = args.feature_extractor
        self.normalizer = TorchRunningNormalizer(n_states)
        
        if self.feature_type == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            self.poly = PolynomialFeatures(degree=args.poly_degree, include_bias=True)
            self.n_features = self.poly.fit_transform(np.zeros((1, n_states))).shape[1]
            print(self.n_features)
        elif self.feature_type == 'rbf':
            self.n_centers = args.rbf_centers
            self.n_features = self.n_centers * 6 + 2
            print(self.n_features)
            self.centers = {
                i: torch.linspace(-1, 1, self.n_centers).to(self.device)
                for i in range(6)
            }
            # Use single width parameter from args
            self.width = args.rbf_width
            self.widths = torch.full((6,), self.width).to(self.device)
        else:
            self.n_features = n_states
    
    def extract(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        normalized = torch.from_numpy(self.normalizer.normalize(state)).float().to(self.device)
        
        if self.feature_type == 'polynomial':
            poly_features = self.poly.transform(normalized.cpu().numpy().reshape(1, -1))[0]
            return torch.from_numpy(poly_features).float().to(self.device)
        elif self.feature_type == 'rbf':
            features = []
            normalized = normalized.reshape(-1)
            for dim in range(6):
                dist = normalized[dim] - self.centers[dim]
                rbf = torch.exp(-(dist ** 2) / (2 * self.widths[dim] ** 2))
                features.extend(rbf.cpu().tolist())
            features.extend(normalized[6:].cpu().tolist())
            return torch.tensor(features).float().to(self.device)
        else:
            return normalized.reshape(-1)

    def process_batch_features(self, states):
        features_list = []
        for state in states:
            features = self.extract(state)
            features_list.append(features)
        return torch.stack(features_list)

def create_agent(args):
    """Create agent based on arguments"""
    env = gym.make('LunarLander-v2')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    
    if args.method == 'linear':
        return TorchLinearQLearning(n_states, n_actions, args)
    elif args.method == 'gds-tdm':
        return TorchGDSTDMLearning(n_states, n_actions, args)
    elif args.method == 'gd':
        return TorchGDRepeatLearning(n_states, n_actions, args)
    elif args.method == 'dqn':
        return TorchDQNLearning(n_states, n_actions, args)
    elif args.method == 'dqn-gds-tdm':
        return TorchDQNGDSTDMLearning(n_states, n_actions, args)
    elif args.method == 'dqn-gd':
        return TorchDQNGDLearning(n_states, n_actions, args)

def main():
    args = parse_args()
    train_and_evaluate(args)

if __name__ == "__main__":
    main()