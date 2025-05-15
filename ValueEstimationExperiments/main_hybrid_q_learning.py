import torch
import numpy as np
import gymnasium as gym
import os
from datetime import datetime
import json
import argparse
from copy import deepcopy
import wandb
from helpers import set_seed, plot_learning_curve, save_video, EarlyStoppingMonitor, TorchExperienceReplay
from main import TorchLinearQLearning, TorchGDSTDMLearning, ValidationEnv, validate_agent

class HybridQLearningTrainer:
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('results', args.save_name, self.timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

        set_seed(args.seed)

        self.env = gym.make('LunarLander-v2')
        self.val_env = ValidationEnv(num_episodes=20)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.q_agent = TorchLinearQLearning(self.n_states, self.n_actions, args)
        self.gds_agent = None

        self.current_agent = self.q_agent
        self.current_method = 'q_learning'
        self.early_stopping = EarlyStoppingMonitor(patience=args.early_stopping_patience)

        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"hybrid_q_learning_{self.timestamp}",
                config=vars(args)
            )

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
            'current_method': [],
            'switch_episode': -1
        }
        self.best_val_reward = float('-inf')

    def switch_to_gds(self, episode):
        print(f"\n==== Switching from Q-Learning to GDS-TDM at episode {episode} ====\n")
        self.gds_agent = TorchGDSTDMLearning(self.n_states, self.n_actions, self.args)
        self.gds_agent.weights = self.q_agent.weights.clone()
        self.gds_agent.target_weights = self.q_agent.target_weights.clone()
        self.gds_agent.velocity = self.q_agent.velocity.clone()
        self.gds_agent.replay_buffer = deepcopy(self.q_agent.replay_buffer)
        self.gds_agent.feature_extractor = deepcopy(self.q_agent.feature_extractor)
        self.gds_agent.epsilon = self.q_agent.epsilon
        self.gds_agent.lr = self.q_agent.lr
        self.gds_agent.steps = self.q_agent.steps

        self.current_agent = self.gds_agent
        self.current_method = 'gds-tdm'
        self.training_history['switch_episode'] = episode

        if self.args.use_wandb:
            wandb.log({
                'switch/episode': episode,
                'switch/epsilon': self.q_agent.epsilon,
                'switch/lr': self.q_agent.lr
            })

    def train(self):
        for episode in range(self.args.episodes):
            print(episode)
            if episode == self.args.switch_episode and self.current_method == 'q_learning':
                self.switch_to_gds(episode)

            state, _ = self.env.reset(seed=self.args.seed + episode)
            total_reward = 0
            done = False
            episode_td_errors = []

            while not done:
                action = self.current_agent.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                update_info = self.current_agent.update(state, action, reward, next_state, done)
                if update_info and 'td_errors' in update_info:
                    episode_td_errors.extend(update_info['td_errors'])
                total_reward += reward
                state = next_state
                if truncated:
                    break

            self.training_history['rewards'].append(total_reward)
            self.training_history['epsilons'].append(self.current_agent.epsilon)
            self.training_history['learning_rates'].append(self.current_agent.lr)
            self.training_history['success_rates'].append(float(total_reward > 200))
            self.training_history['current_method'].append(self.current_method)

            if episode_td_errors:
                mean_td_error = np.mean(np.abs(episode_td_errors))
                self.training_history['td_errors'].append(mean_td_error)
                window = min(100, len(self.training_history['td_errors']))
                self.training_history['mean_td_errors'].append(
                    np.mean(self.training_history['td_errors'][-window:])
                )

            if total_reward > self.training_history['best_episode_reward']:
                self.training_history['best_episode_reward'] = total_reward
                self.training_history['best_episode_num'] = episode

            if self.args.use_wandb:
                wandb.log({
                    'training/episode': episode,
                    'training/reward': total_reward,
                    'training/epsilon': self.current_agent.epsilon,
                    'training/learning_rate': self.current_agent.lr,
                    'training/method': self.current_method,
                    'training/mean_td_error': self.training_history['mean_td_errors'][-1] if self.training_history['mean_td_errors'] else 0,
                    'training/success_rate': np.mean(self.training_history['success_rates'][-100:])
                })

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

                if val_results['mean_reward'] > self.best_val_reward:
                    self.best_val_reward = val_results['mean_reward']
                    torch.save(self.current_agent.weights, os.path.join(self.save_dir, 'best_model.pt'))

                if episode + 1 >= self.args.min_episodes:
                    if self.early_stopping.update(val_results['mean_reward']):
                        print(f"Early stopping at episode {episode + 1}")
                        break

        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)

        plot_learning_curve(self.training_history['rewards'], os.path.join(self.save_dir, 'rewards.png'), 'Episode Rewards')
        plot_learning_curve(self.training_history['td_errors'], os.path.join(self.save_dir, 'td_errors.png'), 'TD Errors')
        self.env.close()
        self.val_env.close()
        if self.args.use_wandb:
            wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000000)
    parser.add_argument('--switch_episode', type=int, default=10)
    parser.add_argument('--validate_freq', type=int, default=50)
    parser.add_argument('--min_episodes', type=int, default=5000)
    parser.add_argument('--save_name', type=str, default='hybrid_q_learning')
    parser.add_argument('--wandb_name', type=str, default='hybrid_q_learning')
    parser.add_argument('--wandb_project', type=str, default='Q Learning Experiments')
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.999)
    parser.add_argument('--min_epsilon', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=0.7)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_buffer_size', type=int, default=10000)
    parser.add_argument('--min_buffer_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--update_freq', type=int, default=50)
    parser.add_argument('--feature_extractor', type=str, default='polynomial')
    parser.add_argument('--poly_degree', type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = HybridQLearningTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
