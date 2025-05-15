import torch
import numpy as np
import gymnasium as gym
import os
import json
import wandb
from datetime import datetime
from argparse import Namespace
from helpers import set_seed, plot_learning_curve, save_video, EarlyStoppingMonitor, parse_args_sarsa
import gc
from main import TorchFeatureExtractor

class OnPolicySARSALearner:
    def __init__(self, n_states, n_actions, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.args = args
        self.feature_extractor = TorchFeatureExtractor(n_states, args)
        self.n_features = self.feature_extractor.n_features
        self.weights = torch.randn(n_actions, self.n_features).to(self.device) * 0.1
        self.velocity = torch.zeros_like(self.weights)
        self.lr = args.learning_rate
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon
        self.epsilon_min = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.lr_decay = args.lr_decay
        self.lr_min = args.min_lr
        self.grad_clip = args.grad_clip
        self.momentum = 0.95
        self.repeat_num = args.repeat_num

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

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def get_q_values(self, state):
        features = self.feature_extractor.extract(state)
        return torch.mv(self.weights, features).detach().cpu().numpy()

    def compute_td_error(self, state, action, reward, next_state, next_action, done):
        current_q = self.get_q_values(state)[action]
        next_q = self.get_q_values(next_state)[next_action] if not done else 0.0
        target = reward + self.gamma * next_q
        return target - current_q

    def update_td(self, state, action, reward, next_state, next_action, done):
        current_features = self.feature_extractor.extract(state)
        next_features = self.feature_extractor.extract(next_state)
        q_current = self.weights[action].dot(current_features)
        q_next = self.weights[next_action].dot(next_features) if not done else 0.0
        target = reward + self.gamma * q_next
        td_error = target - q_current.item()

        grad = td_error * current_features
        grad_norm = torch.norm(grad)
        if grad_norm > self.grad_clip:
            grad = grad * (self.grad_clip / grad_norm)

        self.velocity[action] = self.momentum * self.velocity[action] + self.lr * grad
        self.weights[action] += self.velocity[action]

        return td_error

    def update_gd(self, state, action, reward, next_state, next_action, done):
        current_features = self.feature_extractor.extract(state)
        next_features = self.feature_extractor.extract(next_state)
        q_current = self.weights[action].dot(current_features)
        q_next = self.weights[next_action].dot(next_features) if not done else 0.0
        target = reward + self.gamma * q_next
        td_error = target - q_current.item()

        grad = td_error * (current_features - self.gamma * next_features)

        grad_norm = torch.norm(grad)
        if grad_norm > self.grad_clip:
            grad = grad * (self.grad_clip / grad_norm)

        self.velocity[action] = self.momentum * self.velocity[action] + self.lr * grad
        self.weights[action] += self.velocity[action]

        return td_error

    def update_gds_tdm(self, state, action, reward, next_state, next_action, done):
        current_features = self.feature_extractor.extract(state)
        next_features = self.feature_extractor.extract(next_state)
        q_current = self.weights[action].dot(current_features)
        q_next = self.weights[next_action].dot(next_features) if not done else 0.0
        target = reward + self.gamma * q_next
        td_error = target - q_current.item()

        grad_td = td_error * current_features
        grad_gd = td_error * (current_features - self.gamma * next_features)
        grad_gds = torch.sign(grad_gd) * torch.abs(grad_td)

        grad_norm = torch.norm(grad_gds)
        if grad_norm > self.grad_clip:
            grad_gds *= self.grad_clip / grad_norm

        self.velocity[action] = self.momentum * self.velocity[action] + self.lr * grad_gds
        self.weights[action] += self.velocity[action]

        return td_error

def validate_agent(agent, env_name):
    rewards = []
    success_count = 0
    for i in range(20):
        env = gym.make(env_name)
        state, _ = env.reset(seed=1000 + i)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated = agent.repeat_action(env, state, action)
            state = next_state
            total_reward += reward
            if truncated:
                break
        rewards.append(total_reward)
        if total_reward > 200:
            success_count += 1
        env.close()
    return np.mean(rewards), np.std(rewards), success_count / 20

def train_sarsa(args: Namespace):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', args.save_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name or f"sarsa_onpolicy_{timestamp}", config=vars(args))

    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = OnPolicySARSALearner(n_states, n_actions, args)
    early_stopper = EarlyStoppingMonitor(patience=args.early_stopping_patience)

    training_rewards = []
    td_errors = []
    val_rewards = []
    val_success_rates = []

    if(args.method == "hybrid_repeat"):
        update_fn = getattr(agent, f"update_td")
    else:
        update_fn = getattr(agent, f"update_{args.method}")

    for episode in range(args.episodes):
        print(episode)
        if(args.method == "hybrid_repeat" and episode == 4000):
            agent.repeat_num = 1
            
        state, _ = env.reset(seed=args.seed + episode)
        action = agent.select_action(state)
        total_reward = 0
        episode_td_errors = []
        done = False

        while not done:
            next_state, reward, done, truncated = agent.repeat_action(env, state, action)
            next_action = agent.select_action(next_state) if not done else None
            td_error = update_fn(state, action, reward, next_state, next_action, done)
            episode_td_errors.append(abs(td_error))
            total_reward += reward
            state = next_state
            action = next_action
            if truncated:
                break

        agent.lr = max(agent.lr_min, agent.lr * agent.lr_decay)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        training_rewards.append(total_reward)
        td_errors.append(np.mean([float(x) if torch.is_tensor(x) else x for x in episode_td_errors]))

        if (episode + 1) % 50 == 0:
            torch.save(agent.weights, os.path.join(save_dir, f'weights_ep{episode+1}.pt'))

        if (episode + 1) % args.validate_freq == 0:
            mean_val, std_val, success_rate = validate_agent(agent, args.env)
            val_rewards.append(mean_val)
            val_success_rates.append(success_rate)
            if args.use_wandb:
                wandb.log({
                    'validation/reward': mean_val,
                    'validation/std_reward': std_val,
                    'validation/success_rate': success_rate
                })
            if early_stopper.update(mean_val):
                print(f"Early stopping at episode {episode}")
                break

        if args.use_wandb:
            wandb.log({
                'episode': episode,
                'reward': total_reward,
                'epsilon': agent.epsilon,
                'lr': agent.lr,
                'td_error': td_errors[-1]
            })

    torch.save(agent.weights, os.path.join(save_dir, 'final_weights.pt'))
    with open(os.path.join(save_dir, 'training_rewards.json'), 'w') as f:
        json.dump(training_rewards, f)

    plot_learning_curve(training_rewards, os.path.join(save_dir, 'rewards.png'), 'SARSA On-Policy Rewards')
    plot_learning_curve(td_errors, os.path.join(save_dir, 'td_errors.png'), 'SARSA On-Policy TD Errors')
    plot_learning_curve(val_rewards, os.path.join(save_dir, 'val_rewards.png'), 'Validation Rewards')
    plot_learning_curve(val_success_rates, os.path.join(save_dir, 'val_success_rates.png'), 'Validation Success Rate')

def test_agent(agent, env_name, save_dir):
    print("testing")
    test_env = gym.make(env_name, render_mode='rgb_array')
    test_results = []
    test_td_errors = []
    test_success_count = 0

    best_model_path = os.path.join(save_dir, 'best_weights.pt')
    if os.path.exists(best_model_path):
        agent.weights = torch.load(best_model_path)
    else:
        print("Best model not found, falling back to final model.")
        agent.weights = torch.load(os.path.join(save_dir, 'final_weights.pt'))

    for test_ep in range(20):
        state, _ = test_env.reset(seed=2000 + test_ep)
        frames = []
        total_reward = 0
        episode_td_errors = []
        done = False

        while not done:
            frames.append(test_env.render())
            action = agent.select_action(state)
            next_state, reward, done, truncated = agent.repeat_action(test_env, state, action)

            next_action = agent.select_action(next_state) if not done else None
            td_error = agent.compute_td_error(state, action, reward, next_state, next_action, done)
            episode_td_errors.append(abs(td_error))

            total_reward += reward
            state = next_state
            if truncated:
                break

        test_results.append(total_reward)
        if episode_td_errors:
            test_td_errors.append(np.mean(np.abs(episode_td_errors)))
        if total_reward > 200:
            test_success_count += 1

        if frames:
            save_video(frames, os.path.join(save_dir, f'test_episode_{test_ep}.mp4'))

    test_statistics = {
        'mean_reward': float(np.mean(test_results)),
        'std_reward': float(np.std(test_results)),
        'min_reward': float(np.min(test_results)),
        'max_reward': float(np.max(test_results)),
        'success_rate': float(test_success_count / 20),
        'mean_td_error': float(np.mean(test_td_errors)) if test_td_errors else 0,
        'std_td_error': float(np.std(test_td_errors)) if test_td_errors else 0
    }

    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_statistics, f, indent=2)

    if agent.args.use_wandb:
        wandb.log({f'test/{k}': v for k, v in test_statistics.items()})

    print("Test Results:")
    print(f"Mean reward: {test_statistics['mean_reward']:.2f} ± {test_statistics['std_reward']:.2f}")
    print(f"Success rate: {test_statistics['success_rate']:.2%}")
    print(f"Mean TD error: {test_statistics['mean_td_error']:.4f} ± {test_statistics['std_td_error']:.4f}")
    print(f"Reward range: [{test_statistics['min_reward']:.1f}, {test_statistics['max_reward']:.1f}]")
    test_env.close()

def main():
    args = parse_args_sarsa()
    train_sarsa(args)

    # Load and test final model
    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = OnPolicySARSALearner(n_states, n_actions, args)

    latest_dir = sorted(os.listdir(os.path.join('results', args.save_name)))[-1]
    save_dir = os.path.join('results', args.save_name, latest_dir)
    
    test_agent(agent, args.env, save_dir)

if __name__ == "__main__":
    main()
