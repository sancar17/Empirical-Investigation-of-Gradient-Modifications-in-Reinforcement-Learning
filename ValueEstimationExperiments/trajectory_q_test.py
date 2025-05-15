import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
from helpers import parse_args

def create_test_env(seed):
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    env.reset(seed=seed)
    return env

def load_model(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if isinstance(checkpoint, dict) and 'weights' in checkpoint:
        agent.weights = checkpoint['weights']
        agent.target_weights = checkpoint['weights'].clone()

        if 'normalizer_mean' in checkpoint:
            agent.feature_extractor.normalizer.mean = checkpoint['normalizer_mean']
            agent.feature_extractor.normalizer.var = checkpoint['normalizer_var']
            agent.feature_extractor.normalizer.count = checkpoint['normalizer_count']
    elif isinstance(checkpoint, torch.Tensor):
        # Raw tensor
        agent.weights = checkpoint
        agent.target_weights = checkpoint.clone()
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    agent.epsilon = 0
    agent.lr = 0

def test_agent(agent, env):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if truncated:
            break

    return total_reward

def main(save_dir, agent_class):
    print(f"Starting testing in directory: {save_dir}")

    args = parse_args()
    args.num_trajectories = 10
    args.trajectory_shift_threshold = 0
    args.max_episodes_per_trajectory = 0

    seeds = [1000 + i for i in range(10)]
    all_rewards = []
    episode_numbers = []

    for n in range(11):
        episode_number = (n * 50000) + 10
        checkpoint_filename = f"model_ep{episode_number}.pt"
        checkpoint_path = os.path.join(save_dir, checkpoint_filename)

        if(n == 10):
            episode_number = (n * 50000) - 10
            checkpoint_filename = f"model_ep{episode_number}.pt"
            checkpoint_path = os.path.join(save_dir, checkpoint_filename)

        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\nTesting model saved at episode {episode_number} (file: {checkpoint_filename})")

        agent = agent_class(8, 4, args)
        load_model(agent, checkpoint_path)

        traj_rewards = []
        for seed in seeds:
            env = create_test_env(seed)
            reward = test_agent(agent, env)
            traj_rewards.append(reward)
            env.close()

        all_rewards.append(traj_rewards)
        if(n == 10):
            episode_numbers.append(episode_number)
        else:
            episode_numbers.append(episode_number-10)

    if not all_rewards:
        print("No rewards collected. Exiting.")
        return

    all_rewards = np.array(all_rewards).T  # Shape becomes (10 seeds, N checkpoints)
    print(episode_numbers)

    # Plotting
    plt.figure(figsize=(12, 6))
    for idx in range(all_rewards.shape[0]):
        plt.plot(episode_numbers, all_rewards[idx], marker='o', label=f'Trajectory {idx}')
    
    mean_rewards = all_rewards.mean(axis=0)
    plt.plot(episode_numbers, mean_rewards, marker='x', linestyle='--', color='black', label='Mean Reward')

    plt.title('Reward Across Trajectories', fontsize=18)
    plt.xlabel('Training Episode', fontsize=16)
    plt.ylabel('Total Reward', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('reward_over_envs_td.png')
    plt.close()

    print("\nTesting completed and plot saved to:", os.path.join('./', 'reward_over_envs.png'))


if __name__ == "__main__":
    from main_from_trajectory_q_learning import TorchLinearTrajectoryQLearning
    save_dir = "/home/sancar/Improving-Policy-Learning-with-Gradient-Stopping/current_q_learning/traj_results/trajectory_q_gd_50k/20250416_234736"
    main(save_dir, TorchLinearTrajectoryQLearning)
