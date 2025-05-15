import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from main_from_trajectory_dqn import DQNTrajectoryTD
from helpers import parse_args

def create_test_env(seed):
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    env.reset(seed=seed)
    return env

def load_model(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if hasattr(agent, 'q_network'):
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
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
    print(f"Starting DQN testing in directory: {save_dir}")

    args = parse_args()
    args.num_trajectories = 10
    args.trajectory_shift_threshold = 0
    args.max_episodes_per_trajectory = 0

    seeds = [1000 + i for i in range(10)]
    all_rewards = []
    episode_numbers = []

    # === Evaluate random-initialized agent (Episode 0) ===
    print("\nEvaluating initial random weights (episode 0)")
    agent = agent_class(8, 4, args)
    agent.epsilon = 0
    agent.lr = 0

    initial_rewards = []
    for seed in seeds:
        env = create_test_env(seed)
        reward = test_agent(agent, env)
        initial_rewards.append(reward)
        env.close()

    all_rewards.append(initial_rewards)
    episode_numbers.append(0)

    # ===  evaluate checkpoints ===
    for n in range(10):
        checkpoint_filename = f"model_trajectory_{n}.pt"
        checkpoint_path = os.path.join(save_dir, checkpoint_filename)

        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\nTesting model saved at trajectory {n} (file: {checkpoint_filename})")

        agent = agent_class(8, 4, args)
        load_model(agent, checkpoint_path)

        traj_rewards = []
        for seed in seeds:
            env = create_test_env(seed)
            reward = test_agent(agent, env)
            traj_rewards.append(reward)
            env.close()

        all_rewards.append(traj_rewards)
        episode_numbers.append(50000 * (n + 1))  # Episode count aligned with training progress

    if not all_rewards:
        print("No rewards collected. Exiting.")
        return

    all_rewards = np.array(all_rewards).T  # Shape: (10 seeds, N checkpoints + 1)

    # === Plotting ===
    plt.figure(figsize=(12, 6))
    for idx in range(all_rewards.shape[0]):
        plt.plot(episode_numbers, all_rewards[idx], marker='o', label=f'Seed {idx}')

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
    plt.savefig(os.path.join('./', 'reward_over_envs_dqn_gds.png'))
    plt.close()

    print("\nTesting completed and plot saved to:", 'reward_over_envs_dqn.png')


if __name__ == "__main__":
    save_dir = "/home/sancar/Improving-Policy-Learning-with-Gradient-Stopping/current_q_learning/traj_results/trajectory_dqn_gds-tdm_50k/20250425_143535"
    main(save_dir, DQNTrajectoryTD)
