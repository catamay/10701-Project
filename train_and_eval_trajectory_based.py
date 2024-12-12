import sddpg_trajectory_based as sddpg
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cpu")
torch.set_num_threads(12)
torch.set_num_interop_threads(12)
# Hyperparameters
N_EPISODES = 250
# default timesteps in an episode
BATCH_SIZE = 128
MEMORY_SIZE = 1000000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
WEIGHT_DECAY = 0.005
TAU = 1e-3
Q_LR = 1e-4
POLICY_LR = 3e-5
NUM_TRAJ = 5

env = gym.make("HalfCheetah-v5", render_mode="rgb_array", max_episode_steps=100)

n_actions = env.action_space.shape[0] 
x0, _ = env.reset()

n_obs = len(x0)

def d(x: torch.tensor):
    return (torch.linalg.vector_norm(x[:,8:10],dim=1)>=1).unsqueeze(-1)

d0 = 50

# Training loop
agent = sddpg.SDDPG(n_obs, n_actions, BATCH_SIZE, GAMMA, TAU, POLICY_LR, Q_LR, WEIGHT_DECAY,d, d0, x0, NUM_TRAJ)

memory = sddpg.ReplayMemory(MEMORY_SIZE)
losses = []
constraint_violation_costs = []
constraint_violation_counts = []

for i_episode in (pbar := tqdm(range(N_EPISODES))):
    print(f"Current Episode: {i_episode}")
    if i_episode == 0:
        last_policy = agent
    last_policy, actor_losses, seed, constraint_costs, constraint_violations = sddpg.train(i_episode, EPS_START, EPS_END, EPS_DECAY, last_policy, env, NUM_TRAJ, BATCH_SIZE)
    # need to consider only until done = True
    avg_losses = np.mean([np.mean(losses) for losses in actor_losses])
    avg_costs = np.mean(constraint_costs)
    avg_violations = np.mean(constraint_violations)
    print(f"Average Actor Loss: {avg_losses}")
    print(f"Average Constraints Cost: {avg_costs}")
    print(f"Average Constraints Violated: {avg_violations}")
    losses.append(avg_losses)
    constraint_violation_costs.append(avg_costs)
    constraint_violation_counts.append(avg_violations)

# Plot 1: Training Loss Over Time
print("Training completed.")
fig = plt.figure()
ax = fig.subplots(1)
ax.plot(losses, label="Losses")

# Add trendline
x = np.arange(len(losses))
trendline = np.poly1d(np.polyfit(x, losses, 1))  # Linear trendline (degree 1)
ax.plot(x, trendline(x), linestyle="--", color="red", label="Trendline")

ax.set_xlabel("Episode")
ax.set_ylabel("Loss")
fig.suptitle("Training Loss Over Time")
ax.legend()

if os.path.isdir('figures') is False:
    os.mkdir('figures')
fig.savefig("figures/last_fig_long.png")

# Plot 2: Constraint Violations Over Time
fig = plt.figure()
ax = fig.subplots(1)
ax.plot(constraint_violation_counts, label="Violations")

# Add trendline
x = np.arange(len(constraint_violation_counts))
trendline = np.poly1d(np.polyfit(x, constraint_violation_counts, 1))
ax.plot(x, trendline(x), linestyle="--", color="red", label="Trendline")

ax.set_xlabel("Episode")
ax.set_ylabel("Number of Constraint Violations")
fig.suptitle("Constraint Violations Over Time")
ax.legend()

if os.path.isdir('figures') is False:
    os.mkdir('figures')
fig.savefig("figures/constraints_violated.png")

# Plot 3: Cost of Constraint Violations Over Time
fig = plt.figure()
ax = fig.subplots(1)
ax.plot(constraint_violation_costs, label="Costs")

# Add trendline
x = np.arange(len(constraint_violation_costs))
trendline = np.poly1d(np.polyfit(x, constraint_violation_costs, 1))
ax.plot(x, trendline(x), linestyle="--", color="red", label="Trendline")

# Add horizontal line at y=50
ax.axhline(y=50, color='green', linestyle=':', label='50 Threshold')

ax.set_xlabel("Episode")
ax.set_ylabel("Average Cost of Constraint Violation")
fig.suptitle("Cost of Constraint Violations Over Time")
ax.legend()

if os.path.isdir('figures') is False:
    os.mkdir('figures')
fig.savefig("figures/constraints_cost.png")


print("Saving model...")
agent.save("humanoid")
print("Saved Model Weights!")

num_eval_episodes=5
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
env = RecordVideo(env, video_folder="video_renders", name_prefix="cheetah", episode_trigger=lambda _: True)

for episode_num in range(num_eval_episodes):
    env.action_space.seed(seed.item())
    obs, _ = env.reset()
    done = False
    constraint_satisfied = []
    rewards = []
    timestep = 0

    while not done:
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = agent.select_action(state, 0, timestep).numpy(force=True)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        timestep += 1
    print(np.sum(rewards))

env.close()
