import sddpg
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import os
from tqdm import tqdm
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
N_EPISODES = 500
N_STEPS = 200
BATCH_SIZE = 128
MEMORY_SIZE = 1000000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
WEIGHT_DECAY = 0.005
TAU = 1e-3
LR = 1e-4

env = gym.make("HalfCheetah-v5", render_mode="rgb_array", max_episode_steps=N_STEPS)

n_actions = env.action_space.shape[0] 
state, _ = env.reset()

n_obs = len(state)

# clip robot velocity
def c(x: torch.tensor):
    return torch.abs(x[:,8]) # absolute value of x velocity

c0 = 1

# Training loop
agent = sddpg.SDDPG(n_obs, n_actions, BATCH_SIZE, GAMMA, TAU, LR, WEIGHT_DECAY, c, c0, device=device)

criterion = nn.MSELoss()
memory = sddpg.ReplayMemory(MEMORY_SIZE)
rewards = []

for i_episode in (pbar := tqdm(range(N_EPISODES))):
    episode_reward = sddpg.train(i_episode, EPS_START, EPS_END, EPS_DECAY, criterion, agent, memory, env, device=device)
    rewards.append(episode_reward)
    if i_episode>5:
        pbar.set_postfix({
                'average last 5 rewards': round(np.mean(rewards[-6:-1]), 5),
                })

print("Training completed.")
fig = plt.figure()
ax  = fig.subplots(1)
ax.plot(rewards)
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
fig.suptitle("Episode rewards over time")
if os.path.isdir('figures') is False:
            os.mkdir('figures')
fig.savefig(f"figures/rewards.png")

print("Saving model...")
agent.save("humanoid")
print("Saved Model Weights!")

num_eval_episodes=5
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
env = RecordVideo(env, video_folder="video_renders", name_prefix="cheetah", episode_trigger=lambda _: True)

fig = plt.figure()
ax  = fig.subplots(1)
velocities = np.zeros((num_eval_episodes, N_STEPS))
for episode_num in range(num_eval_episodes):
    obs, _ = env.reset()
    done = False

    t=0
    while not done:
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = agent.select_action(state).numpy(force=True).squeeze(0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        velocities[episode_num, t] = abs(obs[8])
        t+=1

T = np.arange(0, N_STEPS)
mean_vol = np.mean(velocities,axis=0)
moving_avg = np.convolve(mean_vol, np.ones(5), 'same')/5
std_vol = np.std(velocities, axis=0)

ax.plot(T,moving_avg, 'b-', label='smoothed velocity mean')
ax.fill_between(T,moving_avg - std_vol, moving_avg + std_vol, color='b', alpha=0.2)

ax.legend()
fig.suptitle(f"Velocity from {num_eval_episodes} eval episodes")
if os.path.isdir('figures') is False:
    os.mkdir('figures')
fig.savefig(f"figures/velocities.png")

env.close()
