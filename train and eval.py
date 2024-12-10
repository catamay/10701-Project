import sddpg
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import math
import os
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cpu")
# Hyperparameters
N_EPISODES = 200
BATCH_SIZE = 128
MEMORY_SIZE = 1000000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
WEIGHT_DECAY = 0.005
TAU = 1e-3
LR = 1e-4

env = gym.make("HalfCheetah-v5", render_mode="rgb_array", max_episode_steps=200)

n_actions = env.action_space.shape[0] 
state, _ = env.reset()

n_obs = len(state)

def d(x: torch.tensor):
    v = x[:,8:10]
    v = torch.norm(v,dim=1).unsqueeze(-1)

    return torch.relu(v-1)

d0 = 50

# Training loop
agent = sddpg.SDDPG(n_obs, n_actions, BATCH_SIZE, GAMMA, TAU, LR, WEIGHT_DECAY,d, d0, state)
criterion_critic = nn.MSELoss()
criterion_constraint = nn.MSELoss()
memory = sddpg.ReplayMemory(MEMORY_SIZE)
losses = []

progress_bar = tqdm(total=N_EPISODES, position=0, leave=True)
for i_episode in range(N_EPISODES):
    progress_bar.update(1)
    episode_reward = sddpg.train(i_episode, EPS_START, EPS_END, EPS_DECAY, criterion_critic, criterion_constraint, agent, memory, env)
    losses.append(episode_reward)
    if i_episode>5:
        progress_bar.set_postfix({
                'average last 5 rewards': round(np.mean(losses[-6:-1]), 5),
                })
    
del progress_bar

print("Training completed.")
plt.plot(losses)
plt.xlabel("Episodes")
plt.ylabel("Training Reward")
plt.show()

print("Saving model...")
agent.save("humanoid")
print("Saved Model Weights!")

num_eval_episodes=5
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
env = RecordVideo(env, video_folder="video_renders", name_prefix="cheetah", episode_trigger=lambda _: True)

for episode_num in range(num_eval_episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = agent.select_action(state).numpy(force=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()