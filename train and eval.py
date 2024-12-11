import sddpg
import ddpg
import memory 
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import os
import math
import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description='Train or Load.')
parser.add_argument('--no_train', default=False, action='store_true',
                    help='Don\'t train the model')
parser.add_argument('--save', dest='save', default='name', help='Designate saved file name, defaults to name')
parser.add_argument('--safe', dest='safe', default=True, help='Designate model. Defaults to sDDPG')
parser.add_argument('--path', dest='path',
                    help='Designate path file. You MUST provide a path if you passed in --no_train')

args = parser.parse_args()


train = not args.no_train

path = args.path

file_name = args.save

safe = args.safe


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

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

max_steps = 500

env = gym.make("HalfCheetah-v5", render_mode='rgb_array', max_episode_steps=max_steps)

env = gym.wrappers.NumpyToTorch(env, device=device)
n_actions = env.action_space.shape[0] 
state, _ = env.reset()

n_obs = len(state)

def d(x: torch.tensor):
    return (torch.abs(x[:,1])>=np.pi/4).unsqueeze(-1).to(device)

d0 = 50

# Training loop
if safe:
    agent = sddpg.SDDPG(n_obs, n_actions, BATCH_SIZE, GAMMA, TAU, LR, WEIGHT_DECAY,d, d0, state, device=device)
else:
    agent = ddpg.DDPG(n_obs, n_actions, BATCH_SIZE, GAMMA, TAU, LR, WEIGHT_DECAY, device=device)

def train(episode, eps_start, eps_end, eps_decay, criterion_critic, agent, memory, env, device=torch.device("cpu")):
    state, info = env.reset()
    state = state.float().unsqueeze(0)

    if isinstance(agent, sddpg.SDDPG):
        x0 = torch.zeros_like(state)
        x0.copy_(state)
        x0 = x0.to(device)
        agent.set_init(x0)

    episode_loss = 0
    episode_reward = 0
    while True:
        # Select an action in the current state and
        # add the resulting observations to the
        # memory buffer
        eps = eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)

        if random.random()<eps:
            action = torch.tensor(env.action_space.sample()).to(device)

        else:
            action = agent.select_action(state)

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        done = terminated or truncated

        next_state = None if terminated else observation.float().unsqueeze(0)
        memory.push(state, action.unsqueeze(0), next_state, reward)

        state = next_state
        agent.update(memory, criterion_critic)

        if done:
            return episode_reward

if train:
    if path is not None:
        agent.load(path)
    criterion_critic = nn.MSELoss()
    memory = memory.ReplayMemory(MEMORY_SIZE)
    losses = []

    progress_bar = tqdm(total=N_EPISODES, position=0, leave=True)
    for i_episode in range(N_EPISODES):
        progress_bar.update(1)
        episode_reward = train(i_episode, EPS_START, EPS_END, EPS_DECAY, criterion_critic, agent, memory, env, device=device)
        losses.append(episode_reward)
        if i_episode>5:
            progress_bar.set_postfix({
                    'reward moving average': round(np.mean(losses[-6:-1]), 5),
                    })
        
    del progress_bar

    print("Training completed.")
    fig = plt.figure()
    ax  = fig.subplots(1)
    ax.plot(losses)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    fig.suptitle("Training loss over time")
    if os.path.isdir('figures') is False:
                os.mkdir('figures')
    fig.savefig(f"figures/{file_name}_losses.png")
    print("Saving model...")
    agent.save(file_name)
    print("Saved Model Weights!")
else:
    assert path is not None
    agent.load(path)


num_eval_episodes=5
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
env = RecordVideo(env, video_folder="video_renders", name_prefix=f"cheetah_{file_name}", episode_trigger=lambda _: True)
fig = plt.figure()
ax  = fig.subplots(1)
velocities = np.zeros((num_eval_episodes, max_steps))
for episode_num in range(num_eval_episodes):
    obs, _ = env.reset()
    x0 = obs.float().unsqueeze(0)

    done = False
    t=0
    if(isinstance(agent, sddpg.SDDPG)):
        agent.set_init(x0)
    while not done:
        state = obs.unsqueeze(0)
        action = agent.select_action(state)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        velocities[episode_num, t] = torch.abs(obs[1]).item()
        t+=1

T = np.arange(0,max_steps)
mean_vol = np.mean(velocities,axis=0)
moving_avg = np.convolve(mean_vol, np.ones(5), 'same')/5
std_vol = np.std(velocities, axis=0)

ax.plot(T,moving_avg, 'b-', label='smoothed angle mean')
ax.fill_between(T,moving_avg - std_vol, moving_avg + std_vol, color='b', alpha=0.2)

ax.legend()
fig.suptitle(f"Angle of head from {num_eval_episodes} eval episodes")
if os.path.isdir('figures') is False:
    os.mkdir('figures')
fig.savefig(f"figures/{file_name}_angle.png")

env.close()