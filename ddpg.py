import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import math
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import namedtuple, deque

# Memory class
# Used to create batches on which to train the agent
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Alternatively we could pre-concatenate the obs-action pair
# and make this just a sequential but oh well
class Critic(nn.Module):
    def __init__(self, n_obs, n_hidden, n_actions):
        super().__init__()

        self.input_layer = nn.Linear(n_obs, n_hidden)
        self.batch_norm = nn.BatchNorm1d(n_hidden)
        self.hidden_layer = nn.Linear(n_hidden + n_actions, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, obs, action):
        obs_ff = self.input_layer(obs)
        obs_ff = self.relu(self.batch_norm(obs_ff))
        x = self.hidden_layer(torch.cat([obs_ff, action], 1))
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Deep Deterministic Policy Gradient
class DDPG:
    def __init__(self, n_obs, n_actions, batch_size, gamma, tau, lr, weight_decay, device=torch.device("cpu"), n_hidden=300):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.actor_local = nn.Sequential(
                nn.Linear(n_obs, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions),
                nn.Tanh()
            ).to(device)
        self.actor_target = nn.Sequential(
                nn.Linear(n_obs, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions),
                nn.Tanh()
            ).to(device)
        self.actor_optim = optim.AdamW(self.actor_local.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.critic_local = Critic(n_obs, n_hidden, n_actions).to(device)
        self.critic_target = Critic(n_obs, n_hidden, n_actions).to(device)
        self.critic_optim = optim.AdamW(self.critic_local.parameters(), lr=lr, weight_decay=weight_decay)

    def select_action(self, obs, eps=0):
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(obs).squeeze(0)
        self.actor_local.train()
        return action
            

    def update(self, memory, criterion):
        if len(memory) < self.batch_size:
            return 0 # no loss until we can make a batch

        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Batch a sample of the memory
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        # Use critic to compute q value after actor action
        next_actions_batch = self.actor_target(next_state_batch)
        target_q_values = reward_batch.unsqueeze(1) + self.gamma * self.critic_target(next_state_batch, next_actions_batch)

        # Critic update
        q_values = self.critic_local(state_batch, action_batch)
        critic_loss = criterion(q_values, target_q_values)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        # Actor update
        actor_loss = -self.critic_local(state_batch, self.actor_local(state_batch)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Perform soft updates of target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
        return actor_loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, name, path='models'):
        if 'models' in path and os.path.isdir('models') is False:
            os.mkdir('models')
        torch.save({'actor_weights': self.actor_local.state_dict(),
                    'critic_weights': self.critic_local.state_dict()
                    }, f"{path}/name.pt")

    def load(self, path):
        model_dict = torch.load(path)
        self.actor_local.load_state_dict(model_dict['actor_weights'])
        self.actor_target.load_state_dict(model_dict['actor_weights'])
        self.critic_local.load_state_dict(model_dict['critic_weights'])
        self.critic_target.load_state_dict(model_dict['critic_weights'])

def train(episode, eps_start, eps_end, eps_decay, criterion, agent, memory, env, device=torch.device("cpu")):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_loss = 0
    episode_reward = 0
    while True:
        # Select an action in the current state and
        # add the resulting observations to the
        # memory buffer
        eps = eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)

        if random.random()<eps:
            action = torch.Tensor(env.action_space.sample())
        else:
            action = agent.select_action(state, eps)
        observation, reward, terminated, truncated, _ = env.step(action.numpy(force=True))
        episode_reward += reward
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        done = terminated or truncated

        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action.unsqueeze(0), next_state, reward)

        state = next_state
        agent.update(memory, criterion)

        if done:
            return episode_reward            

# # Algorithm
# env = gym.make("HumanoidStandup-v5", render_mode="rgb_array", max_episode_steps=200)

# n_actions = env.action_space.shape[0] 
# state, _ = env.reset()
# n_obs = len(state)

# # Training loop
# agent = DDPG(n_obs, n_actions)
# criterion = nn.MSELoss()
# memory = ReplayMemory(MEMORY_SIZE)
# losses = []

# progress_bar = tqdm(total=N_EPISODES, position=0, leave=True)
# for i_episode in range(N_EPISODES):
#     progress_bar.update(1)
#     state, info = env.reset()
#     state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#     episode_loss = 0
#     episode_reward = 0
#     while True:
#         # Select an action in the current state and
#         # add the resulting observations to the
#         # memory buffer
#         eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)

#         if random.random()<eps:
#             action = torch.Tensor(env.action_space.sample())
#         else:
#             action = agent.select_action(state, eps)
#         observation, reward, terminated, truncated, _ = env.step(action.numpy(force=True))
#         episode_reward += reward
#         reward = torch.tensor([reward], device=device, dtype=torch.float32)
#         done = terminated or truncated

#         next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
#         memory.push(state, action.unsqueeze(0), next_state, reward)

#         state = next_state
#         agent.update(memory, criterion)

#         if done:
#             losses.append(episode_reward)
#             break
#     if i_episode>0:
#         progress_bar.set_postfix({
#                 'last reward': round(episode_reward, 5),
#                 'best reward': round(min(losses), 5),
#                 })
    
# del progress_bar

# print("Training completed.")
# plt.plot(losses)
# plt.xlabel("Episodes")
# plt.ylabel("Training Reward")
# plt.show()

# print("Saving model...")
# agent.save()
# print("Saved Model Weights!")

# num_eval_episodes=5
# env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
# env = RecordVideo(env, video_folder="video_renders", name_prefix="cheetah", episode_trigger=lambda _: True)

# for episode_num in range(num_eval_episodes):
#     obs, _ = env.reset()
#     done = False

#     while not done:
#         state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
#         action = agent.select_action(state).cpu().numpy()

#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

# env.close()
