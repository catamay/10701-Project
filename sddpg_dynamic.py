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

class SafetyLayer():
    def __init__(self, n_obs, n_hidden, n_actions, constraint_fn, constraint_bound, lr, weight_decay, device=torch.device("cpu")):
        super().__init__()

        self.constraint_fn = constraint_fn
        self.constraint_bound = constraint_bound

        self.constraint_gs = nn.Sequential(
                nn.Linear(n_obs, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions)
            ).to(device)
        self.optim = optim.AdamW(self.constraint_gs.parameters(), lr=lr, weight_decay=weight_decay)

    def __call__(self, obs, action):
        cs = self.constraint_fn(obs).unsqueeze(-1)
        gs = self.constraint_gs(obs)
        csprime = (gs * action).sum(dim=-1, keepdim=True) + cs

        l = nn.ReLU()((csprime - self.constraint_bound)/(gs * gs).sum(dim=-1, keepdim=True))
        return action - l*gs

    def update(self, state_batch, action_batch, next_state_batch, criterion):
        cs = self.constraint_fn(state_batch).unsqueeze(-1)
        gs = self.constraint_gs(state_batch)
        csprime_target = (gs * action_batch).sum(dim=-1, keepdim=True) + cs
        csprime = self.constraint_fn(next_state_batch).unsqueeze(-1)

        loss = criterion(csprime, csprime_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

# Safe Deep Deterministic Policy Gradient
class SDDPG:
    def __init__(self, n_obs, n_actions, batch_size, gamma, tau, lr, weight_decay, constraint_fn, constraint_bound, device=torch.device("cpu"), n_hidden=300):
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

        self.constraint_fn = constraint_fn
        self.constraint_bound = constraint_bound
        self.safety_layer = SafetyLayer(n_obs, n_hidden, n_actions, constraint_fn, constraint_bound, lr, weight_decay, device=device)

    def select_action(self, obs):
        with torch.no_grad():
            self.actor_local.eval()
            action = self.actor_local(obs)
            safe_action = self.safety_layer(obs, action)

        self.actor_local.train()
        return safe_action
            

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

        next_actions_batch = self.actor_target(next_state_batch)
        next_actions_batch = self.safety_layer(next_state_batch, next_actions_batch)

        target_q_values = reward_batch.unsqueeze(1) + self.gamma * self.critic_target(next_state_batch, next_actions_batch)
        q_values = self.critic_local(state_batch, action_batch)

        critic_loss = criterion(q_values, target_q_values)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        safe_action_batch = self.actor_local(state_batch)
        safe_action_batch = self.safety_layer(state_batch, safe_action_batch)
        actor_loss = -self.critic_local(state_batch, safe_action_batch).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Perform soft updates of target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        # Update safety layer
        self.safety_layer.update(state_batch, action_batch, next_state_batch, criterion)
        
        return actor_loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, name, path='models'):
        if 'models' in path and os.path.isdir('models') is False:
            os.mkdir('models')
        torch.save({'actor_weights': self.actor_local.state_dict(),
                    'critic_weights': self.critic_local.state_dict(),
                    'safety_weights': self.safety_layer.constraint_gs.state_dict()
                    }, f"{path}/model.pt")


    def load(self, path):
        model_dict = torch.load(path)
        self.actor_local.load_state_dict(model_dict['actor_weights'])
        self.actor_target.load_state_dict(model_dict['actor_weights'])
        self.critic_local.load_state_dict(model_dict['critic_weights'])
        self.critic_target.load_state_dict(model_dict['critic_weights'])
        self.safety_layer.constraint_gs.load_state_dict(model_dict['safety_weights'])

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
            action = torch.tensor(env.action_space.sample(), device=device)
        else:
            action = agent.select_action(state).squeeze(0)

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
