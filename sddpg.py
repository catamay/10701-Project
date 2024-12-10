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

# Safe Deep Deterministic Policy Gradient
class SDDPG:
    def __init__(self, n_obs, n_actions, batch_size, gamma, tau, lr, weight_decay, d, d0, x0, device=torch.device("cpu"), n_hidden=300):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.d = d
        self.d0 = d0
        self.aux = 0
        self.x0 = x0
        self.baseline = lambda x: torch.zeros(n_actions)

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

        self.const_critic_local = Critic(n_obs, n_hidden, n_actions).to(device)
        self.const_critic_target = Critic(n_obs, n_hidden, n_actions).to(device)
        self.const_critic_optim = optim.AdamW(self.const_critic_local.parameters(), lr=lr, weight_decay=weight_decay)

    def select_action(self, obs):
        self.actor_local.eval()
        self.const_critic_local.eval()
        self.actor_local.zero_grad()

        action = self.actor_local(obs)

        v = self.const_critic_local(obs, action)
        gL = torch.autograd.grad(outputs=v, inputs=action)[0].squeeze(0).detach()
        action = action.squeeze(0).detach()
        
        lambda_star = (torch.dot(gL, action-self.baseline(obs)) - self.aux)/torch.norm(gL,p=2)**2

        safe_action = action + lambda_star*gL


        self.actor_local.train()
        self.const_critic_local.train()
        return safe_action.squeeze(0)
            

    def update(self, memory, criterion_critic, criterion_constraint):
        if len(memory) < self.batch_size:
            return 0 # no loss until we can make a batch
        
        self.critic_optim.zero_grad()
        self.const_critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Batch a sample of the memory
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)


        next_actions_batch = self.actor_target(next_state_batch)
        target_safe_q_values = self.d(state_batch) + self.aux + self.gamma * self.const_critic_target(next_state_batch, next_actions_batch)
        target_q_values = reward_batch.unsqueeze(1) + self.gamma * self.critic_target(next_state_batch, next_actions_batch)

        safe_q_values = self.const_critic_local(state_batch, action_batch)
        q_values = self.critic_local(state_batch, action_batch)



        const_critic_loss = criterion_constraint(safe_q_values, target_safe_q_values)
        critic_loss = criterion_critic(q_values, target_q_values)

        loss = critic_loss **2 + const_critic_loss **2
        loss.backward()


        self.critic_optim.step()
        self.const_critic_optim.step()


        actor_loss = self.const_critic_local(state_batch, self.actor_local(state_batch)).mean()-self.critic_local(state_batch, self.actor_local(state_batch)).mean()
        actor_loss.backward()
        self.actor_optim.step()

        nn.utils.clip_grad_norm_(self.const_critic_local.parameters(), 1)
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        # Perform soft updates of target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.const_critic_local, self.const_critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
        return actor_loss.item()

    def set_auxillary(self, x0):
        self.x0 = x0

        action = self.select_action(x0).unsqueeze(0)

        self.const_critic_local.eval()
        with torch.no_grad():
            self.aux = (1-self.gamma)*(self.d0-self.const_critic_local(x0, action))
        self.const_critic_local.train()

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

def train(episode, eps_start, eps_end, eps_decay, criterion_critic, criterion_constraint, agent, memory, env, device=torch.device("cpu")):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    x0 = torch.zeros_like(state)
    x0 = state.copy_(x0)
    agent.set_auxillary(x0)
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
            action = agent.select_action(state)

        observation, reward, terminated, truncated, _ = env.step(action.numpy(force=True))
        episode_reward += reward
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        done = terminated or truncated

        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action.unsqueeze(0), next_state, reward)

        state = next_state
        agent.update(memory, criterion_critic, criterion_constraint)

        if done:
            return episode_reward
