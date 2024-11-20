import gymnasium as gym
import recorder as rc

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
N_EPISODES = 100
BATCH_SIZE = 64
MEMORY_SIZE = 1000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

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
        self.hidden_layer = nn.Linear(n_hidden + n_actions, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, obs, action):
        obs_ff = self.input_layer(obs)
        obs_ff = self.relu(obs_ff)
        x = self.hidden_layer(torch.cat([obs_ff, action], 1))
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Deep Deterministic Policy Gradient
class DDPG:
    def __init__(self, n_obs, n_actions, n_hidden=30):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.is_training = True
        self.action_space = gym.make("HalfCheetah-v5").action_space

        self.actor = nn.Sequential(
                nn.Linear(n_obs, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions),
                nn.Tanh()
            ).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR)
        
        self.critic = Critic(n_obs, n_hidden, n_actions).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR)

    def select_action(self, obs, eps):
        if random.random() < eps and self.is_training:
            return torch.tensor(self.action_space.sample(), device=device, dtype=torch.float32)
        else:
            with torch.no_grad():
                return self.actor(obs).squeeze(0)

    def update(self, memory, criterion):
        if len(memory) < BATCH_SIZE:
            return 0 # no loss until we can make a batch

        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Batch a sample of the memory
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        # Use critic to compute q value after actor action
        next_q_values = reward_batch.unsqueeze(1) + GAMMA * self.critic(next_state_batch, self.actor(next_state_batch))

        # Critic update
        self.critic.zero_grad()
        q_values = self.critic(state_batch, action_batch)
        critic_loss = criterion(q_values.float(), next_q_values.float())
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        actor_loss.backward()
        self.actor_optim.step()

        return critic_loss.item()
                

# Algorithm
env = gym.make("HalfCheetah-v5", render_mode="rgb_array", max_episode_steps=1000)
recorder = rc.Recorder(env, "cheetah")

n_actions = env.action_space.shape[0] 
state, _ = env.reset()
n_obs = len(state)

# Training loop
agent = DDPG(n_obs, n_actions)
criterion = nn.MSELoss()
memory = ReplayMemory(MEMORY_SIZE)
losses = []

for i_episode in range(N_EPISODES):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"The current episode is: {i_episode}")
    episode_loss = 0
    episode_reward = 0
    for t in count():
        # Select an action in the current state and
        # add the resulting observations to the
        # memory buffer
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)
        action = agent.select_action(state, eps)
        observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action.unsqueeze(0), next_state, reward)

        state = next_state
        episode_loss += agent.update(memory, criterion)

        if done:
            print(f"Episode stats: loss = {episode_loss}")
            losses.append(episode_loss)
            break

print("Training completed.")
plt.plot(losses)
plt.show()

# Video rendering using the recorder
recorder.render_video(num_eval_episodes=5, agent=agent)
