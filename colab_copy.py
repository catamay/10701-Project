import gymnasium as gym
import recorder as rc

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from itertools import count

# Environment setup
env = gym.make("HalfCheetah-v5", render_mode="rgb_array", max_episode_steps=1000)
recorder = rc.Recorder(env, "cheetah")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Model and memory setup
n_actions = env.action_space.shape[0]
state, info = env.reset()
n_observations = len(state)

model = nn.Sequential(
    nn.Linear(n_observations, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, n_actions)
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, amsgrad=True)
criterion = nn.MSELoss()
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() > eps_threshold:
        with torch.no_grad():
            return model(state).squeeze(0)
    else:
        return torch.tensor(env.action_space.sample(), device=device, dtype=torch.float32)

def optimize_model(optimizer, criterion):
    if len(memory) < BATCH_SIZE:
        return 0 # no loss while we're batching

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).view(BATCH_SIZE, n_actions)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = model(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = criterion(state_action_values, action_batch)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 100)
    optimizer.step()

    return loss.item()

# Define an Agent class
# This is needed since prior we had an inline object 
# {"select_action": select_action, "device": device} hat was used
# to model the agent that led to a problem with the select_action method passing
# The issue: when called "agent.select_action(state)" implicitly passed a self parameter along with
# the required state paramater causing it to throw an error
class Agent:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.steps_done = 0  # Store steps_done as part of the agent's state
    # TODO: We define select_action twice which must be changed in the future
    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.model(state).squeeze(0)
        else:
            return torch.tensor(env.action_space.sample(), device=self.device, dtype=torch.float32)

# Create agent instance
agent = Agent(model, device)

# Training loop
num_episodes = 100
episode_durations = []

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"The current episode is: {i_episode}")
    episode_loss = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)

        state = next_state
        episode_loss += optimize_model(optimizer, criterion)

        if done:
            episode_durations.append(t + 1)
            print(f"Episode loss: {episode_loss}")
            break

print("Training completed. Episode durations:", episode_durations)

# Video rendering using the recorder
recorder.render_video(num_eval_episodes=5, agent=agent)
