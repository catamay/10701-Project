import ddpg
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_num_threads(12)
torch.set_num_interop_threads(12)
# Hyperparameters
N_EPISODES = 250
BATCH_SIZE = 128
MEMORY_SIZE = 1000000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
WEIGHT_DECAY = 0.005
TAU = 1e-3
LR = 1e-4

env = gym.make("HumanoidStandup-v5", render_mode="rgb_array", max_episode_steps=200)

n_actions = env.action_space.shape[0] 
state, _ = env.reset()
n_obs = len(state)

# Training loop
agent = ddpg.DDPG(n_obs, n_actions, BATCH_SIZE, GAMMA, TAU, LR, WEIGHT_DECAY)
agent.load("./models/humanoid.pt")
criterion = nn.MSELoss()
memory = ddpg.ReplayMemory(MEMORY_SIZE)
losses = []

progress_bar = tqdm(total=N_EPISODES, position=0, leave=True)
for i_episode in range(N_EPISODES):
    progress_bar.update(1)
    episode_reward = ddpg.train(i_episode, EPS_START, EPS_END, EPS_DECAY, criterion, agent, memory, env)
    losses.append(episode_reward)
    if i_episode>0:
        progress_bar.set_postfix({
                'last reward': round(episode_reward, 5),
                'best reward': round(max(losses), 5),
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
fig.savefig(f"figures/last_fig_long.png")

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
        action = agent.select_action(state).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()