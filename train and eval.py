import sddpg
import ddpg
import memory 

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import ray

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



    # Hyperparameters
N_EPISODES = 200
BATCH_SIZE = 120
ASYNC_SIZE = 12
SYNC_SIZE = BATCH_SIZE//ASYNC_SIZE

MEMORY_SIZE = 1000000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
WEIGHT_DECAY = 0.005
TAU = 1e-3
LR = 1e-4

max_steps = 500
d0 = 50


def d(x: torch.tensor):
    return (x[:,8] <=1).unsqueeze(-1).to(device)

@ray.remote
class Simulator(object):
    def __init__(self):
        self.env = gym.make("HalfCheetah-v5", max_episode_steps=max_steps)
        self.env.reset()
    
    def generate_trajectory(self, batch_size, episode, state, max_steps, eps_start, eps_end, eps_decay, agent, memory, device=torch.device("cpu")):
        t = 0
        state, _ = self.env.reset()
        cur_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        traj_reward=0
        for _ in range(batch_size):
            states = []
            actions = []
            rewards = []
            next_states = []
            while t< max_steps:
                # Select an action in the current state and
                # add the resulting observations to the
                # memory buffer
                eps = eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)

                if random.random()<eps:
                    action = torch.tensor(self.env.action_space.sample()).to(device)

                else:
                    action = agent.select_action(cur_state)

                observation, reward, terminated, truncated, _ = self.env.step(action.numpy(force=True))
                traj_reward += reward
                reward = torch.tensor(reward, device=device, dtype=torch.float32)

                t += 1
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                states.append(cur_state)
                actions.append(action.unsqueeze(0))
                rewards.append(reward.unsqueeze(0))
                next_states.append(next_state)

                cur_state = next_state
            t=0
            memory.push(torch.cat(states), torch.cat(actions), torch.cat(next_states), torch.cat(rewards))

        return traj_reward

def train(episode, eps_start, eps_end, eps_decay, criterion_critic, agent, memory, init_env, sim, device=torch.device("cpu")):
    state, info = init_env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    episode_reward = 0

    if isinstance(agent, sddpg.SDDPG):
        x0 = torch.zeros_like(state)
        x0.copy_(state)
        x0 = x0.unsqueeze(0).to(device)
        agent.set_init(x0)

    result_ids = [sim[i].generate_trajectory.remote(SYNC_SIZE, episode, state, max_steps, eps_start, eps_end, eps_decay, agent, memory, device) for i in range(ASYNC_SIZE)]

    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        episode_reward += ray.get(done_id[0]).item()

    agent.update(memory, criterion_critic)
    
    return episode_reward


def main():
    parser = argparse.ArgumentParser(description='Train or Load.')
    parser.add_argument('--no_train', default=False, action='store_true',
                        help='Don\'t train the model')
    parser.add_argument('--save', dest='save', default='name', help='Designate saved file name, defaults to name')
    parser.add_argument('--safe', dest='safe', default=True, help='Designate model. Defaults to sDDPG')
    parser.add_argument('--path', dest='path',
                        help='Designate path file. You MUST provide a path if you passed in --no_train')

    args = parser.parse_args()


    no_train = args.no_train

    path = args.path

    file_name = args.save

    safe = args.safe


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    ray.init()

    if not no_train:
        init_env = gym.make("HalfCheetah-v5", render_mode='rgb_array', max_episode_steps=max_steps)
        n_actions = init_env.action_space.shape[0] 
        state, _ = init_env.reset()
        n_obs = state.shape[0]
        sim = []
        for i in range(ASYNC_SIZE):
            sim.append(Simulator.remote())
    else:

        parallel_envs = gym.make_vec("HalfCheetah-v5",num_envs=ASYNC_SIZE, vectorization_mode='async',wrappers=(gym.wrappers.NumpyToTorch,))
        env = parallel_envs
        n_actions = env.action_space.shape[1] 
        state, _ = env.reset()

        n_obs = state.shape[1]

    if safe:
        agent = sddpg.SDDPG(n_obs, n_actions, BATCH_SIZE, max_steps, GAMMA, TAU, LR, WEIGHT_DECAY,d, d0, state, device=device)
    else:
        agent = ddpg.DDPG(n_obs, n_actions, BATCH_SIZE, GAMMA, TAU, LR, WEIGHT_DECAY, device=device)



    if not no_train:
        if path is not None:
            agent.load(path)
        criterion_critic = nn.MSELoss()
        buffer = memory.ReplayMemory(MEMORY_SIZE)
        losses = []

        for i_episode in (pbar := tqdm(range(N_EPISODES))):
            episode_reward = train(i_episode, EPS_START, EPS_END, EPS_DECAY, criterion_critic, agent, buffer, init_env, sim, device=device)
            losses.append(episode_reward)
            if i_episode>5:
                pbar.set_postfix({
                        'average last 5 rewards': round(np.mean(losses[-6:-1]), 5),
                        })


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

    # Eval
    env = gym.make("HalfCheetah-v5", render_mode='rgb_array', max_episode_steps=max_steps)
    env = gym.wrappers.NumpyToTorch(env, device=device)
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
            state = obs.unsqueeze(0).float()
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


if __name__ == '__main__':
    main()
