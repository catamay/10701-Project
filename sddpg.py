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
        self.loss_fn = nn.MSELoss()
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
    def __init__(self, n_obs, n_actions, batch_size, gamma, tau, policy_lr, q_lr, weight_decay, d, d0, x0, num_traj, device=torch.device("cpu"), n_hidden=300):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.d = d
        self.d0 = d0
        self.x0 = x0

        # Initialize all memoizers as tensors of 0's
        self.aux = torch.zeros(num_traj, batch_size)
        self.baseline_action =  torch.zeros((num_traj, batch_size,n_actions))
        self.last_gL = torch.zeros((num_traj, batch_size,n_actions))
        

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
        self.actor_optim = optim.AdamW(self.actor_local.parameters(), lr=q_lr, weight_decay=weight_decay)
        
        self.critic_local = Critic(n_obs, n_hidden, n_actions).to(device)
        self.critic_target = Critic(n_obs, n_hidden, n_actions).to(device)
        self.critic_optim = optim.AdamW(self.critic_local.parameters(), lr=q_lr, weight_decay=weight_decay)

        self.const_critic_local = Critic(n_obs, n_hidden, n_actions).to(device)
        self.const_critic_target = Critic(n_obs, n_hidden, n_actions).to(device)
        self.const_critic_optim = optim.AdamW(self.const_critic_local.parameters(), lr=policy_lr, weight_decay=weight_decay)

    def select_action(self, state_obs, trajectory, timestep):
        self.actor_local.eval()
        self.const_critic_local.eval()
        self.actor_local.zero_grad()

        baseline_action = self.baseline_action[trajectory][timestep]
        baseline_gradient = self.last_gL[trajectory][timestep]
        auxillary_constant = self.aux[trajectory][timestep]
        baseline_norm = torch.norm(baseline_gradient,p=2)**2

        if (baseline_norm == 0):
            baseline_norm = 1

        with torch.enable_grad():
            action = self.actor_local(state_obs)
            action.requires_grad_(True)
            action_baseline_diff = (action-baseline_action).squeeze(0)

            v = self.const_critic_local(state_obs, action)
            v.requires_grad_(True)  # Explicitly require gradients for value

            # Compute gradient with respect to action
            next_gL = torch.autograd.grad(
                outputs=v, 
                inputs=action, 
                create_graph=True,  # Allow higher-order gradients
                retain_graph=True   # Retain the computational graph
            )[0].squeeze(0)

            lambda_star = (torch.dot(baseline_gradient, action_baseline_diff) - auxillary_constant)/baseline_norm
            lambda_star = torch.clamp(lambda_star, min=0)

            safe_action = (action - lambda_star*baseline_gradient).detach()
            if torch.isnan(safe_action).any():
                print(f"trajectory: {trajectory}\ntimestep:{timestep}")
                print(f"safe_action: {safe_action}")
                print(f"lambda_star: {lambda_star}")
                print(f"action_baseline_diff: {action_baseline_diff}")
                print(f"baseline_gradient: {baseline_gradient}")
                print(f"auxillary_constant: {auxillary_constant}")
                print(f"baseline_norm: {baseline_norm}\n\n\n")
                exit(0)

        # calculate new baseline gradient eagerly for next iteration
        self.last_gL[trajectory][timestep] = next_gL.detach()


        self.actor_local.train()
        self.const_critic_local.train()
        return safe_action.squeeze(0)
    
    def update_critics(self, trajectories):
        const_costs_over_traj = []
        const_vio_traj = []
        for traj_num, t in enumerate(trajectories):
            states = t[0]
            actions = t[1]
            rewards = t[2] 
            next_states = t[3] 
            dones= t[4].squeeze()
            constraints_violations = 0

            next_actions = self.actor_target(next_states)

            target_safe_q_values = (self.d(states).squeeze() + self.aux[traj_num] + self.gamma * self.const_critic_target(next_states, next_actions).squeeze()).unsqueeze(1)
            target_q_values = rewards + self.gamma * self.critic_target(next_states, next_actions)

            # Filter so only states where trajectory is not finished are considered
            # Bizarre off by one error
            filtered_target_safe_q_values = torch.cat([target_safe_q_values[i] for i in range(len(target_safe_q_values)) if not dones[i-1].item()], dim=0)
            filtered_target_q_values = torch.cat([target_q_values[i] for i in range(len(target_q_values)) if not dones[i-1].item()], dim=0)
            safe_q_values = self.const_critic_local(states, actions)
            q_values = self.critic_local(states, actions)

            constraints_cost = self.d(states).sum().item()
            if constraints_cost > self.d0:
                constraints_violations = math.floor(constraints_cost / self.d0)

            filtered_q_values = torch.cat([q_values[i] for i in range(len(q_values)) if not dones[i-1].item()], dim=0)
            filtered_safe_q_values = torch.cat([safe_q_values[i] for i in range(len(safe_q_values)) if not dones[i-1].item()], dim=0)

            # print(f"filtered_q_values: {filtered_q_values.shape}\nfiltered_target_q_values: {filtered_target_q_values.shape}\n\nfiltered_safe_q_values: {filtered_safe_q_values.shape}\nfiltered_target_safe_q_values: {filtered_target_safe_q_values.shape}")
            critic_loss = self.critic_local.loss_fn(filtered_q_values, filtered_target_q_values)
            const_critic_loss = self.const_critic_local.loss_fn(filtered_safe_q_values, filtered_target_safe_q_values)

            # need to consider only until end of trajectory somehow for loss + reward
            loss = critic_loss + const_critic_loss
    
            loss.backward()


            self.critic_optim.step()
            self.const_critic_optim.step()


            nn.utils.clip_grad_norm_(self.const_critic_local.parameters(), 1)
            nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.const_critic_local, self.const_critic_target)
            const_costs_over_traj.append(constraints_cost)
            const_vio_traj.append(constraints_violations)

        return const_costs_over_traj, const_vio_traj

    def update_agent(self, trajectories):
        actor_losses = []
        predicted_actions_per_trajectory = []
        for t in trajectories:
            states = t[0]
            dones = t[4]
            # Compute actor loss
            self.actor_optim.zero_grad()
            predicted_actions = self.actor_local(states)
            critic_q_values = -self.critic_local(states, predicted_actions)
            filtered_q_values = [critic_q_values[i] for i in range(len(critic_q_values)) if not dones[i-1]]
            actor_loss = torch.cat(filtered_q_values, dim=0).mean()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), max_norm=1.0)

            # Perform optimization step
            self.actor_optim.step()

            # Perform soft updates of target networks
            self.soft_update(self.actor_local, self.actor_target)
            actor_losses.append(actor_loss.item())
            predicted_actions_per_trajectory.append(predicted_actions)



        return actor_losses, predicted_actions_per_trajectory


    def set_auxillary(self, x0, N, T):
        self.x0 = x0
        for traj_num in range(N):
            for timestep in range(T):
                raw_actions = self.select_action(x0, traj_num, timestep).unsqueeze(0)

                self.const_critic_local.eval()
                with torch.no_grad():
                    self.aux[traj_num][timestep] = ((1-self.gamma)*(self.d0-self.const_critic_local(x0, raw_actions)))
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

def generate_trajectories(policy_agent, env, N, T, episode, eps_start, eps_end, eps_decay, device):
    trajectories = []
    seed = torch.randint(1,1000, (1,1))
    env.action_space.seed(seed.item())

    for traj_num in range(N):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for timestep in range(T):
            eps = eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)

            if random.random()<eps:
                action = torch.Tensor(env.action_space.sample())
            else:
                action = policy_agent.select_action(state, traj_num, timestep)

            
            observation, reward, terminated, truncated, _ = env.step(action.numpy(force=True))
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state if next_state is not None else state)
            dones.append(torch.tensor([done], dtype=torch.float32, device=device))

            state = next_state

            if done:
                # Pad trajectory tensors to ensure consistent length
                # Use the last valid state for padding
                last_valid_state = states[-1] if states else torch.zeros_like(state)
                last_valid_action = actions[-1] if actions else torch.zeros_like(action)
                last_valid_reward = rewards[-1] if rewards else torch.zeros_like(reward)
                while len(states) < T:
                    states.append(last_valid_state)
                    actions.append(last_valid_action)
                    rewards.append(last_valid_reward)
                    next_states.append(last_valid_state)
                    dones.append(torch.tensor([True], dtype=torch.float32, device=device))
                    
                    timestep += 1
                break

        # print(f"States: {states}")
        # print(f"Actions: {actions}")
        # print(f"rewards: {rewards}")
        # print(f"dones: {dones}")
        # Convert collected data into tensors
        states = torch.cat(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.stack(dones, dim=0)

        trajectories.append([states, actions, rewards, next_states, dones])
    
    return trajectories, seed

def train(episode, eps_start, eps_end, eps_decay, last_policy, env, N, T, device=torch.device("cpu")):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    x0 = torch.zeros_like(state)
    x0 = x0.copy_(state)
    pi_theta_k = last_policy
    pi_theta_k.set_auxillary(x0, N, T)

    # Step 0: generate N trajectories of length T using the previous policy
    trajectories, seed = generate_trajectories(pi_theta_k, env, N, T, episode, eps_start, eps_end, eps_decay, device)

    # Step 1: Using trajectories estimate critic and constraint critic
    constraint_cost, constraint_violations = pi_theta_k.update_critics(trajectories)

    # Step 2: Update Policy Parameters
    actor_losses, predicted_actions_per_traj = pi_theta_k.update_agent(trajectories)

    # Step 3: Update baseline
    pi_theta_k.baseline_action = predicted_actions_per_traj
    return pi_theta_k, actor_losses, seed, constraint_cost, constraint_violations
