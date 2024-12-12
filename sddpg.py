import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import memory

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

class SafetyLayer(nn.Module):
    def __init__(self, d0, critic, n_actions, gamma):
        super().__init__()

        self.critic = critic 
        self.n_actions = n_actions
        self.actor = None
        self.aux = None
        self.x0 = None
        self.d0 = d0
        self.gamma = gamma
        self.has_data = False

    def updateQA(self, constraint_critic, actor):
        self.critic = constraint_critic
        self.actor = actor
        self.has_data = True
        self.critic.eval()
        self.actor.eval()
        if self.x0 is not None:
            self.set_aux()

    
    def set_x0(self, x0):
        self.x0 = x0

    def set_aux(self):
        self.critic.eval()
        if self.actor is not None:
            self.actor.eval()
            action = self.actor(self.x0)
        else: 
            action = torch.zeros(self.n_actions, device=self.x0.device).unsqueeze(0)

        self.aux = (1-self.gamma)*(self.d0-self.critic(self.x0, action))

    def forward(self, obs, action):
        grad_update = 0
        self.critic.eval()

        if self.has_data:
            self.actor.eval()
            action_prev = self.actor(obs)
        else:
            action_prev = torch.zeros_like(action)
        
        action_prev.requires_grad_()
        Q_val = self.critic(obs, action_prev)
        Q_val = Q_val.requires_grad_()

        Q_val.backward(inputs=action_prev)
        gL = action_prev.grad.detach().squeeze(0)
        action = action.squeeze(0)
        action_prev = action_prev.squeeze(0)

        lambda_star = torch.relu((torch.dot(gL, action - action_prev) - self.aux)/torch.linalg.vector_norm(gL)**2)
        grad_update = lambda_star * gL

        return (action + grad_update).detach()

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

        self.safety_layer = SafetyLayer(d0, copy.deepcopy(self.const_critic_local), self.n_actions, self.gamma).to(device)

    def set_init(self, x0):
        self.safety_layer.set_x0(x0)
        self.safety_layer.set_aux()
    
    def select_action(self, obs):
        self.actor_local.eval()

        action = self.actor_local(obs)
        safe_action = self.safety_layer(obs, action)

        return safe_action.squeeze(0)
            

    def update(self, memory, criterion_critic):
        if len(memory) < self.batch_size:
            return 0 # no loss until we can make a batch

        self.safety_layer.updateQA(copy.deepcopy(self.const_critic_local), copy.deepcopy(self.actor_local))

        self.critic_local.train()
        self.const_critic_local.train()
        self.actor_local.train()

        self.critic_target.train()
        self.const_critic_target.train()
        self.actor_target.train()
        
        self.critic_optim.zero_grad()
        self.const_critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        transitions = memory.sample(self.batch_size)
        batch = memory.Transition(*zip(*transitions))

        # Batch a sample of the memory
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)

        reward_batch = torch.cat(batch.reward)


        next_actions_batch = self.actor_target(next_state_batch)

        target_safe_q_values = self.d(state_batch) + self.safety_layer.aux + self.gamma * self.const_critic_target(next_state_batch, next_actions_batch)
        target_q_values = reward_batch.unsqueeze(1) + self.gamma * self.critic_target(next_state_batch, next_actions_batch)

        safe_q_values = self.const_critic_local(state_batch, action_batch)
        q_values = self.critic_local(state_batch, action_batch)


        critic_loss = criterion_critic(q_values, target_q_values)
        const_critic_loss = criterion_critic(safe_q_values, target_safe_q_values)

        loss = critic_loss

        loss.backward()


        self.critic_optim.step()
        self.const_critic_optim.step()


        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        actor_loss = -self.critic_local(state_batch, self.actor_local(state_batch)).mean()
        actor_loss.backward()
        self.actor_optim.step()


        # Perform soft updates of target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.const_critic_local, self.const_critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
        return actor_loss.item()



    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, name, path='models'):
        if 'models' in path and os.path.isdir('models') is False:
            os.mkdir('models')
        torch.save({'actor_weights': self.actor_local.state_dict(),
                    'critic_weights': self.critic_local.state_dict(),
                    'const_critic_weights': self.const_critic_local.state_dict(),
                    }, f"{path}/{name}.pt")


    def load(self, path):
        model_dict = torch.load(path)
        self.actor_local.load_state_dict(model_dict['actor_weights'])
        self.actor_target.load_state_dict(model_dict['actor_weights'])
        self.critic_local.load_state_dict(model_dict['critic_weights'])
        self.critic_target.load_state_dict(model_dict['critic_weights'])
        self.const_critic_local.load_state_dict(model_dict['const_critic_weights'])
        self.const_critic_target.load_state_dict(model_dict['const_critic_weights'])
        self.safety_layer.updateQA(self.const_critic_local, self.actor_local)


