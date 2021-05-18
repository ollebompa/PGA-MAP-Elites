'''
Copyright (c) 2020 Scott Fujimoto
Based on Twin Delayed Deep Deterministic Policy Gradients (TD3)
Implementation by Scott Fujimoto https://github.com/sfujim/TD3 Paper: https://arxiv.org/abs/1802.09477
Modified by Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Process, Queue, Event
from joblib import Parallel, delayed

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, neurons_list=[128, 128], normalise=False, affine=False, init=False):
		super(Actor, self).__init__()
		self.weight_init_fn = nn.init.xavier_uniform_
		self.num_layers = len(neurons_list)
		self.normalise = normalise
		self.affine = affine
		
		if self.num_layers == 1:

			self.l1 = nn.Linear(state_dim, neurons_list[0], bias=(not self.affine))
			self.l2 = nn.Linear(neurons_list[0], action_dim, bias=(not self.affine))

			if self.normalise:
				self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=self.affine)
				self.n2 = nn.LayerNorm(action_dim, elementwise_affine=self.affine)

		if self.num_layers == 2:
			self.l1 = nn.Linear(state_dim, neurons_list[0], bias=(not self.affine))
			self.l2 = nn.Linear(neurons_list[0], neurons_list[1], bias=(not self.affine))
			self.l3 = nn.Linear(neurons_list[1], action_dim, bias=(not self.affine))

			if self.normalise:
				self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=self.affine)
				self.n2 = nn.LayerNorm(neurons_list[1], elementwise_affine=self.affine)
				self.n3 = nn.LayerNorm(action_dim, elementwise_affine=self.affine)

		self.max_action = max_action

		for param in self.parameters():
			param.requires_grad = False

		self.apply(self.init_weights)

		
		self.type = None
		self.id = None
		self.parent_1_id = None
		self.parent_2_id = None
		self.novel = None
		self.delta_f = None
	

	def forward(self, state):
		if self.num_layers == 1:
			if self.normalise:
				a = F.relu(self.n1(self.l1(state)))
				return self.max_action * torch.tanh(self.n2(self.l2(a)))
			else:
				a = F.relu(self.l1(state))
				return self.max_action * torch.tanh(self.l2(a))
		if self.num_layers == 2:
			if self.normalise:
				a = F.relu(self.n1(self.l1(state)))
				a = F.relu(self.n2(self.l2(a)))
				return self.max_action * torch.tanh(self.n3(self.l3(a)))

			else:
				a = F.relu(self.l1(state))
				a = F.relu(self.l2(a))
				return self.max_action * torch.tanh(self.l3(a))


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self(state).cpu().data.numpy().flatten()


	def save(self, filename):
		torch.save(self.state_dict(), filename)


	def load(self, filename):
		self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

	
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			self.weight_init_fn(m.weight)
		if isinstance(m, nn.LayerNorm):
			pass


	def disable_grad(self):
		for param in self.parameters():
			param.requires_grad = False

	
	def enable_grad(self):
		for param in self.parameters():
			param.requires_grad = True


	def return_copy(self):
		return copy.deepcopy(self)

	


class CriticNetwork(nn.Module):

	def __init__(self, state_dim, action_dim):
		super(CriticNetwork, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

	
	def save(self, filename):
		torch.save(self.state_dict(), filename)



class Critic(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.critic = CriticNetwork(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.total_it = 0
		self.actors_set = set()
		self.actors = []
		self.actor_targets = []
		self.actor_optimisers = []

	

	def train(self, archive, replay_buffer, nr_of_steps, batch_size=256):
		# check if found new species
		diff = set(archive.keys()) - self.actors_set
		for desc in diff:
			# add new species to the critic training pool
			self.actors_set.add(desc)
			new_actor = archive[desc].x
			a = copy.deepcopy(new_actor)
			for param in a.parameters():
				param.requires_grad = True
			a.parent_1_id = new_actor.id
			a.parent_2_id = None
			a.type = "critic_training"
			target = copy.deepcopy(a)
			optimizer = torch.optim.Adam(a.parameters(), lr=3e-4)
			self.actors.append(a)
			self.actor_targets.append(target)
			self.actor_optimisers.append(optimizer)

		for _ in  range(nr_of_steps):
			self.total_it += 1
			# Sample replay buffer 
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			all_target_Q = torch.zeros(batch_size, len(self.actors))
			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				
				for idx, actor in enumerate(self.actors):
					next_action = (
						self.actor_targets[idx](next_state) + noise
					).clamp(-self.max_action, self.max_action)
					# Compute the target Q value
					target_Q1, target_Q2 = self.critic_target(next_state, next_action)
					target_Q = torch.min(target_Q1, target_Q2)
					all_target_Q[:,idx] = target_Q.squeeze()
				# print(all_target_Q)
				target_Q = torch.max(all_target_Q, dim=1, keepdim=True)[0]
				target_Q = reward + not_done * self.discount * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if self.total_it % self.policy_freq == 0:
				for idx, actor in enumerate(self.actors):
					# Compute actor loss
					actor_loss = -self.critic.Q1(state, actor(state)).mean()
					
					# Optimize the actor 
					self.actor_optimisers[idx].zero_grad()
				
					actor_loss.backward()
					self.actor_optimisers[idx].step()
					
					for param, target_param in zip(actor.parameters(), self.actor_targets[idx].parameters()):
						target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return critic_loss



	def save(self, filename):
		torch.save(self.critic.state_dict(), filename)
		torch.save(self.critic_optimizer.state_dict(), filename + "_optimizer")
		

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
		self.critic_target = copy.deepcopy(self.critic)
