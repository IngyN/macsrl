import numpy as np
import pandas as pd

from copy import deepcopy

# class to help record video

class Trace:


	def __init__(self, nagents, n_episodes, n_steps):
		self.n_agents = nagents
		self.n_episodes = n_episodes
		self.n_steps = n_steps

		# ep + iter+ labels + #agents*(rewards, states, actions)
		self.row_shape = 3 + self.n_agents* 3

		self.columns = [ 'iteration', 'episode', 'step', 'labels', 'labels_seen']
		for i in range(self.n_agents):
			self.columns.append(f'state_{i}')

		for i in range(self.n_agents):
			self.columns.append(f'reward_{i}')

		for i in range(self.n_agents):
			self.columns.append(f'action_{i}')

		for i in range(self.n_agents):
			self.columns.append(f'available_actions_{i}')

		self.df = pd.DataFrame(columns=self.columns)


	def add_episode(self, step, episode, it, rewards, state, actions, labels, labels_seen, available_actions):

		row = {}

		row[self.columns[0]] = it
		row[self.columns[1]] = episode
		row[self.columns[2]] = step
		row[self.columns[3]] = labels
		
		row[self.columns[4]] = labels_seen

		ind = 5
		for i in range(self.n_agents):
			row[self.columns[ind]] = state[i]
			ind+=1 

		for i in range(self.n_agents):
			row[self.columns[ind]] = rewards[i]
			ind +=1

		for i in range(self.n_agents):
			row[self.columns[ind]] = actions[i]
			ind +=1 

		for i in range(self.n_agents):
			row[self.columns[ind]] = available_actions[i]
			ind +=1

		# print('row', row)

		self.df = self.df.append(deepcopy(row), ignore_index=True)

	def get_episode(self, i):
		return self.df.loc[self.df[self.columns[1]] == i]

	def get_iteration(self, i):
		return self.df.loc[self.df[self.columns[0]] == i]

	def save(self, file):
		self.df.to_csv(file, index=False)

	def load(self, file):
		self.df = pd.read_csv(file) 