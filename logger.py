import numpy as np
import pandas as pd


# class to help record video

class Trace:


	def __init__(self, nagents, n_episodes, n_iterations):
		self.n_agents = nagents
		self.n_episodes = n_episodes
		self.n_iterations = n_iterations

		# ep + iter+ labels + #agents*(rewards, states, actions)
		self.row_shape = 3 + self.n_agents* 3

		self.columns = ['iteration', 'episode', 'labels', 'labels_seen']
		for i in range(self.n_agents):
			self.columns.append(f'state_{i}')

		for i in range(self.n_agents):
			self.columns.append(f'reward_{i}')

		for i in range(self.n_agents):
			self.columns.append(f'action_{i}')

		for i in range(self.n_agents):
			self.columns.append(f'available_actions_{i}')

		self.df = pd.DataFrame(columns=self.columns)
		print('potato')


	def add_episode(self, iteration, episode, rewards, state, actions, labels, labels_seen, available_actions):

		row = {}

		row[self.columns[0]] = iteration
		row[self.columns[1]] = episode
		row[self.columns[2]] = labels
		
		row[self.columns[3]] = labels_seen

		ind = 4
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

		print('row', row)

		self.df = self.df.append(row, ignore_index=True)

	def get_iteration(self, i):
		return self.df.loc[self.df[self.columns[0]] == i]

	def save(self, file):
		self.df.to_csv(file, index=False)