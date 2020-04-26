import numpy as np
from tabular.agent import Agent


class EpsilonGreedyAgent(Agent):
	""" 
	Implementation of Agent that selects action based on ε-Greedy Method.
	"""

	def __init__(self, action_space, n_actions, epsilon = 0.01, init_bias = 0):
		
		Agent.__init__(self, action_space)
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.init_bias = init_bias # initial bias, provide greater values to encourage exploration at begining
		
		self.Q_actions = [self.init_bias for i in range(self.n_actions)]	# stores action value estimates
		self.N_actions = [0 for i in range(self.n_actions)]	# stores number of time each action taken until current timestep
		
	def explore(self):
		# Exploration (non-greedy): choose action randomly from the available action space
		
		return(self.action_space.sample())

	def exploit(self):
		# Exploitation (greedy): choose action with max estimated action value 
		
		return(np.argmax(self.Q_actions))	

	def act(self, state, reward, done):
		# Return chosen action 

		probability = np.random.rand()
		if (probability < self.epsilon):
			action = self.explore()
		else:
			action = self.exploit()
		
		self.N_actions[action] += 1
		self.Q_actions[action] += (reward - self.Q_actions[action]) / float(self.N_actions[action])  
		return(action)

	def reset_memory(self, epsilon, init_bias):
		# Reset memory & set new epsion & bias values
		
		self.epsilon = epsilon
		self.init_bias = init_bias
		self.Q_actions = [self.init_bias for i in range(self.n_actions)]
		self.N_actions = [0 for i in range(self.n_actions)]
