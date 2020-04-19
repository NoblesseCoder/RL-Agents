import numpy as np
from tabular.agent import Agent

class EpsilonGreedyAgent(Agent):
	""" 
	Implementation of Epsilon Greedy Agent class
	Algorithm: Agent selects action based on epsilon greedy method
	"""

	def __init__(self, action_space, n_actions, epsilon=0.1):
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.Q_actions = [0 for i in range(n_actions)] # Q(a) value estimates for all actions
		self.N_actions = [0 for i in range(n_actions)] # N(a) for all actions
		Agent.__init__(self, action_space)

	def explore_act(self, state, reward, done):
		# Exploration (non-greedy): choose action randomly from the action space
		action = Agent.act(self,state, reward, done)
		return action

	def exploit_act(self):
		# Exploitation (greedy): choose action with max estimated action value 
		return(np.argmax(self.Q_actions))	

	def act(self, state, reward, done):
		# Return chosen action 
		if(np.random.rand() < self.epsilon):
			action = self.explore_act(state, reward, done)
		else:
			action = self.exploit_act()
		self.N_actions[action] += 1
		self.Q_actions[action] += (reward - self.Q_actions[action])/self.N_actions[action]  
		return action