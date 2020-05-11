import numpy as np
from tabular.agent import Agent

class UpperConfidenceBoundAgent(Agent):
	""" 
	Implementation of Upper Confidence Bound Agent class
	Algorithm: Agent selects action based on Upper Confidence Bound (UCB) method
	"""

	def __init__(self, action_space, n_actions, init_bias=0, confidence = 2):
		
		Agent.__init__(self, action_space)
		self.n_actions = n_actions
		self.init_bias = init_bias # initial bias, provide greater values to encourage exploration at begining
		self.confidence = confidence # confidence > 0, controls the degree of exploration
		self.Q_actions = [self.init_bias for i in range(self.n_actions)]	# stores action value estimates
		self.N_actions = [0 for i in range(self.n_actions)]	# stores number of time each action taken until current timestep


	def get_uncertaininty(self,timestep, action):
		#	Returns uncertainity term to be used for action selection 
		uncertainity_term = self.confidence * (np.sqrt(np.log(timestep)/self.N_actions[action])) 
		return(uncertainity_term)

	def exploit(self):
		# UCB Method(greedy): choose action with max estimated action value
		timestep = len(self.N_actions)
		values = np.array([self.Q_actions[i]+self.get_uncertaininty(timestep,i) for i in range(len(self.Q_actions))])
		return(np.argmax(values))

	def act(self, state, reward, done):
		# Return chosen action 
		action = self.exploit()
		self.N_actions[action] += 1
		self.Q_actions[action] += (reward - self.Q_actions[action]) / float(self.N_actions[action])  
		return(action)

	def reset_memory(self, confidence, init_bias):
		# Reset memory & set confidence level & bias values
		self.confidence = confidence
		self.init_bias = init_bias
		self.Q_actions = [self.init_bias for i in range(self.n_actions)]
		self.N_actions = [0 for i in range(self.n_actions)]
	


