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
		self.reward_history = []
		self.action_history = []
		Agent.__init__(self, action_space)

	def explore_act(self, state, reward, done):
		# Exploration (non-greedy): choose action randomly from the action space
		action = Agent.act(self,state, reward, done)
		return action

	def _action_val_estimation_SAM(self, action, time_step):
		# Action value estimation via Sample Average Method (SAM)

		numerator = np.sum([self.reward_history[i] for i in range(len(self.reward_history)) if self.action_history[i]==action])
		denominator = np.sum([self.action_history[i]==action for i in range(len(self.action_history))])

		if (denominator == 0):
			return 0	
		return(numerator/denominator) 

	def exploit_act(self, time_step):
		# Exploitation (greedy): choose action with max estimated action value 
		estimated_action_vals = []
		for action in range(self.n_actions):
			estimated_action_vals.append(self._action_val_estimation_SAM(action,time_step))
		action_ind = np.argmax(np.array(estimated_action_vals))
		return(action_ind)	

	def act(self, state, time_step, reward, done):
		# Return chosen action 
		if(np.random.rand() < self.epsilon):
			action = self.explore_act(state, reward, done)
		else:
			action = self.exploit_act(time_step)
		self.reward_history.append(reward)
		self.action_history.append(action)
		#print(self.action_history)
		return action