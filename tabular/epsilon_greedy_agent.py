import numpy as np
from agent import Agent

class EpsilonGreedyAgent(Agent):
	""" 
	Implementation of Epsilon Greedy Agent class
	Algorithm: Agent selects action based on epsilon greedy method
	"""

	def __init__(self, action_space, epsilon=0.01):
		Agent.__init__(action_space)
		self.epsilon = epsilon
		self.action_history = []
		self.rewards_history = []

	def is_exploit(self):
		# Return 1 if exploit, else return 0
		choices = [0,1]
		probs = [self.epsilon, 1-self.epsilon]
		return(np.random.choice(choices, 1, p=probs))
	
	def explore_act(self):
		# Exploration (non-greedy): choose action randomly from the action space
		action = Agent.act()
		return action

	def action_val_estimation_SAM(self, action, time_step):
		# Action value estimation via Sample Average Method (SAM)
		numerator, denominator = 0, 0
		for t in range(time_step):
			predicate = (action == self.ac)
			numerator += rewards_history[t] * predicate
			denominator += predicate
		if (denominator == 0):
			return 0
		q_value = numerator/denominator	
		return q_value 

	def exploit_act(self, time_step):
		# Exploitation (greedy): choose action with max estimated action value 
		estimated_action_vals = []
		for action in self.action_space:
			estimated_action_vals.append(self.estimate_action_val(action,time_step))
		action_ind = np.argmax(np.array(estimated_action_vals))
		return(self.action_space[action_ind])	

	def act(self, time_step, reward, done):
		# Return chosen action 
		action_type = self.is_exploit()
		if (action_type == 0):
			action = self.explore_act()
		if (action_type == 1):
			action = self.exploit_act(time_step)
		self.action_history.append(action)
		self.rewards_history.append(reward)
		return action