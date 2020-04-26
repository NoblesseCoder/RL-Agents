import numpy as np
from tabular.epsilon_greedy_agent import EpsilonGreedyAgent

class UpperConfidenceBoundAgent(EpsilonGreedyAgent):
	""" 
	Implementation of Upper Confidence Bound Agent class
	Algorithm: Agent selects action based on Upper Confidence Bound (UCB) method
	"""

	def __init__(self, action_space, n_actions, epsilon=0.1, init_bias=0, confidence = 3):
		self.confidence = confidence
		EpsilonGreedyAgent.__init__(self, action_space,n_actions, epsilon, init_bias)


	def get_uncertaininty(self,timestep, action):
		#	Returns uncertainity term to be used for action selection 
		uncertainity_term = self.confidence * (np.sqrt(np.log(timestep)/self.N_actions[action])) 
		return(uncertainity_term)

	def exploit(self):
		# UCB Method(greedy): choose action with max estimated action value
		timestep = len(self.N_actions)
		values = np.array([self.Q_actions[i]+self.get_uncertaininty(timestep,i) for i in range(len(self.Q_actions))])
		return(np.argmax(values))	


