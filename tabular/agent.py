class Agent(object):
	""" 
	Implementation of Base Agent class
	Algorithm: Agent randomly selects action from available action space (default agent)
	"""

	def __init__(self, action_space):
		self.action_space = action_space

	def act(self, state, reward, done):
		# Return chosen action
		return self.action_space.sample()

		