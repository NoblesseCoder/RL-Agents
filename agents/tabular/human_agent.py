from tabular.agent import Agent

class HumanAgent(Agent):
	""" 
	Implementation of Human Agent class
	Algorithm: Agent selects action based on human input 
	"""

	def __init__(self, action_space):
		Agent.__init__(self, action_space)

	def act(self, state, reward, done):
		# Return chosen action 
		print("Choose action from available actions: " + self.action_space +"\n")
		action = input()
		while action not in self.action_space:
			print("Action doesn't belong to action space. Try again")
			action = input()
		return action