import gym
import gym_karmedbandits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../agents')
from tabular.epsilon_greedy_agent import EpsilonGreedyAgent 

sns.set(color_codes=True, style="whitegrid")

# Creating the Multi Armed Bandits gym environment
env = gym.make('KArmedBandits-v0')
env.seed(0)

'''
# Initializing the Learning Agent
agent = EpsilonGreedyAgent(env.action_space)

# Training phase
reward = 0
prev_state = env.reset()
done = False
for _ in range(1000):
	while True:
		next_state, reward, done, _ = env.step(agent.act(prev_state, reward, done))
		print(reward)
		if done:
			break
env.close()
'''