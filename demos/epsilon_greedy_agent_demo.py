import gym
import gym_karmedbandits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../agents')
from tabular.epsilon_greedy_agent import EpsilonGreedyAgent 
from tabular.agent import Agent

sns.set(color_codes=True, style="whitegrid")

# Creating the Multi Armed Bandits gym environment
env = gym.make('KArmedBandits-v0')
env.seed(0)

def train_epsilon_greedy_agent(episodes = 100, epsilon=0.1):
	# Initializing the Learning Agent
	agent = EpsilonGreedyAgent(env.action_space, env.k, epsilon)
	
	# Training phase
	reward = 0
	prev_state = env.reset()
	done = False

	average_rewards = []
	for episode in range(episodes):
		while True:
			next_state, reward, done, _ = env.step(agent.act(prev_state, episode, reward, done))
			print("Episode:" + str(episode) + " Reward:" + str(reward))
			if done:
				break
		average_rewards.append(np.mean(agent.reward_history))
	env.close()
	return(average_rewards)


epsilons = [0, 0.01, 0.5, 0.1]
episodes = 1000
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
for epsilon in range(len(epsilons)):
	average_rewards = train_epsilon_greedy_agent(episodes,epsilon)
	plt.plot(average_rewards[1:], label="epsilon = "+str(epsilons[epsilon]))
plt.legend(loc="upper right")
plt.show()















