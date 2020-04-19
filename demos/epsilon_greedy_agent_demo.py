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

	total_rewards = [0]
	average_rewards = [0]
	init_reward = 0
	for episode in range(episodes):
		while True:
			next_state, reward, done, _ = env.step(agent.act(prev_state, reward, done))
			print("Episode:" + str(episode) + " Reward:" + str(reward))
			if done:
				break
		init_reward += reward
		total_rewards.append(init_reward)
		average_rewards.append(init_reward/(episode+1))
	env.close()
	return(average_rewards,total_rewards)


def run(episodes,epsilons):
	average_rewards_for_epsilons = []
	for epsilon in range(len(epsilons)):
		average_rewards = train_epsilon_greedy_agent(episodes,epsilon)[0]
		average_rewards_for_epsilons.append(average_rewards)
	return(np.array(average_rewards_for_epsilons))	



epsilons = [0, 0.01, 0.5, 0.1]
episodes = 1000
num_runs = 10

avg_rewards_for_epsilons_num_runs = []
for i in range(num_runs):
	avg_reward_for_epsilons = run(episodes,epsilons)
	avg_rewards_for_epsilons_num_runs.append(avg_reward_for_epsilons)

avg_rewards_for_epsilons_num_runs = np.array(avg_rewards_for_epsilons_num_runs)
avg_of_avg_rewards_for_epsilons = np.mean(avg_rewards_for_epsilons_num_runs, axis=0)


plt.xlabel("Episodes")
plt.ylabel("Average Reward")

for i in range(len(avg_of_avg_rewards_for_epsilons)):
	plt.plot(avg_of_avg_rewards_for_epsilons[i], label="epsilon = "+str(epsilons[i]))
	plt.legend(loc="upper right")
plt.show()














