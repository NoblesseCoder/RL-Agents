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


def train_epsilon_greedy_agent(env, epsilon, init_bias, episodes):
	''' Function trains the ε-Greedy agent on the environment'''

	# Instantiating the epsilon greedy agent
	agent = EpsilonGreedyAgent(env.action_space, env.k, epsilon, init_bias)
	reward = 0
	prev_state = env.reset()
	done = False
	total_rewards = []
	average_rewards = []
	cumilative_reward = 0
	
	for episode in range(episodes):
		while True:
			next_state, reward, done, _ = env.step(agent.act(prev_state, reward, done))
			print("Episode:" + str(episode) + " Reward:" + str(reward))
			if done:
				break

		cumilative_reward += reward
		total_rewards.append(cumilative_reward)
		average_rewards.append(cumilative_reward / (episode+1))
	env.close()

	return(np.array(average_rewards), np.array(total_rewards))


def run(env, init_bias, episodes, epsilons):
	''' Runs training for multiple epsilon values'''

	average_rewards_for_respective_epsilons = []
	total_rewards_for_respective_epsilons = []

	for epsilon in range(len(epsilons)):
		rewards = train_epsilon_greedy_agent(env, epsilons[epsilon], init_bias, episodes)
		average_rewards_for_respective_epsilons.append(rewards[0]) 
		total_rewards_for_respective_epsilons.append(rewards[1])

	return(np.array(average_rewards_for_respective_epsilons), np.array(total_rewards_for_respective_epsilons))	



# Creating the Multi Armed Bandits gym environment
env = gym.make('KArmedBandits-v0')
env.seed(0)


epsilons = [0, 0.01, 0.5, 0.1]
init_bias = 0
episodes = 1000
num_runs = 2000

# Running multiple runs of training for multiple epsilon values
avg_rewards_for_epsilons_over_num_runs = []
for i in range(num_runs):
	avg_reward_for_epsilons = run(env, init_bias, episodes,epsilons)[0]
	avg_rewards_for_epsilons_over_num_runs.append(avg_reward_for_epsilons)

avg_rewards_for_epsilons_over_num_runs = np.array(avg_rewards_for_epsilons_over_num_runs)
avg_of_avg_rewards_for_epsilons_over_num_runs = np.mean(avg_rewards_for_epsilons_over_num_runs, axis=0)


# Visualisation of results
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
for i in range(len(avg_of_avg_rewards_for_epsilons_over_num_runs)):
	plt.plot(avg_of_avg_rewards_for_epsilons_over_num_runs[i], label = "epsilon = " + str(epsilons[i]))
	plt.legend(loc = "upper right")
plt.show()














