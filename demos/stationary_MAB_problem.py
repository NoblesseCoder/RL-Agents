import gym
import gym_karmedbandits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../agents')
from tabular.epsilon_greedy_agent import EpsilonGreedyAgent
from tabular.upper_confidence_bound_agent import UpperConfidenceBoundAgent 

sns.set(color_codes=True, style="whitegrid")


def train(env, agent, epsilon, init_bias, episodes):
	''' Function trains the ε-Greedy agent on the environment '''

	# Initial parameters
	agent.reset_memory(epsilon, init_bias)
	reward = 0
	prev_state = env.reset()
	done = False
	total_rewards = []
	avg_rewards = [] # store average rewards
	cum_reward = 0 
	
	# Reinforcement Learning Training 
	for episode in range(episodes):
		while True:
			next_state, reward, done, _ = env.step(agent.act(prev_state, 
										reward, done))
			print("Episode:" + str(episode) + " Reward:" + str(reward))
			if done:
				break

		cum_reward += reward
		total_rewards.append(cum_reward)
		avg_rewards.append(cum_reward / (episode + 1))
	
	env.close()
	return(np.array(avg_rewards), np.array(total_rewards))


def train_for_different_epsilons(env, agent, init_bias, episodes, epsilons):
	''' Runs training for multiple epsilon values '''

	avg_rewards_for_epsilons = []
	total_rewards_for_epsilons = []

	for ind in range(len(epsilons)):
		rewards = train(env, agent, epsilons[ind], init_bias, episodes)
		avg_rewards_for_epsilons.append(rewards[0]) 
		total_rewards_for_epsilons.append(rewards[1])

	return(np.array(avg_rewards_for_epsilons), 
			np.array(total_rewards_for_epsilons))

 

# Creating the Multi Armed Bandits(MAB) gym environment
env = gym.make('KArmedBandits-v0')
env.seed(0)

# Instantiating Agent 
epsilon_greedy_agent = EpsilonGreedyAgent(env.action_space, env.k, None, None)
ucb_greedy_agent = UpperConfidenceBoundAgent(env.action_space, 
					env.k, None, None, 2)

# Initializing Parameters
epsilons = [0, 0.01, 0.5, 0.1]
init_bias = 0
episodes = 2000
num_runs = 1


# Running multiple runs of training for multiple epsilon values
avg_rewards_for_epsilons_over_num_runs_epsilon_greedy = []
avg_rewards_for_epsilons_over_num_runs_ucb_greedy = []

for i in range(num_runs):
	print("Run:" + str(i+1))
	avg_reward_for_epsilons_epsilon_greedy = train_for_different_epsilons(
											env, epsilon_greedy_agent, 
											init_bias, episodes,epsilons)[0]
	avg_reward_for_epsilons_ucb_greedy = train_for_different_epsilons(env, 
										ucb_greedy_agent, init_bias, 
										episodes,epsilons)[0]
	avg_rewards_for_epsilons_over_num_runs_epsilon_greedy.append(
							avg_reward_for_epsilons_epsilon_greedy)
	avg_rewards_for_epsilons_over_num_runs_ucb_greedy.append(
							avg_reward_for_epsilons_ucb_greedy)



avg_rewards_for_epsilons_over_num_runs_epsilon_greedy = np.array(
						avg_rewards_for_epsilons_over_num_runs_epsilon_greedy)
avg_rewards_for_epsilons_over_num_runs_ucb_greedy = np.array(
							avg_rewards_for_epsilons_over_num_runs_ucb_greedy)


# Final Averages
avg_of_avg_rewards_for_epsilons_over_num_runs_epsilon_greedy = np.mean(
				avg_rewards_for_epsilons_over_num_runs_epsilon_greedy, axis=0)
avg_of_avg_rewards_for_epsilons_over_num_runs_ucb_greedy = np.mean(
					avg_rewards_for_epsilons_over_num_runs_ucb_greedy, axis=0)


# Visualisation of results
plt.title("Stationary MAB problem (Average reward v/s Episodes plot) for "  
			+ str(num_runs)
			+ " runs")

plt.xlabel("Episodes")
plt.ylabel("Average Reward")

for i in range(len(avg_of_avg_rewards_for_epsilons_over_num_runs_epsilon_greedy)
				):
	plt.plot(avg_of_avg_rewards_for_epsilons_over_num_runs_epsilon_greedy[i],
	 		label = "(ε-Greedy) ε = " + str(epsilons[i]))
	plt.legend(loc = "lower right")
for i in range(len(avg_of_avg_rewards_for_epsilons_over_num_runs_ucb_greedy)):
	plt.plot(avg_of_avg_rewards_for_epsilons_over_num_runs_ucb_greedy[i], 
			label = "(UCB-Greedy) ε = " + str(epsilons[i]) + ", c = 2")
	plt.legend(loc = "lower right") 
plt.show()














