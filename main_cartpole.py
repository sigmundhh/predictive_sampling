# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_cartpole_swingup
from predictive_sampling import predictiveSampler
from mpc_visualizer import MPCVisualizer
import torch


# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("CartPoleSwingUp-v0")
done = False

predictive_sampler = predictiveSampler(horizon=100, dt=0.01, sample_variance=0.01)
#mpc_visualizer = MPCVisualizer(predictive_sampler)

obs = env.reset()
# Logging state for a small sim
sim_steps = 1000
poss = np.zeros(sim_steps)
pos_dots = np.zeros(sim_steps)
thetas = np.zeros(sim_steps)
theta_dots = np.zeros(sim_steps)

# make interactive plot
plt.ion()

for i in range(thetas.shape[0]):
    action = predictive_sampler.predict(obs)
    obs, rew, done, info = env.step(action)
    poss[i], pos_dots[i], thetas[i], theta_dots[i] = predictive_sampler.convert_state(obs)
    #plot all the predicted state trajectories and highlight the best one

    plt.clf()
    for i in range (predictive_sampler.states.shape[0]):
        if i == predictive_sampler.best_index:
            plt.plot(predictive_sampler.states[i,:,2 ], c='r')
            ## Also add the trajectory cost to the plot
            trajectory_cost = predictive_sampler.evaluator.total_trajectory_cost(predictive_sampler.states[i,:,:], predictive_sampler.best_actions)
            plt.title("Trajectory cost: " + str(trajectory_cost))
        else:
            plt.plot(predictive_sampler.states[i,:,2], c='b', alpha=0.1)

    env.render()
    #print(obs)
    
## plot pos over time
plt.plot(poss)
# plot cos(theta) over time
#plt.plot(np.cos(thetas))    # we want this to be close to 1

#Legend
plt.legend(['pos'])


plt.show()