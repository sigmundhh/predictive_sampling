# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_cartpole_swingup
from gym_cartpole_swingup.envs.cartpole_swingup import State
from predictive_sampling import PredictiveSampler
from mpc_visualizer import MPCVisualizer
import torch


# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("CartPoleSwingUp-v0")
done = False

predictive_sampler = PredictiveSampler(horizon=100, dt=0.01, sample_variance=0.1)
#mpc_visualizer = MPCVisualizer(predictive_sampler)

obs = env.reset()
env.render()
#new_state = State(0,0,np.pi,0)
#env.state = new_state
#env.render()
# Logging state for a small siVil 
sim_steps = 100
poss = np.zeros(sim_steps)
pos_dots = np.zeros(sim_steps)
thetas = np.zeros(sim_steps)
theta_dots = np.zeros(sim_steps)

# make interactive plot
#plt.ion()
for i in range(thetas.shape[0]):
    action = predictive_sampler.predict(obs)
    action = 1.0
    obs, rew, done, info = env.step(action)
    poss[i], pos_dots[i], thetas[i], theta_dots[i] = predictive_sampler.convert_state(obs)
    env.render()
    #print(obs)
    
## plot the state over time in the same plot
plt.figure()
plt.plot(poss)
plt.plot(pos_dots)
plt.plot(thetas)
plt.plot(theta_dots)
plt.legend(["pos", "pos_dot", "theta", "theta_dot"])


plt.show()