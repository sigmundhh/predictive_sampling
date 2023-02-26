# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_cartpole_swingup
from gym_cartpole_swingup.envs.cartpole_swingup import State
from predictive_sampling import PredictiveSampler
from mpc_visualizer import MPCVisualizer
import torch

env = gym.make("CartPoleSwingUp-v0")
done = False

obs = env.reset()
env.render()
sim_steps = 1000

predictive_sampler = PredictiveSampler(horizon=100, dt=0.01, sample_variance=0.1)
mpc_visualizer = MPCVisualizer(predictive_sampler, sim_steps)

for i in range(sim_steps):
    action = predictive_sampler.predict(obs)
    obs, rew, done, info = env.step(action)
    mpc_visualizer.log_state(obs, i)
    env.render()
    mpc_visualizer.visualize_theta_trajs()


## plot the state history
mpc_visualizer.visualize_state_history()
