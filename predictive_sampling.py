from action_sampling import actionSampler
from numerical_integration import implicitIntegrator
from trajectory_evaluation import trajectoryEvaluator
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class predictiveSampler:
    """
    A predictive sampling algorithm for the cartpole-environment.
    """
    def __init__(self, horizon, dt, sample_variance=1, mean = None):
        """Initialize the predictive sampler.
        Args:
            horizon (int): The number of time steps to sample actions for.
            dt (float): The time step size.
            sample_variance (float): The variance of the gaussian distribution.
            mean (torch.tensor): The mean of the gaussian distribution. Shape: (horizon,)."""
        self.horizon = horizon
        self.dt = dt
        self.action_sampler = actionSampler(horizon, sample_variance, mean=mean)
        self.integrator = implicitIntegrator(horizon, dt)
        self.evaluator = trajectoryEvaluator()
        self.N_trajs = 10
        self.states = np.zeros((self.N_trajs, self.horizon, self.integrator.x0.shape[0]))
    
    def predict(self, state):
        ## Sample a batch of actions
        action_trajs = self.action_sampler.sample_batch(self.N_trajs) # (N_trajs, horizon)
        ## Integrate the actions
        self.states = self.integrator.integrate_trajs(action_trajs, self.convert_state(state)) # (N_trajs, horizon, 4)
        ## Evaluate the trajectories
        self.best_actions, self.best_index = self.evaluator.find_best_control(self.states, action_trajs) # (N_trajs,)
        ## Update the sampling distribution mean
        self.action_sampler.update_mu(self.best_actions)
        return self.best_actions

    def convert_state(self, state):
        """Converting from [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]
        to [pos, pos_dot, theta, theta_dot]
        We also need to offset theta, as 0 is defined as straight up in the envirement, whereas we define it as straight down."""
        theta = np.arctan2(state[3], state[2])
        theta = theta - np.pi
        return np.array([state[0], state[1], theta, state[4]])
        


if __name__ == "__main__":
    predictive_sampler = predictiveSampler(horizon=1000, dt=0.001)
    state = np.array([0, 0, np.pi - 0.25, 0])
    # Convert to [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]])
    best_actions = predictive_sampler.predict(state_cartpole)
