import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from predictive_sampling import PredictiveSampler


class MPCVisualizer:
    def __init__(self, predictive_sampler, sim_steps):
        self.predictive_sampler = predictive_sampler
        self.fig = plt.figure()
        plt.ion()
        self.poss = np.zeros(sim_steps)
        self.pos_dots = np.zeros(sim_steps)
        self.thetas = np.zeros(sim_steps)
        self.theta_dots = np.zeros(sim_steps)

    def log_state(self, obs, i):
        """
        Log the state of the system
        """
        (
            self.poss[i],
            self.pos_dots[i],
            self.thetas[i],
            self.theta_dots[i],
        ) = self.predictive_sampler.convert_state(obs)

    def visualize_state_history(self):
        """
        Visualize the state history
        """
        plt.ioff()
        plt.close()
        plt.figure()
        plt.plot(self.poss)
        plt.plot(self.pos_dots)
        plt.plot(self.thetas)
        plt.plot(self.theta_dots)
        plt.legend(["pos", "pos_dot", "theta", "theta_dot"])
        plt.title("State history")
        plt.xlabel("Time step")
        plt.ylabel("State")
        plt.show()

    def visualize_theta_trajs(self):
        """
        Given an MPC object, visualize the trajectories of theta over the horizon.
        The states are stored in the mpc object as mpc.states as a (N_trajs, horizon, 4) matrix
        Each state is on the form [pos, pos_dot, theta, theta_dot]

        Note: This pauses for 0.001 second, so turn off for better performance
        """
        plt.clf()
        for i in range(self.predictive_sampler.states.shape[0]):
            if i == self.predictive_sampler.best_index:
                plt.plot(self.predictive_sampler.states[i, :, 2], c="r")
            else:
                plt.plot(self.predictive_sampler.states[i, :, 2], c="b", alpha=0.1)

        plt.title("Predicted theta trajectory")
        plt.xlabel("Time step")
        plt.ylabel("Theta")
        plt.pause(0.001)


if __name__ == "__main__":
    predictive_sampler = PredictiveSampler(
        horizon=500, dt=0.01, sample_variance=0.1, shift=False
    )
    predictive_sampler_visualizer = MPCVisualizer(predictive_sampler)
    state = np.array([0, 0, np.pi - 0.25, 0])
    # Convert to [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array(
        [state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]
    )
    for i in range(100):
        best_actions = predictive_sampler.predict(state_cartpole)
        predictive_sampler_visualizer.visualize_theta_trajs()
