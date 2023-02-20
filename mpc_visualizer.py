import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from predictive_sampling import PredictiveSampler

class MPCVisualizer:
    def __init__(self, predictive_sampler):
        self.predictive_sampler = predictive_sampler
        self.fig = plt.figure()
        plt.ion()
        # Set the axis limits to [0, horizon]
        #self.ax = plt.axes(xlim=(0, self.predictive_sampler.horizon), ylim=(-np.pi, np.pi))
        #self.line, = self.ax.plot(mpc.states[0, :, 2])


    def visualize_theta_trajs(self):
        """
        Given an MPC object visualize the trajectories of theta over the horizon.
        The states are stored in the mpc object as mpc.states which is a (N_trajs, horizon, 4) matrix
        Each state is on the form [pos, pos_dot, theta, theta_dot]
        """
        plt.clf()
        for i in range (self.predictive_sampler.states.shape[0]):
            if i == self.predictive_sampler.best_index:
                plt.plot(self.predictive_sampler.states[i,:,2], c='r')
            else:
                plt.plot(self.predictive_sampler.states[i,:,2], c='b', alpha=0.1)

        plt.title("Predicted theta trajectory")
        plt.xlabel("Time step")
        plt.ylabel("Theta")
        plt.pause(0.001)


    def update_plot(self):
        self.line.set_ydata(predictive_sampler.states[0, :, 2])
        print("updated plot!")
        return self.line,

    

if __name__ == "__main__":
    predictive_sampler = PredictiveSampler(horizon=500, dt=0.01, sample_variance=0.1, shift=False)
    predictive_sampler_visualizer = MPCVisualizer(predictive_sampler)
    state = np.array([0, 0, np.pi - 0.25, 0])
    # Convert to [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]])
    for i in range(100):
        best_actions = predictive_sampler.predict(state_cartpole)
        predictive_sampler_visualizer.visualize_theta_trajs()
        


