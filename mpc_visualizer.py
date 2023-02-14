import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from predictive_sampling import predictiveSampler

class MPCVisualizer:
    def __init__(self, mpc):
        self.mpc = mpc
        self.fig = plt.figure()
        # Set the axis limits to [0, horizon]
        self.ax = plt.axes(xlim=(0, self.mpc.horizon), ylim=(-np.pi, np.pi))
        #self.line, = self.ax.plot(mpc.states[0, :, 2])


    def visualize_theta_trajs(self):
        """
        Given an MPC object visualize the trajectories of theta over the horizon.
        The states are stored in the mpc object as mpc.states which is a (N_trajs, horizon, 4) matrix
        Each state is on the form [pos, pos_dot, theta, theta_dot]
        """
        plt.clf()
        # Add all trajecories to the line
        for i in range(self.mpc.N_trajs):
            plt.plot(self.mpc.states[i, :, 2])
        
        plt.pause(0.0001)

    def update_plot(self):
        self.line.set_ydata(mpc.states[0, :, 2])
        print("updated plot!")
        return self.line,

    

if __name__ == "__main__":
    mpc = predictiveSampler(horizon=1000, dt=0.001, sample_variance=10)
    mpc_visualizer = MPCVisualizer(mpc)
    state = np.array([0, 0, np.pi - 0.25, 0])
    # Convert to [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]])
    for i in range(100):
        best_actions = mpc.predict(state_cartpole)
        mpc_visualizer.visualize_theta_trajs()
    plt.show()


