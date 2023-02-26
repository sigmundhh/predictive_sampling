import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# Define a named tuple to store the state of the cartpole
#State = namedtuple("State", ["x_pos", "x_dot", "theta", "theta_dot"])

class PendulumEnvironment:
    def __init__(self, dt=0.01):
        self.x0 = np.array([0.0, 0.0, 1.0, 0.0])
        self.dt = dt
        self.g = 9.81
        self.L = 1


        # Set up the matplotlib window to be updated in render()
        plt.ion()
        # Set x and y limits
        self.xlim = (-2, 2)
        self.ylim = (-2, 2)
        # Set up the figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.grid()
            

    def reset(self):
        """

        Returns
        -------
        state (np.array) : [pos, pos_dot, theta, theta_dot]
        """
        self.state = self.x0
        return self.state
    

    def pendulum_derivatives(self, state, desired_vels):
        """
        Calculate the derivatives of theta and omega at a given time t, given the current values of theta, omega, and the velocity of the fixation point v.

        Parameters
        ----------
        state (np.array) : [pos, pos_dot, theta, theta_dot]
        desired_vel (torch.tensor) : Action trajectory of size (Horizon)

        Returns
        -------
        d_state (np.array) : [pos_dot, pos_ddot, theta_dot, theta_ddot]
        """
        desired_vel = desired_vels[0]
        T_v = 0.01
        x_ddot = 1 / T_v * (desired_vel - state[1])
        theta_ddot = 1 / self.L * x_ddot * np.cos(state[2]) - self.g / self.L * np.sin(state[2])

        return np.array([state[1], x_ddot, state[3], theta_ddot])

    
    def step(self, desired_vel):
        dstate = self.pendulum_derivatives(self.state, desired_vel)
        self.state = self.state + self.dt * dstate
        return self.state

    def render(self):
        plt.clf()
        x = [self.state[0], self.state[0] - self.L * np.sin(self.state[2])]
        y = [0, -np.cos(self.state[2])*self.L]
        plt.plot(x, y, 'o-', color='black')
        # Set aspect ratio to 1
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.grid()

        plt.pause(0.01)




if __name__ == "__main__":
    env = PendulumEnvironment()
    env.reset()
    env.render()
    for i in range(100):
        env.step(1.0)
        env.render()
    env.reset()
    for i in range(100):
        env.step(0.0)
        env.render()
