import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# Define a named tuple to store the state of the cartpole
#State = namedtuple("State", ["x_pos", "x_dot", "theta", "theta_dot"])


class ExplicitIntegrator:
    """
    Using the dynamics from gym_cartpole_swingup
    """

    def __init__(self, horizon, dt=0.01):
        #self.x0 = State(0.0, 0.0, 0.0, 0.0)
        self.x0 = np.zeros(4)
        self.horizon = horizon
        self.dt = dt
        self.g = 9.81
        self.L = 10

    def pendulum_derivatives(self, state, desired_vel, L=1, g=9.81):
        """
        Calculate the derivatives of theta and omega at a given time t, given the current values of theta, omega, and the velocity of the fixation point v.

        Parameters
        ----------
        state (np.array) : [pos, pos_dot, theta, theta_dot]
        desired_vel (torch.tensor) : scalar 

        Returns
        -------
        d_state (np.array) : [pos_dot, pos_ddot, theta_dot, theta_ddot]
        """
        T_v = 0.01
        x_ddot = 1 / T_v * (desired_vel - state[1])
        theta_ddot = 1 / L * x_ddot * np.cos(state[2]) - g / L * np.sin(state[2])

        return np.array([state[1], x_ddot, state[3], theta_ddot])

    def integrate_traj(self, control_traj):
        """
        Args:
            F_traj (np.array): (Hx1)
        Returns:
            xs (np.array): state trajectory (H, state_dim)
        """

        # Initialize the state
        state = self.x0
        # Array for logging states
        states = np.zeros((self.horizon, self.x0.shape[0]))
        # Loop through the velocities and calculate the state at each time step
        for i in range(self.horizon):
            # Calculate the derivatives
            dstate = self.pendulum_derivatives(state, control_traj[i])
            # Update the state
            state = state + self.dt * dstate
            """state = State(
                state.x_pos + state.x_dot * self.dt,
                state.x_dot + dstate[1] * self.dt,
                state.theta + state.theta_dot * self.dt,
                state.theta_dot + dstate[3] * self.dt,
            )"""
            # Log the state
            states[i, :]= state
        # Convert the list of states into an array
        #states = np.array(states)
        return states


    def integrate_trajs(self, F_trajs, state):
        """Integrates up multiple state trajectories
        Args:
            F-trajs (np.array): (N x H x state_dim)
            state (np.array) : [pos, pos_dot, theta, theta_dot]
        Returns:
            state_trajs (np.array): (N, H, state_dim) 
        """
        self.x0 = state
        state_trajs = np.zeros((F_trajs.shape[0], F_trajs.shape[1], self.x0.shape[0]))
        for i in range(state_trajs.shape[0]):
            state_trajs[i, :, :] = self.integrate_traj(F_trajs[i, :])
        return state_trajs

        

if __name__ == "__main__":
    horizon = 100
    F_traj = np.ones((2, horizon))

    # Test explicit integrator
    explicit_integrator = ExplicitIntegrator(horizon)
    xs = explicit_integrator.integrate_trajs(F_traj, state=np.array([0.0, 0.0, 0.0, 0.0]))
    plt.plot(xs[1, :, 0])
    plt.plot(xs[1, :, 1])
    plt.plot(xs[1, :, 2])
    plt.plot(xs[1, :, 3])
    # add legend
    plt.legend(["pos", "pos_dot", "theta", "theta_dot"])

    plt.show()
