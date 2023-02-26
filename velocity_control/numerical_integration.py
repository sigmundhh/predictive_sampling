import numpy as np
import matplotlib.pyplot as plt


class explicitIntegrator:
    """
    Using the dynamics from gym_cartpole_swingup
    """

    def __init__(self, horizon, dt=0.01):
        self.x0 = np.array([0, 0, 0, 0])  # initial state
        self.horizon = horizon
        self.dt = dt
        self.g = 9.82  # no idea why
        self.forcemag = 10
        self.friction = 0.1
        self.x_threshold = 2.4
        self.m_cart = 0.5
        self.m_pole = 0.5
        self.pole_length = 0.6
        self.pole_width = 0.05  # Not sure this is relevant for dynamics
        self.pole_mpl = self.m_pole * self.pole_length
        self.mass_total = self.m_cart + self.m_pole

    def x_dot(self, x, F):
        """Returns the state derivatives given the state and forces
        Args:
            x (np.array) : [pos, pos_dot, theta, theta_dot]
            F (float)
        Returns:
            x_dot (np.array) : d/dt([pos, pos_dot, theta, theta_dot])

        Note:
        - The forces are magnified by 10 to match the gym_cartpole_swingup environment
        """

        action = F * self.forcemag

        pos, pos_dot, theta, theta_dot = x[0], x[1], x[2], x[3]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        pos_ddot = (
            -2 * self.pole_mpl * (theta_dot**2) * sin_theta
            + 3 * self.m_pole * self.g * sin_theta * cos_theta
            + 4 * action
            - 4 * self.friction * pos_dot
        ) / (4 * self.mass_total - 3 * self.m_pole * cos_theta**2)
        theta_ddot = (
            -3 * self.pole_mpl * (theta_dot**2) * sin_theta * cos_theta
            + 6 * self.mass_total * self.g * sin_theta
            + 6 * (action - self.friction * pos_dot) * cos_theta
        ) / (
            4 * self.pole_length * self.mass_total - 3 * self.pole_mpl * cos_theta**2
        )

        return np.array([x[1], pos_ddot, x[3], theta_ddot])

    def integrate_traj(self, F_traj):
        """
        Args:
            F_traj (np.array): (Hx1)
        Returns:
            xs (np.array): (H x state_dim) = (H x 4)
        """
        xs = np.zeros((self.horizon, self.x0.shape[0]))
        x = self.x0
        for i in range(self.horizon):
            x_dot = self.x_dot(x, F_traj[i])
            x = x + self.dt * x_dot
            xs[i, :] = x
        return xs

    def integrate_trajs(self, F_trajs, state):
        """Integrates up multiple state trajectories
        Args:
            F-trajs (np.array): (N x H x 1)
            state (np.array): Initial state (1 x state_dim)
        Returns:
            state_trajs (np.array): (N x H x state_dim)
        """
        self.x0 = state
        state_trajs = np.zeros((F_trajs.shape[0], F_trajs.shape[1], self.x0.shape[0]))
        for i in range(state_trajs.shape[0]):
            state_trajs[i, :, :] = self.integrate_traj(F_trajs[i, :])
        return state_trajs


if __name__ == "__main__":
    horizon = 100
    F_traj = np.ones(horizon)  ## Shifting to get negative numbers too

    # Test explicit integrator
    explicit_integrator = explicitIntegrator(horizon)
    xs = explicit_integrator.integrate_traj(F_traj)
    plt.plot(xs[:, 0])
    plt.plot(xs[:, 1])
    plt.plot(xs[:, 2])
    plt.plot(xs[:, 3])
    # add legend
    plt.legend(["pos", "pos_dot", "theta", "theta_dot"])

    plt.show()
