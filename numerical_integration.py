import numpy as np
import matplotlib.pyplot as plt

class implicitIntegrator():
    """
    We define the full state (pos and vel) as x = [pos, pos_dot, theta, theta_dot]
    We solve for x_dot and use forward Euler to integrate the dynamics
    """
    def __init__(self, horizon, dt):
        self.x0 = np.array([0, 0, 0, 0]) # initial state
        self.horizon = horizon
        self.dt = dt
        self.g = 9.81  # should maybe have a global variable for this
        self.m_1 = 0.2
        self.m_2 = 0.1
        self.l = 1

    def solve_x_dot(self, x, F):
        ## Have a set of equations describing the dynamics:
        # A x_dot = b
        theta, theta_dot = x[2], x[3]
        A = np.array([[np.cos(theta), self.l],
                    [self.m_1 + self.m_2, self.m_2*self.l*np.cos(theta)]])   # why is is so tricky to get a scalar from cos?
        b = np.array([[-self.g*np.sin(theta)],
                    [F + self.m_2 * self.l * np.power(theta_dot, 2)*np.sin(theta)]])
        sol = np.linalg.solve(A, b)
        pos_ddot, theta_ddot = sol[0][0], sol[1][0]
        return np.array([x[1], pos_ddot, x[3], theta_ddot])

    def integrate_traj(self, F_traj):
        xs = np.zeros((self.horizon, self.x0.shape[0]))
        x = self.x0
        for i in range(self.horizon):
            x_dot = self.solve_x_dot(x, F_traj[i])
            x = x + self.dt * x_dot
            xs[i, :] = x
        return xs
    
    def integrate_trajs(self, F_trajs, state):
        self.x0 = state
        state_trajs = np.zeros((F_trajs.shape[0], F_trajs.shape[1], self.x0.shape[0]))
        for i in range(state_trajs.shape[0]):
            state_trajs[i,:,:] = self.integrate_traj(F_trajs[i, :])
        return state_trajs

if __name__ == "__main__":
    horizon = 100
    dt = 0.02
    F_traj = np.ones(horizon) ## Shifting to get negative numbers too
    integrator = implicitIntegrator(horizon, dt)

    #plot the state trajectory, only theta and theta_dot and pos
    xs = integrator.integrate_traj(F_traj)
    plt.plot(xs[:,0])
    plt.plot(xs[:,1])
    plt.plot(xs[:,2])
    plt.plot(xs[:,3])
    # add legend
    plt.legend(['pos', 'pos_dot', 'theta', 'theta_dot'])
    
    plt.show()
