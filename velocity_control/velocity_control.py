import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import namedtuple

# Define a named tuple to store the state of the cartpole
State = namedtuple("State", ["x_pos", "x_dot", "theta", "theta_dot"])


def pendulum_derivatives(state, desired_vel, L=1, g=9.81):
    """
    Calculate the derivatives of theta and omega at a given time t, given the current values of theta, omega, and the velocity of the fixation point v.

    Parameters
    ----------
    theta : float
        The current angle of the pendulum in radians.
    omega : float
        The current angular velocity of the pendulum in radians/second.
    t : float
        The current time in seconds.
    v : float
        The velocity of the fixation point at time t.
    L : float, optional
        The length of the pendulum, default 1.
    g : float, optional
        The acceleration due to gravity, default 9.81 m/s^2.

    Returns
    -------
    [x_dot, x_ddot, theta_dot, theta_ddot] : list
    """
    T_v = 0.01
    x_ddot = 1 / T_v * (desired_vel - state.x_dot)
    theta_ddot = 1 / L * x_ddot * np.cos(state.theta) - g / L * np.sin(state.theta)

    return [state.x_dot, x_ddot, state.theta_dot, theta_ddot]


def pendulum_trajectory(velocities, L=1, g=9.81, dt=0.01, theta0=0):
    """
    Calculate the trajectory of a 2D pendulum given a series of velocities for the fixation point.

    Parameters
    ----------
    velocities : array_like
        An array of velocity values for the fixation point, one for each time step.
    L : float, optional
        The length of the pendulum, default 1.
    g : float, optional
        The acceleration due to gravity, default 9.81 m/s^2.
    dt : float, optional
        The time step between each velocity value, default 0.01 s.
    theta0 : float, optional
        The initial angle of the pendulum in radians, default 0.
    omega0 : float, optional
        The initial angular velocity of the pendulum in radians/second, default 0.

    Returns
    -------
    tuple
        A tuple containing two arrays: the times at which the angle is calculated and the corresponding values of the angle.
    """
    # Initialize the state
    state = State(x_pos=0, x_dot=0, theta=theta0, theta_dot=0.0)
    # Array for logging states
    states = []
    # Loop through the velocities and calculate the state at each time step
    for i, v in enumerate(velocities):
        # Calculate the derivatives
        dstate = pendulum_derivatives(state, v, L, g)
        # Update the state
        state = State(
            state.x_pos + state.x_dot * dt,
            state.x_dot + dstate[1] * dt,
            state.theta + state.theta_dot * dt,
            state.theta_dot + dstate[3] * dt,
        )
        # Log the state
        states.append(state)
    # Convert the list of states into an array
    states = np.array(states)

    return states


def visualize_pendulum(velocities, theta_0=0, dt=0.01):
    # Parameters
    L = 1
    states = pendulum_trajectory(velocities, theta0=theta_0, dt=dt)
    ## Visualize the trajectory of the pendulum with an animation
    fig, ax = plt.subplots()

    # Set up the axes
    # set c lim to be the same as the fixation point trajectory, but wider than -1.5 to 1.5
    ax.set_xlim(states[:, 0].min() - 0.5, states[:, 0].max() + 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid()

    # Draw the pendulum hanging from the fixation point
    (pendulum,) = ax.plot([], [], "o-", lw=2)

    def init():
        pendulum.set_data([], [])
        return (pendulum,)

    def animate(i):
        x = [states[i, 0], states[i, 0] - L * np.sin(states[i, 2])]
        y = [0, -np.cos(states[i, 2])]
        pendulum.set_data(x, y)
        return (pendulum,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(velocities), interval=dt * 1000, blit=True
    )

    plt.show()


if __name__ == "__main__":
    velocities = np.sin(np.linspace(0, 10, 1000))
    velocities[500:] = 0
    visualize_pendulum(velocities, theta_0=0.0)
