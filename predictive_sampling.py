from action_sampling import ActionSampler
from numerical_integration import implicitIntegrator, explicitIntegrator
from trajectory_evaluation import TrajectoryEvaluator
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class PredictiveSampler:
    """
    A predictive sampling algorithm for the cartpole-environment.
    """

    def __init__(self, horizon, dt, sample_variance=0.1, mean=None, sample_count=10, shift=True):
        """Initialize the predictive sampler.
        Args:
            horizon (int): The number of time steps to sample actions for.
            dt (float): The time step size.
            sample_variance (float): The variance of the gaussian distribution.
            mean (torch.tensor): The mean of the gaussian distribution. Shape: (horizon,)."""
        self.horizon = horizon
        self.dt = dt
        self.action_sampler = ActionSampler(horizon, sample_variance, mean=mean)
        self.integrator = explicitIntegrator(horizon, dt)
        self.evaluator = TrajectoryEvaluator()
        self.N_trajs = sample_count
        self.states = np.zeros(
            (self.N_trajs, self.horizon, self.integrator.x0.shape[0])
        )
        self.shift = shift

    def predict(self, state):
        ## Sample a batch of actions
        self.action_trajs = self.action_sampler.sample_batch(
            self.N_trajs
        )  # (N_trajs, horizon)
        ## Integrate the actions
        self.states = self.integrator.integrate_trajs(
            self.action_trajs, self.convert_state(state)
        )  # (N_trajs, horizon, 4)
        ## Evaluate the trajectories
        self.best_actions, self.best_index = self.evaluator.find_best_control(
            self.states, self.action_trajs
        )  # (N_trajs,)
        ## Update the sampling distribution mean
        self.action_sampler.update_mu(self.best_actions, shift=self.shift)
        return self.best_actions

    def convert_state(self, state):
        """Converting from [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]
        to [pos, pos_dot, theta_tilde, theta_dot]
        Now, theta is 0 when straight up, and positive counter-clockwise."""
        theta = np.arctan2(state[3], state[2])
        return np.array([state[0], state[1], theta, state[4]])


def test_trajectory_ranking():
    """Test that the trajectory ranking picks out the trajectory with the lowest cost."""
    # For this experiment, remember to sett shift = False in the update_mu function
    predictive_sampler = PredictiveSampler(horizon=1000, dt=0.01)
    state = np.array(
        [0.0, 0.0, np.pi, 0.0]
    )  # [x_pos,.0 x_dot, theta, theta_dot] in env state space, pendulum straight up
    # Convert to [pos, pos_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array(
        [state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]
    )  # in env state space

    plt.figure()
    predictive_sampler.predict(state_cartpole)
    # Color related
    vmin = 200
    vmax = 100

    max_cost = 0
    min_cost = np.inf
    for i in range(predictive_sampler.states.shape[0]):
        states = predictive_sampler.states[i]
        actions = predictive_sampler.action_trajs[i]
        cost = predictive_sampler.evaluator.total_trajectory_cost(states, actions)
        if cost > max_cost:
            max_cost = cost
        if cost < min_cost:
            min_cost = cost

    vmin = min_cost
    vmax = max_cost

    for i in range (predictive_sampler.states.shape[0]):
        states = predictive_sampler.states[i]
        actions = predictive_sampler.action_trajs[i]
        cost = predictive_sampler.evaluator.total_trajectory_cost(states, actions)
        if i == predictive_sampler.best_index:
            # Map the cost to the color of the line
            plt.plot(predictive_sampler.states[i,:,2 ], color=plt.cm.jet((cost-vmin)/(vmax-vmin)))
        else:
            plt.plot(predictive_sampler.states[i,:,2], color=plt.cm.jet((cost-vmin)/(vmax-vmin)), alpha=0.3)
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.title("Predictive sampling")
    plt.xlabel("Time step")
    plt.ylabel("Theta")
    
    plt.show()


def test_convergence():
    """Test that the sampling scheme converges to a sensible solution when we iterate
    Note: mean shifting should be set to false in this experiment, as the state doesn't change."""
    predictive_sampler = PredictiveSampler(horizon=200, dt=0.01)
    state = np.array(
        [0.0, 0.0, np.pi, 0.0]
    )  # [x_pos,.0 x_dot, theta, theta_dot] in env state space, pendulum straight up
    # Convert to [pos, pos_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array(
        [state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]
    )  # in env state space
    iterations = 300
    plt.figure()
    # Plot the the theta trajectory at each iteration
    # Color code the lines by the iteration number
    for i in range(iterations):
        predictive_sampler.predict(state_cartpole)
        plt.plot(predictive_sampler.states[predictive_sampler.best_index,:,2], color=plt.cm.jet(i/iterations))
    plt.title("Predictive sampling")
    plt.xlabel("Time step")
    plt.ylabel("Theta")
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm)

    plt.show()

def test_convergence_cost():
    """Want to create a plot showing the cost over iterations"""

    plt.figure()
    ### BASELINE ###
    predictive_sampler = PredictiveSampler(horizon=200, sample_variance=0.1, dt=0.01, sample_count=10)
    state = np.array(
        [0.0, 0.0, np.pi, 0.0]
    )  # [x_pos,.0 x_dot, theta, theta_dot] in env state space, pendulum straight up
    # Convert to [pos, pos_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array(
        [state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]
    )  # in env state space
    iterations = 300
    # Plot the the theta trajectory at each iteration
    costs = []
    for i in range(iterations):
        predictive_sampler.predict(state_cartpole)
        states = predictive_sampler.states[predictive_sampler.best_index]
        actions = predictive_sampler.action_trajs[predictive_sampler.best_index]
        cost = predictive_sampler.evaluator.total_trajectory_cost(states, actions)
        costs.append(cost)
    plt.plot(costs)

    ### SAMPLE COUNT 5 ###
    predictive_sampler = PredictiveSampler(horizon=200, sample_variance=1.0, dt=0.01, sample_count=5)
    state = np.array(
        [0.0, 0.0, np.pi, 0.0]
    )  # [x_pos,.0 x_dot, theta, theta_dot] in env state space, pendulum straight up
    # Convert to [pos, pos_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array(
        [state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]
    )  # in env state space
    iterations = 300
    # Plot the the theta trajectory at each iteration
    costs = []
    for i in range(iterations):
        predictive_sampler.predict(state_cartpole)
        states = predictive_sampler.states[predictive_sampler.best_index]
        actions = predictive_sampler.action_trajs[predictive_sampler.best_index]
        cost = predictive_sampler.evaluator.total_trajectory_cost(states, actions)
        costs.append(cost)
    plt.plot(costs)

    ### SAMPLE COUNT 7 ###
    predictive_sampler = PredictiveSampler(horizon=200, sample_variance=0.01, dt=0.01, sample_count=7)
    state = np.array(
        [0.0, 0.0, np.pi, 0.0]
    )  # [x_pos,.0 x_dot, theta, theta_dot] in env state space, pendulum straight up
    # Convert to [pos, pos_dot, np.cos(theta), np.sin(theta), theta_dot]
    state_cartpole = np.array(
        [state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]
    )  # in env state space
    iterations = 300
    # Plot the the theta trajectory at each iteration
    costs = []
    for i in range(iterations):
        predictive_sampler.predict(state_cartpole)
        states = predictive_sampler.states[predictive_sampler.best_index]
        actions = predictive_sampler.action_trajs[predictive_sampler.best_index]
        cost = predictive_sampler.evaluator.total_trajectory_cost(states, actions)
        costs.append(cost)
    plt.plot(costs)

    
    # Legend
    #plt.legend(["Baseline variance: 0.1", "Variance 1", "Variance 0.01"])
    plt.legend(["Baseline, sample count: 10", "Sample count: 5", "Sample count: 7"])


    plt.title("Predictive sampling")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()



if __name__ == "__main__":
    test_convergence_cost()
            