import numpy as np


class TrajectoryEvaluator:
    def stage_cost(self, x, u):
        """
        Stage cost for a pendulum swing-up problem
        xi = [pos, pos_dot, theta, theta_dot]
        Want to have theta = 0 and standing still
        Note that every multiple of 2p is also allowed, so we want cos(theta) = 1
        We also want to have the cart not moving too far
        J(xi) = (cos(theta)-1) + 1/2 * theta_dot^2 + 1/2 * x^2

        Args:
            x : shape (state_dim)
            u : 1
        Returns:
            cost : 1
        """
        return (1 - np.cos(x[2])) + 0.5 * x[0] ** 2

    def terminal_cost(self, x, u):
        """
        Terminal cost for a pendulum swing-up problem
        Args:
            x : shape (state_dim)
            u : 1
        """
        return 0

    def total_trajectory_cost(self, xs, us):
        """
        Total trajectory cost for a pendulum swing-up problem
        Args:
            xs (np.array) : (H, state_dim)
            us (np.array) : (H, 1)
        Returns:
            total_cost (float)

        """
        total_cost = 0
        for i in range(xs.shape[0]):  # (H,4)
            total_cost += self.stage_cost(xs[i, :], us[i])
        total_cost += self.terminal_cost(
            xs[-1, :], us[-1]
        )  # I know, the last step has stage + terminal
        return total_cost

    def evaluate_trajectories(self, state_trajs, control_trajs):
        """
        Evaluate a set of trajectories
        Args:
            state_trajs (np.array) : (N, H, state_dim)
            control_trajs (np.array) : (N, H, 1)
        Returns:
            costs (np.array) : (N, 1)
        """
        costs = np.zeros(state_trajs.shape[0])
        for i in range(state_trajs.shape[0]):
            costs[i] = self.total_trajectory_cost(
                state_trajs[i, :, :], control_trajs[i, :]
            )

        return costs

    def find_best_control(self, state_trajs, control_trajs):
        """
        Find the best control for a set of trajectories
        Args:
            state_trajs (np.array) : (N, H, state_dim)
            control_trajs (np.array) : (N, H, 1)
        Returns:
            best_control (np.array) : (H, 1)
            best_control_index (int)
        """
        costs = self.evaluate_trajectories(state_trajs, control_trajs)
        return control_trajs[np.argmin(costs), :], np.argmin(costs)


if __name__ == "__main__":
    # Test the stage cost
    x = np.array([0, 0, np.pi - 0.25, 0])
    u = np.array([0])
    evaluator = TrajectoryEvaluator()
    print(evaluator.stage_cost(x, u))

    # Test the total trajectory cost
    xs = np.array(
        [
            [0, 0, np.pi - 0.25, 0],
            [0, 0, 0, 0],
            [np.pi - 0.25, np.pi - 0.25, np.pi - 0.25, np.pi - 0.25],
            [0, 0, 0, 0],
        ]
    )
    us = np.array([0, 0, 0, 0])
    print(evaluator.total_trajectory_cost(xs, us))

    # Test the terminal cost,
    ones = np.ones((4, 4))
    zeros = np.zeros((4, 4))
    us = np.array([0, 0, 0, 0])
    print(
        evaluator.total_trajectory_cost(ones, us)
    )  # this should be lower than the next
    zeros[-1] = ones[-1]
    print(evaluator.find_best_control(np.array([zeros, ones]), np.array([us, us])))
