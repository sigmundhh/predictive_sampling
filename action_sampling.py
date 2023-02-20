import torch


class ActionSampler:
    def __init__(self, horizon, variance, mean=None):
        """Initialize the action sampler.
        Args:
            horizon (int): The number of time steps to sample actions for.
            variance (float): The variance of the gaussian distribution.
            mean (torch.tensor): The mean of the gaussian distribution. Shape: (horizon,).
        """
        self.horizon = horizon
        if mean is None:
            mu = torch.zeros(horizon)
        else:
            mu = mean
        sigma = torch.eye(horizon) * variance
        self.distr = torch.distributions.MultivariateNormal(mu, sigma)

    def sample(self):
        """Sample a single action from the gaussian distribution.
        The action is clamped to be between -1 and 1.
        Returns:
            torch.tensor: A single action sampled from the gaussian distribution. Shape: (horizon,).
        """
        return torch.clamp(self.distr.sample(), -1, 1)

    def sample_batch(self, batch_size):
        """
        Sample a batch of actions from the gaussian distribution.
        The action is clamped to be between -1 and 1.
        Args:
            batch_size (int): The number of actions to sample.
        Returns:
            torch.tensor: A batch of actions sampled from the gaussian distribution. Shape: (batch_size, horizon).
        """

        if batch_size == 1:
            # If we only want one sample, just return the mean
            return self.distr.loc.unsqueeze(0)
        else:
            # If we want more than one sample, return the mean
            # and then sample the rest
            return torch.cat(
                (
                    self.distr.loc.unsqueeze(0),
                    torch.clamp(self.distr.sample((batch_size - 1,)), -1, 1),
                )
            )

    def update_mu(self, mu, shift=True):
        """Update the mean of the gaussian distribution.
        Args:
            mu (torch.tensor): The new mean of the gaussian distribution. Shape: (horizon,).
            shift (bool, optional): Whether to shift the mean. Defaults to True.
        """
        if shift:
            self.distr.loc = torch.cat((mu[1:], torch.zeros(1)))
        else:
            self.distr.loc = mu


if __name__ == "__main__":
    action_sampler = ActionSampler(horizon=10, variance=10)
    ## print the dimensions of the sampled action
    print(action_sampler.sample_batch(3).size())
