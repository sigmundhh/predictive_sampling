import torch

class actionSampler:
    def __init__(self, horizon, variance, mean=None):
        """Initialize the action sampler.
        Args:
            horizon (int): The number of time steps to sample actions for.  
            variance (float): The variance of the gaussian distribution.
            mean (torch.tensor): The mean of the gaussian distribution. Shape: (horizon,)."""
        self.horizon = horizon
        # Parameters for an n-dimensional gaussian
        if mean is None:
            mu = torch.zeros(horizon)
        else:
            mu = mean
        sigma = torch.eye(horizon)*variance
        # Create a gaussian we can sample
        self.distr = torch.distributions.MultivariateNormal(mu, sigma)

    def sample(self):
        # Sample from the gaussian
        return self.distr.sample()

    def sample_batch(self, batch_size):
        # Sample from the gaussian
        # Always want to include the mean as a sample
        # so we can use it as a baseline

        if batch_size == 1:
            # If we only want one sample, just return the mean
            return self.distr.loc.unsqueeze(0)
        else:
            # If we want more than one sample, return the mean
            # and then sample the rest
            return torch.cat((self.distr.loc.unsqueeze(0), self.distr.sample((batch_size-1,))))

    def update_mu(self, mu):
        # Update the mean of the gaussian
        # To incorporate that the first action of mu is already done, 
        # we just want to sample using the last N-1
        # and appending a 0 to the end of the mean

        self.distr.loc = torch.cat((mu[1:], torch.zeros(1)))


if __name__ == "__main__":
    action_sampler = actionSampler(horizon=10, variance=10)
    ## print the dimensions of the sampled action
    print(action_sampler.sample_batch(3).size())

