# Predictive Sampling
An minimal implementation of Predictive Sampling, as described in "Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo" by DeepMind. 
The repo is constructed primarily for a personal introduction to MPC and it's strengths and weaknesses.

## Cartpole experiment
Using the cart pole environment from this [repo](https://github.com/0xangelo/gym-cartpole-swingup) and a dynamical model from [Matthew Kelly's tutorial](http://www.matthewpeterkelly.com/tutorials/cartPole/index.html), I made an implementation of Predictive Sampling for this setup.

The script shows the environment along with the predicted trajectories:

<img width="600" alt="env_screenshot" src="https://user-images.githubusercontent.com/42750085/218702243-111f4403-c79f-405b-890b-60869fcfc2b3.png">

![mpc_plot](https://user-images.githubusercontent.com/42750085/218702286-6233459e-03e5-4986-834b-5e99b18a9f57.png)

### Remaining challenges
There are a number of things that inhibit control:
- The physical properties (lengths and weights) might not reflect the environment
- The numerical solution of the dynamics might introduce instability
- There are multiple tunable parameters in the Predictive Sampling algorithm:
  - The sampling variance
  - The rollout horizon
  - The stage and terminal cost function
  - Number of samples pr. iteration


