# behavior models

This is the readme for the behavioral models. We developped a series of models, for the moment there are 4:
- the exponential smoothing model based on the previous actions
- the exponential smoothing model based on the previous stimulus sides
- the optimal Bayesian model. This model performs inference in the generative Bayesian process with the correct parameters (mean block size=50, lower bound=20, upper bound=100, p(stimulus on left side | left block) = 0.8)
- the biased Bayesian model. This model performs inference in the generative Bayesian process with parameters fitted to the behavior (unstable)
- the biased Approximate Bayesian model. The model assumes a probability of change: at every trial, the probability that the block has changed is constant and fitted.
- the smoothing model which fits a kernel based on the previous simulus sides (unstable)

See the `main.py` file for an example on prior generation

In the `models` folder, you will find a file called `model.py` from which all models inherits. In this file, you will find all the methods to which you have access. The other files defines the specificities for each model.

The inference takes some minutes but once it has run (and has been saved automatically), computing the prior is very fast (which means you can run this prior computation on a lot of pseudoblocks)
