# bayesian-microlensing
This is a Python library that uses a reversible-jump algorithm which adapts a suite of surrogate posteriors to infer the marginal model probabilities behind gravitational microlensing events.   

## An Example Posterior
To estimate the marginal model probabilities, the adaptive reversible-jump algorithm constructs a discrete posterior that is joint over all candidate models. For example, the following figure joins the single and binary lens models for microlensing:

<img src="source/figures/1-broccoli.png" width="300" height="300">

The data from this figure was constructed with **example_inference.py**

## Dependencies
To model microlensing events, install [MulensModel](https://rpoleski.github.io/MulensModel/install.html).

## Usage
To construct synthetic gravitational microlensing data and initialise a single (1L1S) model:
```python
import distributions
import light_curve_simulation
import sampling

# Generate a discrete synthetic single event with noise.
parameters = [15.0, 1.1, 32.0]
theta = sampling.State(truth=parameters)
n_obs = 720
snr = 23

data = light_curve_simulation.synthetic_single(theta, n_obs, snr)

# Create a list of single parameter prior distributions.
fs_pi = distributions.Uniform(0.1, 1)
t0_pi = distributions.Uniform(0, 72)
u0_pi = distributions.Uniform(0, 2)
tE_pi = distributions.Uniform(1, 100)

single_priors = [fs_pi, t0_pi, u0_pi, tE_pi]

# Initialise a single lens model Gaussian proposal distribution.
single_covariance = [[0.1, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.1, 0.0],
                     [0.0, 0.0, 0.0, 1.0]]

# Initialise a single lens model centre.
single_centre = sampling.State(truth=[0.9, 15.0, 1.1, 32.0])

# Initialise the single model.
m = 0
D = 4
single_Model = sampling.Model(m, D, single_centre, single_priors, single_covariance, \
                                  data, light_curve_simulation.single_log_likelihood)
```

To sample from a joint single (1L1S)/binary (2L1S) model posterior:

```python
# Create pool of models.
Models = [single_Model, binary_Model]

# Sample from the joint posterior.
joint_model_chain, total_acc_history, inter_model_acc_history = \
                               sampling.ARJMH(Models, iterations, warm_up_iterations)
```
For detailed use cases, see **example_inference.py**.

## Documentation
For detailed documentation, see https://dominickeehan.github.io/bayesian-microlensing/.

## Authors and Acknowledgements
Created by Dominic Keehan.

Joint project with Jack Yarndley.
