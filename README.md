# bayesian-microlensing
This is a Python library that uses an adaptive reversible-jump algorithm to infer the marginal model probabilities behind gravitational microlensing events.   

## An Example Posterior
![What is this](Networks.png)

## Dependencies
To model microlensing events, install [MulensModel](https://rpoleski.github.io/MulensModel/install.html). This works best in a Linux environment.

To use neural network functionality, a .pkl nueral net file is required. The location of this can be specified in **get_nueral_net.py**. For more information, see [Jack](https://github.com/jackyarndley/microlensing).

## Usage
To construct synthetic gravitational microlensing data and initialise a 1L1S model:
```python
import distributions
import light_curve_simulation
import sampling

# Generate a discrete synthetic 1L1S event with noise.
data = light_curve_simulation.synthetic_single([10.0, 1.0, 36.0], 720, 23)

# Create a list of 1L1S parameter prior distributions.
t0_pi = distributions.Uniform(0, 72)
u0_pi = distributions.Uniform(0, 2)
tE_pi = distributions.Uniform(1, 100)
1L1S_priors = [t0_pi, u0_pi, tE_pi]

# Initialise the 1L1S model Gaussian proposal distribution.
1L1S_covariance = [[1.0, 0.0, 0.0],
                   [0.0, 0.1, 0.0],
                   [0.0, 0.0, 1.0]]

# Initialise the 1L1S model centre.
1L1S_centre = sampling.State(truth=[15.0, 1.1, 32.0])

# Initialise the 1L1S model.
1L1S_Model = sampling.Model(0, 3, 1L1S_centre, 1L1S_priors, 1L1S_covariance, data, light_curve_simulation.single_log_likelihood)
```

To sample from a joint 1L1S/2L1S model posterior:

```python
# Create pool of models.
Models = [1L1S_Model, 2L1S_Model]

# Sample from the joint posterior.
joint_model_chain, acceptance_history = sampling.adapt_RJMH(Models, iterations)
```
For more detailed use cases, see **examples.py** and **robustness.py**.

## Authors and Acknowledgements
Created by Dominic Keehan.

Joint project with Jack Yarndley.

## License
[MIT](https://choosealicense.com/licenses/mit/)
