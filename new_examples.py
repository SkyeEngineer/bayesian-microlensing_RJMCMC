"""Library of light curves from Evans, 2019.

Tests the ARJMH algorithm. Plots slices of the posterior and behavioural diagnostics.
"""

import sampling
import light_curve_simulation
import distributions
import autocorrelation as acf
import plotting as pltf
import random
import numpy as np
import matplotlib.pyplot as plt
import surrogate_posteriors
from copy import deepcopy
import time


if __name__ == "__main__":

    random.seed(42)



    """User Settings"""

    n_epochs = 720
    sn_base = 23 #(230-23)/2 + 23 (lower = noisier).



    event_params = sampling.State(truth = [15, 0.1, 10, 0.001, 0.3, 60])

    data = light_curve_simulation.synthetic_binary(event_params, n_epochs, sn_base)

    sp = surrogate_posteriors.Surrogate_Posterior(1, data.flux)
    sp.sample(10000)
    sp.get_modes()
    #sp.samples
    print(sp.modes)




    #samples = surrogate_posteriors.sample_posterior(sp, data.flux, 100000)

    #modes = surrogate_posteriors.get_modes(samples)
    #print(modes)