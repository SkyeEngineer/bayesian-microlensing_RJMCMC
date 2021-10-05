"""Interface with neural network surrogate posteriors."""


import pickle
import numpy as np

# File access. 
import os
import os.path
from pathlib import Path


def posterior(m):
    """Get a single or binary model posterior.
        
    Args:
        m: [int] Model index, single or binary, 0 or 1.

    Returns:
        posterior: [pickle] Posterior object.
    """
    path = os.getcwd()
    #path = (str(Path(path).parents[0]))

    if m == 0:
        with open(path+"/surrogate_posteriors/single_25K_720.pkl", "rb") as handle: posterior = pickle.load(handle)

    if m == 1:
        with open(path+"/surrogate_posteriors/binary_100K_720.pkl", "rb") as handle: posterior = pickle.load(handle)
    
    return posterior



def maximise_posterior(posterior, signal_data):
    """Maximise a posterior.
        
    The input signal_data conditions the posterior to data.

    Args:
        posterior: [pickle] Posterior object.
        signal_data: [list] Measured flux signals at discrete times.

    Returns:
        centre: [list] Estimated parameter values of maximum.
    """
    centre = np.array(np.float64(posterior.map(signal_data, num_iter = 10, num_init_samples = 10, show_progress_bars = False)))
    
    
    print(centre)

    return centre