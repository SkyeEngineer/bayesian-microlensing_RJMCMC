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

    #throw=throw

    """User Settings"""

    n_epochs = 720
    sn_base = 23 #(230-23)/2 + 23 (lower = noisier).



    event_params = sampling.State(truth = [15, 0.1, 10, 0.0001, 0.2, 60])

    data = light_curve_simulation.synthetic_binary(event_params, n_epochs, sn_base)




    # Informative priors in true space (Zhang et al).
    t0_pi = distributions.Uniform(0, 72)
    u0_pi = distributions.Uniform(0, 2)
    tE_pi = distributions.Truncated_Log_Normal(1, 100, 10**1.15, 10**0.45)
    q_pi = distributions.Log_Uniform(10e-6, 1)
    s_pi = distributions.Log_Uniform(0.2, 5)
    alpha_pi = distributions.Uniform(0, 360)
    priors = [t0_pi, u0_pi, tE_pi, q_pi, s_pi, alpha_pi]






    single_sp = surrogate_posteriors.Surrogate_Posterior(0, data.flux)
    single_sp.sample(10000)
    single_sp_avg = sampling.State(truth=np.mean(single_sp.samples.numpy(), 0))

    single_sp.get_modes()
    #print(single_sp.modes[0])
    single_centre = sampling.State(truth=single_sp.modes[0])#max_aposteriori())
    #print(single_sp.samples.numpy())
    single_covariance = np.cov(single_sp.samples.numpy(), rowvar=False)/10e15
    print(np.linalg.det(single_covariance))
    #throw=throw

    single_Model = sampling.Model(0, 3, single_centre, priors, single_covariance, data, light_curve_simulation.single_log_likelihood)
    single_Model.scaled_avg_state = single_sp_avg.scaled
    single_Model.sampled.n = 100
    single_Model.sampled.add_general_state(0, sampling.State(truth=deepcopy(single_sp.samples.numpy()[-1])))



    binary_sp = surrogate_posteriors.Surrogate_Posterior(1, light_curve_simulation.synthetic_binary(event_params, 7200, sn_base).flux-1)
    binary_sp.sample(50000)
    binary_sp.get_modes()
    #print(binary_sp.modes)
    #print(binary_sp.mode_samples)
    #print(binary_sp.mode_samples[0])
    #print(binary_sp.mode_samples[1])
    #throw=throw
    
    fin_rho_0 = binary_sp.modes[0]#max_aposteriori()
    #print(binary_sp.)
    # Remove finite source size parameter from neural network.
    binary_centre_0 = sampling.State(truth = np.array([fin_rho_0[0], fin_rho_0[1], fin_rho_0[2], fin_rho_0[4], fin_rho_0[5], fin_rho_0[6]]))
    #print(binary_centre_0)

    binary_samples_0 = binary_sp.mode_samples[0]
    binary_samples_0[:, 4] = np.log10(binary_samples_0[:, 4])
    binary_samples_0 = np.delete(binary_samples_0, [3, 7], 1)
    print(binary_samples_0[1])
    #throw=throw
    binary_covariance_0 = np.cov(binary_samples_0, rowvar=False)/10e15
    print(np.linalg.det(binary_covariance_0))
#print(np.mean(binary_samples, 0))
    #throw=throw
    binary_sp_avg_0 = sampling.State(scaled=np.mean(binary_samples_0, 0))

    binary_Model_0 = sampling.Model(1, 6, binary_centre_0, priors, binary_covariance_0, data, light_curve_simulation.binary_log_likelihood)
    binary_Model_0.scaled_avg_state = binary_sp_avg_0.scaled
    binary_Model_0.sampled.n = 100    
    binary_Model_0.sampled.add_general_state(1, sampling.State(scaled=deepcopy(binary_samples_0[-1])))


    Models = [single_Model, binary_Model_0]#, binary_Model_1]

    # Run algorithm.
    start_time = (time.time())
    random.seed(42)
    joint_model_chain, total_acc, inter_model_history = sampling.ARJMH(Models, 2000, 1, 1, 1, user_feedback=True)
    duration = (time.time() - start_time)/60
    print(duration, ' minutes')
    single_Model, binary_Model = Models