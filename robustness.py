"""Analyses the robustness of the ARJMH algorithm.

Calculates the expected marginalised binary model probability of some ambiguous light curves.
"""

import MulensModel as mm
import sampling
import light_curve_simulation
import distributions
import autocorrelation as acf
import plotting as pltf
import random
import numpy as np
import matplotlib.pyplot as plt
import surrogate_posterior
from copy import deepcopy

import time




def P_m2(event_params, sn_base, n_epochs):
    """Calculates the marginalised binary model probability for a specefic light curve.

    Args:
        event_params: [list] Single lens model parameters.
        sn_base: [float] Signal to noise baseline of data.
        n_epochs: [int] Number of observations.

    Returns:
        P_m2: [float] Marginalised binary model probability.
    """


    """User Settings"""

    # Synthetic light curve to generate.
    n_suite = 0

    use_surrogate_posterior = False#True

    # Warm up parameters.
    fixed_warm_up_iterations = 25#25
    adaptive_warm_up_iterations = 75#975
    warm_up_repititions = 1#2

    # Algorithm parameters.
    iterations = 100#20000

    # Output parameters.
    truncate = False # Truncate a burn in period based on IACT.
    user_feedback = True

    """Sampling Process"""

    # Generate synthetic light curve. Could otherwise use f.Read_Light_Curve(file_name).
    data = light_curve_simulation.synthetic_single(event_params, n_epochs, sn_base)

    # Informative priors in true space (Zhang et al).
    t0_pi = distributions.Uniform(0, 72)
    u0_pi = distributions.Uniform(0, 2)
    tE_pi = distributions.Truncated_Log_Normal(1, 100, 10**1.15, 10**0.45)
    q_pi = distributions.Log_Uniform(10e-6, 1)
    s_pi = distributions.Log_Uniform(0.2, 5)
    alpha_pi = distributions.Uniform(0, 360)
    priors = [t0_pi, u0_pi, tE_pi, q_pi, s_pi, alpha_pi]


    # Get initial centre points.
    if use_surrogate_posterior == True:
        single_centre = sampling.State(truth = surrogate_posterior.maximise_posterior(surrogate_posterior.posterior(0), data.flux))
        fin_rho = surrogate_posterior.maximise_posterior(surrogate_posterior.posterior(1), data.flux)
        # Remove finite source size parameter from neural network.
        binary_centre = sampling.State(truth = np.array([fin_rho[0], fin_rho[1], fin_rho[2], fin_rho[4], fin_rho[5], fin_rho[6]]))

    else: # Use known values for centres.
        single_centres = [
        [15.0245, 0.1035, 10**1.0063], # 0
        [15.0245, 0.1035, 10**1.0063], # 1
        [15.0245, 0.1035, 10**1.0063], # 2
        [15.0245, 0.1035, 10**1.0063]] # 3
        single_centre = sampling.State(truth = np.array(single_centres[n_suite]))

        binary_centres = [
        [1.50424747e+01, 1.04854599e-01, 1.00131283e+01, 4.51699379e-05, 9.29979384e-01, 6.72737579e+01], # 0
        [15.0245, 0.1035, 10**1.0063, 10**-2.3083, 10**0.5614, 161.0036],    # 1
        [15.0186, 0.1015, 10**1.0050, 10**-1.9734, 10**-0.3049, 60.4598],  # 2
        [14.9966, 0.1020, 10**1.0043, 10**-1.9825, 10**-0.1496, 60.2111]]   # 3
        binary_centre = sampling.State(truth = np.array(binary_centres[n_suite]))

    # Initial diagonal covariances.
    covariance_scale = 0.001 # Reduce values by scalar
    single_covariance = np.zeros((3, 3))
    np.fill_diagonal(single_covariance, np.multiply(covariance_scale, [1, 0.1, 1]))
    binary_covariance = np.zeros((6, 6))
    np.fill_diagonal(binary_covariance, np.multiply(covariance_scale, [1, 0.1, 1, 0.1, 0.1, 10]))

    # Models.
    single_Model = sampling.Model(0, 3, single_centre, priors, single_covariance, data, light_curve_simulation.single_log_likelihood)
    binary_Model = sampling.Model(1, 6, binary_centre, priors, binary_covariance, data, light_curve_simulation.binary_log_likelihood)
    Models = [single_Model, binary_Model]

    # Run algorithm.
    start_time = (time.time())
    random.seed(42)
    joint_model_chain, total_acc, inter_model_history = sampling.ARJMH(Models, iterations, adaptive_warm_up_iterations, fixed_warm_up_iterations, warm_up_repititions, user_feedback)
    duration = (time.time() - start_time)/60
    print(duration, ' minutes')
    single_Model, binary_Model = Models

    P_m2 = np.sum(joint_model_chain.model_indices)/joint_model_chain.n

    return P_m2




"""Expectation Process"""

n = 5 # Number of samples in expectation.

# Parameter range to vary.
theta = [36, 1.0, 5.5]
tE_pi = distributions.Truncated_Log_Normal(1, 100, 10**1.15, 10**0.45)
tE_range = np.linspace(5, 10, n)

with open("results/robustness-run.txt", "w") as file:

    for n_epochs in [720, 360, 72]: # Cadence to calculate expectations at.
        for sn_base in [23, 126.5, 230]: # Noise to calculate expectations at.
            
            # Initialise for current expectation.
            prior_density = []
            binary_probability =[]

            for i in range(n): # Samples in expectation.
                tE = tE_range[i]
                prior_density.append(np.exp(tE_pi.log_pdf(tE)))

                # Create new light curve and run ARJMH.                
                theta_tE = deepcopy(theta)
                theta_tE[2] = tE
                event_params = sampling.State(truth = theta_tE)
                binary_probability.append(P_m2(event_params, sn_base, n_epochs))

                # Prior density weighted expectation and standard deviation. 
                EPM2 = sum([a*b for a,b in zip(binary_probability, prior_density)])/sum(prior_density)
                sdEPM2 = (sum([(a-EPM2)**2*b for a,b in zip(binary_probability, prior_density)])/sum(prior_density))**0.5
            
            # Store expectation.
            file.write("n_epochs: "+str(n_epochs)+" sn_base: "+str(sn_base)+" E: "+str(EPM2)+" sd+-: "+str(sdEPM2)+"\n")