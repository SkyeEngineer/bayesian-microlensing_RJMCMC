"""Library of light curves from Evans, 2019.

Tests the ARJMH algorithm. Saves output.
"""


import sampling
import light_curve_simulation
import distributions
import random
import numpy as np
import surrogate_posteriors
import time


if __name__ == "__main__":


    """User Settings"""

    random.seed(42)

    # Synthetic light curve to generate.
    n_suite = 4
    n_epochs = 720
    sn_base = 23 #(230-23)/2 + 23 (lower = noisier).

    # Warm up parameters.
    fixed_warm_up_iterations = 250
    adaptive_warm_up_iterations = 750 #975
    warm_up_repititions = 1

    # Algorithm parameters.
    iterations = 10000 #20000

    # Output parameters.
    dpi = 100
    user_feedback = True


    """Sampling Process"""

    # Synthetic event parameters.
    model_parameters = [
        [0.5, 15, 0.1, 10, 0.01, 0.3, 60],  # 0
        [0.5, 15, 0.1, 10, 0.01, 0.4, 60],  # 1
        [0.5, 15, 0.1, 10, 0.01, 0.5, 60],  # 2
        [0.5, 15, 0.1, 10, 0.01, 0.6, 60],  # 3
        [0.5, 15, 0.1, 10, 0.01, 0.467, 60]]  # 4
    event_params = sampling.State(truth = model_parameters[n_suite])

    # Model index for parameters.
    model_types = [1, 1, 1, 1, 1]
    model_type = model_types[n_suite]

    # Generate synthetic light curve.
    if model_type == 0:
        data = light_curve_simulation.synthetic_single(event_params, n_epochs, sn_base)
    else: 
        data = light_curve_simulation.synthetic_binary(event_params, n_epochs, sn_base)

    # Informative priors in true space (Zhang et al).
    t0_pi = distributions.Uniform(0, 72)
    u0_pi = distributions.Uniform(0, 2)
    tE_pi = distributions.Truncated_Log_Normal(1, 100, 10**1.15, 10**0.45)
    q_pi = distributions.Log_Uniform(10e-6, 1)
    s_pi = distributions.Log_Uniform(0.2, 5)
    alpha_pi = distributions.Uniform(0, 360)
    fs_pi = distributions.Log_Uniform(0.1, 1)
    priors = [fs_pi, t0_pi, u0_pi, tE_pi, q_pi, s_pi, alpha_pi]

    # Get centre points and initial covariance from surrogate posteriors.
    # Drop the fourth parameter (finite source size).
    single_sp = surrogate_posteriors.Surrogate_Posterior(0, data.flux-1)
    single_sp.sample(50000)
    single_sp.get_modes()
    single_centre = sampling.State(truth=single_sp.modes[0][:4])

    binary_sp = surrogate_posteriors.Surrogate_Posterior(1, light_curve_simulation.synthetic_binary(event_params, 7200, sn_base).flux-1)
    binary_sp.sample(50000)
    binary_sp.get_modes()
    fin_rho = binary_sp.modes[0]
    binary_centre = sampling.State(truth = np.delete(binary_sp.modes[0], [4]))

    single_samples = single_sp.mode_samples[0]
    single_samples = np.delete(single_samples, [4], 1)    
    single_covariance = np.cov(single_samples, rowvar=False)

    binary_samples = binary_sp.mode_samples[0] 
    binary_samples = np.delete(binary_samples, [4], 1)
    binary_samples[:, 4] = np.log10(binary_samples[:, 4])
    binary_covariance = np.cov(binary_samples, rowvar=False)

    # Models.
    single_Model = sampling.Model(0, 4, single_centre, priors, single_covariance, data, light_curve_simulation.single_log_likelihood)
    binary_Model = sampling.Model(1, 7, binary_centre, priors, binary_covariance, data, light_curve_simulation.binary_log_likelihood)
    Models = [single_Model, binary_Model]

    # Run algorithm.
    start_time = (time.time())
    random.seed(42)
    joint_model_chain, MAPests, total_acc, inter_model_history = sampling.ARJMH(Models, iterations, adaptive_warm_up_iterations, fixed_warm_up_iterations, warm_up_repititions, user_feedback)
    duration = (time.time() - start_time)/60
    print(duration, ' minutes')
    single_Model, binary_Model = Models


    """Results"""

    # Text.
    labels = ['fs', 'Impact Time [days]', 'Minimum Impact Parameter', 'Einstein Crossing Time [days]', r'$log_{10}(Mass Ratio)$', 'Separation', 'Alpha']
    letters = ['fs', 't0', 'u0', 'tE', 'log10(q)', 's', 'a']
    symbols = [r'$f_s$', r'$t_0$', r'$u_0$', r'$t_E$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']
    names = ['1/1', '2/2', '3/3', '4/4', '5/5']
    name = names[n_suite]

    # Drop AMH samples.
    warm_up_iterations = adaptive_warm_up_iterations + fixed_warm_up_iterations
    binary_states = binary_Model.sampled.states_array(scaled = True)[:, warm_up_iterations:]
    single_states = single_Model.sampled.states_array(scaled = True)[:, warm_up_iterations:]

    # Drop finite source parameter and reorder for all surrogate posterior samples.
    single_sp_states = np.transpose(single_samples)
    
    binary_sp_samples = np.concatenate((np.array(binary_sp.samples)[:,-1:], np.array(binary_sp.samples)[:,:-1]), axis=1)
    binary_sp_samples = np.delete(binary_sp_samples, [4], 1)
    binary_sp_samples[:, 4] = np.log10(binary_sp_samples[:, 4])
    binary_sp_states = np.transpose(binary_sp_samples)

    # Save results.
    import pickle 
    object = [joint_model_chain, MAPests, binary_states, single_states, binary_sp_states, single_sp_states, warm_up_iterations, symbols, event_params, data, name, dpi]
    filehandler = open('results/'+name+'_stored_run.mcmc', 'wb') 
    pickle.dump(object, filehandler)
