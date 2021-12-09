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

    # Synthetic light curve to generate.
    n_suite = 0
    n_epochs = 720
    sn_base = 23 #(230-23)/2 + 23 (lower = noisier).

    use_surrogate_posterior = True

    # Warm up parameters.
    fixed_warm_up_iterations = 25#25
    adaptive_warm_up_iterations = 975#975
    warm_up_repititions = 1#2

    # Algorithm parameters.
    iterations = 4000#20000

    # Output parameters.
    n_pixels = 2#25 # Density for posterior contour plot.
    dpi = 250
    user_feedback = True



    """Sampling Process"""

    # Synthetic event parameters.
    model_parameters = [
        [15, 0.1, 10, 0.00001, 0.3, 60],  # 0
        [15, 0.1, 10, 0.01, 0.3, 60],  # 1
        [15, 0.1, 10, 0.01, 0.5, 60],  # 2
        [15, 0.1, 10, 0.01, 0.7, 60]]  # 3
    event_params = sampling.State(truth = model_parameters[n_suite])

    # Model index for parameters.
    model_types = [1, 1, 1, 1]
    model_type = model_types[n_suite]

    # Generate synthetic light curve. Could otherwise use f.Read_Light_Curve(file_name).
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
    priors = [t0_pi, u0_pi, tE_pi, q_pi, s_pi, alpha_pi]


    # Get initial centre points.
    if use_surrogate_posterior == True:
        single_sp = surrogate_posteriors.Surrogate_Posterior(0, data.flux)
        single_sp.sample(50000)
        single_sp.get_modes()
        #print(single_sp.modes[0])
        single_centre = sampling.State(truth=single_sp.modes[0])#max_aposteriori())#single_sp.modes[0])
        #single_centre = sampling.State(truth = surrogate_posteriors.maximise_posterior(surrogate_posteriors.posterior(0), data.flux))
        binary_sp = surrogate_posteriors.Surrogate_Posterior(1, light_curve_simulation.synthetic_binary(event_params, 7200, sn_base).flux-1)
        binary_sp.sample(50000)
        binary_sp.get_modes()
        fin_rho = binary_sp.modes[0]#max_aposteriori()
        # Remove finite source size parameter from neural network.
        binary_centre = sampling.State(truth = np.array([fin_rho[0], fin_rho[1], fin_rho[2], fin_rho[4], fin_rho[5], fin_rho[6]]))
        fin_rho2 = binary_sp.modes[1]#max_aposteriori()
        # Remove finite source size parameter from neural network.
        ternary_centre = sampling.State(truth = np.array([fin_rho2[0], fin_rho2[1], fin_rho2[2], fin_rho2[4], fin_rho2[5], fin_rho2[6]]))


        #fin_rho = surrogate_posteriors.maximise_posterior(surrogate_posteriors.posterior(1), data.flux)
        # Remove finite source size parameter from neural network.
        #binary_centre = sampling.State(truth = np.array([fin_rho[0], fin_rho[1], fin_rho[2], fin_rho[4], fin_rho[5], fin_rho[6]]))

    else: # Use known values from previous maximising of posterior for centres.
        single_centres = [
        [15.0424, 0.1048, 10**1.0068], # 0
        [15.0245, 0.1035, 10**1.0063], # 1
        [15.0186, 0.1015, 10**1.0050], # 2
        [14.9966, 0.1020, 10**1.0043]] # 3
        single_centre = sampling.State(truth = np.array(single_centres[n_suite]))

        binary_centres = [
        [15.0424, 0.1048, 10**1.0068, 10**-2.4101, 10**-0.0311, 67.2737], # 0
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
    ternary_covariance = np.zeros((6, 6))
    np.fill_diagonal(ternary_covariance, np.multiply(covariance_scale, [1, 0.1, 1, 0.1, 0.1, 10]))

    # Models.
    single_Model = sampling.Model(0, 3, single_centre, priors, single_covariance, data, light_curve_simulation.single_log_likelihood)
    binary_Model = sampling.Model(1, 6, binary_centre, priors, binary_covariance, data, light_curve_simulation.binary_log_likelihood)
    ternary_Model = sampling.Model(2, 6, ternary_centre, priors, ternary_covariance, data, light_curve_simulation.binary_log_likelihood)
    Models = [single_Model, binary_Model, ternary_Model]

    # Run algorithm.
    start_time = (time.time())
    random.seed(42)
    joint_model_chain, total_acc, inter_model_history = sampling.ARJMH(Models, iterations, adaptive_warm_up_iterations, fixed_warm_up_iterations, warm_up_repititions, user_feedback)
    duration = (time.time() - start_time)/60
    print(duration, ' minutes')
    single_Model, binary_Model = Models




    """Plot Results"""

    # Initialise text.
    pltf.style()
    labels = ['Impact Time [days]', 'Minimum Impact Parameter', 'Einstein Crossing Time [days]', r'$log_{10}(Mass Ratio)$', 'Separation', 'Alpha']
    letters = ['t0', 'u0', 'tE', 'log10(q)', 's', 'a']
    symbols = [r'$t_0$', r'$u_0$', r'$t_E$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']
    shifted_symbols = [r'$t_0-\hat{\theta}$', r'$u_0-\hat{\theta}$', r'$t_E-\hat{\theta}$', r'$\rho-\hat{\theta}$', r'$log_{10}(q)-\hat{\theta}$', r'$s-\hat{\theta}$', r'$\alpha-\hat{\theta}$']
    names = ['1/1', '2/2', '3/3', '4/4']
    name = names[n_suite]

    #pltf.adaption_contraction(binary_Model, adapt_MH_warm_up+adapt_MH+iterations, name+'-binary', dpi)
    #pltf.adaption_contraction(single_Model, adapt_MH_warm_up+adapt_MH+iterations, name+'-single', dpi)
    pltf.adaption_contraction(inter_model_history, iterations, name+'-inter', dpi)

    acf.plot_act(joint_model_chain, symbols, name, dpi)
    #acf.attempt_truncation(Models, joint_model_chain)
    sampling.output_file(Models, adaptive_warm_up_iterations + fixed_warm_up_iterations, joint_model_chain, total_acc, n_epochs, sn_base, letters, name, event_params)

    # Trace of model index.
    plt.plot(np.linspace(0, joint_model_chain.n, joint_model_chain.n), joint_model_chain.model_indices, linewidth = 0.25, color = 'purple')
    plt.xlabel('Samples')
    plt.ylabel(r'$m_i$')
    plt.locator_params(axis = "y", nbins = 2) # only two ticks
    plt.savefig('results/'+name+'-mtrace.png', bbox_inches = 'tight', dpi = dpi, transparent=True)
    plt.clf()

    pltf.density_heatmaps(binary_Model, n_pixels, data, symbols, event_params, 1, name, dpi)
    pltf.joint_samples_pointilism(binary_Model, single_Model, joint_model_chain, symbols, name, dpi)
    #pltf.centre_offsets_pointilism(binary_Model, single_Model, shifted_symbols, name, dpi)
