# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Main]



import MulensModel as mm
import sampling
import light_curve_simulation
import distributions
import autocorrelation as acf
import plotting as pltf
import random
import numpy as np
import matplotlib.pyplot as plt
#import get_neural_network as neural_net
from copy import deepcopy

import time



#-----------
## INPUTS ##
#-----------


suite_n = 0
adapt_MH_warm_up = 25 #25 # mcmc steps without adaption
adapt_MH = 975  #475 # mcmc steps with adaption
initial_n = 1 # times to repeat mcmc optimisation of centers to try to get better estimate
iterations = 1000 # rjmcmc steps
n_epochs = 720
epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
sn_base = 23 #(230-23)/2 + 23 # np.random.uniform(23.0, 230.0) # lower means noisier
n_pixels = 5 # density for posterior contour plot
n_sampled_curves = 5 # sampled curves for viewing distribution of curves
uniform_priors = False 
informative_priors = True
use_neural_net = False # use neural net to get maximum aposteriori estimate for centreing points
dpi = 250
user_feedback = True

#---------------
## END INPUTS ##
#---------------
names = ['1/1', '2/2', '3/3', '4/4']
name = names[suite_n]

# GENERATE DATA
# synthetic event parameters
model_parameters = [
    [15, 0.1, 10, 0.01, 0.2, 60],                  # 0
    [15, 0.1, 10, 0.01, 0.3, 60],   # 1
    [15, 0.1, 10, 0.01, 0.5, 60],  # 2
    [15, 0.1, 10, 0.01, 0.7, 60]]  # 3
event_params = sampling.State(truth = model_parameters[suite_n])

model_types = [0, 1, 1, 1] # model type associated with synethic event suite above
model_type = model_types[suite_n]
# store a synthetic lightcurve. Could otherwise use f.Read_Light_Curve(file_name)
if model_type == 0:
    data = light_curve_simulation.synthetic_single(event_params, n_epochs, sn_base)
else: 
    data = light_curve_simulation.synthetic_binary(event_params, n_epochs, sn_base)


# SET PRIORS
# priors in truth space informative priors (Zhang et al)
t0_pi = distributions.Uniform(0, 72)
u0_pi = distributions.Uniform(0, 2)
tE_pi = distributions.Truncated_Log_Normal(1, 100, 10**1.15, 10**0.45)
q_pi = distributions.Log_Uniform(10e-6, 1)
s_pi = distributions.Log_Uniform(0.2, 5)
alpha_pi = distributions.Uniform(0, 360)
priors = [t0_pi, u0_pi, tE_pi, q_pi, s_pi, alpha_pi]


# GET CENTERS
if use_neural_net == True:

    # centreing points for inter-model jumps
    single_center = sampling.State(truth = neural_net.get_model_centers(neural_net.get_posteriors(0), data.flux))
    fin_rho = neural_net.get_model_centers(neural_net.get_posteriors(1), data.flux)
    binary_center = sampling.State(truth = np.array([fin_rho[0], fin_rho[1], fin_rho[2], fin_rho[4], fin_rho[5], fin_rho[6]]))

else: # use known values for centers 
    single_centers = [
    [15.0245, 0.1035, 10**1.0063], # 0
    [15.0245, 0.1035, 10**1.0063], # 1
    [15.0245, 0.1035, 10**1.0063], # 2
    [15.0245, 0.1035, 10**1.0063]] # 3
    single_center = sampling.State(truth = np.array(single_centers[suite_n]))

    binary_centers = [
    [1.50424747e+01, 1.04854599e-01, 1.00131283e+01, 4.51699379e-05, 9.29979384e-01, 6.72737579e+01], # 0
    [15.0245, 0.1035, 10**1.0063, 10**-2.3083, 10**0.5614, 161.0036],    # 1
    [15.0186, 0.1015, 10**1.0050, 10**-1.9734, 10**-0.3049, 60.4598],  # 2
    [14.9966, 0.1020, 10**1.0043, 10**-1.9825, 10**-0.1496, 60.2111]]   # 3
    binary_center = sampling.State(truth = np.array(binary_centers[suite_n]))


# MODEL COVARIANCES
# initial covariances (diagonal)
covariance_scale = 0.001 # reduce diagonals by a multiple
single_covariance = np.zeros((3, 3))
np.fill_diagonal(single_covariance, np.multiply(covariance_scale, [1, 0.1, 1]))
binary_covariance = np.zeros((6, 6))
np.fill_diagonal(binary_covariance, np.multiply(covariance_scale, [1, 0.1, 1, 0.1, 0.1, 10]))

# MODELS
single_Model = sampling.Model(0, 3, single_center, priors, single_covariance, data, light_curve_simulation.single_log_likelihood)
binary_Model = sampling.Model(1, 6, binary_center, priors, binary_covariance, data, light_curve_simulation.binary_log_likelihood)
Models = [single_Model, binary_Model]

start_time = (time.time())
random.seed(42)
joint_model_chain, total_acc, inter_info = sampling.adapt_RJMH(Models, adapt_MH_warm_up, adapt_MH, initial_n, iterations, user_feedback)
duration = (time.time() - start_time)/60
print(duration, ' minutes')
single_Model, binary_Model = Models

#-----------------
## PLOT RESULTS ##
#-----------------

#print(single_Model.acc, len(single_Model.acc))
#print(Models[0].acc, len(Models[0].acc))
# plotting resamplings
pltf.style()
labels = ['Impact Time [days]', 'Minimum Impact Parameter', 'Einstein Crossing Time [days]', r'$log_{10}(Mass Ratio)$', 'Separation', 'Alpha']
letters = ['t0', 'u0', 'tE', 'log10(q)', 's', 'a']
symbols = [r'$t_0$', r'$u_0$', r'$t_E$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']
shifted_symbols = [r'$t_0-\hat{\theta}$', r'$u_0-\hat{\theta}$', r'$t_E-\hat{\theta}$', r'$\rho-\hat{\theta}$', r'$log_{10}(q)-\hat{\theta}$', r'$s-\hat{\theta}$', r'$\alpha-\hat{\theta}$']

#pltf.adaption_contraction(binary_Model, adapt_MH_warm_up+adapt_MH+iterations, name+'-binary', dpi)
#pltf.adaption_contraction(single_Model, adapt_MH_warm_up+adapt_MH+iterations, name+'-single', dpi)
pltf.adaption_contraction(inter_info, iterations, name+'-inter', dpi)



acf.plot_act(joint_model_chain, symbols, name, dpi)
#acf.attempt_truncation(Models, joint_model_chain)
sampling.output_file(Models, adapt_MH_warm_up + adapt_MH, joint_model_chain, total_acc, n_epochs, sn_base, letters, name, event_params)

# trace of model index
plt.plot(np.linspace(0, joint_model_chain.n, joint_model_chain.n), joint_model_chain.model_indices, linewidth = 0.25, color = 'purple')
plt.xlabel('Samples')
plt.ylabel(r'$m_i$')
plt.locator_params(axis = "y", nbins = 2) # only two ticks
plt.savefig('results/'+name+'-mtrace.png', bbox_inches = 'tight', dpi = dpi, transparent=True)
plt.clf()

pltf.density_heatmaps(binary_Model, n_pixels, data, event_params, symbols, 1, name, dpi)
pltf.joint_samples_pointilism(binary_Model, single_Model, joint_model_chain, symbols, name, dpi)
#pltf.center_offsets_pointilism(binary_Model, single_Model, shifted_symbols, name, dpi)
