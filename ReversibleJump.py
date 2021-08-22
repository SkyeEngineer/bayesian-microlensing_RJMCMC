# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Main]

import math
from pickle import FALSE

from numpy.core.numeric import Inf
import MulensModel as mm
import Functions as f
import Autocorrelation as AC
import PlotFunctions as pltf
import emcee as MC
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
import interfaceing as interf
from multiprocessing import Pool
from scipy.optimize import minimize
from copy import deepcopy

from scipy.stats import chi2
import scipy

import PlotFunctions as pltf

import os
import os.path
import shutil
from pathlib import Path


#-----------
## INPUTS ##
#-----------

suite_n = 1

adaptive_warmup_iterations = 25 # mcmc steps without adaption
adaptive_iterations = 1975 # mcmc steos with adaption
warmup_loops = 1 # times to repeat mcmc optimisation of centers to try to get better estimate
iterations = 4000 # rjmcmc steps

n_epochs = 720
epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]

signal_to_noise_baseline = (230-23)/2 + 23 # np.random.uniform(23.0, 230.0) # lower means noisier

n_points = 2 # density for posterior contour plot
n_sampled_curves = 5 # sampled curves for viewing distribution of curves

uniform_priors = False 
informative_priors = True

sbi = False # use neural net to get maximum aposteriori estimate for centreing points

truncate = False # automatically truncate burn in period based on autocorrelation of m

#---------------
## END INPUTS ##
#---------------

## INITIALISATION ##

# synthetic event parameters
model_parameter_suite = [
    [0.1, 36, 0.833, 31.5, 0.0096], # 0 single
    [0.8, 36, 0.833, 31.5, 0.0096, 0.0001, 1.27, 210.8], # 1 weak binary
    [0.1, 36, 0.833, 31.5, 0.001, 0.03, 1.10, 180] # 2 caustic crossing binary
    ]
model_type_suite = [0, 1, 1]


light_curve_type = model_type_suite[suite_n]
true_theta = np.array(model_parameter_suite[suite_n])


if light_curve_type == 0: single_true = f.scale(true_theta)
else: single_true = False

if light_curve_type == 1: binary_true = f.scale(true_theta)
else: binary_true = False

# store a synthetic lightcurve. Could otherwise use f.Read_Light_Curve(file_name)
data = f.Synthetic_Light_Curve(true_theta, light_curve_type, n_epochs, signal_to_noise_baseline)


# priors in true space
if informative_priors == True:

    # informative priors (Zhang et al)
    s_pi = f.logUniDist(0.2, 5)
    q_pi = f.logUniDist(10e-6, 1)
    #log_q_pi = f.uniDist(np.log10(10e-6), np.log10(1))
    alpha_pi = f.uniDist(0, 360)
    u0_pi = f.uniDist(0, 2)
    t0_pi = f.uniDist(0, 72)
    tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
    rho_pi =  f.logUniDist(10**-4, 10**-2)
    fs_pi = f.logUniDist(0.1, 1)

    priors = [fs_pi, t0_pi, u0_pi, tE_pi, rho_pi, q_pi, s_pi, alpha_pi]

elif uniform_priors == True:
    
    # uninformative priors
    s_upi = f.uniDist(0.2, 3)
    q_upi = f.uniDist(10e-6, 0.1)
    alpha_upi = f.uniDist(0, 360)
    u0_upi = f.uniDist(0, 2)
    t0_upi = f.uniDist(0, 72)
    tE_upi = f.uniDist(1, 100)
    rho_upi =  f.uniDist(10**-4, 10**-2)
    
    priors = [t0_upi, u0_upi,  tE_upi, rho_upi,  q_upi, s_upi, alpha_upi]


if sbi == True:

    # get nueral network posteriors for each model
    single_surrogate_posterior = interf.get_posteriors(0)
    binary_surrogate_posterior = interf.get_posteriors(1)

    # centreing points for inter-model jumps
    single_center = interf.get_model_centers(single_surrogate_posterior, data.flux)
    binary_center = interf.get_model_centers(binary_surrogate_posterior, data.flux)

else: 
    
    # use known values for centers 
    single_center = np.array([0.8, 36, 0.833, 31.5, 0.0096])
    binary_center = np.array([0.8, 36, 0.833, 31.5, 0.0096, 0.0001, 1.27, 210.8])

# plot unoptimised centers
#pltf.Light_Curve_Fit_Error(0, single_center, priors, data, True, "SingleCenterSurr")
#pltf.Light_Curve_Fit_Error(1, binary_center, priors, data, True, "BinaryCenterSurr")


# initial covariances (diagonal)
covariance_scale = 0.001 # reduce diagonals by a multiple
single_covariance = np.zeros((f.D(0), f.D(0)))
np.fill_diagonal(single_covariance, np.multiply(covariance_scale, [0.01, 0.1, 0.01, 0.1, 0.0001]))
binary_covariance = np.zeros((f.D(1), f.D(1)))
np.fill_diagonal(binary_covariance, np.multiply(covariance_scale, [0.01, 0.1, 0.01, 0.1, 0.0001, 0.1, 0.01, 1]))

# use adaptiveMCMC to calculate initial covariances and optimise centers
w_single_covariance, w_s_chain_states, w_s_chain_means, w_s_acceptance_history, w_s_covariance_history, w_s_best_posterior, w_s_best_theta =\
    f.Loop_Adaptive_Warmup(warmup_loops, 0, data, single_center, priors, single_covariance, adaptive_warmup_iterations, adaptive_iterations)
w_binary_covariance, w_b_chain_states, w_b_chain_means, w_b_acceptance_history, w_b_covariance_history, w_b_best_posterior, w_b_best_theta =\
    f.Loop_Adaptive_Warmup(warmup_loops, 1, data, binary_center, priors, binary_covariance, adaptive_warmup_iterations, adaptive_iterations)

# plot optimised centers
#pltf.LightcurveFitError(2, f.unscale(2, bestt_2), priors, Data, Model, epochs, error, True, "BinaryCenterMCMC")
#pltf.LightcurveFitError(1, bestt_1, priors, Data, Model, epochs, error, True, "SingleCenterMCMC")


# Load resources for RJMCMC
centers = [w_s_best_theta, w_b_best_theta]
initial_states = [w_s_chain_states[:, -1], w_b_chain_states[:, -1]]
initial_means = [w_s_chain_means[:, -1], w_b_chain_means[:, -1]]
n_warmup_iterations = adaptive_warmup_iterations + adaptive_iterations
initial_covariances = [w_single_covariance, w_binary_covariance]

# run RJMCMC
chain_states, chain_ms, best_thetas, best_pi, cov_histories, acc_history, inter_j_acc_histories, intra_j_acc_histories, inter_cov_history =\
    f.Run_Adaptive_RJ_Metropolis_Hastings(initial_states, initial_means, n_warmup_iterations, initial_covariances, centers, priors, iterations, data)


#-----------------
## PLOT RESULTS ##
#-----------------

# plotting resources
pltf.Style()
labels = ['Source Flux Fraction', 'Impact Time [days]', 'Minimum Impact Parameter', 'Einstein Crossing Time [days]', 'Rho', r'$log_{10}(Mass Ratio)$', 'Separation', 'Alpha']
symbols = [r'$f_s$', r'$t_0$', r'$u_0$', r'$t_E$', r'$\rho$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']
letters = ['fs', 't0', 'u0', 'tE', 'p', 'log10(q)', 's', 'a']
marker_size = 75

# construct the generalised state signal to analyse
auxiliary_states = []
auxiliary_states.append(initial_states[1])
print(initial_states[1])

for i in range(1, iterations):
    if chain_ms[i] == 0: # fill most recent binary non shared parameters if single
        auxiliary_states.append(np.concatenate((chain_states[i], auxiliary_states[i - 1][f.D(0):])))

    if chain_ms[i] == 1: # currently binary
        auxiliary_states.append(chain_states[i])

auxiliary_states = np.array(auxiliary_states)

# truncate once m below 50 auto correlation times
if truncate == True:
    n_ac = 25
    N = np.exp(np.linspace(np.log(int(iterations/n_ac)), np.log(iterations), n_ac)).astype(int)

    ac_time_m = np.zeros(len(N))
    y_m = np.array(chain_ms)

    for i, n in enumerate(N):
        ac_time_m[i] = MC.autocorr.integrated_time(y_m[:n], c = 5, tol = 5, quiet = True)
        
        if ac_time_m[i] < N[i]/50: # linearly interpolate truncation point
            if i == 0:
                truncated = N[i]
            else:
                slope = (ac_time_m[i] - ac_time_m[i-1]) / (N[i] - N[i-1])
                truncated = int(math.ceil((ac_time_m[i] - slope * N[i]) / (1/50 - slope)))
            break

        truncated = math.nan

else: truncated = 0


# construct untruncated auto correlation functions
n_ac = 10
N = np.exp(np.linspace(np.log(int(iterations/n_ac)), np.log(iterations), n_ac)).astype(int)

for p in range(f.D(1)):
    ac_time_p = np.zeros(len(N))
    y_p = np.array(auxiliary_states[:, p])

    for i, n in enumerate(N):
        ac_time_p[i] = MC.autocorr.integrated_time(y_p[:n], c = 5, tol = 5, quiet = True)

    plt.loglog(N, ac_time_p, "o-", label = symbols[p], color = plt.cm.autumn(p/6), linewidth = 2, markersize = 5)


# again for m
ac_time_m = np.zeros(len(N))
y_m = np.array(chain_ms)

for i, n in enumerate(N):

    ac_time_m[i] = MC.autocorr.integrated_time(y_m[:n], c = 5, tol = 5, quiet = True, )

plt.loglog(N, ac_time_m, "o-b", label=r"$M$",  linewidth = 2, markersize = 5)


# plot details
ylim = plt.gca().get_ylim()
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.plot(N, N / 50.0, "--k", label = r"$\tau = N/50$")
plt.ylim(ylim)
#plt.axvline(truncated, alpha = 0.5)
#plt.gca().set_yticks([])
#plt.gca().set_xticks([])
#plt.title('Adptv-RJMH convergence assessment')
plt.xlabel(r"Samples, $N$")
plt.ylabel(r"Autocorrelation time, $\tau$")
#plt.grid()
plt.legend(fontsize = 7)
plt.tight_layout()
plt.savefig('Plots/ACTime.png')
plt.clf()

# record states

single_states = []
binary_states = []

for i in range(iterations):
    if chain_ms[i] == 0: 
        single_states.append(chain_states[i]) # record all single model states in the truncated chain

    if chain_ms[i] == 1: 
        binary_states.append(chain_states[i]) # record all binary model states in the truncated chain

single_states = np.array(single_states)
binary_states = np.array(binary_states)

# record truncated states
tr_single_states = []
tr_binary_states = []

for i in range(truncated, iterations):
    if chain_ms[i] == 0: 
        tr_single_states.append(chain_states[i]) # record all single model states in the truncated chain

    if chain_ms[i] == 1: 
        tr_binary_states.append(chain_states[i]) # record all binary model states in the truncated chain

tr_single_states = np.array(tr_single_states)
tr_binary_states = np.array(tr_binary_states)



# output File:
with open('results/run.txt', 'w') as file:
    # inputs
    file.write('Inputs:\n')
    file.write('Parameters: ' + str(true_theta)+'\n')
    file.write('Number of observations: ' + str(n_epochs)+', Signal to noise baseline: '+str(signal_to_noise_baseline)+'\n')
    
    if informative_priors == True:
        type_priors = 'Informative'
    elif uniform_priors == True:
        type_priors = 'Uninformative'
    file.write('Priors: '+type_priors+'\n')

    # run info
    file.write('\n')
    file.write('Run information:\n')
    file.write('Iterations: '+str(iterations)+', Burn in: '+str(truncated)+' \n')
    file.write('Accepted move fraction; Total: '+str(np.sum(acc_history)/iterations)+',\
        Inter-model: ' + str(np.sum(inter_j_acc_histories) / (len(inter_j_acc_histories))) + ',\
        Intra-model: ' + str(np.sum(np.sum(intra_j_acc_histories)) / (len(intra_j_acc_histories[0]) + len(intra_j_acc_histories[1]))) + ' \n')

    # results
    P_S = 1-np.sum(chain_ms[truncated:]) / (iterations-truncated)
    P_B = np.sum(chain_ms[truncated:]) / (iterations-truncated)
    file.write('\n')
    file.write('Results:\n')
    file.write('Classifications; P(single|y): '+str(P_S)+', P(binary|y): '+str(P_B)+' \n')
    
    if P_S >= P_B:
        probable_states = tr_single_states
        probable_m = 0

    elif P_S < P_B:
        probable_states = tr_binary_states
        probable_m = 1

    for i in range(f.D(probable_m)):

        mu = np.average(probable_states[:, i])
        sd = np.std(probable_states[:, i])

        file.write(letters[i]+': mean: '+str(mu)+', sd: '+str(sd)+' \n')



# adaptive progression plots
# want to include full history with warmup (w) too

single_cov_histories = np.array(deepcopy(w_s_covariance_history))
single_cov_histories = np.concatenate((np.array(w_s_covariance_history), np.array(cov_histories[0][:])))
#print(cov_histories[0][:][:])

#single_cov_histories = np.stack(np.array(single_cov_histories), 0)
#print(single_cov_histories)

#binary_cov_histories = deepcopy(w_b_covariance_history)
#binary_cov_histories.append(cov_histories[1][:])

#binary_cov_histories = np.stack(np.array(binary_cov_histories), 0)

binary_cov_histories = np.array(deepcopy(w_b_covariance_history))
binary_cov_histories = np.concatenate((np.array(w_b_covariance_history), np.array(cov_histories[1][:])))



pltf.Adaptive_Progression(np.concatenate((w_s_acceptance_history, intra_j_acc_histories[0])), single_cov_histories, 'single')
pltf.Adaptive_Progression(np.concatenate((w_b_acceptance_history, intra_j_acc_histories[1])), binary_cov_histories, 'binary')

conditioned_cov_histories = []
n_shared = f.D(0)

print(inter_cov_history[:][:][0], inter_cov_history[:][:][1], inter_cov_history[:][:][-1])

for i in range(len(inter_cov_history)):
    covariance = inter_cov_history[:][:][i]


    c_11 = covariance[:n_shared, :n_shared] # covariance matrix of (shared) dependent variables
    c_12 = covariance[:n_shared, n_shared:] # covariances, not variances
    c_21 = covariance[n_shared:, :n_shared] # same as above
    c_22 = covariance[n_shared:, n_shared:] # covariance matrix of independent variables
    c_22_inv = np.linalg.inv(c_22)
    
    conditioned_cov_histories.append(c_11 - c_12.dot(c_22_inv).dot(c_21))

conditioned_cov_histories = np.array(conditioned_cov_histories)

pltf.Adaptive_Progression(inter_j_acc_histories, conditioned_cov_histories, 'conditioned') # progression for between model jump distribution



# plot of randomly sampled curves
sampled_curves = random.sample(range(truncated, iterations), n_sampled_curves)
sampled_params =[]
for i in sampled_curves:
    pltf.PlotLightcurve(chain_ms[i], f.unscale(np.array(chain_states[i])), False, 'red', 0.01, False, [0,72])
    sampled_params.append(np.append(chain_ms[i], f.unscale(np.array(chain_states[i]))))
print(sampled_params)
#sampled_params = np.array(sampled_params)
plt.scatter(epochs, data.flux, label = 'signal', color = 'grey', s = 1)
#plt.title('Joint distribution samples, N = '+str(n_sampled_curves))
plt.xlabel('time [days]')
plt.ylabel('Magnification')
plt.tight_layout()
plt.savefig('results/RJMH-Samples.png')
plt.clf()


# plot of model index trace
plt.plot(np.linspace(truncated, iterations, num = iterations - truncated ), chain_ms[truncated:] + 1, linewidth = 0.25)
plt.title('RJMH Model Trace')
plt.xlabel('Iterations')
plt.ylabel('Model Index')
plt.locator_params(axis = "y", nbins = 2) # only two ticks
plt.tight_layout()
#plt.savefig('Plots/M-Trace.png')
plt.clf()



# begin corner plots
# note that these destroy the style environment (plot these last)

#pltf.Walk_Plot(6, single_states, np.delete(binary_states, 3, 1), np.delete(auxiliary_states, 3, 1), data, np.delete(np.array(symbols), 3), 'binary-corner', sampled_params)

#print(binary_cov_histories[:][:][-1])

#pltf.Contour_Plot(6, n_points, np.delete(tr_binary_states, 3, 1), np.delete(np.delete(binary_cov_histories[-1], 3, 1), 3, 0), binary_true, np.delete(centers[1], 3), 1, np.delete(np.array(priors), 3), data, np.delete(np.array(symbols), 3), 'binary-contour', P_B)
pltf.Contour_Plot(8, n_points, tr_binary_states, binary_cov_histories[-1], binary_true, centers[1], 1, priors, data, symbols, 'binary-contour', P_B)


shifted_symbols = [r'$t_0-\hat{\theta}$', r'$u_0-\hat{\theta}$', r'$t_E-\hat{\theta}$', r'$\rho-\hat{\theta}$', r'$log_{10}(q)-\hat{\theta}$', r'$s-\hat{\theta}$', r'$\alpha-\hat{\theta}$']

#print(single_states)
#print(centers[0])

pltf.Double_Plot(3, tr_single_states - centers[0], tr_binary_states - centers[1], shifted_symbols, 'shifted-overlay')











