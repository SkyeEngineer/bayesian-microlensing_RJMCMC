# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing

import math
from pickle import FALSE

from numpy.core.numeric import Inf
import MulensModel as mm
import main_functions as f
import autocorrelation_functions as acf
import plot_functions as pltf
import NN_interfaceing as interf
import emcee as MC
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
#import interfaceing as interf
from multiprocessing import Pool
from scipy.optimize import minimize
from copy import deepcopy

import time

from scipy.stats import chi2
import scipy

import plot_functions as pltf

import os
import os.path
import shutil
from pathlib import Path


#-----------
## INPUTS ##
#-----------

adaptive_warmup_iterations = 25#25 # mcmc steps without adaption
adaptive_iterations = 475#475 # mcmc steps with adaption
warmup_loops = 1#5 # times to repeat mcmc optimisation of centers to try to get better estimate
iterations = 500 # rjmcmc steps

truncate = True # automatically truncate burn in period based on autocorrelation of m


n_epochs = 720
epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]

signal_to_noise_baseline = 23#(230-23)/2 + 23 # np.random.uniform(23.0, 230.0) # lower means noisier



#---------------
## END INPUTS ##
#---------------

## INITIALISATION ##

# priors in true space
# informative priors (Zhang et al)

t0_pi = f.uniDist(0, 72)
u0_pi = f.uniDist(0, 2) # reflected
tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
q_pi = f.logUniDist(10e-6, 1)
s_pi = f.logUniDist(0.2, 5)
alpha_pi = f.uniDist(0, 360)

priors = [t0_pi, u0_pi, tE_pi, q_pi, s_pi, alpha_pi]



#QUANTITATIVE RESULTS

def P_B(adaptive_warmup_iterations, adaptive_iterations, warmup_loops, iterations,\
    truncate, true_theta, binary_true, single_true, data, priors, binary_center, single_center):

    # initial covariances (diagonal)
    covariance_scale = 0.001 # reduce diagonals by a multiple
    single_covariance = np.zeros((f.D(0), f.D(0)))
    np.fill_diagonal(single_covariance, np.multiply(covariance_scale, [0.1, 0.01, 0.1]))
    binary_covariance = np.zeros((f.D(1), f.D(1)))
    np.fill_diagonal(binary_covariance, np.multiply(covariance_scale, [0.1, 0.01, 0.1, 0.01, 0.01, 1]))

    start_time = (time.time())

    # use adaptiveMCMC to calculate initial covariances and optimise centers
    w_single_covariance, w_s_chain_states, w_s_chain_means, w_s_acceptance_history, w_s_covariance_history, w_s_best_posterior, w_s_best_theta =\
        f.Loop_Adaptive_Warmup(warmup_loops, 0, data, single_center, priors, single_covariance, adaptive_warmup_iterations, adaptive_iterations)
    w_binary_covariance, w_b_chain_states, w_b_chain_means, w_b_acceptance_history, w_b_covariance_history, w_b_best_posterior, w_b_best_theta =\
        f.Loop_Adaptive_Warmup(warmup_loops, 1, data, binary_center, priors, binary_covariance, adaptive_warmup_iterations, adaptive_iterations)

    # Load resources for RJMCMC

    centers = [w_s_best_theta, w_b_best_theta]
    initial_states = [w_s_chain_states[:, -1], w_b_chain_states[:, -1]]
    initial_means = [w_s_chain_means[:, -1], w_b_chain_means[:, -1]]
    n_warmup_iterations = adaptive_warmup_iterations + adaptive_iterations
    initial_covariances = [w_single_covariance, w_binary_covariance]

    print(centers)

    # run RJMCMC
    chain_states, chain_ms, best_thetas, best_pi, cov_histories, acc_history, inter_j_acc_histories, intra_j_acc_histories, inter_cov_history =\
        f.Run_Adaptive_RJ_Metropolis_Hastings(initial_states, initial_means, n_warmup_iterations, initial_covariances, centers, priors, iterations, data)

    print((time.time() - start_time)/60, 'minutes')

    #-----------------
    ## PLOT RESULTS ##
    #-----------------

    # plotting resources
    pltf.Style()

    # truncate once m below 50 auto correlation times
    if truncate == True:
        n_ac = 25
        N = np.exp(np.linspace(np.log(int(iterations/n_ac)), np.log(iterations), n_ac)).astype(int)

        ac_time_ms = np.zeros(len(N))
        y_ps = np.array(chain_ms)

        for i, n in enumerate(N):
            ac_time_ms[i] = MC.autocorr.integrated_time(y_ps[:n], c = 5, tol = 5, quiet = True)
            
            if ac_time_ms[i] < N[i]/50: # linearly interpolate truncation point
                truncated = N[i]

                break

            truncated = 0
            #print("Not enough iterations to converge to the limiting distribution")

    else: truncated = 0

    truncated = 0

    # results
    # P_S = 1-np.sum(chain_ms[truncated:]) / (iterations-truncated)
    P_B = np.sum(chain_ms[truncated:]) / (iterations-truncated)

    return P_B




theta_r = [36, 0.1, 61.5, 0.01, 0.2, 25]
P_Bs =[]
n = 3
ss = np.linspace(0.2, 3.0, n)
density = []

colors=['blue', 'purple', 'green', 'blue', 'purple', 'green']

s_p_1 = interf.get_posteriors(0)
s_p_2 = interf.get_posteriors(1)

for i in range(n):

    true_theta = deepcopy(theta_r)
    true_theta[4] = ss[i]
    binary_true = deepcopy(true_theta)
    single_true = False
    data = f.Synthetic_Light_Curve(true_theta, 1, n_epochs, signal_to_noise_baseline)

    #single_center = interf.get_model_centers(s_p_1, data.flux)
    #binary_center_rho = interf.get_model_centers(s_p_2, data.flux)
    #binary_center = [binary_center_rho[0], binary_center_rho[1], binary_center_rho[2], binary_center_rho[4], binary_center_rho[5], binary_center_rho[6]]
    #print(binary_center)
    binary_center = deepcopy(true_theta)
    single_center = [36, 0.1, 61.5]


    P_Bs.append(P_B(adaptive_warmup_iterations, adaptive_iterations, warmup_loops, iterations,\
    truncate, true_theta, binary_true, single_true, data, priors, binary_center, single_center))

    density.append(np.exp(priors[4].log_PDF(ss[i])))


    ts = [0, 72]
    pltf.PlotLightcurve(1, true_theta, "te="+str(ss[i]), colors[i], 1, False, ts)


pltf.PlotLightcurve(0, single_center, "single", 'black', 1, False, ts)
plt.legend()
plt.xlabel('Time [days]')
plt.ylabel('Flux')
plt.tight_layout()
plt.savefig('Plots/EPm2.png')
plt.clf()

print(P_Bs)
print(density)
print(sum([a*b for a,b in zip(P_Bs, density)])/sum(density))

