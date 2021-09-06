# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Main]

import math
from pickle import FALSE

from numpy.core.numeric import Inf
import MulensModel as mm
import source
import autocorrelation_functions as acf
import plot_functions as pltf
import emcee as MC
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
import NN_interfaceing as interf
from multiprocessing import Pool
from scipy.optimize import minimize
from copy import deepcopy

import time

from scipy.stats import chi2
import scipy


import os
import os.path
import shutil
from pathlib import Path


#-----------
## INPUTS ##
#-----------

suite_n = 1

adapt_MH_warm_up = 25 #25 # mcmc steps without adaption
adapt_MH = 975  #475 # mcmc steps with adaption
initial_n = 1 # times to repeat mcmc optimisation of centers to try to get better estimate
iterations = 1000 # rjmcmc steps

n_epochs = 720
epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]

signal_to_noise_baseline = 23 #(230-23)/2 + 23 # np.random.uniform(23.0, 230.0) # lower means noisier

n_pixels = 3 # density for posterior contour plot
n_sampled_curves = 5 # sampled curves for viewing distribution of curves

uniform_priors = False 
informative_priors = True

sbi = False # use neural net to get maximum aposteriori estimate for centreing points

truncate = False # automatically truncate burn in period based on autocorrelation of m


user_feedback = True

#---------------
## END INPUTS ##
#---------------

## INITIALISATION ##



def test(n_suite, adapt_MH_warm_up, adapt_MH, initial_n, iterations, user_feedback, sbi, truncate, signal_to_noise_baseline):


    # GENERATE DATA
    # synthetic event parameters
    model_parameters = [
        [36, 0.1, 10],                  # 0
        [36, 0.1, 10, 0.01, 0.2, 60],   # 1
        [36, 0.1, 10, 0.001, 0.5, 60],  # 2
        [36, 0.1, 36, 0.8, 0.25, 123]]  # 3
    event_params = source.State(truth = model_parameters[n_suite])
    
    model_types = [0, 1, 1, 1] # model type associated with synethic event suite above
    model_type = model_types[n_suite]
    # store a synthetic lightcurve. Could otherwise use f.Read_Light_Curve(file_name)
    if model_type == 0:
        data = source.synthetic_single(event_params, n_epochs, signal_to_noise_baseline)
    else: 
        data = source.synthetic_binary(event_params, n_epochs, signal_to_noise_baseline)


    # SET PRIORS
    # priors in truth space informative priors (Zhang et al)
    t0_pi = source.Uniform(0, 72)
    u0_pi = source.Uniform(0, 2)
    tE_pi = source.Truncated_Log_Normal(1, 100, 10**1.15, 10**0.45)
    q_pi = source.Log_Uniform(10e-6, 1)
    s_pi = source.Log_Uniform(0.2, 5)
    alpha_pi = source.Uniform(0, 360)
    priors = [t0_pi, u0_pi, tE_pi, q_pi, s_pi, alpha_pi]


    # GET CENTERS
    if sbi == True:

        # get nueral network posteriors for each model
        single_surrogate_posterior = interf.get_posteriors(0)
        binary_surrogate_posterior = interf.get_posteriors(1)

        # centreing points for inter-model jumps
        single_center = interf.get_model_centers(single_surrogate_posterior, data.flux)
        binary_center_rho = interf.get_model_centers(binary_surrogate_posterior, data.flux)
        binary_center = [binary_center_rho[0], binary_center_rho[1], binary_center_rho[2], binary_center_rho[4], binary_center_rho[5], binary_center_rho[6]]

    else: # use known values for centers 
        single_centers = [
        [36, 0.1, 10], # 0
        [36, 0.1, 10], # 1
        [45, 0.2, 20], # 2
        [36, 0.1, 36]] # 4
        single_center = source.State(truth = np.array(single_centers[n_suite]))

        binary_centers = [
        [36, 0.1, 10, 0.0001, 0.2, 60], # 0
        [36, 0.1, 10, 0.01, 0.2, 60],    # 1
        [45, 0.2, 20, 0.001, 1.0, 300],  # 2
        [36, 0.1, 36, 0.8, 0.25, 123]]   # 3
        binary_center = source.State(truth = np.array(binary_centers[n_suite]))


    # MODEL COVARIANCES
    # initial covariances (diagonal)
    covariance_scale = 0.0001 # reduce diagonals by a multiple
    single_covariance = np.zeros((3, 3))
    np.fill_diagonal(single_covariance, np.multiply(covariance_scale, [1, 0.1, 1]))
    binary_covariance = np.zeros((6, 6))
    np.fill_diagonal(binary_covariance, np.multiply(covariance_scale, [1, 0.1, 1, 0.1, 0.1, 10]))


    # MODELS
    single_Model = source.Model(0, 3, single_center, priors, single_covariance, data, source.single_log_likelihood)
    binary_Model = source.Model(1, 6, binary_center, priors, binary_covariance, data, source.binary_log_likelihood)
    Models = [single_Model, binary_Model]

    start_time = (time.time())

    total_acc, joint_model_chain = source.adapt_RJMH(Models, adapt_MH_warm_up, adapt_MH, initial_n, iterations, user_feedback = user_feedback)

    # use adaptiveMCMC to calculate initial covariances and optimise centers
    #w_single_covariance, w_s_chain_states, w_s_chain_means, w_s_acceptance_history, w_s_covariance_history, w_s_best_posterior, w_s_best_theta =\
    #    f.Loop_Adaptive_Warmup(warmup_loops, 0, data, single_center, priors, single_covariance, adaptive_warmup_iterations, adaptive_iterations)
    #w_binary_covariance, w_b_chain_states, w_b_chain_means, w_b_acceptance_history, w_b_covariance_history, w_b_best_posterior, w_b_best_theta =\
    #    f.Loop_Adaptive_Warmup(warmup_loops, 1, data, binary_center, priors, binary_covariance, adaptive_warmup_iterations, adaptive_iterations)

    # plot optimised centers
    #pltf.LightcurveFitError(2, f.unscale(2, bestt_2), priors, Data, Model, epochs, error, True, "BinaryCenterMCMC")
    #pltf.LightcurveFitError(1, bestt_1, priors, Data, Model, epochs, error, True, "SingleCenterMCMC")


    # Load resources for RJMCMC
    #centers = [w_s_best_theta, w_b_best_theta]
    #initial_states = [w_s_chain_states[:, -1], w_b_chain_states[:, -1]]
    ##initial_means = [w_s_chain_means[:, -1], w_b_chain_means[:, -1]]
    #n_warmup_iterations = adaptive_warmup_iterations + adaptive_iterations
    #initial_covariances = [w_single_covariance, w_binary_covariance]



    # run RJMCMC
    #chain_states, chain_ms, best_thetas, best_pi, cov_histories, acc_history, inter_j_acc_histories, intra_j_acc_histories, inter_cov_history =\
    #    f.Run_Adaptive_RJ_Metropolis_Hastings(initial_states, initial_means, n_warmup_iterations, initial_covariances, centers, priors, iterations, data)

    print((time.time() - start_time)/60, 'minutes')


    #-----------------
    ## PLOT RESULTS ##
    #-----------------

    # plotting resources
    pltf.style()
    labels = ['Impact Time [days]', 'Minimum Impact Parameter', 'Einstein Crossing Time [days]', r'$log_{10}(Mass Ratio)$', 'Separation', 'Alpha']
    symbols = [r'$t_0$', r'$u_0$', r'$t_E$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']
    letters = ['t0', 'u0', 'tE', 'log10(q)', 's', 'a']
    marker_size = 75

    pltf.density_heatmaps(binary_Model, 3, event_params, symbols)
    pltf.joint_samples_pointilism(binary_Model, single_Model, joint_model_chain, symbols)
    pltf.center_offsets_pointilism(binary_Model, single_Model, symbols)

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


    chain_ps = np.zeros((iterations))
    for i in range(iterations):
        chain_ps[i] = np.sum(chain_ms[:i])/(i+1)

    # truncate once m below 50 auto correlation times
    if truncate == True:
        n_ac = 25
        N = np.exp(np.linspace(np.log(int(iterations/n_ac)), np.log(iterations), n_ac)).astype(int)

        ac_time_ms = np.zeros(len(N))
        y_ps = np.array(chain_ms)

        for i, n in enumerate(N):
            ac_time_ms[i] = MC.autocorr.integrated_time(y_ps[:n], c = 5, tol = 5, quiet = True)
            
            if ac_time_ms[i] < N[i]/50: # linearly interpolate truncation point
                #if i == 0:
                truncated = N[i]
                #else:
                #    slope = (ac_time_m[i] - ac_time_m[i-1]) / (N[i] - N[i-1])
                #    truncated = int(math.ceil((ac_time_m[i] - slope * N[i]) / (1/50 - slope)))
                break

            truncated = 0
            print("Not enough iterations to converge to the limiting distribution")

    else: truncated = 0

    #print('We recommend truncating '+str(truncated)+' iterations off of '+str(truncated)+'. Type the amount you wish for:')
    #truncated = int(input('Burn in: '))

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
    ac_time_ms = np.zeros(len(N))
    y_ms = np.array(chain_ms)

    for i, n in enumerate(N):

        ac_time_ms[i] = MC.autocorr.integrated_time(y_ms[:n], c = 5, tol = 5, quiet = True, )

    plt.loglog(N, ac_time_ms, "o-b", label=r"$m$",  linewidth = 2, markersize = 5)


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
    plt.savefig('results/'+run_name+'-ACTime.png')
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
    with open('results/'+run_name+'-run.txt', 'w') as file:
        # inputs
        file.write('Inputs:\n')
        file.write('Parameters: ' + str(truth_theta)+'\n')
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



    pltf.Adaptive_Progression(adaptive_warmup_iterations+adaptive_iterations+iterations, np.concatenate((w_s_acceptance_history, intra_j_acc_histories[0])), single_cov_histories, run_name+'-single-')
    pltf.Adaptive_Progression(adaptive_warmup_iterations+adaptive_iterations+iterations, np.concatenate((w_b_acceptance_history, intra_j_acc_histories[1])), binary_cov_histories, run_name+'-binary-')

    conditioned_cov_histories = []
    n_shared = f.D(0)

    #print(inter_cov_history[:][:][0], inter_cov_history[:][:][1], inter_cov_history[:][:][-1])

    for i in range(len(inter_cov_history)):
        covariance = inter_cov_history[:][:][i]


        c_11 = covariance[:n_shared, :n_shared] # covariance matrix of (shared) dependent variables
        c_12 = covariance[:n_shared, n_shared:] # covariances, not variances
        c_21 = covariance[n_shared:, :n_shared] # same as above
        c_22 = covariance[n_shared:, n_shared:] # covariance matrix of independent variables
        c_22_inv = np.linalg.inv(c_22)
        
        conditioned_cov_histories.append(c_11 - c_12.dot(c_22_inv).dot(c_21))

    conditioned_cov_histories = np.array(conditioned_cov_histories)

    pltf.Adaptive_Progression(iterations, inter_j_acc_histories, conditioned_cov_histories, run_name+'-conditioned-') # progression for between model jump distribution



    # plot of randomly sampled curves
    sampled_curves = random.sample(range(truncated, iterations), n_sampled_curves)
    sampled_params =[]
    #for i in sampled_curves:
    #    pltf.PlotLightcurve(chain_ms[i], f.unscale(np.array(chain_states[i])), False, 'red', 0.01, False, [0,72])
    #    sampled_params.append(np.append(chain_ms[i], f.unscale(np.array(chain_states[i]))))
    #print(sampled_params)
    #sampled_params = np.array(sampled_params)
    #plt.scatter(epochs, data.flux, label = 'signal', color = 'grey', s = 1)
    #plt.title('Joint distribution samples, N = '+str(n_sampled_curves))
    #plt.xlabel('time [days]')
    #plt.ylabel('Magnification')
    #plt.tight_layout()

    plt.figure()
    pltf.Draw_Light_Curve_Noise_Error(data, plt.gca())
    plt.savefig('results/'+run_name+'-RJMH-Samples.png', bbox_inches="tight")
    plt.clf()


    # plot of model index trace
    plt.plot(np.linspace(0, iterations, num = iterations), chain_ms + 1, linewidth = 0.25)
    #plt.title('RJMH Model Trace')
    plt.xlabel('Iterations')
    plt.ylabel('Model Index')
    plt.locator_params(axis = "y", nbins = 2) # only two ticks
    plt.tight_layout()
    plt.savefig('results/'+run_name+'M-Trace.png')
    plt.clf()

    # plot of model index trace
    plt.plot(np.linspace(0, iterations, num = iterations), chain_ps, linewidth = 0.25, color='purple')
    #plt.title('RJMH Model Trace')
    plt.xlabel('Iterations')
    plt.ylabel('P')
    #plt.locator_params(axis = "y", nbins = 2) # only two ticks
    plt.tight_layout()
    plt.savefig('results/'+run_name+'P-Trace.png')
    plt.clf()

    # begin corner plots
    # note that these destroy the style environment (plot these last)

    #pltf.Walk_Plot(6, single_states, np.delete(binary_states, 3, 1), np.delete(auxiliary_states, 3, 1), data, np.delete(np.array(symbols), 3), 'binary-corner', sampled_params)
    pltf.Walk_Plot(6, single_states, binary_states, auxiliary_states, data, symbols, run_name+'point-corner', sampled_params)


    #print(binary_cov_histories[:][:][-1])

    #pltf.Contour_Plot(6, n_points, np.delete(tr_binary_states, 3, 1), np.delete(np.delete(binary_cov_histories[-1], 3, 1), 3, 0), binary_truth, np.delete(centers[1], 3), 1, np.delete(np.array(priors), 3), data, np.delete(np.array(symbols), 3), 'binary-contour', P_B)
    if P_B>P_S:
        pltf.Contour_Plot(6, n_points, tr_binary_states, binary_cov_histories[-1], binary_truth, centers[1], 1, priors, data, symbols, run_name+'binary-contour', P_B)
    else:
        pltf.Contour_Plot(3, n_points, tr_single_states, single_cov_histories[-1], single_truth, centers[0], 0, priors, data, symbols, run_name+'single-contour', P_S)

    shifted_symbols = [r'$t_0-\hat{\theta}$', r'$u_0-\hat{\theta}$', r'$t_E-\hat{\theta}$', r'$\rho-\hat{\theta}$', r'$log_{10}(q)-\hat{\theta}$', r'$s-\hat{\theta}$', r'$\alpha-\hat{\theta}$']

    #print(single_states)
    #print(centers[0])

    pltf.Double_Plot(3, tr_single_states - centers[0], tr_binary_states - centers[1], shifted_symbols, 'shifted-overlay')
    
    return

# RESULTS

#test(0, adapt_MH_warm_up, adapt_MH, initial_n, iterations, user_feedback, sbi, truncate, signal_to_noise_baseline)

test(suite_n, adapt_MH_warm_up, adapt_MH, initial_n, iterations, user_feedback, sbi, truncate, signal_to_noise_baseline)

#truth_theta, binary_truth, single_truth, data, binary_center, single_center = Suite(1)
#Run('2/', adaptive_warmup_iterations, adaptive_iterations, warmup_loops, iterations,\
#    truncate, truth_theta, binary_truth, single_truth, data, priors, binary_center, single_center)

#truth_theta, binary_truth, single_truth, data, binary_center, single_center = Suite(2)
#Run('3/', adaptive_warmup_iterations, adaptive_iterations, warmup_loops, iterations,\
#    truncate, truth_theta, binary_truth, single_truth, data, priors, binary_center, single_center)

#truth_theta, binary_truth, single_truth, data, binary_center, single_center = Suite(3)
#Run('4/', adaptive_warmup_iterations, adaptive_iterations, warmup_loops, iterations,\
#    truncate, truth_theta, binary_truth, single_truth, data, priors, binary_center, single_center)