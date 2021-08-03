# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Main]

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
import corner
from scipy.stats import chi2
import scipy

import PlotFunctions as pltf

import os
import os.path
import shutil
from pathlib import Path


pltf.Style()


def ParralelMain(arr):

    sn, Data, signal_data, priors, binary_Sposterior, single_Sposterior, m_pi, iterations, Model, error, epochs, burns, iters = arr


    sbi = False
    if sbi == True:

        single_Sposterior = interf.get_posteriors(1)
        binary_Sposterior = interf.get_posteriors(2)

        # centreing points for inter-model jumps
        center_1s = interf.get_model_centers(single_Sposterior, signal_data)
        center_2s = interf.get_model_centers(binary_Sposterior, signal_data)
    
    else:
        

        center_1s = np.array([36.37017441,  0.83766584, 31.54711723]) #1
        #center_1s = np.array([35.93706894,  0.83814418, 30.89567947])#3
        #center_1s = np.array([35.98891449,  0.83486302, 30.986166])#4

        




        center_2s = np.array([3.64606857e+01, 8.32321227e-01, 3.14062214e+01,  0.001, 0.004, 1.40, 175]) #1
        #center_2s = np.array([3.64606857e+01, 8.32321227e-01, 3.14062214e+01, 1.09751643e-04, 8.47467631e-02, 2.32403427e-01, 1.16953224e+02]) #3, #4
        #center_2s = np.array([3.57069664e+01, 8.28740132e-01, 3.13619919e+01, 1.06470281e-04, 2.02446827e-03, 1.91855073e+00, 2.07568512e+02]) #4

    
    #center_1s = np.array([36., 0.133, 31.5])

    center_2ss = [
        [36, 0.133, 61.5, 0.0096, np.log(0.002), 1.27, 210.8], # strong binary
        [36., 0.133, 61.5, 0.00963, np.log(0.00092), 1.31, 210.8], # weak binary 1
        [36., 0.133, 61.5, 0.0052, np.log(0.0006), 1.29, 210.9], # weak binary 2
        [36., 0.133, 61.5, 0.0096, np.log(0.00002), 4.25, 223.8], # indistiguishable from single
        ]
    #center_2 = np.array(center_2s[sn])



    #throw=throw

    #center_2s[4] = np.log(center_2s[4])
    #center_2s = f.scale(center_2s)

    #print("\n", center_2, " hi")

    #print(Data.flux)
    #binary_ensemble = interf.get_model_ensemble(binary_posterior, Data.flux, 100000)

    pltf.LightcurveFitError(2, center_2s, priors, Data, Model, epochs, error, True, "BinaryCenterSurr")
    pltf.LightcurveFitError(1, center_1s, priors, Data, Model, epochs, error, True, "SingleCenterSurr")
    #throw=throw
    
    #fun_1 = lambda x: -f.logLikelihood(1, Data, x, priors)
    #min_center_1 = minimize(fun_1, center_1s, method='Nelder-Mead')
    #print(min_center_1)
    #center_1 = min_center_1.x

    #fun_2 = lambda x: -2*f.logLikelihood(2, Data, x, priors)
    #min_center_2 = minimize(fun_2, center_2s, method = 'Nelder-Mead', options={'maxfev': 1000})
    #print(min_center_2)
    #center_2 = min_center_2.x

    #pltf.LightcurveFitError(2, center_2, priors, Data, Model, epochs, error, True, "BinaryCenterOpt")

    #throw=throw

 
    #pltf.LightcurveFitError(1, center_1, priors, Data, Model, epochs, error, True, "SingleCenterOpt")


    #center_2 = f.scale(center_2)

    # initial covariances (diagonal)
    cov_scale = 0.001 #0.01

    covariance_1 = np.zeros((3, 3))
    np.fill_diagonal(covariance_1, np.multiply(cov_scale, [0.1, 0.01, 0.1]))

    covariance_2 = np.zeros((7, 7))
    np.fill_diagonal(covariance_2, np.multiply(cov_scale, [0.1, 0.01, 0.1, 0.0001, 0.1, 0.01, 1])) #0.5

    #covariance_1s = np.multiply(1, [0.01, 0.01, 0.1])
    #covariance_2s = np.multiply(1, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])#0.5
    #covariance_1 = np.outer(covariance_1s, covariance_1s)
    #covariance_2 = np.outer(covariance_2s, covariance_2s)
    #covariance_p = [covariance_1, covariance_2]

    # Use adaptiveMCMC to calculate initial covariances
    #burns = 50 #25
    #iters = 50 #250
    theta_1i = center_1s
    theta_2i = f.scale(center_2s)
    covariance_1p, states_1, means_1, c_1, covs_1, bests, bestt_1 = f.AdaptiveMCMC(1, Data, theta_1i, priors, covariance_1, burns, iters)
    covariance_2p, states_2, means_2, c_2, covs_2, bests, bestt_2 = f.AdaptiveMCMC(2, Data, theta_2i, priors, covariance_2, burns, iters)

    covariance_p = [covariance_1p, covariance_2p]
    #print(covariance_1p)
    #print(covariance_2p)
    #throw=throw

    center_1 = bestt_1
    center_2 = bestt_2
    
    print(center_1)
    print(center_2)

    #print("Center 1", center_1, "True Chi", -(f.logLikelihood(1, Data, center_1, priors)))
    #print("Center 2", center_2, "True Chi", -(f.logLikelihood(2, Data, f.unscale(2, center_2), priors)))
    centers = [center_1, center_2]


    pltf.LightcurveFitError(2, f.unscale(2, bestt_2), priors, Data, Model, epochs, error, True, "BinaryCenterMCMC")
    pltf.LightcurveFitError(1, bestt_1, priors, Data, Model, epochs, error, True, "SingleCenterMCMC")


    # loop specific values

    #print(states_1[:, -1])
    theta = center_2#states_2[:, -1]#[36., 0.133, 61.5]#, 0.0014, 0.0009, 1.26, 224.]
    #print(theta)
    m = 2
    pi = (f.logLikelihood(m, Data, f.unscale(m, theta), priors))
    #print(pi)

    ms = np.zeros(iterations, dtype=int)
    ms[0] = m
    states = []
    score = 0
    Dscore = 0
    Dtotal = 1
    J_2 = np.prod(center_2[0:3])
    J_1 = np.prod(center_1)
    J = np.abs([J_1/J_2, J_2/J_1])
    #print(J)

    #adaptive params
    t=[burns+iters, burns+iters]
    I = [np.identity(3), np.identity(7)] 
    s = [2.4**2 / 3, 2.4**2 / 7] # Arbitrary(ish), good value from Haario et al
    eps = 1e-12 # Needs to be smaller than the scale of parameter values
    #means = [np.zeros((3, iters+burns+iterations)), np.zeros((7, iters+burns+iterations))]
    #print(means[0][:,0:2])
    #print(means_1)
    #means[0][:, 0:burns+iters] = means_1
    #means[1][:, 0:burns+iters] = means_2
    stored_mean = [means_1[:, -1], means_2[:, -1]]


    bests = [0, 0]
    bestt = [[], []]

    print('Running RJMCMC')

                
    mem_2 = center_2#states_2[:, -1]
    adaptive_score = [c_1.tolist(), c_2.tolist()]
    inter_props = [[], []]

    n_samples = 10000
    samples, log_prob_samples = True, True#interf.get_model_ensemble(binary_Sposterior, signal_data, n_samples)
    samples = True#(samples - f.unscale(2, center_2))/1 + f.unscale(2, center_2)

    covs = [covs_1, covs_2]

    for i in range(iterations): # loop through RJMCMC steps
        
        #diagnostics
        #print(f'\rLikelihood: {np.exp(pi):.3f}', end='')
        cf = i/(iterations-1);
        print(f'Current: Likelihood {np.exp(pi):.4f}, M {m} | Progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')

        mProp = random.randint(1,2) # since all models are equally likelly, this has no presence in the acceptance step
        #thetaProp = f.RJCenteredProposal(m, mProp, theta, covariance_p[mProp-1], centers, mem_2, priors) #states_2)

        #priorRatio = np.exp(f.PriorRatio(m, mProp, f.unscale(m, theta), f.unscale(mProp, thetaProp), priors))
        
        #piProp = (f.logLikelihood(mProp, Data, f.unscale(mProp, thetaProp), priors))

        #print(piProp, pi, priorRatio, mProp)
        #scale = 1
        
        #if mProp == 2 and m == 1: 
        #    l = (theta - centers[m-1]) / centers[m-1]
        #    scale = 1/(np.max(l) - np.min(l))
        #elif mProp == 1 and m == 2:
        #    l = (thetaProp - centers[mProp-1]) / centers[mProp-1]
        #    scale = 1/(np.max(l[0:3]) - np.min(l[0:3]))

        #scale = 1
        thetaProp, piProp, acc = f.Propose(Data, signal_data, m, mProp, theta, pi, covariance_p, centers, binary_Sposterior, samples, log_prob_samples, n_samples, priors, mem_2, stored_mean, False)
        #if random.random() <= scale * np.exp(piProp-pi) * priorRatio * m_pi[mProp-1]/m_pi[m-1] * J[mProp-1]: # metropolis acceptance
        if random.random() <= acc * m_pi[mProp - 1] / m_pi[m - 1]: #*q!!!!!!!!!!!!# metropolis acceptance
            if m == mProp: 
                adaptive_score[mProp - 1].append(1)
            else:
                inter_props[mProp - 1].append(1)

            theta = thetaProp
            m = mProp
            score += 1
            pi = piProp
            if mProp == 2: mem_2 = thetaProp

            if bests[mProp-1] < np.exp(piProp): 
                bests[mProp-1] = np.exp(piProp)
                bestt[mProp-1] = f.unscale(mProp, thetaProp)

        elif m == mProp: 
            adaptive_score[mProp - 1].append(0)
        else:
            inter_props[mProp - 1].append(0)

        '''
        elif m != mProp and random.random() <= 0.5 and False: #Delayed rejection for Jump False: #
            Dtotal += 1

            thetaProp_2, piProp_2, acc_2 = f.Propose(Data, m, mProp, theta, pi, covariance_p, centers, mem_2, priors, True)

            pi_2 = f.logLikelihood(mProp, Data, f.unscale(mProp, thetaProp_2), priors)
            thetaProp_25, piProp_25, acc_25 = f.Propose(Data, mProp, m, thetaProp_2, pi_2, covariance_p, centers, mem_2, priors, False)
            
            if random.random() <= acc_2 * (1-acc_25)/(1 - acc) * 1/(2**3): #*q!!!!!!!!!!!!# delayed metropolis acceptance
                theta = thetaProp_2
                m = mProp
                score += 1
                Dscore += 1
                pi = piProp_2
                
            if mProp==2: mem_2 = thetaProp_2

            if bests[mProp-1] < np.exp(piProp_2): 
                bests[mProp-1] = np.exp(piProp_2)
                bestt[mProp-1] = f.unscale(mProp, thetaProp_2)
        '''

        #scale = 1
        states.append(theta)
        ms[i] = m


        tr = t[m-1]
        
        #means[m-1][:, tr] = (means[m-1][:, tr-1]*tr + theta)/(tr + 1) # recursive mean (offsets indices starting at zero by one)    
        # update step (recursive covariance)

        #covariance_p[m-1] = (tr - 1)/tr * covariance_p[m-1] + s[m-1]/tr * (tr*means[m-1][:, tr - 1]*np.transpose(means[m-1][:, tr - 1]) - (tr + 1)*means[m-1][:, tr]*np.transpose(means[m-1][:, tr]) + theta*np.transpose(theta)) #+ eps*I[m-1]
        covs[m-1].append(covariance_p[m-1])
        covariance_p[m-1] = (tr - 1)/tr * covariance_p[m - 1] + s[m-1]/(tr + 1) * np.outer(theta - stored_mean[m-1], theta - stored_mean[m-1]) + s[m-1]*eps*I[m-1]/tr
        #(t*means[:, t - 1]*np.transpose(means[:, t - 1]) - (t + 1)*means[:, t]*np.transpose(means[:, t]) + states[:, t]*np.transpose(states[:, t]) + eps*I)
        
        #print('My Formula: ', f.check_symmetric(covariance_p[m-1], tol=1e-8))
        
        stored_mean[m-1] = (stored_mean[m-1]*tr + theta)/(tr + 1)

        t[m-1] += 1

    # performance diagnostics:
    print("\nIterations: "+str(iterations))
    print("Accepted Move Fraction: "+str(score/iterations))
    print("Accepted Delayed Move Fraction: "+str(Dscore/Dtotal))
    print("P(Singular): "+str(1-np.sum(ms-1)/iterations))
    print("P(Binary): "+str(np.sum(ms-1)/iterations))
    #print(states)

    return states, adaptive_score, inter_props, ms, bestt, bests, centers, covs, score


labels = [r'Impact Time [$days$]', r'Minimum Impact Parameter', r'Einstein Crossing Time [$days$]', r'Rho', r'ln(Mass Ratio)', r'Separation', r'Alpha']
symbols = [r'$t_0$', r'$u_0$', r'$t_E$', r'$\rho$', r'$q$', r'$s$', r'$\alpha$']
letters = ['t0', 'u0', 'tE', 'p', 'q', 's', 'a']

## INITIALISATION ##



# Synthetic Event Parameters
theta_Models = [
    [36, 0.633, 31.5, 0.0096, 0.01, 1.27, 210.8], # 0 strong binary
    [36, 0.833, 31.5,  0.001, 0.03, 1.10, 180], # 1 weak binary 1
    [36, 0.933, 21.5, 0.0056, 0.065, 1.1, 210.8], # 2 weak binary 2
    [36, 0.833, 31.5, 0.0096, 0.0001, 4.9, 223.8], # 3 indistiguishable from single
    [36, 0.833, 31.5]  # 4 single
    ]

sn = 1
theta_Model = np.array(theta_Models[sn])

single_true = False#f.scale(theta_Model)
binary_true = f.scale(theta_Model)

burns = 25
iters = 975
iterations = 5000
truncation_iterations = 0

n_epochs = 720

n_points = 2

signal_to_noise_baseline = 60.0#np.random.uniform(23.0, 230.0) # Lower means noisier

uniform_priors = False
informative_priors = True


if isinstance(single_true, np.ndarray):

    Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta_Model)))
    Model.set_magnification_methods([0., 'point_source', 72.])

elif isinstance(binary_true, np.ndarray):

    Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta_Model)))
    Model.set_magnification_methods([0., 'VBBL', 72.])


Model.plot_magnification(t_range = [0, 72], subtract_2450000 = False, color = 'black')
plt.savefig('temp.jpg')
plt.clf()
#throw=throw


#0, 50, 25, 0.3
# Generate "Synthetic" Lightcurve
#epochs = Model.set_times(n_epochs = 720)

epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
true_data = Model.magnification(epochs)
#epochs = Model.set_times(n_epochs = 100)
#error = Model.magnification(epochs) * 0 + np.max(Model.magnification(epochs))/60 #Model.magnification(epochs)/100 + 0.5/Model.magnification(epochs)
random.seed(a = 99, version = 2)


noise = np.random.normal(0.0, np.sqrt(true_data) / signal_to_noise_baseline, n_epochs) 
noise_sd = np.sqrt(true_data) / signal_to_noise_baseline
error = deepcopy(noise_sd)
model_data = true_data + noise

'''
with open("OB110251.csv") as file:
    array = np.loadtxt(file, delimiter=",")

array = array[1008:3168][:]
array = array[::3][:]
array[:, 0] = array[:, 0] - array[0][0]
print(array)


Data = mm.MulensData(data_list = [array[:, 0], array[:, 1], array[:, 2]], phot_fmt = 'flux', chi2_fmt = 'flux')
signal_n_epochs = 720
signal_epochs = np.linspace(0, 72, signal_n_epochs + 1)[:signal_n_epochs]
true_signal_data = array[:, 1]
signal_data = array[:, 1]
'''

Data = mm.MulensData(data_list = [epochs, model_data, noise_sd], phot_fmt = 'flux', chi2_fmt = 'flux')
signal_n_epochs = 720
signal_epochs = np.linspace(0, 72, signal_n_epochs + 1)[:signal_n_epochs]
true_signal_data = Model.magnification(signal_epochs)
signal_data = model_data


plt.scatter(epochs, signal_data, color = 'grey', s = 1, label='signal')
plt.ylabel('Magnification')
plt.xlabel('Time [days]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('ObsTru.png', transparent=True)
plt.clf()
#throw=throw
'''
plt.scatter(epochs, signal_data, color = 'grey', s = 1, label='signal')
plt.plot(epochs, true_data, color = 'red', label='true')
plt.ylabel('Magnification')
plt.xlabel('Time [days]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Tru.png', transparent=True)
plt.clf()
'''

#throw=throw
#print(Model.magnification(epochs))




if informative_priors == True:
    # informative priors (Zhang et al)
    s_pi = f.logUniDist(0.2, 5)
    q_pi = f.logUniDist(10e-6, 1)
    #q_pi = f.uniDist(10e-6, 0.1)
    alpha_pi = f.uniDist(0, 360)
    u0_pi = f.uniDist(0, 2)
    t0_pi = f.uniDist(0, 72)
    tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
    rho_pi =  f.logUniDist(10**-4, 10**-2)
    a = 0.5
    m_pi = [1 - a, a]
    priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

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
    m_pi = [0.5, 0.5]

#print(np.exp(f.logLikelihood(2, Data, theta_Models[0], priors)), "hi")
#g=g

#full_path = os.getcwd()
#out_path = (str(Path(full_path).parents[0]))
#with open(out_path+"/microlensing/output/binary_100K_720.pkl", "rb") as handle: binary_posterior = pickle.load(handle)

single_Sposterior = True#interf.get_posteriors(1)
binary_Sposterior = True#interf.get_posteriors(2)



#u_full = binary_Sposterior.sample((1, ), x = Data.flux)
#u_full.numpy
#u = np.float64(u_full[0])[3:]
#print(u)
#throw=throw

#arr, l_arr = interf.get_model_ensemble(binary_Sposterior, Data.flux, 1)




params = [sn, Data, signal_data, priors, binary_Sposterior, single_Sposterior, m_pi, iterations,  Model, error, epochs, burns, iters]

states, adaptive_score, inter_props, ms, bestt, bests, centers, covs, score = ParralelMain(params)
center_1, center_2 = centers




'''
p = 2
pool = Pool(p)

poolSol = pool.map(ParralelMain, np.tile(params, (p, 1)))

#print(poolSol[0][0])
#print(poolSol[1][0])
#print(poolSol[0][1])
#print(poolSol[1][1])


states = np.array(poolSol)[:, 0]
adaptive_history = np.array(poolSol)[:, 1]
#print(adaptive_history)
#print(states)

## AUTO CORR ANALYSIS TRUNCATION + PLOTS ##


# Plot the comparisons
#N=iterations
N = np.exp(np.linspace(np.log(1000), np.log(iterations), 10)).astype(int)
#N = np.linspace((100), iterations, 20).astype(int)

new = np.zeros((p, len(N)))
#newm = np.empty(len(N))
#newu = np.empty(len(N))
for chain in range(p):
    y = np.array(AC.scalarPolyProjection(states[chain]))
#y = states
#y= states

    for i, n in enumerate(N):
    #gw2010[i] = a.autocorr_gw2010(y[:, :n])
        new[chain][i] = MC.autocorr.integrated_time(y[:n], c=5, tol=50, quiet=True)#AC.autocorr_new(y[:n])#autocorr_new(y[:, :n])
    #newm[i] = MC.autocorr.integrated_time(jumpStates_2[:n, 4], c=5, tol=50, quiet=True)
    #newu[i] = MC.autocorr.integrated_time(states_u[:n], c=5, tol=50, quiet=True)


#plt.loglog(N, gw2010, "o-", label="G&W 2010")

plt.plot(N, np.average(new, 0), "o-", label="Average Scalar")
plt.plot(N, new[0][:], "o-", label="0")
plt.plot(N, new[1][:], "o-", label="1")
#plt.plot(N, newm, "--", label="M")
#plt.plot(N, newu, label="newu")
#plt.plot(np.linspace(1, iterations, num=iterations), y, "o-", label="new")

ylim = plt.gca().get_ylim()
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
#plt.axhline(true_tau, color="k", label="truth", zorder=-100)
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14)
plt.savefig('Plots/AutoCorr.png')
plt.clf()

#plt.plot(np.linspace(1, iterations, num=iterations), m, "o-", label="m")
plt.plot(np.linspace(1, iterations, num=iterations), y, "-", label="scalar")
plt.legend(fontsize=14)
plt.savefig('Plots/Temp.png')
plt.clf()


pltf.AdaptiveProgression(adaptive_history, ['Single', 'Binary'], p)

g=g
'''









## OTHER PLOT RESULTS ##

markerSize=75

states_2 = []
jumpStates_2 = []
h_ind = []
h=0
for i in range(iterations): # record all binary model states in the chain
    if ms[i] == 2: 
        states_2.append((states[i]))
        if ms[i-1] == 1: 
            jumpStates_2.append((states[i]))
            h_ind.append(len(states_2))


states_2=np.array(states_2)
jumpStates_2 = np.array(jumpStates_2)



states_1 = []
h_states_1 = []
h_ind1=[]
for i in range(iterations): # record all single model states in the chain
    if ms[i] == 1: 
        states_1.append((states[i]))
        if ms[i-1] == 2: 
            h_states_1.append((states[i]))
            h_ind1.append(len(states_1))

states_1 = np.array(states_1)
h_states_1 = np.array(h_states_1)



details = True


states_u = []
for i in range(iterations): # record all single model states in the chain
    states_u.append(states[i][1])
states_u=np.array(states_u)


#keep = deepcopy(plt.rcParams)



#plt.rcParams.update(plt.rcParamsDefault)
#plt.rcdefaults()
#plt.style.use('default')
#plt.rc_file_defaults()
##plt.rcParams=keep

#pltf.Style()
#throw=throw

#states_2df = pd.DataFrame(data = states_2, columns = labels)
#print(states_2df)

#grid = sns.PairGrid(data = states_2df, vars = labels, height = 7)
#grid = grid.map_upper(pltf.PPlotWalk)
#plt.savefig('Plots/RJ-binary-pplot.png')
#plt.clf()



#pltf.LightcurveFitError(2, bestt[1][:], priors, Data, Model, epochs, error, details, 'BestBinary')

#pltf.LightcurveFitError(1, bestt[0][:], priors, Data, Model, epochs, error, details, 'BestSingle')


# Output File:

with open('results/run.txt', 'w') as file:
    # Inputs
    file.write('Inputs:\n')
    file.write('Parameters: '+str(theta_Model)+'\n')
    file.write('Number of observations: '+str(n_epochs)+', Signal to noise baseline: '+str(signal_to_noise_baseline)+'\n')
    
    if informative_priors == True:
        type_priors = 'Informative'
    elif uniform_priors == True:
        type_priors = 'Uninformative'
    file.write('Priors: '+type_priors+'\n')

    # Run info
    file.write('\n')
    file.write('Run information:\n')
    file.write('RJMCMC iterations: '+str(iterations-truncation_iterations)+', RJMCMC burn in: '+str(truncation_iterations)+' \n')
    file.write('Accepted move fraction; Total'+str(score/iterations)+', Intra-model: '+str(-99)+', Inter-model: '+str(-99)+' \n')

    # Results
    P_S = 1-np.sum(ms-1)/(iterations-truncation_iterations)
    P_B = np.sum(ms-1)/(iterations-truncation_iterations)
    file.write('\n')
    file.write('Results:\n')
    file.write('Classifications; P(Singular): '+str(P_S)+', P(Binary): '+str(P_B)+' \n')
    
    if P_S >= P_B:
        states_p = states_1
        m_p = 1
    elif P_S < P_B:
        states_p = states_2
        m_p = 2

    for i in range(f.D(m_p)):
        #no truncations yet!!!!!!!!!
        mu = np.average(states_p[:, i])
        sd = np.std(states_p[:, i])

        file.write(letters[i]+': mean: '+str(mu)+', sd: '+str(sd)+' \n')




n_density = 5

pltf.AdaptiveProgression(adaptive_score[1], inter_props[1], covs[1][:], 'binary')
pltf.AdaptiveProgression(adaptive_score[0], inter_props[0], covs[0][:], 'single')

#throw=throw

if False:





    for i in range(7):


        
        #pltf.TracePlot(i, states_2, jumpStates_2, h_ind, labels, symbols, letters, 'binary', center_2, binary_true)
        pltf.DistPlot(i, states_2, labels, symbols, letters, 2, 'binary', center_2, binary_true, priors, Data)
        
        for j in range(i+1, 7):
            pltf.PlotWalk(i, j, states_2, labels, symbols, letters, 'binary', center_2, binary_true)

            pltf.contourPlot(i, j, states_2, labels, symbols, letters, 'binary', center_2, binary_true, 2, priors, Data, n_density)

    ## SINGLE MODEL ##



    for i in range(3):

        #pltf.TracePlot(i, states_1, h_states_1, h_ind1, labels, symbols, letters, 'single', center_1, single_true)
        pltf.DistPlot(i, states_1, labels, symbols, letters, 1, 'single', center_1, single_true, priors, Data)

        for j in range(i+1, 3):
            pltf.PlotWalk(i, j, states_1, labels, symbols, letters, 'single', center_1, single_true)

            pltf.contourPlot(i, j, states_1, labels, symbols, letters, 'single', center_1, single_true, 1, priors, Data, n_density)


sampled_curves = random.sample(range(0, np.size(ms, 0)), 100)#int(0.1*np.size(states_2, 0)))
for i in sampled_curves:
    #print(states[i])
    #print(states[i, :])
    pltf.PlotLightcurve(ms[i], f.unscale(ms[i], np.array(states[i])), 'Samples', 'red', 0.05, False, [0,72])

#if len(theta_Model)>5:
#    pltf.PlotLightcurve(2, theta_Model, 'True', 'black', 1, False, [0, 72])
#else:
#    pltf.PlotLightcurve(1, theta_Model, 'True', 'black', 1, False, [0, 72])
#plt.legend()
plt.scatter(epochs, Data.flux, label = 'signal', color = 'grey', s=1)
plt.title('Joint Dist Samples')
plt.xlabel('time [days]')
plt.ylabel('Magnification')
plt.tight_layout()
plt.savefig('results/RJMCMC-Samples.png')
plt.clf()



plt.plot(np.linspace(1, iterations, num = iterations), ms, linewidth=0.5)
plt.title('RJMCMC Model Trace')
plt.xlabel('Iterations')
plt.ylabel('Model Index')
plt.locator_params(axis="y", nbins=2)
plt.tight_layout()
plt.savefig('Plots/M-Trace.png')
plt.clf()





plt.grid()
plt.plot(np.linspace(1, iterations, num=iterations), AC.scalarPolyProjection(states), "-", label="scalar")
#plt.legend(fontsize=14)
plt.title('RJMCMC scalar indicator trace')
plt.xlabel('Iterations')
plt.ylabel('Scalar Indicator')
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
plt.tight_layout()
plt.savefig('Plots/Temp.png')
plt.clf()


# Plot the comparisons
n_ac = 10
N = np.exp(np.linspace(np.log(int(iterations/n_ac)), np.log(iterations), n_ac)).astype(int)

new_v = np.zeros(len(N))
yv = np.array(AC.scalarPolyProjection(states))

new_m = np.zeros(len(N))
ym = np.array(ms)

for i, n in enumerate(N):
    new_v[i] = MC.autocorr.integrated_time(yv[:n], c=5, tol=50, quiet=True)

    new_m[i] = MC.autocorr.integrated_time(ym[:n], c=5, tol=50, quiet=True)



#
plt.loglog(N, new_v, "o-g", label=r"$\tau (V)$")
plt.loglog(N, new_m, "o-m", label=r"$\tau (M)$")

ylim = plt.gca().get_ylim()
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.plot(N, N / 50.0, "--k", label = r"$\tau = N/50$")

plt.ylim(ylim)
#plt.gca().set_yticks([])
#plt.gca().set_xticks([])
plt.title('Adpt-RJMCMC convergence assessment')
plt.xlabel("Samples, N")
plt.ylabel(r"Autocorrelation time, $\tau$")
#plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/AutoCorr.png')
plt.clf()





figure = corner.corner(states_2)

ndim = 7
# Extract the axes
plt.rcParams['font.size'] = 9
axes = np.array(figure.axes).reshape((ndim, ndim))
#figure.clf()
#plt.rcParams['font.size'] = 9
# Loop over the diagonal

for i in range(ndim):
    ax = axes[i, i]

    ax.cla()
    ax.grid()
    ax.plot(np.linspace(1, len(states_2), len(states_2)), states_2[:, i], linewidth = 0.5)

    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    
# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.cla()
        ax.grid()
        ax.scatter(states_2[:, xi], states_2[:, yi], c = np.linspace(0.0, 1.0, len(states_2)), cmap = 'spring', alpha = 0.25, marker = ".", s = 20)#, markeredgewidth=0.0)
        #ax.set_title(str(yi)+str(xi))
            
        if yi == ndim - 1:
            ax.set_xlabel(symbols[xi])
            #ax.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
            ax.tick_params(axis='x', labelrotation = 45)

        else:    
            ax.axes.get_xaxis().set_ticklabels([])

        if xi == 0:
            ax.set_ylabel(symbols[yi])
            #ax.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
            ax.tick_params(axis='y', labelrotation = 45)

        else:    
            ax.axes.get_yaxis().set_ticklabels([])

figure.savefig('results/corner.png', dpi=500)
figure.clf()




figure = corner.corner(states_2)

ndim = 7
# Extract the axes
plt.rcParams['font.size'] = 9
axes = np.array(figure.axes).reshape((ndim, ndim))
#figure.clf()
#plt.rcParams['font.size'] = 9
# Loop over the diagonal

if isinstance(binary_true, np.ndarray):
    base = f.scale(theta_Model)
else:
    base = center_2

#states = states_2
m = 2

for i in range(ndim):
    ax = axes[i, i]

    ax.cla()
    ax.grid()
    #ax.plot(np.linspace(1, len(states_2), len(states_2)), states_2[:, i], linewidth = 0.5)

    ax.hist(states_2[:, i], bins = 50, density = True)



    mu = np.average(states_2[:, i])
    sd = np.std(states_2[:, i])
    ax.axvline(mu, label = r'$\mu$', color = 'cyan')
    ax.axvspan(mu - sd, mu + sd, alpha = 0.25, color='cyan', label = r'$\sigma$')

    if isinstance(binary_true, np.ndarray):
        ax.axvline(base[xi], label = 'True', color = 'red')

    ax.xaxis.tick_top()
    #ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    
# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.cla()
        #ax.grid()
        #ax.scatter(states_2[:, xi], states_2[:, yi], c = np.linspace(0.0, 1.0, len(states_2)), cmap = 'spring', alpha = 0.25, marker = ".", s = 10)#, markeredgewidth=0.0)
            
            
        yLower = np.min([np.min(states_2[:, yi]), base[yi]])
        yUpper = np.max([np.max(states_2[:, yi]), base[yi]])
        xLower = np.min([np.min(states_2[:, xi]), base[xi]])
        xUpper = np.max([np.max(states_2[:, xi]), base[xi]])

        yaxis = np.linspace(yLower, yUpper, n_points)
        xaxis = np.linspace(xLower, xUpper, n_points)
        density = np.zeros((n_points, n_points))
        x = -1
        y = -1

        for i in yaxis:
            x += 1
            y = -1
            for j in xaxis:
                y += 1
                theta = deepcopy(base)

                theta[xi] = j
                theta[yi] = i

                theta = f.unscale(m, theta)
                #print(theta[4], xaxis, yaxis)
                density[x][y] = np.exp(f.logLikelihood(m, Data, theta, priors)+f.priorProduct(m, Data, theta, priors))

        density = np.sqrt(np.flip(density, 0)) # So lower bounds meet
        #density = np.flip(density, 1) # So lower bounds meet
        ax.imshow(density, interpolation='none', extent=[xLower, xUpper, yLower, yUpper,], aspect=(xUpper-xLower) / (yUpper-yLower))#, cmap = plt.cm.BuPu_r) #
        #cbar = plt.colorbar(fraction = 0.046, pad = 0.04, ticks = [0, 1]) # empirical nice auto sizing
        #ax = plt.gca()
        #cbar.ax.set_yticklabels(['Initial\nStep', 'Final\nStep'], fontsize=9)
        #cbar.ax.yaxis.set_label_position('right')


        #https://stats.stackexchange.com/questions/60011/how-to-find-the-level-curves-of-a-multivariate-normal

        mu = [np.mean(states_2[:, xi]), np.mean(states_2[:, yi])]
        K = np.cov([states_2[:, xi], states_2[:, yi]])
        angles = np.linspace(0, 2*np.pi, 360)
        R = [np.cos(angles), np.sin(angles)]
        R = np.transpose(np.array(R))

        for levels in [0.38, 0.86, 0.98]:

            rad = np.sqrt(chi2.isf(levels, 2))
            level_curve = rad*R.dot(scipy.linalg.sqrtm(K))
            ax.plot(level_curve[:, 0]+mu[0], level_curve[:, 1]+mu[1], color = 'White')
            

        if isinstance(binary_true, np.ndarray):
            ax.scatter(base[xi], base[yi], marker = '*', s = markerSize, c = 'red', alpha = 1)
            ax.axvline(base[xi], color = 'red')
            ax.axhline(base[yi], color = 'red')



        if yi == ndim - 1:
            ax.set_xlabel(symbols[xi])
            #ax.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
            ax.tick_params(axis='x', labelrotation = 45)

        else:    
            ax.axes.get_xaxis().set_ticklabels([])

        if xi == 0:
            ax.set_ylabel(symbols[yi])
            #ax.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
            ax.tick_params(axis='y', labelrotation = 45)

        else:    
            ax.axes.get_yaxis().set_ticklabels([])

figure.savefig('results/density_corner.png', dpi=500)
figure.clf()





figure = corner.corner(states_1)

ndim = 3
# Extract the axes
plt.rcParams['font.size'] = 9
axes = np.array(figure.axes).reshape((ndim, ndim))
#figure.clf()
#plt.rcParams['font.size'] = 9
# Loop over the diagonal

for i in range(ndim):
    ax = axes[i, i]

    ax.cla()
    #ax.grid()
    #ax.plot(np.linspace(1, len(states_2), len(states_2)), states_2[:, i], linewidth = 0.5)

    ax.patch.set_alpha(0.0)
    ax.axis('off')
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    
# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.cla()
        ax.grid()
        ax.scatter(states_2[:, xi]-center_2[xi], states_2[:, yi]-center_2[yi], c = np.linspace(0.0, 1.0, len(states_2)), cmap = 'winter', alpha = 0.25, marker = ".", s = 20)#, markeredgewidth=0.0)
        ax.scatter(states_1[:, xi]-center_1[xi], states_1[:, yi]-center_1[yi], c = np.linspace(0.0, 1.0, len(states_1)), cmap = 'spring', alpha = 0.25, marker = ".", s = 20)

        #ax.set_title(str(yi)+str(xi))
            
        if yi == ndim - 1:
            ax.set_xlabel(symbols[xi])
            #ax.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
            ax.tick_params(axis='x', labelrotation = 45)

        #else:    
            ax.axes.get_xaxis().set_ticklabels([])

        if xi == 0:
            ax.set_ylabel(symbols[yi])
            #ax.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
            ax.tick_params(axis='y', labelrotation = 45)

        #else:    
            ax.axes.get_yaxis().set_ticklabels([])

figure.savefig('Plots/overlayed_corner.png', dpi=500)
figure.clf()








