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



import PlotFunctions as pltf

import os
import os.path
import shutil
from pathlib import Path


pltf.Style()




labels = [r'Impact Time [$days$]', r'Minimum Impact Parameter', r'Einstein Crossing Time [$days$]', r'Rho', r'ln(Mass Ratio)', r'Separation', r'Alpha']
symbols = [r'$t_0$', r'$u_0$', r'$t_E$', r'$\rho$', r'$q$', r'$s$', r'$\alpha$']


## INITIALISATION ##

sn = 4

# Synthetic Event Parameters
theta_Models = [
    [36, 0.133, 31.5, 0.0096, 0.002, 1.27, 210.8], # strong binary
    [36, 0.133, 31.5, 0.0096, 0.00091, 1.3, 210.8], # weak binary 1
    [36, 0.833, 21.5, 0.0056, 0.025, 1.3, 210.8], # weak binary 2
    [36, 0.133, 31.5, 0.0096, 0.0002, 4.9, 223.8], # indistiguishable from single
    [36, 0.533, 21.5]  # single
    ]
theta_Model = np.array(theta_Models[sn])
# 36, 'u_0': 0.833, 't_E': 21.5, 'rho': 0.0056, 'q': 0.025, 's': 1.3, 'alpha': 210.8
#Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta_Model)))
#Model.set_magnification_methods([0., 'VBBL', 72.])

Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta_Model)))
Model.set_magnification_methods([0., 'point_source', 72.])

#Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('temp.jpg')
#plt.clf()


#0, 50, 25, 0.3
# Generate "Synthetic" Lightcurve
#epochs = Model.set_times(n_epochs = 720)
n_epochs = 720
epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
true_data = Model.magnification(epochs)
#epochs = Model.set_times(n_epochs = 100)
#error = Model.magnification(epochs) * 0 + np.max(Model.magnification(epochs))/60 #Model.magnification(epochs)/100 + 0.5/Model.magnification(epochs)
random.seed(a = 99, version = 2)

signal_to_noise_baseline = np.random.uniform(23.0, 230.0)
noise = np.random.normal(0.0, np.sqrt(true_data) / signal_to_noise_baseline, n_epochs) 
noise_sd = np.sqrt(true_data) / signal_to_noise_baseline
error = noise_sd
model_data = true_data + noise
Data = mm.MulensData(data_list = [epochs, model_data, noise_sd], phot_fmt = 'flux', chi2_fmt = 'flux')

signal_n_epochs = 720
signal_epochs = np.linspace(0, 72, signal_n_epochs + 1)[:signal_n_epochs]

true_signal_data = Model.magnification(signal_epochs)
signal_data = model_data


#print(Model.magnification(epochs))

iterations = 10000



# priors (Zhang et al)
s_pi = f.logUniDist(0.2, 5)
q_pi = f.logUniDist(10e-6, 1)
#q_pi = f.uniDist(10e-6, 0.1)
alpha_pi = f.uniDist(0, 360)
u0_pi = f.uniDist(0, 2)
t0_pi = f.uniDist(0, 72)
tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  f.logUniDist(10**-4, 10**-2)
a = 0.5
#m_pi = [1 - a, a]
#priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

# uninformative priors
s_upi = f.uniDist(0.2, 5)
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

single_Sposterior = interf.get_posteriors(1)
binary_Sposterior = interf.get_posteriors(2)



#u_full = binary_Sposterior.sample((1, ), x = Data.flux)
#u_full.numpy
#u = np.float64(u_full[0])[3:]
#print(u)
#throw=throw

#arr, l_arr = interf.get_model_ensemble(binary_Sposterior, Data.flux, 1)

def ParralelMain(arr):

    sn, Data, signal_data, priors, binary_Sposterior, single_Sposterior, m_pi, iterations, Model, error, epochs = arr

    # centreing points for inter-model jumps
    #center_1s = np.array([36., 0.133, 31.5])
    #center_1 = np.array([36., 0.133, 31.5])

    center_2ss = [
        [36, 0.133, 61.5, 0.0096, np.log(0.002), 1.27, 210.8], # strong binary
        [36., 0.133, 61.5, 0.00963, np.log(0.00092), 1.31, 210.8], # weak binary 1
        [36., 0.133, 61.5, 0.0052, np.log(0.0006), 1.29, 210.9], # weak binary 2
        [36., 0.133, 61.5, 0.0096, np.log(0.00002), 4.25, 223.8], # indistiguishable from single
        ]
    #center_2 = np.array(center_2s[sn])

    center_2s = interf.get_model_centers(binary_Sposterior, signal_data)
    #center_2s = np.array([3.60166321e+01, 1.33796528e-01, 3.12476940e+01, 1.09757202e-04, 9.51249094e-04, 1.06277907e+00, 2.07451248e+02])


    #center_2s[4] = np.log(center_2s[4])
    #center_2s = f.scale(center_2s)

    #print("\n", center_2, " hi")
    center_1s = interf.get_model_centers(single_Sposterior, signal_data)
    #print(Data.flux)
    #binary_ensemble = interf.get_model_ensemble(binary_posterior, Data.flux, 100000)

    pltf.LightcurveFitError(2, center_2s, priors, Data, Model, epochs, error, True, "BinaryCenterSurr")
    pltf.LightcurveFitError(1, center_1s, priors, Data, Model, epochs, error, True, "SingleCenterSurr")

    
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
    cov_scale = 0.01 #0.01
    covariance_1 = np.multiply(cov_scale, [0.1, 0.01, 0.1])
    covariance_2 = np.multiply(cov_scale, [0.1, 0.01, 0.1, 0.0001, 0.1, 0.01, 1])#0.5

    #covariance_1s = np.multiply(1, [0.01, 0.01, 0.1])
    #covariance_2s = np.multiply(1, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])#0.5
    #covariance_1 = np.outer(covariance_1s, covariance_1s)
    #covariance_2 = np.outer(covariance_2s, covariance_2s)
    #covariance_p = [covariance_1, covariance_2]

    # Use adaptiveMCMC to calculate initial covariances
    burns = 25
    iters = 2500 #250
    theta_1i = center_1s
    theta_2i = f.scale(center_2s)
    covariance_1p, states_1, means_1, c_1, null, bests, bestt_1 = f.AdaptiveMCMC(1, Data, theta_1i, priors, covariance_1, burns, iters)
    covariance_2p, states_2, means_2, c_2, null, bests, bestt_2 = f.AdaptiveMCMC(2, Data, theta_2i, priors, covariance_2, burns, iters)

    covariance_p = [covariance_1p, covariance_2p]
    #print(covariance_1p)
    #print(covariance_2p)
    #throw=throw

    center_1 = bestt_1
    center_2 = bestt_2
    
    #print(center_1s, center_1)
    #print(center_2s, center_2)

    #print("Center 1", center_1, "True Chi", -(f.logLikelihood(1, Data, center_1, priors)))
    #print("Center 2", center_2, "True Chi", -(f.logLikelihood(2, Data, f.unscale(2, center_2), priors)))
    centers = [center_1, center_2]


    pltf.LightcurveFitError(2, f.unscale(2, bestt_2), priors, Data, Model, epochs, error, True, "BinaryCenterMCMC")
    pltf.LightcurveFitError(1, bestt_1, priors, Data, Model, epochs, error, True, "SingleCenterMCMC")


    # loop specific values

    #print(states_1[:, -1])
    theta = states_2[:, -1]#[36., 0.133, 61.5]#, 0.0014, 0.0009, 1.26, 224.]
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
    adaptive_score = [[], []]
    #inter_props = [0, 0]

    n_samples = 10000
    samples, log_prob_samples = interf.get_model_ensemble(binary_Sposterior, signal_data, n_samples)
    samples = (samples - f.unscale(2, center_2))/1 + f.unscale(2, center_2)

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
            if m == mProp: adaptive_score[mProp - 1].append(1)

            theta = thetaProp
            m = mProp
            score += 1
            pi = piProp
            if mProp == 2: mem_2 = thetaProp

            if bests[mProp-1] < np.exp(piProp): 
                bests[mProp-1] = np.exp(piProp)
                bestt[mProp-1] = f.unscale(mProp, thetaProp)

        elif m == mProp: adaptive_score[mProp - 1].append(0)


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

        #scale = 1
        states.append(theta)
        ms[i] = m


        tr = t[m-1]
        
        #means[m-1][:, tr] = (means[m-1][:, tr-1]*tr + theta)/(tr + 1) # recursive mean (offsets indices starting at zero by one)    
        # update step (recursive covariance)

        #covariance_p[m-1] = (tr - 1)/tr * covariance_p[m-1] + s[m-1]/tr * (tr*means[m-1][:, tr - 1]*np.transpose(means[m-1][:, tr - 1]) - (tr + 1)*means[m-1][:, tr]*np.transpose(means[m-1][:, tr]) + theta*np.transpose(theta)) #+ eps*I[m-1]
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

    return states, adaptive_score, ms, bestt, bests, centers


params = [sn, Data, signal_data, priors, binary_Sposterior, single_Sposterior, m_pi, iterations,  Model, error, epochs]

states, adaptive_score, ms, bestt, bests, centers = ParralelMain(params)
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









#states_2df = pd.DataFrame(data = states_2, columns = labels)
#print(states_2df)

#grid = sns.PairGrid(data = states_2df, vars = labels, height = 7)
#grid = grid.map_upper(pltf.PPlotWalk)
#plt.savefig('Plots/RJ-binary-pplot.png')
#plt.clf()



#pltf.LightcurveFitError(2, bestt[1][:], priors, Data, Model, epochs, error, details, 'BestBinary')

#pltf.LightcurveFitError(1, bestt[0][:], priors, Data, Model, epochs, error, details, 'BestSingle')

letters = ['t0', 'u0', 'tE', 'p', 'q', 's', 'a']
if True:

    binary_true = f.scale(theta_Model)

    for i in range(7):


        
        pltf.TracePlot(i, states_2, jumpStates_2, h_ind, labels, symbols, letters, 'binary', center_2, binary_true)
        pltf.DistPlot(i, states_2, labels, symbols, letters, 'binary', center_2, binary_true)
        
        for j in range(i+1, 7):
            pltf.PlotWalk(i, j, states_2, labels, symbols, letters, 'binary', center_2, binary_true)

    ## SINGLE MODEL ##

    single_true = False#f.scale(theta_Model)

    for i in range(3):

        pltf.TracePlot(i, states_1, h_states_1, h_ind1, labels, symbols, letters, 'single', center_1, single_true)
        pltf.DistPlot(i, states_1, labels, symbols, letters, 'single', center_1, single_true)

        for j in range(i+1, 3):
            pltf.PlotWalk(i, j, states_1, labels, symbols, letters, 'single', center_1, single_true)


sampled_curves = random.sample(range(0, np.size(states_2, 0)), 50)#int(0.1*np.size(states_2, 0)))
for i in sampled_curves:
    pltf.PlotLightcurve(2, f.unscale(2, states_2[i, :]), 'Samples', 'red', 0.1, False, [0,72])
pltf.PlotLightcurve(2, theta_Model, 'True', 'black', 1, False, [0, 72])
#plt.legend()
plt.tight_layout()
plt.title('Joint Dist Samples | m = 2')
plt.xlabel('time [days]')
plt.ylabel('Magnification')
plt.savefig('Plots/RJMCMC-Samples.png')
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
N = np.exp(np.linspace(np.log(1000), np.log(iterations), 10)).astype(int)

new = np.zeros(len(N))
y = np.array(AC.scalarPolyProjection(states))
for i, n in enumerate(N):
    new[i] = MC.autocorr.integrated_time(y[:n], c=5, tol=50, quiet=True)

#
plt.loglog(N, new, "o-")#, label=r"$\tau$")

ylim = plt.gca().get_ylim()
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")

plt.ylim(ylim)
#plt.gca().set_yticks([])
#plt.gca().set_xticks([])
plt.title('RJMCMC Bias')
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
#plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/AutoCorr.png')
plt.clf()

