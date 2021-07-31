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
import ReversibleJump as RJ


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
    [36, 0.633, 31.5, 0.0096, 0.025, 1.27, 210.8], # 0 strong binary
    [36, 0.133, 31.5, 0.0096, 0.00091, 1.3, 210.8], # 1 weak binary 1
    [36, 0.933, 21.5, 0.0056, 0.065, 1.1, 210.8], # 2 weak binary 2
    [36, 0.833, 31.5, 0.0096, 0.0001, 4.9, 223.8], # 3 indistiguishable from single
    [36, 1.633, 31.5]  # 4 single
    ]
theta_Model = np.array(theta_Models[sn])
binary_true = False#f.scale(theta_Model)
single_true = f.scale(theta_Model)


# 36, 'u_0': 0.833, 't_E': 21.5, 'rho': 0.0056, 'q': 0.025, 's': 1.3, 'alpha': 210.8
#Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta_Model)))
#Model.set_magnification_methods([0., 'VBBL', 72.])

Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta_Model)))
Model.set_magnification_methods([0., 'point_source', 72.])

Model.plot_magnification(t_range = [0, 72], subtract_2450000 = False, color = 'black')
plt.savefig('temp.jpg')
plt.clf()

throw=throw

#0, 50, 25, 0.3
# Generate "Synthetic" Lightcurve
#epochs = Model.set_times(n_epochs = 720)
n_epochs = 720
epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
true_data = Model.magnification(epochs)
#epochs = Model.set_times(n_epochs = 100)
#error = Model.magnification(epochs) * 0 + np.max(Model.magnification(epochs))/60 #Model.magnification(epochs)/100 + 0.5/Model.magnification(epochs)
random.seed(a = 99, version = 2)

signal_to_noise_baseline = 123.0#np.random.uniform(23.0, 230.0)
noise = np.random.normal(0.0, np.sqrt(true_data) / signal_to_noise_baseline, n_epochs) 
noise_sd = np.sqrt(true_data) / signal_to_noise_baseline
error = noise_sd
model_data = true_data + noise
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

iterations = 25000



# priors (Zhang et al)
s_pi = f.logUniDist(0.2, 5)
#q_pi = f.logUniDist(10e-6, 1)
q_pi = f.uniDist(10e-6, 0.1)
alpha_pi = f.uniDist(0, 360)
u0_pi = f.uniDist(0, 2)
t0_pi = f.uniDist(0, 72)
tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  f.logUniDist(10**-4, 10**-2)
a = 0.5
#m_pi = [1 - a, a]
priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

# uninformative priors
s_upi = f.uniDist(0.2, 3)
q_upi = f.uniDist(10e-6, 0.1)
alpha_upi = f.uniDist(0, 360)
u0_upi = f.uniDist(0, 2)
t0_upi = f.uniDist(0, 72)
tE_upi = f.uniDist(1, 100)
rho_upi =  f.uniDist(10**-4, 10**-2)

#priors = [t0_upi, u0_upi,  tE_upi, rho_upi,  q_upi, s_upi, alpha_upi]
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




params = [sn, Data, signal_data, priors, binary_Sposterior, single_Sposterior, m_pi, iterations,  Model, error, epochs]

states, adaptive_score, ms, bestt, bests, centers, covs = RJ.ParralelMain(params)
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



    n_density = 10

    pltf.AdaptiveProgression(adaptive_score[1], covs[1][:], 'binary')

    for i in range(7):


        
        #pltf.TracePlot(i, states_2, jumpStates_2, h_ind, labels, symbols, letters, 'binary', center_2, binary_true)
        pltf.DistPlot(i, states_2, labels, symbols, letters, 2, 'binary', center_2, binary_true, priors, Data)
        
        for j in range(i+1, 7):
            pltf.PlotWalk(i, j, states_2, labels, symbols, letters, 'binary', center_2, binary_true)

            pltf.contourPlot(i, j, states_2, labels, symbols, letters, 'binary', center_2, binary_true, 2, priors, Data, n_density)

    ## SINGLE MODEL ##

    pltf.AdaptiveProgression(adaptive_score[0], covs[0][:], 'single')

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
    pltf.PlotLightcurve(ms[i], f.unscale(ms[i], np.array(states[i])), 'Samples', 'red', 0.1, False, [0,72])

if len(theta_Model)>5:
    pltf.PlotLightcurve(2, theta_Model, 'True', 'black', 1, False, [0, 72])
else:
    pltf.PlotLightcurve(1, theta_Model, 'True', 'black', 1, False, [0, 72])
#plt.legend()
plt.scatter(epochs, Data.flux, label = 'signal', color = 'grey', s=1)
plt.title('Joint Dist Samples')
plt.xlabel('time [days]')
plt.ylabel('Magnification')
plt.tight_layout()
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

