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

