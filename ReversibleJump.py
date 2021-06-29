# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Main]

import MulensModel as mm
import Functions as f
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform



## INITIALISATION ##

# Synthetic Event Parameters
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8}) # strong binary
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0056, 'q': 0.0009, 's': 1.3, 'alpha': 210.8}) # weak binary
Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.00002, 's': 1.3, 'alpha': 223.8}) # indistiguishable from single
Model.set_magnification_methods([0., 'VBBL', 72.])

Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
plt.savefig('temp.jpg')
plt.clf()

# Generate "Synthetic" Lightcurve
t = Model.set_times(n_epochs = 100)
error = Model.magnification(t)/20 + 0.25
Data = mm.MulensData(data_list=[t, Model.magnification(t), error], phot_fmt='flux', chi2_fmt='flux')

# priors (Zhang et al)
s_pi = f.logUniDist(0.2, 5)
q_pi = f.logUniDist(10e-6, 1)
alpha_pi = f.uniDist(0, 360)
u0_pi = f.uniDist(0, 2)
t0_pi = f.uniDist(0, 72)
tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  f.logUniDist(10**-4, 10**-2)
priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

# centreing points for inter-model jumps
center_1 = np.array([36., 0.133, np.log(61.5)])
#center_1 = np.array([36., 0.133, 61.5])
#center_2 = np.array([36, 0.133, 61.5, 0.00096, 0.000002, 3.3, 223.8]) # nice results for adaption
#center_2 = np.array([36., 0.133, np.log(61.5), np.log(0.005), np.log(0.0007), np.log(1.25), 210.]) # nice results for model
center_2 = np.array([36., 0.133, np.log(61.5), np.log(0.0009), np.log(0.00007), np.log(1.25), 221.])

# print(np.exp(f.logLikelihood(1, Data, center_1)))
# print(np.exp(f.logLikelihood(2, Data, center_2)))
centers = [center_1, center_2]

# initial covariances (diagonal)
covariance_1 = np.multiply(0.0001, [0.01, 0.01, 0.1])
covariance_2 = np.multiply(0.0001, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])#0.5
covariance_p = [covariance_1, covariance_2]

'''
# Use adaptiveMCMC to calculate initial covariances
burns = 200
iters = 200
#covariance_1p, states_1, c_1 = f.AdaptiveMCMC(1, Data, theta_1i, priors, covariance_1, 200, 200)
covariance_2p, states_2, c_2 = f.AdaptiveMCMC(2, Data, theta_2i, priors, covariance_2, burns, iters)

#covariance_p = [covariance_1p, covariance_2p]
'''

# loop specefic values
iterations = 100

theta = [36., 0.133, np.log(61.5)]#, 0.0014, 0.0009, 1.26, 224.]
m = 1
pi = np.exp(f.logLikelihood(m, Data, theta, priors))

ms = np.zeros(iterations, dtype=int)
ms[0] = m
states = []
score = 0
J_2 = np.prod(center_2[0:3])
J_1 = np.prod(center_1)
J = np.abs([J_1/J_2, J_2/J_1])
print(J)


for i in range(iterations): # loop through RJMCMC steps

    mProp = random.randint(1,2) # since all models are equally likelly, this has no presence in the acceptance step
    thetaProp = f.RJCenteredProposal(m, mProp, theta, covariance_p[mProp-1], centers) #print('prop: '+str(thetaProp))
    piProp = np.exp(f.logLikelihood(mProp, Data, thetaProp, priors))

    priorRatio = np.exp(f.PriorRatio(m, mProp, theta, thetaProp, priors))
    
    if random.random() <= piProp/pi * priorRatio * J[mProp-1]: # metropolis acceptance
        theta = thetaProp
        m = mProp
        score += 1
        pi = piProp
    
    states.append(theta)
    ms[i] = m

# performance diagnostics:
print("acc: "+str(score/iterations))
print("1: "+str(1-np.sum(ms-1)/iterations))
print("2: "+str(np.sum(ms-1)/iterations))
#print(states)



## PLOT RESULTS ##

markerSize=50

states_2 = [] 
for i in range(iterations): # record all binary model states in the chain
    if ms[i] == 2: states_2.append(f.unscale(2, states[i]))
states_2=np.array(states_2)


meanState_2 = np.median((states_2), axis=0)
print(meanState_2)
MeanModel_2 = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], meanState_2)))
MeanModel_2.set_magnification_methods([0., 'VBBL', 72.])
Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
MeanModel_2.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red')
plt.savefig('Plots/BinaryFit.png')
plt.clf()


plt.scatter(states_2[:,5], ((states_2[:,4])), alpha=0.25)
plt.xlabel('s [Einstein Ring Radius]')
plt.ylabel('q [Unitless]')
plt.title('RJ binary model walk through Mass Fraction / Separation space\nwith centreing function')
plt.scatter((states_2[0,5]), (states_2[0,4]), c=[(0,1,0)], marker='2', label='Initial\nPosition', s=markerSize)
plt.scatter((center_2[5]), (center_2[4]), c=[(1,0,0)], marker='2', label='Centreing\npoint', s=markerSize)
plt.scatter(1.3, 0.0009, c=[(0,0,1)], marker='2', label='True', s=markerSize)
plt.legend()
plt.savefig('Plots/RJ-binary-Walk.png')
plt.clf()


states_1 = []
for i in range(iterations): # record all single model states in the chain
    if ms[i] == 1: states_1.append(f.unscale(1, states[i]))
states_1=np.array(states_1)


meanState_1 = np.median((states_1), axis=0)
MeanModel_1 = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], meanState_1)))
MeanModel_1.set_magnification_methods([0., 'point_source', 72.])
Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
MeanModel_1.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red')
plt.savefig('Plots/SingleFit.png')
plt.clf()



plt.scatter((states_1[:,2]), (states_1[:,1]), alpha=0.25)
plt.xlabel('u0 [?]')
plt.ylabel('tE [?]')
plt.title('RJ single model walk through minimum impact parameter / Einstein crossing time \nwith centreing function')
plt.scatter(states_1[0,2], states_1[0,1], c=[(0,1,0)], marker='2', label='Initial\nPosition', s=markerSize)
plt.scatter(center_1[2], center_1[1], c=[(1,0,0)], marker='2', label='Centreing\npoint', s=markerSize)
#plt.scatter(1.33, 0.0004, c=[(0,0,1)], marker='2', label='True', s=markerSize)
plt.savefig('Plots/RJ-single-Walk.png')
plt.clf()


plt.plot(np.linspace(1, iterations, num=iterations), ms)
plt.title('RJ Weak binary event with centreing function\nmodel trace')
plt.xlabel('Iteration')
plt.ylabel('Model Index')
plt.locator_params(axis="y", nbins=2)
plt.savefig('Plots/Trace.png')
plt.clf()
