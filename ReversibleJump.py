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
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8}) # strong binary
Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0056, 'q': 0.0009, 's': 1.3, 'alpha': 210.8}) # weak binary1
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0056, 'q': 0.0007, 's': 1.3, 'alpha': 210.8}) # weak binary2
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0096, 'q': 0.00002, 's': 4.9, 'alpha': 223.8}) # indistiguishable from single
Model.set_magnification_methods([0., 'VBBL', 72.])

#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5}) #  Single
#Model.set_magnification_methods([0., 'point_source', 72.])

Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
plt.savefig('temp.jpg')
plt.clf()

# Generate "Synthetic" Lightcurve
t = Model.set_times(n_epochs = 100)
error = Model.magnification(t)/50 + 0.1
Data = mm.MulensData(data_list=[t, Model.magnification(t), error], phot_fmt='flux', chi2_fmt='flux')


# priors (Zhang et al)
s_pi = f.logUniDist(0.2, 5)
q_pi = f.logUniDist(10e-6, 1)
alpha_pi = f.uniDist(0, 360)
u0_pi = f.uniDist(0, 2)
t0_pi = f.uniDist(0, 72)
tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  f.logUniDist(10**-4, 10**-2)
a = 0.1

#m_pi = [1-a, a]
#priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

# uninformative priors
s_upi = f.uniDist(0.2, 5)
q_upi = f.uniDist(10e-6, 1)
alpha_upi = f.uniDist(0, 360)
u0_upi = f.uniDist(0, 2)
t0_upi = f.uniDist(0, 72)
tE_upi = f.uniDist(1, 100)
rho_upi =  f.uniDist(10**-4, 10**-2)

priors = [t0_upi, u0_upi,  tE_upi, rho_upi,  q_upi, s_upi, alpha_upi]
m_pi = [0.5, 0.5]

# centreing points for inter-model jumps
center_1 = np.array([36., 0.133, (61.5)])
#center_1 = np.array([36., 0.133, 61.5])
#center_2 = np.array([36, 0.133, 61.5, 0.0096, np.log(0.0039), 1.14, 223.8]) # nice results for adaption strong
center_2 = np.array([36., 0.133, (61.5), (0.00563), np.log(0.00091), (1.31), 210.8]) #weak1
#center_2 = np.array([36., 0.133, (61.5), (0.005), np.log(0.0007), (1.25), 210.]) #weak1
#center_2 = np.array([36., 0.133, (61.5), (0.0052), np.log(0.0006), (1.29), 210.9]) #weak2
#center_2 = np.array([36., 0.133, (61.5), (0.0096), np.log(0.00002), (4.25), 223.8]) #single

# print(np.exp(f.logLikelihood(1, Data, center_1)))
# print(np.exp(f.logLikelihood(2, Data, center_2)))
centers = [center_1, center_2]

# initial covariances (diagonal)
covariance_1 = np.multiply(0.0001, [0.01, 0.01, 0.1])
covariance_2 = np.multiply(0.0001, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])#0.5

#covariance_1s = np.multiply(1, [0.01, 0.01, 0.1])
#covariance_2s = np.multiply(1, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])#0.5
#covariance_1 = np.outer(covariance_1s, covariance_1s)
#covariance_2 = np.outer(covariance_2s, covariance_2s)
#covariance_p = [covariance_1, covariance_2]

# Use adaptiveMCMC to calculate initial covariances
burns = 2
iters = 1000
theta_1i = center_1
theta_2i = center_2
covariance_1p, states_1, means_1, c_1 = f.AdaptiveMCMC(1, Data, theta_1i, priors, covariance_1, burns, iters)
covariance_2p, states_2, means_2, c_2 = f.AdaptiveMCMC(2, Data, theta_2i, priors, covariance_2, burns, iters)

covariance_p = [covariance_1p, covariance_2p]


# loop specific values
iterations = 1000
print(states_1[:, -1])
theta = states_1[:, -1]#[36., 0.133, 61.5]#, 0.0014, 0.0009, 1.26, 224.]
m = 1
pi = (f.logLikelihood(m, Data, f.unscale(m, theta), priors))
#print(pi)

ms = np.zeros(iterations, dtype=int)
ms[0] = m
states = []
score = 0
J_2 = np.prod(center_2[0:3])
J_1 = np.prod(center_1)
J = np.abs([J_1/J_2, J_2/J_1])
#print(J)

#adaptive params
t=[burns+iters,burns+iters]
I = [np.identity(3), np.identity(7)] 
s = [2.4**2 / 3, 2.4**2 / 7] # Arbitrary(ish), good value from Haario et al
eps = 1e-12 # Needs to be smaller than the scale of parameter values
means = [np.zeros((3, iters+burns+iterations)), np.zeros((7, iters+burns+iterations))]
#print(means[0][:,0:2])
#print(means_1)
means[0][:, 0:burns+iters] = means_1
means[1][:, 0:burns+iters] = means_2

print('Running RJMCMC')

            



for i in range(iterations): # loop through RJMCMC steps
    
    #diagnostics
    #print(f'\rLikelihood: {np.exp(pi):.3f}', end='')
    cf = i/(iterations-1);
    print(f'Current: Likelihood {np.exp(pi):.4f}, M {m} | Progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')

    mProp = random.randint(1,2) # since all models are equally likelly, this has no presence in the acceptance step
    thetaProp = f.RJCenteredProposal(m, mProp, theta, covariance_p[mProp-1], centers) #print('prop: '+str(thetaProp))

    priorRatio = np.exp(f.PriorRatio(m, mProp, f.unscale(m, theta), f.unscale(mProp, thetaProp), priors))
    
    piProp = (f.logLikelihood(mProp, Data, f.unscale(mProp, thetaProp), priors))

    #print(piProp, pi, priorRatio, mProp)
    
    if random.random() <= np.exp(piProp-pi) * priorRatio * m_pi[mProp-1]/m_pi[m-1] * J[mProp-1]: # metropolis acceptance
        theta = thetaProp
        m = mProp
        score += 1
        pi = piProp
    
    states.append(theta)
    ms[i] = m


    tr = t[m-1]
    means[m-1][:, tr] = (means[m-1][:, tr-1]*tr + theta)/(tr + 1) # recursive mean (offsets indices starting at zero by one)    
    # update step (recursive covariance)

    covariance_p[m-1] = (tr - 1)/tr * covariance_p[m-1] + s[m-1]/tr * (tr*means[m-1][:, tr - 1]*np.transpose(means[m-1][:, tr - 1]) - (tr + 1)*means[m-1][:, tr]*np.transpose(means[m-1][:, tr]) + theta*np.transpose(theta)) #+ eps*I[m-1]



    t[m-1] += 1

# performance diagnostics:
print("\nIterations: "+str(iterations))
print("Accepted Move Fraction: "+str(score/iterations))
print("P(Singular): "+str(1-np.sum(ms-1)/iterations))
print("P(Binary): "+str(np.sum(ms-1)/iterations))
#print(states)



## PLOT RESULTS ##

markerSize=75

states_2 = []
h_states_2=[]
h_ind = []
h=0
for i in range(iterations): # record all binary model states in the chain
    if ms[i] == 2: 
        states_2.append(f.unscale(2, states[i]))
        if ms[i-1] == 1: 
            h_states_2.append(f.unscale(2, states[i]))
            h_ind.append(len(states_2))


states_2=np.array(states_2)
h_states_2=np.array(h_states_2)


meanState_2 = np.median((states_2), axis=0)
#print(meanState_2)
MeanModel_2 = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], meanState_2)))
MeanModel_2.set_magnification_methods([0., 'VBBL', 72.])
Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
MeanModel_2.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red')
plt.title(np.exp(f.logLikelihood(2, Data, meanState_2, priors)))
plt.savefig('Plots/BinaryFit.png')
plt.clf()


plt.scatter(states_2[:,5], states_2[:,4], c=np.linspace(0.0, 1.0, len(states_2)), cmap='spring', alpha=0.25, marker="o")
cbar = plt.colorbar(fraction = 0.046, pad = 0.04) # empirical nice auto sizing
cbar.set_label('Time', rotation = 90)
plt.scatter(h_states_2[:,5], h_states_2[:,4], c='black', alpha=0.1, marker=".", label='Jump from M1', s=1)
plt.xlabel('s [Einstein Ring Radius]')
plt.ylabel('q [Unitless]')
plt.title('RJ binary model walk through Mass Fraction / Separation space\nwith centreing function')
plt.scatter((center_2[5]), np.exp(center_2[4]), marker=r'$\odot$', label='Centre', s=markerSize, c='black', alpha=1)
plt.scatter(1.3, 0.0009, marker='*', label='True', s=markerSize, c='black', alpha=1)#r'$\circledast$'
plt.legend()
plt.savefig('Plots/RJ-binary-Walk.png')
plt.clf()



#plt.scatter(h_ind, h_states_2[:,4], alpha=0.25, marker="*", label='Jump from M1')
plt.plot(np.linspace(1, len(states_2), len(states_2)), states_2[:,4])
plt.xlabel('Binary Iteration')
plt.ylabel('q [Unitless]')
plt.title('RJ binary model walk through q')
#plt.hlines(np.exp(center_2[4]), 0, len(states_2), label='Centre', color='black')
plt.hlines(0.0009, 0, len(states_2), label='True', color='red')#r'$\circledast$'
plt.legend()
plt.savefig('Plots/RJ-q-binary-Walk.png')
plt.clf()



plt.hist(states_2[:,4], bins=50, density=True)
plt.vlines(0.0009, 0, plt.gca().get_ylim()[1], label='True', color='red')
plt.vlines(np.exp(center_2[4]), 0, plt.gca().get_ylim()[1], label='Centre', color='black')
plt.legend()
plt.savefig('Plots/RJ-Binary-q-dist')
plt.clf()



## SINGLE MODEL ##



states_1 = []
h_states_1 = []
for i in range(iterations): # record all single model states in the chain
    if ms[i] == 1: 
        states_1.append(f.unscale(1, states[i]))
        if ms[i-1] == 2: h_states_1.append(f.unscale(1, states[i]))
states_1=np.array(states_1)
h_states_1=np.array(h_states_1)


meanState_1 = np.median(states_1, axis=0)
MeanModel_1 = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], meanState_1)))
MeanModel_1.set_magnification_methods([0., 'point_source', 72.])
Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
MeanModel_1.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red')
plt.title(np.exp(f.logLikelihood(1, Data, meanState_1, priors)))
plt.savefig('Plots/SingleFit.png')
plt.clf()



plt.scatter((states_1[:,2]), (states_1[:,1]), c=np.linspace(0.0, 1.0, len(states_1)), cmap='spring', alpha=0.25, marker="o")
cbar = plt.colorbar(fraction = 0.046, pad = 0.04) # empirical nice auto sizing
cbar.set_label('Time', rotation = 90)
plt.scatter(h_states_1[:,2], h_states_1[:,1], c='black', alpha=0.1, marker=".", label='Jump from M2', s=1)
plt.xlabel('u0 [?]')
plt.ylabel('tE [?]')
plt.title('RJ single model walk through minimum impact parameter / Einstein crossing time \nwith centreing function')

plt.scatter(center_1[2], center_1[1], marker=r'$\odot$', label='Centre', s=markerSize, c='black', alpha=1)
#plt.scatter(61.5, 0.133, marker='*', label='True', s=markerSize, c='black', alpha=1)#r'$\circledast$'

plt.legend()
plt.savefig('Plots/RJ-single-Walk.png')
plt.clf()


plt.plot(np.linspace(1, iterations, num=iterations), ms)
plt.title('RJ Weak binary event with centreing function\nmodel trace')
plt.xlabel('Iteration')
plt.ylabel('Model Index')
plt.locator_params(axis="y", nbins=2)
plt.savefig('Plots/Trace.png')
plt.clf()
