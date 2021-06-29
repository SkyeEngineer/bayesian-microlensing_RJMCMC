# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Adaptive Markov Chain Monte Carlo Testing]

import MulensModel as mm
import Functions as f
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform

# TESTING

states = np.zeros((2, 3))
states[:, 0] = [1, 0.5]

means = np.zeros((2, 3))
means[:, 0] = [1, 0.5]

t=1
theta = [2, 0.75]
states[:, t] = theta
means[:, t] = (means[:, t-1]*t + theta)/(t + 1) # recursive mean (offsets indices starting at zero by one)


#print(states[:, 0:2])

t=2
theta = [0.75, 1]
states[:, 2] = theta
means[:, t] = (means[:, t-1]*t + theta)/(t + 1) # recursive mean (offsets indices starting at zero by one)
#print(means)

covariance = np.cov(np.transpose(states[:, 0:2]))
print(covariance)

# update step (recursive covariance)
covariance = (t-1)/t * covariance + (1/t) * (t*np.outer(means[:, t - 1], means[:, t - 1]) - (t + 1)*np.outer(means[:, t-0], means[:, t-0]) + np.outer(states[:, t-0], states[:, t-0]))

print(covariance)
print(np.cov(states))
print((1/t) * (t*(means[:, t - 1].T).dot(means[:, t - 1]) - (t + 1)*(means[:, t-0].T).dot(means[:, t-0]) + (states[:, t-0].T).dot(states[:, t-0])))
print(np.outer(means[:, t-0], means[:, t-0].T))

g=h

# INITIALISATION

# Synthetic Event Parameters
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8}) # strong binary
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0004, 's': 1.33, 'alpha': 223.8}) # weak binary
Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.000002, 's': 4.9, 'alpha': 223.8}) # indistiguishable from single
Model.set_magnification_methods([0., 'VBBL', 72.])

# Generate "Synthetic" Lightcurve
t = Model.set_times(n_epochs = 100)
error = Model.magnification(t)/5
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

# initial points
theta_1i = np.array([36., 0.133, 61.5])
#theta_1i = np.array([36., 0.133, 61.5])
theta_2i = np.array([36, 0.133, 61.5, 0.00096, 0.000002, 3.3, 223.8]) # nice results for adaption
#theta_2i = np.array([36., 0.133, 61.5, 0.0014, 0.00096, 1.2, 224.]) # nice results for model
# print(np.exp(f.logLikelihood(1, Data, theta_1i)))
# print(np.exp(f.logLikelihood(2, Data, theta_2i)))

# initial covariances (diagonal)
covariance_1i=np.multiply(0.0001, [0.01, 0.01, 0.1])
covariance_2i=np.multiply(0.0001, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])

burns=200
iters=200

#covariance_1p, states_1, c_1 = f.AdaptiveMCMC(1, Data, theta_1i, priors, covariance_1i, 200, 200)
covariance_2p, states_2, c_2 = f.AdaptiveMCMC(2, Data, theta_2i, priors, covariance_2i, burns, iters)

# Create and plot the accepatnce ratio over time
acc = []
size = 40
bins = int((burns + iters) / size)

for bin in range(bins): # record the ratio of acceptance for each bin
    acc.append(np.sum(c_2[size*bin:size*(bin+1)]) / size)

plt.plot(np.linspace(1, bins, num=bins), acc)
plt.xlabel('Iterations [bins]')
plt.ylabel('Acceptance rate')
plt.title('Adaptive MCMC acceptance timeline')
plt.savefig('Plots/Adaptive-MCMC-acceptance-progression.png')
plt.clf()



#print(np.cov(states_1))
#print(np.cov(states_2))

#print((covariance_1p))
print((covariance_2p))

#print(np.prod(covariance_1i))
print(np.prod(covariance_2i))
#print(np.linalg.det(covariance_1p))
#print(np.linalg.det(covariance_2p))

#print(np.linalg.det(np.cov(states_1)))
print((np.cov(states_2))) # clearly, adaption is calculating wrong

# plot the points visited during the walk
plt.scatter((states_2[5,:]), (states_2[4,:]), alpha=0.25)
plt.xlabel('s [Einstein Ring Radius]')
plt.ylabel('q')
plt.title('Adaptive f walk over binary model space')
plt.scatter(1.33, 0.0004, c=[(0,0,1)], marker='2', label='True', s=50)
plt.scatter(theta_2i[5], theta_2i[4], c=[(1,0,0)], marker='2', label='Initial\nPosition', s=50)
plt.legend()
plt.savefig('Plots/Adaptive-Covariance-Sampleing-Walk.png')
plt.clf()
