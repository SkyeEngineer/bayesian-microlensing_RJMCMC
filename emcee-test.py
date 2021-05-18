from numpy.core.numeric import Inf
import MulensModel as mm
import numpy as np
import emcee

import random


from scipy.stats import truncnorm, loguniform, uniform

import matplotlib.pyplot as plt

random.seed(99)

# Synthetic Event Parameters/Initialisation
SBModel = mm.Model({'t_0': 2452848.06, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8})
SBModel.set_magnification_methods([2452833., 'VBBL', 2452845.])

t=SBModel.set_times()
# Generate Synthetic Lightcurve
SBData = mm.MulensData(data_list=[t, SBModel.magnification(t), SBModel.magnification(t)*0+0.003]) #orignally 0.03


# Get likelihood by creating a new model and fitting to previously generated data
def log_prob(p):
    t_0, u_0, t_E, q, s, alpha = p

    # Transform from log units
    #s=np.exp(s)
    #q=np.exp(q)
    #t_E=np.exp(t_E)
    #return -Inf
    z=-Inf

    # Uniform Priors
    if (s<0.2 or s>5):
        return z
        #s=abs(s)

    if (q<10e-6 or q>1):
        return z
        #q=abs(q)

    if (alpha<0 or alpha>360):
        return z
        #alpha=abs(alpha)

    if (u_0<0 or u_0>2):
        return z
        #u_0=abs(u_0)

    if (t_0<0 or t_0>72):
        return z
        #t_0=abs(t_0)

    if (t_E<1 or t_E>100):
        #print("-ve")
        return z
        #t_E=abs(t_E)




    #Model = mm.Model({'t_0': t_0+2450000, 'u_0': u_0, 't_E': t_E, 'rho': 0.00096, 'q': q, 's': s, 'alpha': alpha})
    try: #for when moves out of bounds of model valididty
        Model = mm.Model({'t_0': t_0+2452848.06-72/2, 'u_0': u_0, 't_E': t_E, 'rho': 0.00096, 'q': q, 's': s, 'alpha': alpha})
        Model.set_magnification_methods([2452833., 'VBBL', 2452845.]) #?
        Event = mm.Event(datasets=SBData, model=Model)
    except: #make more specific
        return z

    return -Event.get_chi2()/2 # takes log prob


#EMCEE
burn=10#50
runs=100

true=[2452848.06-2450000, 0.133, 61.5, 0.0039, 1.120, 223.8] # Not searching on rho?

# Diagonal
cov=np.multiply(0.1,[0.72, 0.02, 1, 0.01, 0.05, 5])#0.5
#genrange=[72, 2, 0, 1-10e-6, 4.8, 360]
#base=[0, 0, 0, 10e-6, 0.2, 0]

n=12

#i = np.array(base+np.multiply(np.random.rand(n, 6), genrange))
#i = true+np.multiply(np.random.rand(n, 6)-1,cov)

s = loguniform.rvs(0.2, 5, size=n)
q = loguniform.rvs(10e-6, 1, size=n)

alpha = uniform.rvs(0, 360, size=n)
u_0 = uniform.rvs(0, 2, size=n)
t_0 = uniform.rvs(0, 72, size=n)



a, b = (np.log(1) - np.log(10**1.15)) / np.log(10**0.45), (np.log(100) - np.log(10**1.15)) / np.log(10**0.45)
t_E = truncnorm.rvs(a, b, size=n)
t_E = np.exp(t_E)

#print(s)
#print(alpha)
#print(t_E)

i=np.column_stack((t_0, u_0, t_E, q, s, alpha))
#print(i)

#print(r)

#sampler = emcee.EnsembleSampler(n, 6, log_prob)
sampler = emcee.EnsembleSampler(n, 6, log_prob, moves=[emcee.moves.GaussianMove(cov, mode='vector', factor=None)])



state = sampler.run_mcmc(i, burn, skip_initial_state_check=True)
sampler.reset()
sampler.run_mcmc(state, runs, skip_initial_state_check=True)
#, skip_initial_state_check=True

walk=sampler.get_chain(flat=True)

walk=np.array(walk)
#walk[:,2]=np.exp(walk[:,2])


print(walk)

acc=np.mean(sampler.acceptance_fraction)
print(acc)








#PLOTTING



#plt.imshow(result, cmap='hot', interpolation='none', extent=[1, 1.3, 0.2, 0.001,])

#print(walk[:,4])
plt.scatter(np.abs(walk[:,4]), np.abs(walk[:,3]), alpha=0.5)
plt.xlabel('s [Einstein Ring Radius]') # Set the y axis label of the current axis.
plt.ylabel('q') # Set a title.
plt.title('sep/mfrac Walk: Acc: ['+str(acc)+']'+' Runs: ['+str(runs*n)+']'+' Burns: ['+str(burn)+']')
plt.savefig('sqwalkAFFC.png')
