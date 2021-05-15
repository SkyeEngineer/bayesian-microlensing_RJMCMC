import MulensModel as mm
import numpy as np
import emcee

import matplotlib.pyplot as plt

# Synthetic Event Parameters/Initialisation
SBModel = mm.Model({'t_0': 2452848.06, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8})
SBModel.set_magnification_methods([2452833., 'VBBL', 2452845.])

t=SBModel.set_times()
# Generate Synthetic Lightcurve
SBData = mm.MulensData(data_list=[t, SBModel.magnification(t), SBModel.magnification(t)*0+0.003]) #orignally 0.03


# Get likelihood by creating a new model and fitting to previously generated data
def log_prob(p):
    t_0, u_0, t_E, q, s, alpha = p
    if q<0:
        q=-q
    if u_0<0:
        u_0=-u_0
    if s<0:
        s=-s
    Model = mm.Model({'t_0': t_0+2450000, 'u_0': u_0, 't_E': t_E, 'rho': 0.00096, 'q': q, 's': s, 'alpha': alpha})
    Model.set_magnification_methods([2452833., 'VBBL', 2452845.])

    Event = mm.Event(datasets=SBData, model=Model)

    return np.exp(-Event.get_chi2()/2)


#EMCEE
burn=20
runs=50

true=[2452848.06-2450000, 0.133, 61.5, 0.0039, 1.120, 223.8] # Not searching on rho?

# Diagonal
cov=np.multiply(1,[5, 0.1, 5, 0.001, 0.1, 4])


n=10
i = true+np.multiply(np.random.rand(n, 6)-1,cov)



sampler = emcee.EnsembleSampler(n, 6, log_prob, moves=[emcee.moves.GaussianMove(cov, mode='vector', factor=None)])



state = sampler.run_mcmc(i, burn, skip_initial_state_check=True)
sampler.reset()
sampler.run_mcmc(state, runs, skip_initial_state_check=True);

walk=sampler.get_chain(flat=True)
#walk=np.array(walk)
#print(walk)

acc=np.mean(sampler.acceptance_fraction)
print(acc)








#PLOTTING



#plt.imshow(result, cmap='hot', interpolation='none', extent=[1, 1.3, 0.2, 0.001,])

#print(walk[:,4])
plt.scatter(np.abs(walk[:,4]), np.abs(walk[:,3]), alpha=0.5)
plt.xlabel('s [Einstein Ring Radius]') # Set the y axis label of the current axis.
plt.ylabel('q') # Set a title.
plt.title('sep/mfrac Gaussian Walk: Acc: ['+str(acc)+']'+' Runs: ['+str(runs*n)+']'+' Burns: ['+str(burn)+']')
plt.savefig('sqwalk.png')
