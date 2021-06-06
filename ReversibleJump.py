import MCMC
import Functions
import random
import numpy as np




m = random.randint(1,3)
J = 1 #THIS IS NOT CORRECT

ms = np.zeros((iterations))
ms[0] = m

covProp
covProp[1], h = AdaptiveMCMC(1, data, SurrogatePosterior[1].rvs, covariance, 200)
covProp[2], h = AdaptiveMCMC(2, data, SurrogatePosterior[2].rvs, covariance, 200)

states = []#np.zeros((iterations))



for i in range(iterations):]

    mProp = random.randint(1,3)
    thetaProp = RJCenteredProposal(m, mProp, theta, covProp[mProp], priors[mProp], centers)

    acc = Likelihood(mProp, Data, thetaProp)/Likelihood(m, Data, theta) * PriorRatio(m, mProp, theta, thetaProp, priors) * J#???
    
    if random.rand <= acc:
        theta = thetaProp
        m = mProp
    
    states.append(theta)
    ms[i] = m






def AdaptiveMCMC(m, data, theta, covariance, iterations):
    '''
    Performs Adaptive MCMC as described in Haario et al “An adaptive Metropolis algorithm”.
    Currently only used to initialise a covariance matrix for RJMCMC, but could be extended.
    '''

    initialRuns = 10  # Arbitrary Value
    if iterations <= initialRuns:
        raise ValueError("Not enough iterations to establish an empirical covariance matrix")
    
    # initialise
    d = len(theta)

    states = np.zeros((d, iterations))
    states[:, 0] = theta

    means = np.zeros((d, iterations))
    means[:, 0] = theta

    s = 2.4**2/d # Arbitrary, good value from Haario et al
    eps = 1e-2 #* s? or the size of the prior space according to paper 
    I = np.identity(d)

    pi = Functions.Likelihood(m, data, theta, covariance)


    for i in range(1, initialRuns): # warm up walk
        proposed = MCMC.GaussianProposal(theta, covariance)
        piProposed = Functions.Likelihood(m, data, proposed, covariance)

        if random.rand < piProposed/pi: # metropolis acceptance
            theta = proposed
            pi = piProposed
        
        states[:, i] = theta
        means[:, i] = (means[:, i]*i + theta)/(i + 1) # recursive mean (offsets indices starting at zero by one)


    covariance = s*np.cov(states) + s*eps*I # emperical adaption

    for i in range(iterations-initialRuns): # adaptive walk
        proposed = MCMC.GaussianProposal(theta, covariance)
        piProposed = Functions.Likelihood(m, data, proposed, covariance)

        if random.rand < piProposed/pi: # metropolis acceptance
            theta = proposed
            pi = piProposed
 
        t = i + initialRuns # global index
        
        states[:, t] = theta
        means[:, t] = (means[:, t]*t + theta)/(t + 1) # recursive mean (offsets indices starting at zero by one)

        # update
        covariance = (t + 1)/t * covariance + s/t * (t*means[:, t - 1]*np.transpose(means[:, t - 1]) - (t + 1)*means[:, t]*np.transpose(means[:, t]) + states[:, t]*np.transpose(states[:, t]) + eps*I)


    return covariance, states