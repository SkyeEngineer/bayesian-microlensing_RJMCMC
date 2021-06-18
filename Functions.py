import math
import random
import MulensModel as mm
import numpy as np
from numpy.core.numeric import Inf
from scipy.stats import truncnorm, loguniform, uniform, multivariate_normal

class uni(object):
    def __init__(self,left,right):
        self.left, self.right = left, right
        self.dist = uniform(left,right)
    def draw(self):
        return self.dist.rvs(self.left, self.right)
    def pdf(self,x):
        return self.dist.pdf(x)

class loguni(object):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.dist = loguniform(a, b)
    def draw(self):
        return self.dist.rvs(self.a, self.b)
    def pdf(self, x):
        return self.dist.pdf(x)

class trunclognorm(object):
    def __init__(self, left, right, x, y):
        self.a, self.b = (np.log(left) - np.log(10**y)) / np.log(10**x), (np.log(right) - np.log(10**y)) / np.log(10**x)
        self.dist = truncnorm(self.a, self.b)
    def draw(self):
        return np.exp(self.dist.rvs(self.a, self.b))
    def pdf(self, x):
        return self.dist.pdf(np.log(x))


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

    pi = Likelihood(m, data, theta)


    for i in range(1, initialRuns): # warm up walk
        proposed = GaussianProposal(theta, covariance)
        piProposed = Likelihood(m, data, proposed)

        if random.rand < piProposed/pi: # metropolis acceptance
            theta = proposed
            pi = piProposed
        
        states[:, i] = theta
        means[:, i] = (means[:, i]*i + theta)/(i + 1) # recursive mean (offsets indices starting at zero by one)


    covariance = s*np.cov(states) + s*eps*I # emperical adaption

    for i in range(iterations-initialRuns): # adaptive walk
        proposed = GaussianProposal(theta, covariance)
        piProposed = Likelihood(m, data, proposed, covariance)

        if random.rand < piProposed/pi: # metropolis acceptance
            theta = proposed
            pi = piProposed
 
        t = i + initialRuns # global index
        
        states[:, t] = theta
        means[:, t] = (means[:, t]*t + theta)/(t + 1) # recursive mean (offsets indices starting at zero by one)

        # update
        covariance = (t + 1)/t * covariance + s/t * (t*means[:, t - 1]*np.transpose(means[:, t - 1]) - (t + 1)*means[:, t]*np.transpose(means[:, t]) + states[:, t]*np.transpose(states[:, t]) + eps*I)


    return covariance, states

def GaussianProposal(theta,covp):
    '''comment'''
    return multivariate_normal.rvs(mean=theta, cov=covp)

def RJCenteredProposal(m, mProp, theta, covProp, center):
    
    if m == mProp: return multivariate_normal.rvs(mean=theta, cov=covProp)
    
    else:
        l = (theta - center[m])/center[m]

        if mProp == 0: return l[0:2] * center[mProp]
        
        if mProp == 1: 
            u = [1, 1, 1, 1, 1]#SurrogatePosterior[mProp].rvs #THIS FUNCTION MIGHT NOT BE DIFFERENTIABLE, JACOBIAN TROUBLES?

            return np.concatenate((l * center[mProp][0:2], u[3:9]))






def RJAuxiliaryProposal(m,m_prop,theta,covp,priors,params):
    '''comment'''

    #??????
    if (m==1 and m_prop==1):
        theta_prop = multivariate_normal.rvs(mean=theta, cov=covp[0:3,0:3])

        return theta_prop

    elif (m==2 and m_prop==2):
        theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)

        return theta_prop

    elif (m==1 and m_prop==2):
        theta = np.append(theta,np.zeros(np.shape(theta)))
        theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)

        return theta_prop

    elif (m==2 and m_prop==1):
        theta_prop = multivariate_normal.rvs(mean=theta[0:3], cov=covp[0:3,0:3])

        return theta_prop

#priors currently dict

def D(m):
    D = [0, 3, 8]
    return D[m]

    #if m == 1: return 3
    #elif m == 2: return 8
    #else: return 0 

def Posterior(m,t,y,theta,cov,priors):
    '''comment'''
    
    product=Likelihood(m,t,y,theta,cov)
    for p in range(D): product*=priors[p].pdf(theta[p])
    
    return product



def PriorRatio(m,mProp,theta,thetaProp,priors):
    '''comment'''
    
    productNum=1.
    productDen=1.
    for p in range(D(mProp)): productNum*=priors[p].pdf(thetaProp[p])
    for p in range(D(m)): productDen*=priors[p].pdf(theta[p])
    
    return productNum/productDen



def PosteriorRatio(t,y,m,m_prop,theta,theta_prop,cov,priors):
    '''comment'''
    return Posterior(m_prop,t,y,theta_prop,cov,priors)/Posterior(m,t,y,theta,cov,priors)



def Likelihood(m, Data, theta):
    '''comment'''
    z=-Inf

    if PriorBounds(m, theta)==0: return z

    #this is possibly very wrong
    if m==1:
        try: #for when moves out of bounds of model valididty
            Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta)))
            #Model.set_magnification_methods([0., 'PSPL', 177.]) #?
            Event = mm.Event(datasets=Data, model=Model)

        except: #make more specific
            return z

        return math.exp(-Event.get_chi2()/2)


    if m==2:
        try: #for when moves out of bounds of model valididty
            Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta)))
            Model.set_magnification_methods([0., 'VBBL', 177.]) #?
            Event = mm.Event(datasets=Data, model=Model)

        except: #make more specific
            return z
            
        return math.exp(-Event.get_chi2()/2)

def PriorBounds(m, theta):

    if m==1:

        t_0, u_0, t_E = theta
        if (u_0<0 or u_0>2):
            return 0

        if (t_0<0 or t_0>72):
            return 0

        if (t_E<1 or t_E>100):
            return 0
    
    elif m==2:

        t_0, u_0, t_E, q, s, alpha = theta
        if (s<0.2 or s>5):
            return 0

        if (q<10e-6 or q>1):
            return 0

        if (alpha<0 or alpha>360):
            return 0

        if (u_0<0 or u_0>2):
            return 0

        if (t_0<0 or t_0>72):
            return 0

        if (t_E<1 or t_E>100):
            return 0
    
    return 1

def ProposalRatio(m,m_prop,theta,thetaProp,priors):
    '''comment'''
    #??????????

    productNum=1.
    productDen=1.
    for p in range(DProp-1, D): productNum*=priors[p].pdf(thetaProp[p])
    for p in range(D-1, Dprop): productDen*=priors[p].pdf(theta[p])

    # If Auxiliary proposal, need?

    #   elif(m_prop==1 and m==1):
    #    return priors['phi'].pdf(theta_prop[3])*priors['q'].pdf(theta_prop[4])*priors['d'].pdf(theta_prop[5])/priors['phi'].pdf(theta[3])*priors['q'].pdf(theta[4])*priors['d'].pdf(theta[5])

    return productNum/productDen
