import math
import random
import matplotlib.pyplot as plt
from numpy.lib.function_base import cov
import MulensModel as mm
import numpy as np
from numpy.core.numeric import Inf
from scipy.stats import truncnorm, lognorm, norm, loguniform, uniform, multivariate_normal

class uni(object):
    def __init__(self,left,right):
        self.left, self.right = left, right
        self.dist = uniform(left,right)
    def pdf(self,x):
        return self.dist.logpdf(x)

class loguni(object):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.dist = loguniform((a), (b))
    def pdf(self, x):
        return self.dist.logpdf(x)

class trunclognorm(object):
    def __init__(self, left, right, mu, sd):
        #mu = np.exp(mud+sdd**2/2)
        #sd = ((np.exp(sdd**2)-1)*(np.exp(2*mud+sdd**2)))**0.5
        #self.a, self.b = (np.log(left) - np.log(mu)) / np.log(sd), (np.log(Right) - np.log(mu)) / np.log(sd)
        #self.a, self.b = (np.log(left) - mu) / sd, (np.log(Right) - mu) / sd
        #self.dist = truncnorm(self.a, self.b)
        self.a=left
        self.b=right
        self.dist = lognorm(scale=np.exp(np.log(mu)), s=(np.log(sd)))
        self.trunc = (self.dist.cdf((left))+1-self.dist.cdf((right)))/((right)-(left))
    def pdf(self, x):
        if self.a<=x<=self.b: return np.log((self.dist.pdf(x))+self.trunc)
        else: return -Inf


def AdaptiveMCMC(m, data, theta, priors, covariance, burns, iterations):
    '''
    Performs Adaptive MCMC as described in Haario et al “An adaptive Metropolis algorithm”.
    Currently only used to initialise a covariance matrix for RJMCMC, but could be extended.
    '''

    initialRuns = burns  # Arbitrary Value
    if burns <= 0:
        raise ValueError("Not enough iterations to establish an empirical covariance matrix")
    
    # initialise
    d = len(theta)

    states = np.zeros((d, iterations+burns))
    states[:, 0] = theta
    dec = np.zeros((iterations+burns))

    means = np.zeros((d, iterations+burns))
    means[:, 0] = theta

    s = 2.4**2/d # Arbitrary, good value from Haario et al
    eps = 1e-12 #-6/-2?* s? or the size of the prior space according to paper 
    I = np.identity(d)

    pi = Likelihood(m, data, theta, 5)
    yes=0

    for i in range(1, initialRuns): # warm up walk
        #print(theta)
        #print(covariance)
        proposed = GaussianProposal(theta, covariance)
        piProposed = Likelihood(m, data, proposed, 5)
        priorRatio = np.exp(PriorRatio(m, m, theta, proposed, priors))
        #print(priorRatio)

        if random.random() < priorRatio * np.exp(piProposed - pi): # metropolis acceptance
            yes+=1
            theta = proposed
            pi = piProposed
            dec[i]=1

        else: dec[i]=0
        
        states[:, i] = theta
        means[:, i] = (means[:, i-1]*i + theta)/(i + 1) # recursive mean (offsets indices starting at zero by one)


    covariance = s*np.cov(states) + s*eps*I # emperical adaptionnp.cov(states)#
    #print(np.linalg.det(covariance))
    
    t = initialRuns
    for i in range(iterations): # adaptive walk
        proposed = GaussianProposal(theta, covariance)
        piProposed = Likelihood(m, data, proposed, 5)

        priorRatio = np.exp(PriorRatio(m, m, theta, proposed, priors))
        #print(priorRatio)

        if random.random() < priorRatio * np.exp(piProposed - pi): # metropolis acceptance
            yes+=1
            theta = proposed
            pi = piProposed
            dec[t]=1

        else: dec[t]=0
        
        #else: print('No :(')
 
        
        
        states[:, t] = theta
        means[:, t] = (means[:, t-1]*t + theta)/(t + 1) # recursive mean (offsets indices starting at zero by one)
        
        # update

        covariance = (t - 1)/t * covariance + s/t * (t*means[:, t - 1]*np.transpose(means[:, t - 1]) - (t + 1)*means[:, t]*np.transpose(means[:, t]) + states[:, t]*np.transpose(states[:, t])) + eps*I
        
        #covariance = s*np.cov(states) + s*eps*I
        
        t +=1 # global index

    print("Adaptive Acc: "+str(yes/(iterations+initialRuns))+", Model: "+str(m))

    return covariance, states, dec

def GaussianProposal(theta,covp):
    '''comment'''
    return multivariate_normal.rvs(mean=theta, cov=covp)

def RJCenteredProposal(m, mProp, theta, covProp, center):
    
    #print(m, mProp, theta, covProp, center)
    if m == mProp: return multivariate_normal.rvs(mean=theta, cov=covProp)
    
    else:
        l = (theta - center[m-1])/center[m-1]
        #print(l)

        if mProp == 1:
            #print('wow!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return l[0:3] * center[mProp-1] + center[mProp-1]
        
        if mProp == 2: 
            r = random.random()
            #u = np.multiply(r, [0.001, 0.00059, 1.238, 223.7])+np.multiply((1-r), [0.00099, 0.0009, 1.2, 223.5])#multivariate_normal.rvs(mean=center[mProp-1][3:], cov=covProp[3:] * np.average(l)) #center[mProp-1][3:] * np.average(l)#SurrogatePosterior[mProp].rvs #THIS FUNCTION MIGHT NOT BE DIFFERENTIABLE, JACOBIAN TROUBLES?
            #u = np.append((center[mProp-1][3:6] + center[mProp-1][3:6] * np.average(l)), center[mProp-1][6])
            u = center[mProp-1][3:] + center[mProp-1][3:] * np.average(l)
            #print(u)
            thetaProp=np.concatenate(((l * center[mProp-1][0:3]+center[mProp-1][0:3]), u))
            #print('l: '+str(l))
            #print('prop: '+str(thetaProp))
            return thetaProp






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
    D = [0, 3, 7]
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
    
    productNum=0.
    productDen=0.
    for p in range(D(mProp)): 
        productNum+=(priors[p].pdf(thetaProp[p]))
        #print(np.exp(priors[p].pdf(thetaProp[p])), thetaProp[p], p)

    for p in range(D(m)): 
        productDen+=(priors[p].pdf(theta[p]))
        #print(np.exp(priors[p].pdf(theta[p])), theta[p], p)

    #print(productNum)
    #print(productDen)
    
    return productNum-productDen



def PosteriorRatio(t,y,m,m_prop,theta,theta_prop,cov,priors):
    '''comment'''
    return Posterior(m_prop,t,y,theta_prop,cov,priors)/Posterior(m,t,y,theta,cov,priors)



def Likelihood(m, Data, theta, noise):
    '''comment'''
    z=-Inf

    if PriorBounds(m, theta)==0: return z

    #this is possibly very wrong
    if m==1:
        try: #for when moves out of bounds of model valididty
            Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta)))
            Model.set_magnification_methods([0., 'point_source', 72.]) #?
            Event = mm.Event(datasets=Data, model=Model)

        except: #make more specific
            return z

        #Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
        #plt.savefig('like.png')

        #pred = Model.magnification(Model.set_times())
        #return multivariate_normal.logpdf(np.zeros(np.shape(pred)), mean = Data-pred, cov = noise)
        return -Event.get_chi2()/2


    if m==2:
        try: #for when moves out of bounds of model valididty
            Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta)))
            Model.set_magnification_methods([0., 'VBBL', 72.]) #?
            Event = mm.Event(datasets=Data, model=Model)

        except: #make more specific
            return z
        
        #Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
        #plt.savefig('pike.png')
        #print(Event.get_chi2())

        #pred = Model.magnification(Model.set_times())
        #X=(Data-pred)
        #return np.exp(-np.dot(X.transpose(),np.dot(np.linalg.pinv(noise),X))/2)
        #return multivariate_normal.logpdf(np.zeros(np.shape(pred)), mean = Data-pred, cov = noise)
        #print(Model.magnification(Model.set_times()))
        return -Event.get_chi2()/2

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
        #print(theta)
        t_0, u_0, t_E, rho, q, s, alpha = theta

        if (s<0.2 or s>5):
            return 0

        if (q<10e-6 or q>1):
            return 0

        if (rho<10**-4 or rho> 10**-2 ):
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
