import math
import random
import MulensModel as mm
import numpy as np
from scipy.stats import truncnorm, loguniform, uniform



def GaussianProposal(theta,covp):
    '''comment'''
    return multivariate_normal.rvs(mean=theta, cov=covp)

def RJCenteredProposal(m, mProp, theta, covProp, priors, center):
    
    if m == mProp: return multivariate_normal.rvs(mean=theta, cov=covProp)
    
    else:
        l = (theta - center[m])/center[m]

        if mProp == 0: return l[0:2] * center[mProp]
        
        if mProp == 1: 
            u = SurrogatePosterior[mProp].rvs #THIS FUNCTION MIGHT NOT BE DIFFERENTIABLE, JACOBIAN TROUBLES?

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
    
    product=likelihood(m,t,y,theta,cov)
    for p in range(D): product*=priors[p].pdf(theta[p])
    
    return product



def PriorRatio(m,m_prop,theta,thetaProp,priors):
    '''comment'''
    
    productNum=1.
    productDen=1.
    for p in range(DProp): productNum*=priors[p].pdf(thetaProp[p])
    for p in range(D): productDen*=priors[p].pdf(theta[p])
    
    return productNum/productDen



def PosteriorRatio(t,y,m,m_prop,theta,theta_prop,cov,priors):
    '''comment'''
    return Posterior(m_prop,t,y,theta_prop,cov,priors)/Posterior(m,t,y,theta,cov,priors)



def Likelihood(m, Data, theta):
    '''comment'''
    z=-Inf

    #this is possibly very wrong
    if m==1:
        try: #for when moves out of bounds of model valididty
            Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta)))
            Model.set_magnification_methods([0., 'PSPL', 177.]) #?
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
