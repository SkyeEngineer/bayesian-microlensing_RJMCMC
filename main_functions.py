# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [functions]


## IMPORTS ##

import MulensModel as mm 
import math
import random
import numpy as np
from numpy.lib.function_base import append, cov
from numpy.core.numeric import Inf
from scipy.stats import lognorm, loguniform, uniform, multivariate_normal
from copy import deepcopy

from types import MethodType



## CLASSES ##

class Uniform(object):
    '''
    Create an instance of a uniform distribution.
    --------------------------------------------
    Attributes:
        left [scalar]: the lower bound for values
        right [scalar]: the upper bound for values 
    '''
    def __init__(self, left, right):
        self.lb = left
        self.rb = right
        self.dist = uniform(left, right)

    def in_bound(self, x):
        '''check if true value is in prior'''
        if self.lb <= x <= self.rb: return 1
        else: return 0

    def log_pdf(self, x):
        return self.dist.logpdf(x)



class Log_Uniform(object):
    '''
    Create an instance of a log uniform distribution. 
    I.e., the log of the data is uniformly distributed
    --------------------------------------------
    Attributes:
        left [scalar]: the lower bound for values in true units
        right [scalar]: the upper bound for values in true units
    '''
    def __init__(self, left, right):
        self.lb = left
        self.rb = right
        self.dist = loguniform(left, right)

    def in_bound(self, x):
        '''check if true value is in prior'''
        if self.lb <= x <= self.rb: return 1
        else: return 0

    def log_pdf(self, x):
        return self.dist.logpdf(x)



class Truncated_Ln_Normal(object):
    '''
    Create an instance of a truncated log normal distribution. 
    I.e., the log of the data is normally distributed, and the 
    distribution is restricted to a certain range
    --------------------------------------------
    Attributes:
        left [scalar]: the lower bound for values in true units
        right [scalar]: the upper bound for values in true units
        mu [scalar]: the mean of the underlying normal distriubtion in true units
        sd [scalar]: the standard deviation of the underlying normal distribution in true units
    '''
    def __init__(self, left, right, mu, sd):
        self.lb = left
        self.rb = right
        self.dist = lognorm(scale = np.exp(np.log(mu)), s = (np.log(sd))) # (Scipy shape parameters)

        # Probability that is otherwise truncated to zero, 
        # distributed uniformly into the valid range (aprroximation)
        self.truncation = (self.dist.cdf(left) + 1 - self.dist.cdf(right)) / (right - left)

    def in_bound(self, x):
        '''check if true value is in prior'''
        if self.lb <= x <= self.rb: return 1
        else: return 0

    def log_pdf(self, x):
        if self.lb <= x <= self.rb: return np.log(self.dist.pdf(x) + self.truncation)
        else: return -Inf


class State(object):

    def __init__(self, true_theta = None, scaled_theta = None):
        
        if true_theta is not None:
            self.true = true_theta
            self.D = len(true_theta)

            self.scaled = self.true
            for p in range(D):
                if p == 3:
                    self.scaled[p] = np.log(self.true[p])
        
        elif scaled_theta is not None:
            self.scaled = scaled_theta
            self.D = len(scaled_theta)

            self.true = self.scaled
            for p in range(D):
                if p == 3:
                    self.true[p] = np.exp(self.scaled[p])

        else:   raise ValueError('Assigned null state')


class Chain(object):

    def __init__(self, m, state):
        self.n = 1
        self.model_indices = [m]
        self.states = [state]
        #self.D = D


    def add_general_state(self, m, state):

        self.model_indices.append(m)
        self.states.append(state)

        self.n += 1

        return



class Model(object):
    '''
    Class that stores a model for MH methods.
    -------------------------------------------------------------------------
    Attributes:
        m [int]: model index
        D [int]: dimensionality of a state in the model
        center [state]: best guess at maximum posterior state of model
        priors [array]: array of priors for variables in model
        covariances [chain]: covariance matrices used for proposal dists of model
        data [MulensData]: photometry readings for microlensing event
    '''

    def __init__(self, m, D, center, priors, covariance, data):#, ln_likelihood_function):
        '''
        Attributes:
            covariance [array]: initial covariance matrix for proposal dist of model
            ln_likelihood_function [function]: likelihood function to assign to instance
        '''
        self.m = m
        self.D = D
        self.n = 1



        self.center = center
        self.priors = priors

        self.sampled = Chain(m, center)
        self.scaled_avg_state = center.scale

        self.covariance = covariance
        self.covariances = [covariance]

        self.I = np.identity(D)
        self.s = 2.4**2 / D # Arbitrary(ish), good value from Haario et al
        
        # data to represent by instances's model
        self.data = data

        # assign instance's model's likelihood
        #self.ln_likelihood = MethodType(ln_likelihood_function, self)
    
    def add_state(self, theta, adapt = True):

        self.n += 1
        self.sampled.states.append(theta)

        if adapt:
            self.covariance = iterative_covariance(self.covariance, theta.scaled, self.scaled_avg_state, self.n, self.s, self.I)
            assert(('Non symmetric covariance', check_symmetric(self.covariance)))

        self.covariances.append(self.covariance)
        self.scaled_avg_state = iterative_mean(self.scaled_avg_state, theta.scaled, self.n)

        return


    def state_check(self, theta):
        assert(('Wrong state dimension for model', self.D != len(theta.true)))

    def log_likelihood(self, theta):
        '''Empty method for object model dependant assignment with MethodType'''
        raise ValueError('No likelihood method assigned for model')

    def log_prior_density(self, theta, v = None, v_D = None):
        '''
        Calculates the product of the priors for a state in this model. 
        Optionally accounts for auxilliary variables.
        ---------------------------------------------------------------
        Inputs:
        theta [state]: values of parameters to calculate prior density for
        
        Optional Args:
        v [state]: values of auxiliary variables, stored from greatest model
        v_D [int]: dimensionality of the larger model

        Returns:
        ln_prior_product [scalar]: log prior probability density of the state
        '''
        self.state_check(theta) # check state dimension
    
        ln_prior_product = 0.

        # cycle through parameters
        for p in range(self.D):

            # product using log rules
            ln_prior_product += (self.priors[p].log_pdf(theta.true[p]))

        # cycle through auxiliary parameters if v and v_D passed
        if v is not None or v_D is not None:
            if v is not None and v_D is not None:
                for p in range(self.D, v_D):
                    
                    # product using log rules
                    ln_prior_product += (self.priors[p].log_pdf(v.true[p]))

            else: raise ValueError('only one of v or v_D passed')

        return ln_prior_product








## HELPER FUNCTIONS ##

def iterative_mean(x_mu, x, n):
    return (x_mu * n + x)/(n + 1)

def iterative_covariance(cov, x, x_mu, n, s, I, eps = 1e-12):
    return (n-1)/n * cov + s/(n+1) * np.outer(x - x_mu, x - x_mu) + s*eps*I/n

def check_symmetric(A, tol = 1e-16):
    return np.all(np.abs(A-A.T) < tol)

## FUNCTIONS ##

def binary_log_likelihood(self, theta):
    '''
    Calculate the ln likelihood that data is binary and produced by theta.
    ---------------------------------------------------------------
    Inputs:
    theta [state]: values of parameters to calculate prior density for

    Returns:
    ln_likelihood [scalar]: ln likelihood state produced data with model
    '''
    self.state_check(theta) # check state dimension

    try:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'q', 's', 'alpha'], theta.true())))
        model.set_magnification_methods([0., 'point_source', 72.])

        a = model.magnification(self.data.time) #proposed magnification signal
        y = self.data.flux # observed flux signal
        
        # fit proposed flux as least squares solution
        A = np.vstack([a, np.ones(len(a))]).T
        f_s, f_b = np.linalg.lstsq(A, y, rcond = None)[0]
        F = f_s*a + f_b # least square signal

        sd = self.data.err_flux # error
        chi2 = np.sum((y - F)**2/sd**2)

    except: # if MulensModel crashes, return true probability zero
        return -Inf

    return -chi2/2 # transform chi2 to ln likelihood
#Binary_model = Model()
#Binary_model.log_likelihood = MethodType(binary_log_likelihood, Binary_model)



def single_log_likelihood(self, theta):
    '''
    Calculate the ln likelihood that data is single and produced by theta.
    ---------------------------------------------------------------
    Inputs:
    theta [state]: values of parameters to calculate prior density for

    Returns:
    ln_likelihood [scalar]: ln likelihood state produced data with model
    '''
    self.state_check(theta) # check state dimension

    try:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta.true())))
        model.set_magnification_methods([0., 'point_source', 72.])

        a = model.magnification(self.data.time) #proposed magnification signal
        y = self.data.flux # observed flux signal
        
        # fit proposed flux as least squares solution
        A = np.vstack([a, np.ones(len(a))]).T
        f_s, f_b = np.linalg.lstsq(A, y, rcond = None)[0]
        F = f_s*a + f_b # least square signal

        sd = self.data.err_flux # error
        chi2 = np.sum((y - F)**2/sd**2)

    except: # if MulensModel crashes, return true probability zero
        return -Inf

    return -chi2/2 # transform chi2 to ln likelihood












def gaussian_proposal(theta, covariance):
    '''
    Takes a single step in a guassian walk process
    ----------------------------------------------
    theta [array like]: the scaled parameter values to step from
    covariance [array like]: the covariance with which to center the multivariate 
                guassian to propose from. Can be the diagonal entries only or a complete matrix

    Returns: [array like]: a new point in scaled parameter space
    '''
    return multivariate_normal.rvs(mean = theta, cov = covariance)









def adapt_MH(Model, warm_up, iterations, user_feedback = False):
    '''
    Performs Adaptive MCMC as described in Haario et al “An adaptive Metropolis algorithm”,
    in the context of microlensing events.
    --------------------------------------------
    m [int]: the index of the microlensing model to use (0 or 1, single or binary)
    data [muLens data]: the data of the microlensing event to analyse
    theta [array like]: the scaled parameter values in the associated model space to start from
    priors [array like]: an array of prior distribution objects for the lensing parameters,
                         in the order of entries in theta
    covariance [array like]: the covariance to initialise with when proposing a move. 
                             Can be the diagonal entries only or a complete matrix.
                             In the order of theta
    burns [int]: how many iterations to perform before beginning to adapt the covariance matrix
    iterations [int]: how many further iterations to perform while adapting the covariance matrix

    Returns:
    covariance [array like] : final adaptive covariance matrix reached 
    chain_states [array like]: array of scaled states visited
    chain_means [array like]: array of mean scaled states of the chain
    acceptance_history [array like]: array of accepted moves. 1 if the proposal was accepted, 0 otherwise.
    covariance_history [array like]: list of scaled states visited
    best_posterior [scalar]: best posterior density visited
    best_theta [array like]: array of scaled state that produced best_posterior
    '''

    if warm_up < 25:
        raise ValueError("Not enough iterations to safely establish an empirical covariance matrix")
    
    # initialise
    acc = np.zeros((iterations + warm_up))
    acc[0] = 1 # first state (move) already accepted

    theta = Model.center
    best_theta = theta

    # initial propbability values
    log_likelihood = Model.log_likelihood(theta)
    log_prior = Model.log_prior_density(theta)
    log_best_posterior = log_likelihood + log_prior

    # warm up walk to establish an empirical covariance
    for i in range(1, warm_up):

        # propose a new state and calculate the resulting density
        proposed = State(scaled_theta = gaussian_proposal(theta.scaled, Model.covariance))
        log_likelihood_proposed = Model.log_likelihood(proposed)
        log_prior_proposed = Model.log_prior_density(proposed)

        # metropolis acceptance
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            theta = proposed
            log_likelihood = log_likelihood_proposed
            acc[i] = 1 # accept proposal

            # store best state
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                best_theta = theta

        else: acc[i] = 0 # reject proposal
        
        # update storage
        Model.add_state(theta, adapt = False)


    # adaptive walk
    for i in range(warm_up, iterations):

        # print progress to screen
        if user_feedback:
            cf = i / (iterations - 1)
            print(f'log score: {log_best_posterior:.4f}, progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')

        # propose a new state and calculate the resulting density
        proposed = State(scaled_theta = gaussian_proposal(theta.scaled, Model.covariance))
        log_likelihood_proposed = Model.log_likelihood(proposed)
        log_prior_proposed = Model.log_prior_density(proposed)

        # metropolis acceptance
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            theta = proposed
            log_likelihood = log_likelihood_proposed
            acc[i] = 1 # accept proposal

            # store best state
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                best_theta = theta

        else: acc[i] = 0 # reject proposal
        
        # update model chain
        Model.add_state(theta, adapt = True)

    # performance
    if user_feedback:
        print(f"\n model: {Model.m}, average acc: {(np.sum(acc) / (iterations + warm_up)):4f}, best score: {log_best_posterior:.4f}")

    return acc, best_theta, log_best_posterior,







def adapt_RJMH_proposal(m, m_prop, covariances, centers, theta, auxiliary_variables):
    '''
    Proposes a new point to jump to when doing RJMCMC using centreing points,
    in the context of single and binary microlensing events.
    --------------------------------------------
    m [int]: the index of the microlensing model to jump from (0 or 1, single or binary)
    m_prop [int]: the index of the microlensing model to jump to
    covariances [array like]: a list of the covariance for each model [single , binary]
    centers [array like]: a list of the parameter values of the centreing point for each model [single , binary]
    theta [array like]: the scaled parameter values in the associated model space to jump from
    auxilliary_variables [array like]: the stored values of the most recent binary state

    Returns: 
    theta_prop [array like]: a new point in the scaled parameter space a jump was proposed too
    g_ratio [scalar]: ratio of proposal distribution densities
    '''
    
    if m == m_prop: return gaussian_proposal(theta, covariances[m_prop]), 1 # intra-model move

    else: # inter-model move

        l = theta - centers[m] # offset from initial model's centre

        if m_prop == 0: # jump to single. Exclude non shared parameters.

            n_shared = D(m_prop)

            theta_prop = centers[m_prop] + l[:n_shared]  # map shared parameters without randomness


            covariance = covariances[m]  
            c_11 = covariance[:n_shared, :n_shared] # covariance matrix of (shared) dependent variables
            c_12 = covariance[:n_shared, n_shared:] # covariances, not variances
            c_21 = covariance[n_shared:, :n_shared] # same as above
            c_22 = covariance[n_shared:, n_shared:] # covariance matrix of independent variables
            c_22_inv = np.linalg.inv(c_22)
            conditioned_covariance = c_11 - c_12.dot(c_22_inv).dot(c_21)
            

            u = Gaussian_Proposal(np.zeros((n_shared)), conditioned_covariance)
            theta_prop = theta_prop + u # add randomness to jump

            #u = Gaussian_Proposal(np.zeros(6), covariances[m])
            #theta_prop = theta[:3]#theta_prop #+ u[:3] # add randomness to jump

            g_ratio = 1 # unity as symmetric forward and reverse jumps

            return theta_prop, g_ratio


        if m_prop == 1: # jump to binary. Include non shared parameters.

            n_shared = D(m)

            v = auxiliary_variables[n_shared:]
            h = l + centers[m_prop][:n_shared]

            theta_prop = np.concatenate((h, v)) # map without randomness


            covariance = covariances[m_prop]  
            c_11 = covariance[:n_shared, :n_shared] # covariance matrix of (shared) dependent variables
            c_12 = covariance[:n_shared, n_shared:] # covariances, not variances
            c_21 = covariance[n_shared:, :n_shared] # same as above
            c_22 = covariance[n_shared:, n_shared:] # covariance matrix of independent variables
            c_22_inv = np.linalg.inv(c_22)
            conditioned_covariance = c_11 - c_12.dot(c_22_inv).dot(c_21)


            u = Gaussian_Proposal(np.zeros((n_shared)), conditioned_covariance)
            theta_prop[:n_shared] = theta_prop[:n_shared] + u # add randomness to jump on shared parameters
            #u = Gaussian_Proposal(np.zeros(6), covariances[m_prop])
            #theta_prop = np.concatenate((theta, v))#theta_prop #+ u[:3] # add randomness to jump


            g_ratio = 1 # unity as symmetric forward and reverse jumps

            return theta_prop, g_ratio

    

    '''
    Calculates the ratio of the product of the priors of the proposed point to the
    initial point, in log units. Accounts for auxilliary variables.
    --------------------------------------------
    m [int]: the index of the microlensing model to jump from (0 or 1, single or binary)
    m_prop [int]: the index of the microlensing model to jump to
    theta [array like]: the scaled parameter values in the associated model space to jump from
    theta_prop [array like]: the scaled parameter values in the associated model space to jump too
    auxiliary_variables [array like / bool]: most recent binary state in scaled space
    priors [array like]: a list of prior distribution objects for the lensing parameters, 
                         in the order of entries in theta, in scaled space

    Returns
    log_prior_ratio [scalar]: log ratio of prior product of poposed and initial states
    '''

    '''
    Calculates the product of the priors for a model and state. 
    Does not accounts for auxilliary variables.
    --------------------------------------------
    m [int]: the index of the microlensing model to jump from (0 or 1, single or binary)
    theta [array like]: the scaled parameter values in the associated model space to jump from
    priors [array like]: a list of prior distribution objects for the lensing parameters, 
                         in the order of entries in theta, in scaled space

    Returns
    log_prior_ratio [scalar]: log ratio of prior product of poposed and initial states
    '''

    '''
    Calculate the log likelihood that a lightcurve represents observed lightcurve data
    --------------------------------------------
    m [int]: the index of the microlensing model to jump from (0 or 1, single or binary)
    theta [array like]: the scaled parameter values in the associated model space to jump from
    priors [array like]: a list of prior distribution objects for the lensing parameters, 
                         in the order of entries in theta, in scaled space
    data [muLens data]: the data of the microlensing event to analyse

    Returns
    log_likelihood [scalar]: log likelihood parameters represent lightcuvre with model
    '''

def initialise_RJMH_model(initial_Model, warm_up, iterations, n_repeat, user_feedback = False):
    '''
    Repeat the adaptive mcmc warmup process used for each model in Adpt-RJMH
    and store the best run for use in Adpt-RJMH
    --------------------------------------------
    n [int]: number of repeats
    m [int]: the index of the microlensing model to use (0 or 1, single or binary)
    data [muLens data]: the data of the microlensing event to analyse
    theta [array like]: the unscaled parameter values in the associated model space to start from
    priors [array like]: an array of prior distribution objects for the lensing parameters, in same order as theta
    covariance [array like]: the covariance to initialise with when proposing a move. 
                             Can be the diagonal entries only or a complete matrix.
                             In the order of theta
    adaptive_warmup_iterations [int]: number of MCMC steps without adapting cov
    adaptive_iterations [int]: remaining number of MCMC steps

    Returns:
    inc_covariance [array like] : final adaptive covariance matrix reached 
    inc_chain_states [array like]: array of scaled states visited
    inc_chain_means [array like]: array of mean scaled states of the chain
    inc_acceptance_history [array like]: array of accepted moves. 1 if the proposal was accepted, 0 otherwise.
    inc_covariance_history [array like]: list of scaled states visited
    inc_best_posterior [scalar]: best posterior density visited
    inc_best_theta [array like]: array of scaled state that produced best_posterior
    '''

    inc_log_best_posterior = -Inf # initialise incumbent value to always lose

    for i in range(n_repeat):
        
        if user_feedback:
            print(str(i)+'/'+str(n_repeat)+' initialisations per model\n')

        Model = deepcopy(initial_Model) # fresh model

        # run adaptive MH
        acc, best_theta, log_best_posterior = adapt_MH(Model, warm_up, iterations, user_feedback = False)

        # keep best posterior
        if inc_log_best_posterior < log_best_posterior:
            inc_Model = deepcopy(Model)
            inc_Model.center = best_theta
            #inc_acc = acc

    return inc_Model#, inc_acc



def adapt_RJMH(Models, adapt_MH_warm_up, adapt_MH, initial_n, iterations): #initial_states, initial_means, n_warmup_iterations, initial_covariances, centers, priors, iterations,  data):
    '''
    Performs Adaptive RJMCMC as described in thesis, in the context of microlensing events.
    --------------------------------------------
    initial_states [array like]: list of the final states in each warmup chain for each model (0 or 1, single or binary)
    initial_means [array like]: list of the means of each parameter in each warmup chain for each model (0 or 1, single or binary)
    n_warmup_iterations [int]: number of states in each warmup chain (0 or 1, single or binary)
    initial_covariances [array like]: list of the covariances as initialised from each warmup chain (0 or 1, single or binary)
    centers [array like]: list of the centreing points for each model (0 or 1, single or binary)
    priors [array like]: an array of prior distribution objects for the lensing parameters, in the order of entries in theta
    iterations [int]: how many iterations to perform
    data [muLens data]: the data of the microlensing event to analyse

    Returns:
    chain_states [array like]: array of scaled states visited
    chain_ms [array like]: array of mean scaled states of the chain
    best_posteriors [array like]: list of best posterior densities visited for each model
    best_thetas [array like]: list of scaled states that produced best_posteriors for each model
    covariances_history [array like]: list of covatainces for each model
    acceptance_history [array like]: list of accepted moves (0 if rejected, 1 if accepted)
    '''

    # initialise model chains
    for Model in Models:
        Model = initialise_RJMH_model(Model, adapt_MH_warm_up, adapt_MH, initial_n)

    Model = random.choice(Models)

    theta = Model.sampled.states[-1] # final state in model's warmup chain
    auxiliary_variables = Models[-1].sampled.states[-1] # final state in super set model

    log_likelihood = Log_Likelihood(m, theta, priors, data)

    # initialse storages
    best_posteriors = [-Inf, -Inf]
    best_posteriors[m] = np.exp(log_likelihood + Log_Prior_Product(m, theta, priors))
    best_thetas = [[], []]
    best_thetas[m] = theta

    chain_ms = np.zeros((iterations))
    chain_ms[0] = m
    chain_states = []
    chain_states.append(theta)
    chain_model_means = [initial_means[0], initial_means[1]] 

    intra_jump_acceptance_histories = [[], []]
    intra_jump_acceptance_histories[m].append(1) # first state counts as jump
    inter_jump_acceptance_histories = []

    acceptance_history = np.zeros((iterations))
    acceptance_history[0] = 1

    covariances = initial_covariances
    covariances_history = [[initial_covariances[0]], [initial_covariances[1]]]
    inter_cov_history = [initial_covariances[1]]



    print('Running Adpt-RJMH')
    for i in range(1, iterations): # loop through Adpt-RJMH steps
        
        #diagnostics
        cf = i / (iterations-1);
        print(f'Current: M {m+1} | Progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')

        m_prop = random.randint(0, 1) # since all models are equally likelly, this has no presence in the acceptance step

        #centers = [mean(), ]

        theta_prop, g_ratio = Adaptive_RJ_Metropolis_Hastings_Proposal(m, m_prop, covariances, centers, theta, auxiliary_variables)
        log_prior_ratio = Log_Prior_Ratio(m, m_prop, theta, theta_prop, auxiliary_variables, priors)
        log_likelihood_prop = Log_Likelihood(m_prop, theta_prop, priors, data)


        J = 1 # Jacobian of sampler
        acc_pi = np.exp(log_likelihood_prop - log_likelihood + log_prior_ratio) * g_ratio * J


        if random.random() <= acc_pi: # metropolis acceptance
            acceptance_history[i] = 1 # accept proposal
            
            if m == m_prop: 
                intra_jump_acceptance_histories[m_prop].append(1)
            else:
                inter_jump_acceptance_histories.append(1)
                inter_cov_history.append(covariances[1])

            theta = theta_prop
            m = m_prop
            
            log_likelihood = log_likelihood_prop

            if m_prop == 1: auxiliary_variables = theta_prop

            posterior_density = np.exp(log_likelihood_prop + Log_Prior_Product(m, theta, priors))
            if best_posteriors[m_prop] < posterior_density: # store best states
                best_posteriors[m_prop] = posterior_density
                best_thetas[m_prop] = theta_prop

        elif m == m_prop: 
            intra_jump_acceptance_histories[m_prop].append(0)
            acceptance_history[i] = 0 # reject move
        
        else:
            inter_jump_acceptance_histories.append(0)
            acceptance_history[i] = 0 # reject jump
            inter_cov_history.append(covariances[1])

        chain_states.append(theta)
        chain_ms[i] = m


        t = ts[m]
        
        covariances_history[m].append(covariances[m])
        covariances[m] = (t-1)/t*covariances[m] + s[m]/(t+1) * np.outer(theta-chain_model_means[m], theta-chain_model_means[m]) + s[m]*eps*I[m]/t

        # print('Is Symmetric?', check_symmetric(covariances[m], tol=1e-8))
        
        chain_model_means[m] = (chain_model_means[m]*t + theta) / (t+1)

        ts[m] += 1

    # performance:
    print("\nIterations: "+str(iterations))
    print("Accepted Move Fraction: " + str(np.sum(acceptance_history) / iterations))
    print("P(Singular|y): " + str(1-np.sum(chain_ms) / iterations))
    print("P(Binary|y): " + str(np.sum(chain_ms) / iterations))

    return chain_states, chain_ms, best_thetas, best_posteriors, covariances_history, acceptance_history, inter_jump_acceptance_histories, intra_jump_acceptance_histories, inter_cov_history



def Read_Light_Curve(file_name):
    '''
    Read in an existing lightcurve. Must be between 0 and 72 days, with 720 observations
    --------------------------------------------
    file_name [string]: csv file in three columns in same directory for light curve

    Returns:
    data [muLens data]: data for true light curve
    '''

    with open(file_name) as file:
        array = np.loadtxt(file, delimiter = ",")

    data = mm.MulensData(data_list = [array[:, 0], array[:, 1], array[:, 2]], phot_fmt = 'flux', chi2_fmt = 'flux')

    return data



def Synthetic_Light_Curve(true_theta, light_curve_type, n_epochs, signal_to_noise_baseline):
    '''
    Generate a synthetic lightcurve for some parameters and signal to noise ratio
    --------------------------------------------
    true_theta [array like]: parameters to generate curve from in unscaled space
    light_curve_type [int]: 0 (single) 1 (binary)
    n_epochs [int]: number of data points in curve
    signal_to_noise_baseline [scalar]: 

    Returns:
    data [muLens data]: data for synthetic light curve
    '''

    if light_curve_type == 0:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], true_theta)))
        model.set_magnification_methods([0., 'point_source', 72.])

    elif light_curve_type == 1:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'q', 's', 'alpha'], true_theta)))
        model.set_magnification_methods([0., 'point_source', 72.])

    # exact signal
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    true_signal = model.magnification(epochs) #(model.magnification(epochs) - 1.0) * true_theta[0] + 1.0 # adjust for fs

    np.random.seed(42)

    # simulate noise in gaussian errored flux space
    noise = np.random.normal(0.0, np.sqrt(true_signal) / signal_to_noise_baseline, n_epochs) 
    noise_sd = np.sqrt(true_signal) / signal_to_noise_baseline
    
    signal = true_signal + noise

    data = mm.MulensData(data_list = [epochs, signal, noise_sd], phot_fmt = 'flux', chi2_fmt = 'flux')

    return data




