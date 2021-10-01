"""Adaptive Reversible-Jump Metropolis Hastings for microlensing.

Implements algorithms for bayesian sampling. Uses the main 
classes: State, Chain, and model.
"""

import MulensModel as mm 
import math
import random
import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy
from types import MethodType


class State(object):
    """State sampled from a model's probability distribution.

    Describes a point in both scaled and unscaled space. The scaling is 
    hardcoded but can be extended per application. Currently log10 scaling 
    the fourth parameter. In microlensing applications, this is q,
    the mass ratio.

    Attributes:
        truth: [list] Parameter values for the state, in true space.
        scaled: [list] Parameter values for the state, in scaled space.
        D: [int] The dimensionality of the state.
    """

    def __init__(self, truth = None, scaled = None):
        """Initialises state with truth, scaled, and D values.
        
        Only one of truth or scaled is needed.
        """        
        if truth is not None:
            self.truth = truth
            self.D = len(truth)

            self.scaled = deepcopy(self.truth)
            for p in range(self.D):
                if p == 3:
                    self.scaled[p] = np.log10(self.truth[p])
        
        elif scaled is not None:
            self.scaled = scaled
            self.D = len(scaled)

            self.truth = deepcopy(self.scaled)
            for p in range(self.D):
                if p == 3:
                    self.truth[p] = 10**(self.scaled[p])

        else:   raise ValueError("Assigned null state")


class Chain(object):
    """Collection of states.

    Describes a markov chain, perhaps from a joint model space.

    Attributes:
        states: [list] State objects in the chain.
        model_indices: [list] models the states are from. 
        n: [int] The number of states in the chain.
    """

    def __init__(self, m, state):
        """Initialises the chain with one state from one model.

        Args:
            state: [state] The state object.
            m: [int] The index of the model the state is from.
        """
        self.states = [state]
        self.model_indices = [m]
        self.n = 1

    def add_general_state(self, m, state):
        """Adds a state in a model to the chain.

        Args:
            state: [state] The state object.
            m: [int] The index of the model the state is from.
        """
        self.states.append(state)
        self.model_indices.append(m)
        self.n += 1
        return

    def states_array(self, scaled = True):
        """Creates a numpy array of all states in the chain.

        Args:
            scale: [optional, bool] Whether the array should be in scaled or 
                                    true space.

        Returns:
            chain_array: [np.array] The numpy array of all state parameters. 
                                    Columns are states, rows are parameters 
                                    for all states.
        """
        n_states = len(self.states)
        D_state = len(self.states[-1].scaled)
        
        chain_array = np.zeros((D_state, n_states))

        if scaled:
            for i in range(n_states):
                chain_array[:, i] = self.states[i].scaled

        else:
            for i in range(n_states):
                chain_array[:, i] = self.states[i].truth

        return chain_array


class Model(object):
    """A model to describe a probability distribution.

    Contains a chain of states from this model, as well as information
    from this. Adapts a covariance matrix iteratively with each new state,
    and stores a guess at a maximum posterior density estimate.

    Attributes:
        m: [int] model index.
        D: [int] Dimensionality of a state in the model.
        priors: [list] Prior distribution objects for state parameter values.
        sampled: [chain] States sampled from the model's distribution.
        scaled_average_state: [list] The scaled average parameter values of the chain.
        centre: [state] Best guess at maximum posterior density.
        covariance: [array] Current covariance matrix, based on all states.
        covariances: [list] All previous covariance matrices.
        acc: [list] Binary values, 1 if the state proposed was accepted,
                    0 if it was rejected.
        data: [mulensdata] Object for photometry readings from the 
            microlensing event.
        log_likelihood: [function] Method to calculate the log likelihood a state is
                                    from this model.
        I: [np.array] Identity matrix the size of D.
        s: [float] Mixing parameter (see Haario et al 2001).
    """

    def __init__(self, m, D, centre, priors, covariance, data, log_likelihood_fnc):
        """Initialises the model."""
        self.m = m
        self.D = D
        self.priors = priors
        self.centre = centre
        self.sampled = Chain(m, centre)
        self.scaled_avg_state = centre.scaled
        self.acc = [1] # First state always accepted
        self.covariance = covariance
        self.covariances = [covariance]

        self.data = data
        # model's custom likelihood function
        self.log_likelihood = MethodType(log_likelihood_fnc, self)

        self.I = np.identity(D)
        self.s = 2.4**2 / D # Arbitrary(ish), good value from Haario et al 2001.
    
    def add_state(self, theta, adapt = True):
        """Adds a sampled state to the model.

        Args:
            theta: [state] Parameters to add.
            adapt: [optional, bool] Whether or not to adjust the covariance 
                                    matrix based on the new state.
        """
        self.sampled.n += 1
        self.sampled.states.append(theta)

        if adapt:
            self.covariance = iterative_covariance(self.covariance, theta.scaled, self.scaled_avg_state, self.sampled.n, self.s, self.I)

        self.covariances.append(self.covariance)
        self.scaled_avg_state = iterative_mean(self.scaled_avg_state, theta.scaled, self.sampled.n)
        self.centre = State(scaled = iterative_mean(self.scaled_avg_state, theta.scaled, self.sampled.n))

        return

    def log_prior_density(self, theta, v = None, v_D = None):
        """Calculates the log prior density of a state in the model.

        Optionally adjusts this log density when using auxilliary vriables.

        Args:
            theta: [state] Parameters to calculate the log prior density for.
            v: [optional, state] The values of all auxiliary variables.
            v_D: [optional, int] The dimensionality to use with auxilliary variables.

        Returns:
            log_prior_product: [float] The log prior probability density.
        """    
        log_prior_product = 0.

        # cycle through parameters
        for p in range(self.D):

            # product using log rules
            log_prior_product += (self.priors[p].log_pdf(theta.truth[p]))

        # cycle through auxiliary parameters if v and v_D passed
        if v is not None or v_D is not None:
            if v is not None and v_D is not None:
                for p in range(self.D, v_D):
                    
                    # product using log rules
                    log_prior_product += (self.priors[p].log_pdf(v.truth[p]))

            else: raise ValueError("Only one of v or v_D passed.")

        return log_prior_product


def iterative_mean(x_mu, x, n):
    return (x_mu * n + x)/(n + 1)

def iterative_covariance(cov, x, x_mu, n, s, I, eps = 1e-12):
    return (n-1)/n * cov + s/(n+1) * np.outer(x - x_mu, x - x_mu) + s*eps*I/n

def check_symmetric(A, tol = 1e-16):
    return np.all(np.abs(A-A.T) < tol)



def gaussian_proposal(theta, covariance):
    """Samples a gaussian move."""
    return multivariate_normal.rvs(mean = theta, cov = covariance)

def AMH(model, adaptive_iterations, fixed_iterations = 25, user_feedback = False):
    """Performs Adaptive Metropolis Hastings.
    
    Produces a posterior distribution by adapting the proposal process within 
    one model, as described in Haario et al (2001).

    Args:
        model: [model] Model object to sample the distribution from.
        adaptive_iterations: [int] Number of steps with adaption.
        fixed_iterations: [optional, int] Number of steps without adaption.
        user_feedback: [optional, bool] Whether or not to print progress.

    Returns:
        best_theta: [state] State producing the best posterior density visited.
        log_best_posterior: [float] Best log posterior density visited. 
    """

    if fixed_iterations < 5:
        raise ValueError("Not enough iterations to safely establish an empirical covariance matrix.")
    

    theta = model.centre
    best_theta = deepcopy(theta)

    # Initial propbability values.
    log_likelihood = model.log_likelihood(theta)
    log_prior = model.log_prior_density(theta)
    log_best_posterior = log_likelihood + log_prior

    # Warm up walk to establish an empirical covariance.
    for i in range(1, fixed_iterations):

        # Propose a new state and calculate the resulting density.
        proposed = State(scaled = gaussian_proposal(theta.scaled, model.covariance))
        log_likelihood_proposed = model.log_likelihood(proposed)
        log_prior_proposed = model.log_prior_density(proposed)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            theta = deepcopy(proposed)
            log_likelihood = log_likelihood_proposed
            model.acc.append(1)

            # Store best state.
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                best_theta = deepcopy(theta)

        else: model.acc.append(0) # Reject proposal.
        
        # Update storage.
        model.add_state(theta, adapt = False)

    # Calculate intial empirical covariance matrix.
    model.covariance = np.cov(model.sampled.states_array(scaled = True))
    model.covariances.pop()
    model.covariances.append(model.covariance)

    # Perform adaptive walk.
    for i in range(fixed_iterations, adaptive_iterations + fixed_iterations):

        if user_feedback:
            cf = i / (adaptive_iterations + fixed_iterations - 1)
            print(f'log score: {log_best_posterior:.4f}, progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end="")

        # Propose a new state and calculate the resulting density.
        proposed = State(scaled = gaussian_proposal(theta.scaled, model.covariance))
        log_likelihood_proposed = model.log_likelihood(proposed)
        log_prior_proposed = model.log_prior_density(proposed)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            theta = deepcopy(proposed)
            log_likelihood = log_likelihood_proposed
            model.acc.append(1)

            # Store the best state.
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                best_theta = deepcopy(theta)

        else: model.acc.append(0) # Reject proposal.
        
        # Update model chain.
        model.add_state(theta, adapt = True)

    if user_feedback:
        print(f"\n model: {model.m}, average acc: {(np.sum(model.acc) / (adaptive_iterations + fixed_iterations)):4f}, best score: {log_best_posterior:.4f}")

    return best_theta, log_best_posterior


def ARJMH_proposal(model, proposed_model, theta, lv):
    """Performs an Adaptive Reversible-Jump Metropolis Hastings proposal.
    
    Args:
        model: [model] Model to jump from.
        proposed_model: [model] Model to jump to.
        theta: [state] State to jump from.
        lv: [list] Current auxilliary variables centre divergence.

    Returns:
        proposed_theta: [state] State proposed to jump to.
    """
    l = theta.scaled - model.centre.scaled # Offset from initial model's centre.

    if model is proposed_model: # Intra-model move.

        # Use the covariance at the proposed model's centre for local shape.
        u = gaussian_proposal(np.zeros((proposed_model.D)), proposed_model.covariance)
        proposed_theta = u + l + proposed_model.centre.scaled
        
        return proposed_theta

    else: # Inter-model move.
        
        s = abs(model.D - proposed_model.D) # Subset size.

        # Use superset model covariance
        if proposed_model.D > model.D: # proposed is superset
            cov = proposed_model.covariance
        else: # proposed is subset
            cov = model.covariance

        conditioned_cov = schur_complement(cov, s)

        # Jump to smaller model. Fix non-shared parameters.
        if proposed_model.D < model.D:

            u = gaussian_proposal(np.zeros((s)), conditioned_cov)
            proposed_theta = u + l[:s] + proposed_model.centre.scaled

            return proposed_theta

        if proposed_model.D > model.D: # Jump to larger model. Append v.

            u = gaussian_proposal(np.zeros((s)), conditioned_cov)
            shared_map = u + l[:s] + proposed_model.centre.scaled[:s]
            non_shared_map = lv[s:] + proposed_model.centre.scaled[s:]
            map = np.concatenate((shared_map, non_shared_map))
            proposed_theta = map

            return proposed_theta

def schur_complement(cov, s):
    c_11 = cov[:s, :s] # Covariance matrix of shared parameters.
    c_12 = cov[:s, s:] # Covariances, not variances.
    c_21 = cov[s:, :s] # Same as above.
    c_22 = cov[s:, s:] # Covariance matrix of non-shared.
    c_22_inv = np.linalg.inv(c_22)

    return c_11 - c_12.dot(c_22_inv).dot(c_21)

def warm_up_model(empty_model, adaptive_iterations, fixed_iterations = 25, repetitions = 1, user_feedback = False):
    """Prepares a model for the ARJMH algorithm.
    
    Repeats the adaptive MH warmup process for a model, storing the best run.

    Args:
        empty_model: [model] Initial model object.
        adaptive_iterations: [int] Number of adaptive steps.
        fixed_iterations: [optional, int] Number of non-adaptive steps.
        repetitions: [optional, int] Number of times to try for a better run.
        user_feedback: [optional, bool] Whether or not to print progress.

    Returns:
        incumbent_model: [model] Model with the states from the best run.
    """

    inc_log_best_posterior = -math.inf # Initialise incumbent posterior to always lose.

    for i in range(repetitions):
        
        if user_feedback:
            print("Running the "+str(i+1)+"/"+str(repetitions)+"th initialisation per model\n")

        model = deepcopy(empty_model) # Fresh model.

        # Run adaptive MH.
        best_theta, log_best_posterior = AMH(model, adaptive_iterations, fixed_iterations = fixed_iterations, user_feedback = user_feedback)

        # Keep the best posterior density run.
        if inc_log_best_posterior < log_best_posterior:
            incumbent_model = deepcopy(model)
            incumbent_model.centre = deepcopy(best_theta)

    return incumbent_model


def ARJMH(models, iterations,  adaptive_warm_up_iterations, fixed_warm_up_iterations = 25, warm_up_repititions = 1, user_feedback = False):
    """Samples from a joint distribution of models.
    
    Initialises each model with multiple adaptive MH runs. Then uses the resulting
    covariances to run adaptive RJMH on all models.

    Args:
        models: [list] Model objects to sample from. 
                    Should be sorted by increasing dimensionality.
        iterations: [int] Number of adaptive RJMH steps.
        adaptive_warm_up_iterations: [int] Number of adaptive steps to initialise with.
        fixed_warm_up_iterations: [int] Number of fixed steps to initilaise with.
        warm_up_repititions: [int] Number of times to try for a better initial run.
        user_feedback: [optional, bool] Whether or not to print progress.

    Returns:
        joint_model_chain: [chain] Generalised chain with states from any model.
        total_acc: [list] Binary values, 1 if the state proposed was accepted,
                          0 if it was rejected, associated with the joint model.
        inter_model_history: [model] Stored inter-model covariances and acc.
    """

    if len(models) == 2:
        inter_model_history = deepcopy(models[1])
        inter_model_history.covariances = [schur_complement(models[1].covariance, models[0].D)]
    else: inter_model_history = None

    # Initialise model chains.
    for m_i in range(len(models)):
        models[m_i] = warm_up_model(models[m_i], adaptive_warm_up_iterations, fixed_warm_up_iterations, warm_up_repititions, user_feedback)

    random.seed(42)

    # Choose a random model to start in.
    model = random.choice(models)
    theta = deepcopy(model.sampled.states[-1]) # Final state in model's warmup chain.

    v = deepcopy(models[-1].sampled.states[-1]) # Auxiliary variables final state in super set model.
    lv = models[-1].sampled.states[-1].scaled - models[-1].centre.scaled # Auxiliary variables offset from centre.

    # Create joint model as initial theta appended to auxiliary variables.
    initial_superset = models[-1].D - model.D
    if initial_superset > 0: # If random choice was a subset model
        theta_v = np.concatenate((theta.scaled, models[-1].sampled.states[-1].scaled[model.D:]))
        joint_model_chain = Chain(model.m, State(scaled = theta_v))
    else:
        joint_model_chain = Chain(model.m, theta)

    total_acc = np.zeros(iterations)
    total_acc[0] = 1

    v_D = models[-1].D # Dimension of largest model is auxilliary variable size.

    # Initial probability values.
    log_likelihood = model.log_likelihood(theta)
    log_prior = model.log_prior_density(theta, v = v, v_D = v_D)


    if user_feedback: print("Running ARJMH.")
    for i in range(1, iterations): # ARJMH algorithm.
        
        if user_feedback:
            cf = i / (iterations - 1)
            print(f'model: {model.m} progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end="")

        # Propose a new model and state and calculate the resulting density.
        proposed_model = random.choice(models)
        proposed = State(scaled = ARJMH_proposal(model, proposed_model, theta, lv))
        log_likelihood_proposed = proposed_model.log_likelihood(proposed)
        log_prior_proposed = proposed_model.log_prior_density(proposed, v = v, v_D = v_D)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            total_acc[i] = 1

            if model is proposed_model: # Intra model move.
                model.acc.append(1)
            elif inter_model_history is not None: # Inter model move.
                inter_model_history.acc.append(1)
                inter_model_history.covariances.append(schur_complement(models[1].covariance, models[0].D))
                inter_model_history.sampled.n += 1

            model = proposed_model
            theta = deepcopy(proposed)

            log_likelihood = log_likelihood_proposed
            log_prior = log_prior_proposed
            
        else: # Reject proposal.
            total_acc[i] = 0
            
            if model is proposed_model: # Intra model move.
                model.acc.append(0)
            elif inter_model_history is not None: # Inter model move.
                inter_model_history.acc.append(0)
                inter_model_history.covariances.append(schur_complement(models[1].covariance, models[0].D))
                inter_model_history.sampled.n += 1
        
        # Update model chain.
        model.add_state(theta, adapt = True)
        v = State(scaled = np.concatenate((theta.scaled, v.scaled[model.D:])))
        joint_model_chain.add_general_state(model.m, v)

        # Update auxilliary centre divergence for new states.
        lv[:model.D] = theta.scaled - model.centre.scaled

    if user_feedback:
        print(f"\n average acc: {np.average(total_acc):4f}")
        print("P(m1|y): " + str(1 - np.sum(joint_model_chain.model_indices) / iterations))
        print("P(m2|y): " + str(np.sum(joint_model_chain.model_indices) / iterations))

    return joint_model_chain, total_acc, inter_model_history



def output_file(models, warm_up_iterations, joint_model_chain, total_acc, n_epochs, sn, letters, name = "", event_params = None):
    
    # Output File.
    with open("results/"+name+"-run.txt", "w") as file:
        file.write("Run "+name+"\n")
        
        # Inputs.
        file.write("Inputs:\n")
        if event_params is not None:
            file.write("Parameters: "+str(event_params.truth)+"\n")
        file.write("Number of observations: "+str(n_epochs)+", Signal to noise baseline: "+str(sn)+"\n")
        file.write("\n")
        file.write("Run information:\n")
        file.write("Iterations: "+str(joint_model_chain.n)+"\n")
        file.write("Average acc; Total: "+str(np.average(total_acc)))

        # Results.
        file.write("\n\nResults:\n")
        for model in models:
            
            # Models.
            P_model = (model.sampled.n-burn_in)/joint_model_chain.n
            sde_model = np.std(np.array(joint_model_chain.model_indices))/(joint_model_chain.n**0.5)
            file.write("\n"+str(model.m)+"\nP(m|y): "+str(P_model)+r"\pm"+str(sde_model)+"\n")

            # Parameters.
            model_states = model.sampled.states_array(scaled = True)
            for i in range(len(model.sampled.states[-1].scaled)):
                mu = np.average(model_states[i, :])
                sd = np.std(model_states[i, :], ddof = 1)
                file.write(letters[i]+": mean: "+str(mu)+", sd: "+str(sd)+" \n")
                
                # Hardcoded unscale q.
                if i == 3:
                    exp10 = np.power(10, model_states[i, :])
                    mu = np.average(exp10)
                    sd = np.std(exp10, ddof = 1)
                    file.write("q: mean: "+str(mu)+", sd: "+str(sd)+" \n")

    return

