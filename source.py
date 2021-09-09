"""Adaptive Reversible-Jump Metropolis Hastings for microlensing.

Implements algorithms for bayesian sampling. Uses the main 
classes: State, Chain, and Model to ensure generality and possible
extension to different RJMH algorithms. Includes code for the
realstic simulation and likelihood calculation of microlensing
events.

  Typical usage example:

  See modules: expected_binary and expected_robustness 

"""

import MulensModel as mm 
import math
import random
import numpy as np
from scipy.stats import lognorm, loguniform, uniform, multivariate_normal
from copy import deepcopy
from types import MethodType


class Uniform(object):
    """A uniform distribution.

    Attributes:
        lb: A float lower bound for support.
        rb: A float upper bound for support.
    """

    def __init__(self, left, right):
        """Initialises Uniform with bounds and sampler."""
        self.lb = left
        self.rb = right
        self.dist = uniform(left, right)

    def in_bound(self, x):
        """Check if value is in support."""
        if self.lb <= x <= self.rb: return 1
        else: return 0

    def log_pdf(self, x):
        """Calculate log probability density."""
        return self.dist.logpdf(x)


class Log_Uniform(object):
    """A log uniform distribution.

    The log of the data is uniformly distributed.

    Attributes:
        lb: A float lower bound for support.
        rb: A float upper bound for support.
    """

    def __init__(self, left, right):
        """Initialises Log uniform with bounds and sampler."""
        self.lb = left
        self.rb = right
        self.dist = loguniform(left, right)

    def in_bound(self, x):
        """Check if value is in support."""
        if self.lb <= x <= self.rb: return 1
        else: return 0

    def log_pdf(self, x):
        """Calculate log probability density."""
        return self.dist.logpdf(x)


class Truncated_Log_Normal(object):
    """A truncated log normal distribution.

    The log of the data is normally distributed, and the data is constrained.

    Attributes:
        lb: A float lower bound for support.
        rb: A float upper bound for support.
    """

    def __init__(self, left, right, mu, sd):
        """Initialises Truncated log normal with bounds and sampler.

        Args:
            mu: The scalar mean of the underlying normal distrubtion in true 
                space.
            sd: The scalar standard deviation of the underlying normal 
                distribution in true space.
        """
        self.lb = left
        self.rb = right
        self.dist = lognorm(scale = np.exp(np.log(mu)), s = (np.log(sd))) # Scipy shape parameters.

        # Probability that is otherwise truncated to zero, distributed uniformly (aprroximation).
        self.truncation = (self.dist.cdf(left) + 1 - self.dist.cdf(right)) / (right - left)

    def in_bound(self, x):
        """Check if value is in support."""
        if self.lb <= x <= self.rb: return 1
        else: return 0

    def log_pdf(self, x):
        """Calculate log probability density."""
        if self.lb <= x <= self.rb: return np.log(self.dist.pdf(x) + self.truncation)
        else: return -math.inf # If out of support.


class State(object):
    """A sampled state from a model's probability distribution.

    Describes a point in both scaled and unscaled space. The scaling is 
    hardcoded but can be extended per application. Currently log10 scaling 
    the fourth parameter. In microlensing applications, this is q,
    the mass ratio.

    Attributes:
        truth: A list of parameter values for the state, in true space.
        scaled: A list of parameter values for the state, in scaled space.
        D: The integer dimensionality of the state.
    """

    def __init__(self, truth = None, scaled = None):
        """Initialises state with truth, scaled, and D values.

        Args:
            either a list of parameter values in the true or scaled states.
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
    """A collection of states.

    Describes a markov chain, perhaps from a joint model space.

    Attributes:
        states: A list of state objects in the chain.
        model_indices: A list of the models the states are from. 
        n: The number of states in the chain.
    """

    def __init__(self, m, state):
        """Initialises the chain with one state from one model.

        Args:
            state: The state object.
            m: The index of the model the state is from.
        """
        self.states = [state]
        self.model_indices = [m]
        self.n = 1

    def add_general_state(self, m, state):
        """Adds a state in a model to the chain.

        Args:
            state: The state object.
            m: The index of the model the state is from.
        """
        self.states.append(state)
        self.model_indices.append(m)
        self.n += 1
        return

    def states_array(self, scaled = True):
        """Creates a numpy array of all states in the chain.

        Args:
            scale: (Optional) whether the array should be in scaled or true 
                    space (Boolean).

        Returns:
            chain_array: The numpy array of all state parameters. Columns are
                        states, rows are parameters for all states.
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
        m: The integer model index.
        D: The integer dimensionality of a state in the model.
        priors: List of prior distributions for state parameter values.
        sampled: A chain of states sampled from the model's distribution.
        scaled_avg_state: The scaled average parameter values of the chain.
        center: The state of the best guess at a maximum posterior density.
        covariance: The current covariance matrix, based on all states.
        covariances: A list of all previous covariance matrices.
        acc: A list of binary values, 1 if the state proposed was accepted,
            0 if it was rejected.
        data: A MulensData object for photometry readings from the 
            microlensing event.
        log_likelihood: A function to calculate the log likelihood a state is
            from this model.
        I: An identity matrix the size of D.
        s: A scalar (see Haario et al 2001).
    """

    def __init__(self, m, D, Center, priors, covariance, data, log_likelihood_fnc):
        """Initialises the model."""
        self.m = m
        self.D = D
        self.priors = priors
        self.Center = Center
        self.Sampled = Chain(m, Center)
        self.scaled_avg_state = Center.scaled
        self.acc = [1] # First state always accepted
        self.covariance = covariance
        self.covariances = [covariance]

        self.data = data
        # Model's custom likelihood function
        self.log_likelihood = MethodType(log_likelihood_fnc, self)

        self.I = np.identity(D)
        self.s = 2.4**2 / D # Arbitrary(ish), good value from Haario et al 2001.
    
    def add_state(self, theta, adapt = True):
        """Adds a sampled state to the model.

        Args:
            theta: The state to add.
            adapt: (Optional) whether to adjust the covariance matrix, based
                on the new state (Boolean).
        """
        self.Sampled.n += 1
        self.Sampled.states.append(theta)

        if adapt:
            self.covariance = iterative_covariance(self.covariance, theta.scaled, self.scaled_avg_state, self.Sampled.n, self.s, self.I)

        self.covariances.append(self.covariance)
        self.scaled_avg_state = iterative_mean(self.scaled_avg_state, theta.scaled, self.Sampled.n)

        return

    def log_likelihood(self, theta):
        """Empty method for object model dependant assignment with MethodType."""
        raise ValueError("No likelihood method assigned for model")

    def log_prior_density(self, theta, v = None, v_D = None):
        """Calculates the log prior density of a state in the model.

        Optionally adjusts this log density when using auxilliary vriables.

        Args:
            theta: The state to calculate the log prior density for.
            v: (Optional) The state of all auxiliary variables.
            v_D: (Optional) the integer dimensionality of to use to adjust v.

        Returns:
            log_prior_product: The log prior probability density.
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

## FUNCTIONS ##

def binary_log_likelihood(self, theta):
    """Calculates the log likelihood of a state in a model.
    
    Uses the point source approximation from MulensModel to calculate
    the log likelihood that a binary state produced the model's data.
    Data must be over the range 0 to 72 days.

    Args:
        theta: The binary model state.

    Returns:
        log_likelihood: The resulting log likelihood.
    """
    try: # MulensModel may throw errors
        model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], theta.truth)))
        model.set_magnification_methods([0., "point_source", 72.])

        a = model.magnification(self.data.time) # The proposed magnification signal.
        y = self.data.flux # The observed flux signal.
        
        # Fit proposed flux as least squares solution.
        A = np.vstack([a, np.ones(len(a))]).T
        f_s, f_b = np.linalg.lstsq(A, y, rcond = None)[0]
        F = f_s*a + f_b # The least squares signal.

        sd = self.data.err_flux
        chi2 = np.sum((y - F)**2/sd**2)

    except: # If MulensModel crashes, return true likelihood zero.
        return -math.inf

    return -chi2/2 # Transform chi2 to log likelihood.

def single_log_likelihood(self, theta):
    """Calculates the log likelihood of a state in a model.
    
    Uses the point source approximation from MulensModel to calculate
    the log likelihood that a single state produced the model's data.
    Data must be over the range 0 to 72 days.

    Args:
        theta: The single model state.

    Returns:
        log_likelihood: The resulting log likelihood.
    """
    try: # MulensModel may throw errors
        model = mm.Model(dict(zip(["t_0", "u_0", "t_E"], theta.truth)))
        model.set_magnification_methods([0., "point_source", 72.])

        a = model.magnification(self.data.time) # The proposed magnification signal.
        y = self.data.flux # The observed flux signal.
        
        # Fit proposed flux as least squares solution.
        A = np.vstack([a, np.ones(len(a))]).T
        f_s, f_b = np.linalg.lstsq(A, y, rcond = None)[0]
        F = f_s*a + f_b # The least squares signal.

        sd = self.data.err_flux
        chi2 = np.sum((y - F)**2/sd**2)

    except: # If MulensModel crashes, return true likelihood zero.
        return -math.inf

    return -chi2/2 # Transform chi2 to log likelihood.

def gaussian_proposal(theta, covariance):
    """Samples a gaussian move."""
    return multivariate_normal.rvs(mean = theta, cov = covariance)

def adapt_MH(Model, warm_up, iterations, user_feedback = False):
    """Performs Adaptive Metropolis Hastings.
    
    Produces a posterior distribution by adapting the proposal process within 
    one model, as described in Haario et al (2001).

    Args:
        Model: The model object to sample the distrbution from.
        warm_up: The integer number of steps without adaption.
        iterations: The integer number of steps with adaption.
        user_feedback: (Optional) whether or not to print porgress (Boolean).

    Returns:
        BestTheta: The state producing the best posterior density visited.
        log_best_posterior: The best log posterior density visited. 
    """

    if warm_up < 5:
        raise ValueError("Not enough iterations to safely establish an empirical covariance matrix.")
    
    Theta = Model.center
    BestTheta = Theta

    # Initial propbability values.
    log_likelihood = Model.log_likelihood(Theta)
    log_prior = Model.log_prior_density(Theta)
    log_best_posterior = log_likelihood + log_prior

    # Warm up walk to establish an empirical covariance.
    for i in range(1, warm_up):

        # Propose a new state and calculate the resulting density.
        Proposed = State(scaled = gaussian_proposal(Theta.scaled, Model.covariance))
        log_likelihood_proposed = Model.log_likelihood(Proposed)
        log_prior_proposed = Model.log_prior_density(Proposed)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            Theta = Proposed
            log_likelihood = log_likelihood_proposed
            Model.acc.append(1)

            # store best state
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                BestTheta = Theta

        else: Model.acc.append(0) # reject proposal
        
        # update storage
        Model.add_state(Theta, adapt = False)

    # Calculate intial empirical covariance matrix.
    Model.covariance = np.cov(Model.sampled.states_array(scaled = True))
    Model.covariances.pop()
    Model.covariances.append(Model.covariance)

    # Perform adaptive walk.
    for i in range(warm_up, iterations):

        if user_feedback:
            cf = i / (iterations - 1)
            print(f"log score: {log_best_posterior:.4f}, progress: [{'#'*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r", end="")

        # Propose a new state and calculate the resulting density.
        Proposed = State(scaled = gaussian_proposal(Theta.scaled, Model.covariance))
        log_likelihood_proposed = Model.log_likelihood(Proposed)
        log_prior_proposed = Model.log_prior_density(Proposed)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            Theta = Proposed
            log_likelihood = log_likelihood_proposed
            Model.acc.append(1)

            # Store the best state.
            log_posterior = log_likelihood_proposed + log_prior_proposed
            if log_best_posterior < log_posterior:
                log_best_posterior = log_posterior
                BestTheta = Theta

        else: Model.acc.append(0) # Reject proposal.
        
        # Update model chain.
        Model.add_state(Theta, adapt = True)

    if user_feedback:
        print(f"\n model: {Model.m}, average acc: {(np.sum(Model.acc) / (iterations + warm_up)):4f}, best score: {log_best_posterior:.4f}")

    return BestTheta, log_best_posterior


def adapt_RJMH_proposal(Model, ProposedModel, Theta, lv):
    """Performs an Adaptive Reversible-Jump Metropolis Hastings proposal.
    
    Args:
        Model: The model to jump from.
        ProposedModel: The model to jump to.
        Theta: The state to jump from.
        lv: A list of the current auxilliary variables center divergence.

    Returns:
        ProposedTheta: A state proposed to jump to.
    """
    l = Theta.scaled - Model.Center.scaled # Offset from initial model's centre.

    if Model.m == ProposedModel.m: # Intra-model move.

        # Use the covariance at the proposed model's center for local shape.
        u = gaussian_proposal(np.zeros((ProposedModel.D)), ProposedModel.covariance)
        ProposedTheta = u + l + ProposedModel.Center.scaled
        
        return ProposedTheta

    else: # Inter-model move.
        
        s = abs(Model.D - ProposedModel.D) # Subset size.

        # Use superset model covariance
        if ProposedModel.D > Model.D: # Proposed is superset
            cov = ProposedModel.covariance
        else: # Proposed is subset
            cov = Model.covariance

        c_11 = cov[:s, :s] # Covariance matrix of shared parameters.
        c_12 = cov[:s, s:] # Covariances, not variances.
        c_21 = cov[s:, :s] # Same as above.
        c_22 = cov[s:, s:] # Covariance matrix of non-shared.
        c_22_inv = np.linalg.inv(c_22)

        conditioned_cov = c_11 - c_12.dot(c_22_inv).dot(c_21)

        if ProposedModel.D < Model.D: # Jump to smaller model. Fix non-shared parameters.

            u = gaussian_proposal(np.zeros((s)), conditioned_cov)
            proposed_theta = u + l[:s] + ProposedModel.Center.scaled

            return proposed_theta

        if ProposedModel.D > Model.D: # Jump to larger model. Append v.

            u = gaussian_proposal(np.zeros((s)), conditioned_cov)
            shared_map = u + l[:s] + ProposedModel.Center.scaled[:s]
            non_shared_map = lv[s:] + ProposedModel.Center.scaled[s:]
            map = np.concatenate((shared_map, non_shared_map))
            proposed_theta = map

            return proposed_theta


def initialise_RJMH_model(EmptyModel, warm_up, iterations, n_repeat, user_feedback = False):
    """Prepares a model for the adaptive RJ algorithm.
    
    Repeats the adaptive MH warmup process for a model, storing the best run.

    Args:
        EmptyModel: The initialised model object.
        warm_up: The integer number of non-adaptive steps.
        iterations: The integer number of adaptive steps.
        n_repeat: The integer number of times to try for a better run.
        user_feedback: (Optional) whether to print progress (Boolean).

    Returns:
        IncumbentModel: The model with the states from the best run.
    """

    inc_log_best_posterior = -math.inf # Initialise incumbent posterior to always lose

    for i in range(n_repeat):
        
        if user_feedback:
            print("Running the "+str(i+1)+"/"+str(n_repeat)+"th initialisation per model\n")

        Model = deepcopy(EmptyModel) # Fresh model.

        # Run adaptive MH.
        BestTheta, log_best_posterior = adapt_MH(Model, warm_up, iterations, user_feedback = user_feedback)

        # Keep the best posterior density run.
        if inc_log_best_posterior < log_best_posterior:
            IncumbentModel = Model
            IncumbentModel.Center = BestTheta

    return IncumbentModel


def adapt_RJMH(models, adapt_MH_warm_up, adapt_MH, initial_n, iterations, user_feedback = False):
    """Samples from a joint distribution of models.
    
    Initialises each model with multiple adaptive MH runs. Then uses the resulting
    covariances to run adaptive RJMH on all models.

    Args:
        models: A list of model objects to sample from. 
            Should be sorted by increasing dimensionality.
        adapt_MH_warm_up: The integer number of non-adaptive steps to initilaise with.
        adapt_MH: The integer number of adaptive steps to initialise with.
        n_repeat: The integer number of times to try for a better initial run.
        iterations: The integer number of adaptive RJMH steps.
        user_feedback: (Optional) whether to print progress (Boolean).

    Returns:
        JointModelChain: A generalised chain with states from any model.
        total_acc: A list of binary values, 1 if the state proposed was accepted,
            0 if it was rejected, associated with the joint model.
    """

    # Initialise model chains.
    for Model in models:
        Model = initialise_RJMH_model(Model, adapt_MH_warm_up, adapt_MH, initial_n, user_feedback = user_feedback)

    random.seed(42)

    # Choose a random model to start in.
    Model = random.choice(models)
    Theta = Model.Sampled.states[-1] # Final state in model's warmup chain.

    v = models[-1].Sampled.states[-1] # Auxiliary variables final state in super set model
    lv = models[-1].Sampled.states[-1].scaled - models[-1].Center.scaled # auxiliary variables offset from center

    # Create joint model as initial theta appended to auxiliary variables.
    initial_superset = models[-1].D - Model.D
    if initial_superset > 0: # If random choice was a subset model
        Theta_v = np.concatenate((Theta.scaled, models[-1].Sampled.states[-1].scaled[Model.D:]))
        JointModelChain = Chain(Model.m, State(scaled = Theta_v))
    else:
        JointModelChain = Chain(Model.m, Theta)

    total_acc = np.zeros(iterations)
    total_acc[0] = 1

    v_D = models[-1].D # Dimension of largest model is auxilliary variable size.

    # Initial propbability values.
    log_likelihood = Model.log_likelihood(Theta)
    log_prior = Model.log_prior_density(Theta, v = v, v_D = v_D)


    if user_feedback: print("Running adapt-RJMH.")
    for i in range(1, iterations): # Adapt-RJMH algorithm.
        
        if user_feedback:
            cf = i / (iterations - 1)
            #print(f"model: {Model.m} progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r", end="")

        # Propose a new model and state and calculate the resulting density.
        ProposedModel = random.choice(models)
        Proposed = State(scaled = adapt_RJMH_proposal(Model, ProposedModel, Theta, lv))
        log_likelihood_proposed = ProposedModel.log_likelihood(Proposed)
        log_prior_proposed = ProposedModel.log_prior_density(Proposed, v = v, v_D = v_D)

        # Metropolis acceptance criterion.
        if random.random() < np.exp(log_likelihood_proposed - log_likelihood + log_prior_proposed - log_prior):
            # Accept proposal.
            total_acc[i] = 1

            if Model == ProposedModel: # Intra model move.
                Model.acc.append(1)

            Model = ProposedModel
            theta = Proposed

            log_likelihood = log_likelihood_proposed
            log_prior = log_prior_proposed
            
        else: # Reject proposal.
            total_acc[i] = 0
            
            if Model == ProposedModel: # Intra model move.
                Model.acc.append(0)
        
        # Update model chain.
        Model.add_state(Theta, adapt = True)
        v = State(scaled = np.concatenate((Theta.scaled, v.scaled[Model.D:])))
        JointModelChain.add_general_state(Model.m, v)

        # Update auxilliary center divergence for new states.
        lv[:Model.D] = Theta.scaled - Model.Center.scaled

    if user_feedback:
        print(f"\n average acc: {np.average(total_acc):4f}")
        print("P(m1|y): " + str(1 - np.sum(JointModelChain.model_indices) / iterations))
        print("P(m2|y): " + str(np.sum(JointModelChain.model_indices) / iterations))

    return JointModelChain


def read_light_curve(file_name):
    """Read in lightcurve data.
    
    Must be between 0 and 72 days, with 720 observations. 
    Photometry data with three columns: time, flux, and error.
    
    Args:
        file_name: String csv file name, in same directory.

    Returns:
        data: MulensData for light curve.
    """
    with open(file_name) as file:
        array = np.loadtxt(file, delimiter = ",")

    data = mm.MulensData(data_list = [array[:, 0], array[:, 1], array[:, 2]], phot_fmt = "flux", chi2_fmt = "flux")

    return data


def synthetic_single(Theta, n_epochs, sn, seed=42):
    """Generate a synthetic single lens lightcurve.
    
    Simulates noise based on guassian flux process.
    In this case, amplification = flux.
    Otherwise based on ROMAN photometric specifications.

    Args:
        Theta: The single model state.
        n_epochs: The number of flux observations.
        sn: The signal to noise baseline.
        seed: (Optional) the integer random seed.
    Returns:
        Data: MulensData for a synthetic lightcurve
    """
    # Create MulensModel.
    model = mm.Model(dict(zip(["t_0", "u_0", "t_E"], Theta.truth)))
    model.set_magnification_methods([0., "point_source", 72.])

    # Exact signal (fs=1, fb=0).
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    truth_signal = model.magnification(epochs)

    # Simulate noise in gaussian errored flux space.
    np.random.seed(seed)
    noise = np.random.normal(0.0, np.sqrt(truth_signal) / sn, n_epochs) 
    noise_sd = np.sqrt(truth_signal) / sn
    
    signal = truth_signal + noise

    data = mm.MulensData(data_list = [epochs, signal, noise_sd], phot_fmt = "flux", chi2_fmt = "flux")

    return data


def synthetic_binary(Theta, n_epochs, sn, seed=42):
    """Generate a synthetic single lens lightcurve.
    
    Simulates noise based on guassian flux process.
    In this case, amplification = flux.
    Otherwise based on ROMAN photometric specifications.

    Args:
        Theta: The single model state.
        n_epochs: The number of flux observations.
        sn: The signal to noise baseline.
        seed: (Optional) the integer random seed.
    Returns:
        Data: MulensData for a synthetic lightcurve
    """
    # Create MulensModel.
    model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], Theta.truth)))
    model.set_magnification_methods([0., "point_source", 72.])

    # Exact signal (fs=1, fb=0).
    epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
    truth_signal = model.magnification(epochs)

    # Simulate noise in gaussian errored flux space.
    np.random.seed(seed)
    noise = np.random.normal(0.0, np.sqrt(truth_signal) / sn, n_epochs) 
    noise_sd = np.sqrt(truth_signal) / sn
    
    signal = truth_signal + noise

    data = mm.MulensData(data_list = [epochs, signal, noise_sd], phot_fmt = "flux", chi2_fmt = "flux")

    return data


def output_file(models, JointModelChain, n_epochs, sn_base, letters, name = "", event_params = None):
    
    # output File:
    with open("results/"+name+"-run.txt", "w") as file:

        file.write("Run "+name+"\n")
        
        # inputs
        file.write("Inputs:\n")
        if event_params is not None:
            file.write("Parameters: "+str(event_params.truth)+"\n")
        file.write("Number of observations: "+str(n_epochs)+", Signal to noise baseline: "+str(sn_base)+"\n")
        
        # priors
        #for Model in Models:
        #    file.write(str(Model.priors)+"\n")

        # run info
        file.write("\n")
        file.write("Run information:\n")
        file.write("Iterations: "+str(joint_model_chain.n)+"\n")
        total_acc = 0
        for Model in Models:
            total_acc += np.sum(Model.acc)
        total_acc /= joint_model_chain.n
        file.write("Average acc; Total: "+str(total_acc))

        # results
        file.write("\n\nResults:\n")
        for Model in Models:
            # models
            P_Model = Model.sampled.n/joint_model_chain.n
            sd_Model = ((Model.sampled.n*(1-P_Model)**2 + (joint_model_chain.n-Model.sampled.n)*(0-P_Model)**2) / (joint_model_chain.n-1))**0.5
            file.write("\n"+str(Model.m)+"\nP(m|y): "+str(P_Model)+"\n")

            # parameters
            Model_states = Model.sampled.states_array(scaled = True)
            for i in range(len(Model.sampled.states[-1].scaled)):
                mu = np.average(Model_states[i, :])
                sd = np.std(Model_states[i, :], ddof = 1)
                file.write(letters[i]+": mean: "+str(mu)+", sd: "+str(sd)+" \n")
    
    return

