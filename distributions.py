"""Basic probability distributions."""

import math
import numpy as np
from scipy.stats import lognorm, loguniform, uniform


class Uniform(object):
    """A uniform distribution.

    Attributes:
        lb: [float] Lower bound for support.
        rb: [float] Upper bound for support.
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
        lb: [float] Lower bound for support.
        rb: [float] Upper bound for support.
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


class Cut_Log_Uniform(object):
    """A log uniform distribution.

    The log of the data is uniformly distributed.

    Attributes:
        lb: [float] Lower bound for support.
        rb: [float] Upper bound for support.
        tl: [float] Lower bound for support.
        tr: [float] Upper bound for support.

    """

    def __init__(self, left, right, t_lower, t_upper):
        """Initialises Log uniform with bounds and sampler."""
        self.lb = left
        self.rb = right
        self.tl = t_lower
        self.tr = t_upper
        self.dist = loguniform(left, right)

        # Probability that is otherwise truncated to zero, distributed uniformly (aprroximation).
        self.truncation = (1-(self.dist.cdf(t_upper)-self.dist.cdf(t_lower))) / (t_upper - t_lower)

    def in_bound(self, x):
        """Check if value is in support."""
        if self.tl <= x <= self.tr: return 1
        else: return 0

    def log_pdf(self, x):
        """Calculate log probability density."""
        if  self.tl <= x <= self.tr: return np.log(self.dist.pdf(x) + self.truncation)
        else: return -math.inf # If out of support.


class Truncated_Log_Normal(object):
    """A truncated log normal distribution.

    The log of the data is normally distributed, and the data is constrained.

    Attributes:
        lb: [float] Lower bound for support.
        rb: [float] Upper bound for support.
    """

    def __init__(self, left, right, mu, sd):
        """Initialises Truncated log normal with bounds and sampler.

        Args:
            mu: [float] The scalar mean of the underlying normal distrubtion in 
                true space.
            sd: [float] The scalar standard deviation of the underlying normal 
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