import MulensModel as mm
import Functions as mc
import random
import numpy as np

from scipy.stats import truncnorm, loguniform, uniform





# Synthetic Event Parameters/Initialisation
SBModel = mm.Model({'t_0': 72, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8})
SBModel.set_magnification_methods([0., 'VBBL', 144.])

t=SBModel.set_times(n_epochs = 100)
# Generate Synthetic Lightcurve
Data = mm.MulensData(data_list=[t, SBModel.magnification(t), SBModel.magnification(t)*0+0.003]) #orignally 0.03

th1 = [71, 0.123, 62]
th2 = [73, 0.14, 62, 0.005, 0.005, 1.05, 222]
#################

#priors
s_pi = mc.loguni(0.2, 5)
q_pi = mc.loguni(10e-6, 1)
alpha_pi = mc.uni(0, 360)
u0_pi = mc.uni(0, 2)
t0_pi = mc.uni(0, 72)
tE_pi = mc.trunclognorm(1, 100, 0.45, 1.15)
priors = [s_pi, q_pi, alpha_pi, u0_pi, t0_pi, tE_pi]

# Initialise
m = random.randint(1,3)
J = 1 #THIS IS NOT CORRECT
iterations = 100

ms = np.zeros((iterations))
ms[0] = m

# Diagonal
covariance=np.multiply(0.1,[0.72, 0.02, 1, 0.01, 0.05, 5])#0.5
#SurrogatePosterior[1].rvs
#SurrogatePosterior[2].rvs
covProp = []
covProp[1], h = mc.AdaptiveMCMC(1, Data, th1, covariance, 200)
covProp[2], h = mc.AdaptiveMCMC(2, Data, th2, covariance, 200)


states = []#np.zeros((iterations))

centers = [th1, th2]

for i in range(iterations):

    mProp = random.randint(1,3)
    thetaProp = mc.RJCenteredProposal(m, mProp, theta, covProp[mProp], centers[mProp])

    acc = mc.Likelihood(mProp, Data, thetaProp)/mc.Likelihood(m, Data, theta) * mc.PriorRatio(m, mProp, theta, thetaProp, priors) * J#???
    
    if random.rand <= acc:
        theta = thetaProp
        m = mProp
    
    states.append(theta)
    ms[i] = m






