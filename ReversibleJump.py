import MulensModel as mm
import Functions as mc
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform





# Synthetic Event Parameters/Initialisation
SBModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8})
SBModel.set_magnification_methods([0., 'VBBL', 72.])

t=SBModel.set_times(n_epochs = 100)
# Generate Synthetic Lightcurve
Data = mm.MulensData(data_list=[t, SBModel.magnification(t), SBModel.magnification(t)/100], phot_fmt='flux', chi2_fmt='flux') #orignally 0.03, last entry represents noise
#Data = SBModel.magnification(t)
#SBModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('RJLike.png')

th1 = np.array([36, 0.133, 61.5])
th2 = np.array([36, 0.134, 61.5, 0.00091, 0.004, 1.119, 223.8])
print(np.exp(mc.Likelihood(1, Data, th1, 5)))
print(np.exp(mc.Likelihood(2, Data, th2, 5)))

#noise=w

#print(Data)
#mc.Likelihood(2, Data, th2, noise)
#print((np.exp(mc.Likelihood(2, Data, th2, noise))))
#print(SBModel.magnification(t))
#################
#g=p
#priors
s_pi = mc.loguni(0.2, 5)
q_pi = mc.loguni(10e-6, 1)
alpha_pi = mc.uni(0, 360)
u0_pi = mc.uni(0, 2)
t0_pi = mc.uni(0, 72)
tE_pi = mc.trunclognorm(1, 100, 10**1.15, 10**0.45)
rho_pi =  mc.loguni(10**-4, 10**-2)
priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

# Initialise
m = 1#random.randint(1,2)
J = 1 #THIS IS NOT CORRECT
iterations = 1000

ms = np.zeros((iterations))
ms[0] = m

# Diagonal
covariance1=np.multiply(0.1, [0.01, 0.01, 0.1])
covariance2=np.multiply(0.01, [0.01, 0.01, 0.1, 0.0001, 0.001, 0.01, 0.1])#0.5
#SurrogatePosterior[1].rvs
#SurrogatePosterior[2].rvs

#covProp1, h1 = mc.AdaptiveMCMC(1, Data, th1, covariance1, noise, 1000)
#covProp2, h2 = mc.AdaptiveMCMC(2, Data, th2, covariance2, noise, 500)
#covProp = [covProp1, covProp2]
covProp = [covariance1, covariance2]

states = []#np.zeros((iterations))

centers = [th1, th2]
theta = th1 #random sample

'''
#print(h1)
#print(h2)
#print((covProp1))
#print((covProp2))
print(np.linalg.det(covProp2))
print(np.linalg.det(covProp1))
print(np.linalg.det(np.cov(h1)))
print(np.linalg.det(np.cov(h2)))

plt.scatter((h2[5,:]), (h2[4,:]), alpha=0.5)
#plt.scatter(np.abs(walkB[:,4]), np.abs(walkB[:,3]), alpha=0.5, c='r')
plt.xlabel('s [Einstein Ring Radius]') # Set the y axis label of the current axis.
plt.ylabel('q') # Set a title.
plt.title('Adaptive Walk:')
plt.savefig('CovSampWalk.png')
'''
#q=w
score=0

for i in range(iterations):

    mProp = random.randint(1,2)
    thetaProp = mc.RJCenteredProposal(m, mProp, theta, covProp[mProp-1], centers)
    #print('prop: '+str(thetaProp))


    piRatio = mc.PriorRatio(m, mProp, theta, thetaProp, priors)
    #print(num, den)
    acc = piRatio + mc.Likelihood(mProp, Data, thetaProp, 5) -  mc.Likelihood(m, Data, theta, 5)#???
    #print(np.exp(acc))
    
    if random.random() <= np.exp(acc)*J:
        theta = thetaProp
        m = mProp
        score+=1
    #else: print('no :(') 
    
    states.append(theta)
    ms[i] = m

print(score/iterations)
print(ms)






