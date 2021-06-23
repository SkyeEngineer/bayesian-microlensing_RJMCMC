from math import pi
import MulensModel as mm
import Functions as mc
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform





# Synthetic Event Parameters/Initialisation
#SBModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8}) #strong
#SBModel.set_magnification_methods([0., 'VBBL', 72.])
#SBModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0004, 's': 1.33, 'alpha': 223.8}) #weak
#SBModel.set_magnification_methods([0., 'VBBL', 72.])
SBModel = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.000002, 's': 4.9, 'alpha': 223.8}) #single
SBModel.set_magnification_methods([0., 'VBBL', 72.])


t=SBModel.set_times(n_epochs = 100)
# Generate Synthetic Lightcurve
Data = mm.MulensData(data_list=[t, SBModel.magnification(t), SBModel.magnification(t)/5], phot_fmt='flux', chi2_fmt='flux') #orignally 0.03, last entry represents noise
#Data = SBModel.magnification(t)
#SBModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('RJLike.png')
iterations = 200
#th1 = np.array([36., 0.133, 61.5])
th2 = np.array([36, 0.133, 61.5, 0.00096, 0.000002, 3.3, 223.8]) # nice reesults for adaption
#th2 = np.array([36., 0.133, 61.5, 0.0014, 0.00096, 1.2, 224.]) # nice results for model!!!!!!!!!!!!!!!
#th2 = np.array([36., 0.133, 61.5, 0.001, 0.00095, 1.23, 223.7])
#print(np.exp(mc.Likelihood(1, Data, th1, 5)))
#print(np.exp(mc.Likelihood(2, Data, th2, 5)))
th1 = np.array([36., 0.133, 61.5])
#th2 = np.array([36., 0.133, 61.5, 0.00096, 0.0004, 1.33, 223.8])
#th2 = np.array([36., 0.133, 61.5, 0.00096, 0.000002, 4.9, 223.8])

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
m = 2#random.randint(1,2)
J = 1 #THIS IS NOT CORRECT


ms = np.zeros((iterations), dtype=int)
ms[0] = m
#ms = [m]

# Diagonal
covariance1=np.multiply(0.0001, [0.01, 0.01, 0.1])
covariance2=np.multiply(0.0001, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])#0.5
#SurrogatePosterior[1].rvs
#SurrogatePosterior[2].rvs
covProp = [covariance1, covariance2]
'''
burns=2
iters=2
#covProp1, h1, dec1 = mc.AdaptiveMCMC(1, Data, th1, priors, covariance1, 200, 200)
covProp2, h2, dec2 = mc.AdaptiveMCMC(2, Data, th2, priors, covariance2, burns, iters)


yes=[]
size=40
bins=int((burns+iters)/size)
for bin in range(bins):
    yes.append(np.sum(dec2[size*bin:size*(bin+1)])/size)

#print(dec2)
plt.plot(np.linspace(1, bins, num=bins), yes)
plt.xlabel('Iterations [bins]')
plt.ylabel('Acceptance rate')
plt.title('Adaptive MCMC acceptance timeline')
plt.savefig('Plots/Adaptive-MCMC-acceptance-progression.png')
plt.clf()

#covProp = [covProp1, covProp2]

'''
states = []#np.zeros((iterations))

centers = [th1, th2]
#random sample
#thet = h2[:,-1]#th1
#theta = thet[0:3]#th1
theta=[36., 0.133, 61.5, 0.0014, 0.0009, 1.26, 224.]

#print(np.cov(h1))
#print(np.cov(h2))
'''

#print((covProp1))
print((covProp2))

#print(np.prod(covariance1))
print(np.prod(covariance2))
#print(np.linalg.det(covProp1))
#print(np.linalg.det(covProp2))

#print(np.linalg.det(np.cov(h1)))
print((np.cov(h2))) # clearly, adaption is calculating wrong


plt.scatter((h2[5,:]), (h2[4,:]), alpha=0.25)
#plt.scatter(np.abs(walkB[:,4]), np.abs(walkB[:,3]), alpha=0.5, c='r')
plt.xlabel('s [Einstein Ring Radius]') # Set the y axis label of the current axis.
plt.ylabel('q') # Set a title.
plt.title('Adaptive MCMC walk over binary model space')
plt.scatter(1.33, 0.0004, c=[(0,0,1)], marker='2', label='True', s=50)#2 is chevron triangle
plt.scatter(th2[5], th2[4], c=[(1,0,0)], marker='2', label='Initial\nPosition', s=50)#2 is chevron triangle
plt.legend()
plt.savefig('Plots/Adaptive-Covariance-Sampleing-Walk.png')
#plt.savefig('Plots/Walk.png')
plt.clf()
'''

score=0


for i in range(iterations):

    mProp = random.randint(1,2)
    thetaProp = mc.RJCenteredProposal(m, mProp, theta, covProp[mProp-1], centers)
    #print('prop: '+str(thetaProp))


    priorRatio = np.exp(mc.PriorRatio(m, mProp, theta, thetaProp, priors))
    #print(priorRatio)
    piProp = np.exp(mc.Likelihood(mProp, Data, thetaProp, 5))
    pi = np.exp(mc.Likelihood(m, Data, theta, 5))
    acc =  piProp/pi * priorRatio #???
    #print(piProp, pi)
    
    if random.random() <= (acc)*J:
        theta = thetaProp
        m = mProp
        score+=1
    #else: print('no :(') 
    
    states.append(theta)
    ms[i] = m
    #ms.append(m)

print("acc: "+str(score/iterations))
print("1: "+str(1-np.sum(ms-1)/iterations))
print("2: "+str(np.sum(ms-1)/iterations))
#print(states)
'''
#ms=ms.astype(int)
#states2=states[ms==2]
s2=[]
for i in range(iterations):
    if ms[i]==2: s2.append(states[i])
#s2=states[(np.where(ms==2)[0])]
s2=np.array(s2)
#print(s2)
markersize=50
plt.scatter((s2[:,5]), (s2[:,4]), alpha=0.25)
plt.xlabel('s [Einstein Ring Radius]')
plt.ylabel('q [Unitless]')
plt.title('RJ binary model walk through Mass Fraction / Separation space\nwith centreing function')
plt.scatter(s2[0,5], s2[0,4], c=[(0,1,0)], marker='2', label='Initial\nPosition', s=markersize)#2 is chevron triangle
plt.scatter(th2[5], th2[4], c=[(1,0,0)], marker='2', label='Centreing\npoint', s=markersize)#2 is chevron triangle
plt.scatter(1.33, 0.0004, c=[(0,0,1)], marker='2', label='True', s=markersize)
plt.legend()
plt.savefig('Plots/RJ-binary-Walk.png')
plt.clf()


s1=[]
for i in range(iterations):
    if ms[i]==1: s1.append(states[i])
#s2=states[(np.where(ms==2)[0])]
s1=np.array(s1)
#print(s2)
plt.scatter((s1[:,2]), (s2[:,1]), alpha=0.5)
plt.xlabel('u0')
plt.ylabel('tE')
plt.title('Walk:')
plt.scatter(th1[2], th1[1], c=[(1,0,0)], marker="2")#2 is chevron triangle
plt.savefig('Plots/RJ-single-Walk.png')
plt.clf()



plt.plot(np.linspace(1, iterations, num=iterations), ms)
plt.title('RJ Weak binary event with centreing function\nmodel trace')
plt.xlabel('Iteration')
plt.ylabel('Model Index')
plt.locator_params(axis="y", nbins=2)
plt.savefig('Plots/Trace.png')
plt.clf()
'''
