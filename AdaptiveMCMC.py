# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Adaptive Markov Chain Monte Carlo Testing]

import MulensModel as mm
import Functions as f
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform
import PlotFunctions as pltf

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# TESTING
'''
states = np.zeros((2, 3))
states[:, 0] = [1, 0.5]

means = np.zeros((2, 3))
means[:, 0] = [1, 0.5]

t=1
theta = [2, 0.75]
states[:, t] = theta
means[:, t] = (means[:, t-1]*t + theta)/(t + 1) # recursive mean (offsets indices starting at zero by one)


#print(states[:, 0:2])

t=2
theta = [0.75, 1]
states[:, 2] = theta
means[:, t] = (means[:, t-1]*t + theta)/(t + 1) # recursive mean (offsets indices starting at zero by one)
print(states, 'states')
print(means, 'mean')

covariance = np.cov((states[:, 0:2]))
print(covariance, 'initial true cov')

# update step (recursive covariance)
covariance = (t-1)/t * covariance + (1/t) * (t*np.outer(means[:, t - 1], means[:, t - 1]) - (t + 1)*np.outer(means[:, t-0], means[:, t-0]) + np.outer(states[:, t-0], states[:, t-0]))

print(covariance, 'final cov')
print(np.cov(states), 'true cov')
#print((1/t) * (t*(means[:, t - 1].T).dot(means[:, t - 1]) - (t + 1)*(means[:, t-0].T).dot(means[:, t-0]) + (states[:, t-0].T).dot(states[:, t-0])))
#print(np.outer(means[:, t-0], means[:, t-0].T))
'''


plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12

plt.style.use('seaborn-bright')

plt.rcParams["legend.edgecolor"] = '0'
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.title_fontsize"] = 10
plt.rcParams["legend.fontsize"] = 9

plt.rcParams["grid.linestyle"] = 'dashed' 
plt.rcParams["grid.alpha"] = 0.25




# INITIALISATION

# Synthetic Event Parameters
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 'q': 0.0039, 's': 1.120, 'alpha': 223.8}) # strong binary
Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0096, 'q': 0.0004, 's': 1.33, 'alpha': 223.8}) # weak binary
#Model = mm.Model({'t_0': 36, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.0096, 'q': 0.02, 's': 1.1, 'alpha': 223.8}) # indistiguishable from single
Model.set_magnification_methods([0., 'VBBL', 72.])

# Generate "Synthetic" Lightcurve
t = Model.set_times(n_epochs = 72)
error = Model.magnification(t)/50 + 0.1
Data = mm.MulensData(data_list=[t, Model.magnification(t), error], phot_fmt='flux', chi2_fmt='flux')

# priors (Zhang et al)
s_pi = f.logUniDist(0.2, 5)
q_pi = f.logUniDist(10e-6, 1)
alpha_pi = f.uniDist(0, 360)
u0_pi = f.uniDist(0, 2)
t0_pi = f.uniDist(0, 72)
tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  f.logUniDist(10**-4, 10**-2)
priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

# initial points
theta_1i = np.array([36., 0.133, (61.5)])
#theta_1i = np.array([36., 0.133, 61.5])
theta_2i = np.array([35, 0.125, 61.1, 0.00988, np.log(0.000305), 1.258, 222.8]) # nice results for adaption
#theta_2i = np.array([36., 0.133, 61.5, 0.0014, 0.00096, 1.2, 224.]) # nice results for model
# print(np.exp(f.logLikelihood(1, Data, theta_1i)))
# print(np.exp(f.logLikelihood(2, Data, theta_2i)))

# initial covariances (diagonal)
covariance_1i=np.multiply(1, [0.01, 0.01, 0.1]) #0.0001
covariance_2id=np.multiply(0.01, [0.1, 0.01, 0.1, 0.0001, 0.01, 0.01, 0.1])#[0.1, 0.01, 0.1, 0.0001, 0.01, 0.1, 1])#[0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001]
covariance_2i = np.zeros((7, 7))
np.fill_diagonal(covariance_2i, covariance_2id)
#should be small to get a quick taste of size (too small makes growth to normal size to slow)

burns = 25 #should be small to not bias later
iters = 25000

#covariance_1p, states_1, c_1 = f.AdaptiveMCMC(1, Data, theta_1i, priors, covariance_1i, 200, 200)
covariance_2p, states_2, means_2, c_2, covs = f.AdaptiveMCMC(2, Data, theta_2i, priors, covariance_2i, burns, iters)

# Create and plot the accepatnce ratio over time
acc = []
size = 50#50
bins = int((burns + iters) / size)

for bin in range(bins): # record the ratio of acceptance for each bin
    acc.append(np.sum(c_2[size*bin:size*(bin+1)]) / size)

plt.plot((np.linspace(1, bins, num=bins)), acc)
plt.ylim((0.0, 1.0))
plt.xlabel(f'Binned Iterations Over Time [Bins, n={size}]')
plt.ylabel('Fraction of Accepted\nProposals '+r'[$Bins^{-1}$]')
plt.title('Adaptive MCMC acceptance timeline')
#plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Plots/Adaptive-MCMC-acceptance-progression.png')
plt.clf()


print(len(covs))
Trs = np.zeros((7, len(covs)))
for i in range(1, len(covs)):
    Mat = (covs[i])
    #print(Mat)
    #print('hi')
    #print(Mat[1, :])
    #print(Mat[:, 1])
    #print('hi')
    for j in range(7):
        Trs[j, i] = Mat[j, j]#(np.sum(np.abs(Mat[j, :])) + np.sum(np.abs(Mat[:, j]))-np.abs(Mat[j,j])) ##

#print(Trs[:, 0])
#print(Trs[0, :])
#Trs[:, 0]=covariance_2i


ij = np.linspace(1, len(covs), num=len(covs))
vars = ['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha']
colors = [cm.rainbow(i) for i in np.linspace(0, 1, 7)]
for j in range(7):
    plt.plot(ij, Trs[j, :], c=colors[j], label = vars[j], alpha=0.9)#/theta_2i[j]

plt.xlabel(f'Iterations [{size}]')
plt.ylabel('Pseudo Trace'+r'[$Bins^{-1}$]')
plt.title('Adaptive MCMC size timeline')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Plots/Adaptive-MCMC-size-progression.png')
plt.clf()



plt.plot(ij, np.sum(Trs, axis=0), c=colors[j], label = 'Trace', alpha=0.9)
plt.xlabel(f'Iterations [{size}]')
plt.ylabel('Trace'+r'[$Bins^{-1}$]')
plt.title('Adaptive MCMC size timeline')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Plots/Adaptive-MCMC-Trace-progression.png')
plt.clf()


#print(np.cov(states_1))
#print(np.cov(states_2))

#print((covariance_1p))
#print((covariance_2p))

#print(np.prod(covariance_1i))
#print(np.prod(covariance_2i), 'inititial')
#print(np.linalg.det(covariance_1p))
#print(np.linalg.det(covariance_2p), 'adapted')

#print(np.linalg.det(np.cov(states_2)), 'empirical')
#print((np.cov(states_2))) # clearly, adaption is calculating wrong

# plot the points visited during the walk

labels = [r'Impact Time [$?$]', r'Minimum Impact Parameter [$?$]', r'Crossing Time [$?$]', r'Rho [$?$]', r'Mass Ratio', r'Separation [$E_r$]', r'Alpha [$Degrees$]', ]
symbols = [r'$t_0$', r'$u_0$', r'$t_E$', r'$\rho$', r'$q$', r'$s$', r'$\alpha$']
details = False


for i in range(7):
    #pltf.PlotWalk(4, 5, states_2, 1, 1, labels, symbols, details)
    pltf.TracePlot(i, np.transpose(states_2), 1, 1, 1, 1, labels, symbols, details)


plt.scatter(states_2[5,:], np.exp(states_2[4,:]), c=np.linspace(0.0, 1.0, iters+burns), cmap='spring', alpha=0.25, marker="o")
cbar = plt.colorbar(fraction = 0.046, pad = 0.04, ticks=[0, 1]) # empirical nice auto sizing
#cbar.set_label('Time', rotation = 90, fontsize=10)
ax=plt.gca()
cbar.ax.set_yticklabels(['Initial\nStep', 'Final\nStep'], fontsize=9)
cbar.ax.yaxis.set_label_position('right')
plt.xlabel(r'Separation [$E_r$]')
plt.ylabel('Mass Ratio')
plt.title('Adaptive MCMC walk\nprojected onto Binary (s, q) space')
#plt.scatter(1.1, 0.02, marker='*', label='True', s=75, c='black', alpha=1)#r'$\circledast$'
plt.legend()
plt.grid()
plt.gca().ticklabel_format(useOffset=False)
plt.tight_layout()
plt.savefig('Plots/Adaptive-Covariance-Sampleing-Walk.png')
plt.clf()




#https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(0, 1, N)
Y = np.linspace(0, 2, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0.1, 1.1])
Sigma = np.array([[covariance_2p[4][4], covariance_2p[4][5]], [covariance_2p[5][4], covariance_2p[5][5]]])
print(Sigma)
H=G
# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.savefig('Plots/Adaptive-Normal.png')
plt.clf()