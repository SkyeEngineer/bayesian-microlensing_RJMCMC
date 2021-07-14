# Author: Dominic Keehan
# Part 4 Project, RJMCMC for Microlensing
# [Main]

import MulensModel as mm
import Functions as f
import Autocorrelation as AC
import PlotFunctions as pltf
import emcee as MC
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, loguniform, uniform
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
from multiprocessing import Pool




plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12

plt.style.use('seaborn-bright')

plt.rcParams["legend.edgecolor"] = '0'
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.title_fontsize"] = 10
plt.rcParams["legend.fontsize"] = 9

plt.rcParams["grid.linestyle"] = 'dashed' 
plt.rcParams["grid.alpha"] = 0.25

plt.rc('axes.formatter', useoffset=False)


labels = [r'Impact Time [$days$]', r'Minimum Impact Parameter [$1$]', r'Einstein Crossing Time [$days$]', r'Rho [$?$]', r'Mass Ratio', r'Separation [$E_r$]', r'Alpha [$Degrees$]', ]
symbols = [r'$t_0$', r'$u_0$', r'$t_E$', r'$\rho$', r'$q$', r'$s$', r'$\alpha$']


## INITIALISATION ##

sn = 0

# Synthetic Event Parameters
theta_Models = [
    [36, 0.133, 61.5, 0.0096, 0.002, 1.27, 210.8], # strong binary
    [36, 0.133, 61.5, 0.0096, 0.00091, 1.3, 210.8], # weak binary 1
    [36, 0.133, 61.5, 0.0056, 0.0007, 1.3, 210.8], # weak binary 2
    [36, 0.133, 61.5, 0.0096, 0.0002, 4.9, 223.8], # indistiguishable from single
    [36, 0.133, 61.5]  # single
    ]
theta_Model = np.array(theta_Models[sn])

Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta_Model)))
Model.set_magnification_methods([0., 'VBBL', 72.])

#Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta)))
#Model.set_magnification_methods([0., 'point_source', 72.])

#Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('temp.jpg')
#plt.clf()


#0, 50, 25, 0.3
# Generate "Synthetic" Lightcurve
epochs = Model.set_times(n_epochs = 50)
error = Model.magnification(epochs)/30 + 0.25
Data = mm.MulensData(data_list=[epochs, Model.magnification(epochs), error], phot_fmt='flux', chi2_fmt='flux')

iterations = 20000



# priors (Zhang et al)
s_pi = f.logUniDist(0.2, 5)
q_pi = f.logUniDist(10e-6, 1)
alpha_pi = f.uniDist(0, 360)
u0_pi = f.uniDist(0, 2)
t0_pi = f.uniDist(0, 72)
tE_pi = f.truncatedLogNormDist(1, 100, 10**1.15, 10**0.45)
rho_pi =  f.logUniDist(10**-4, 10**-2)
a = 0.1
#m_pi = [1-a, a]
#priors = [t0_pi, u0_pi,  tE_pi, rho_pi,  q_pi, s_pi, alpha_pi]

# uninformative priors
s_upi = f.uniDist(0.2, 5)
q_upi = f.uniDist(10e-6, 1)
alpha_upi = f.uniDist(0, 360)
u0_upi = f.uniDist(0, 2)
t0_upi = f.uniDist(0, 72)
tE_upi = f.uniDist(1, 100)
rho_upi =  f.uniDist(10**-4, 10**-2)

priors = [t0_upi, u0_upi,  tE_upi, rho_upi,  q_upi, s_upi, alpha_upi]
m_pi = [0.5, 0.5]



def ParralelMain(arr):

    sn, Data, priors, m_pi, iterations = arr

    # centreing points for inter-model jumps
    center_1 = np.array([36., 0.133, 61.5])
    #center_1 = np.array([36., 0.133, 61.5])

    center_2s = [
        [36, 0.133, 61.5, 0.0096, np.log(0.002), 1.27, 210.8], # strong binary
        [36., 0.133, 61.5, 0.00963, np.log(0.00092), 1.31, 210.8], # weak binary 1
        [36., 0.133, 61.5, 0.0052, np.log(0.0006), 1.29, 210.9], # weak binary 2
        [36., 0.133, 61.5, 0.0096, np.log(0.00002), 4.25, 223.8], # indistiguishable from single
        ]
    center_2 = np.array(center_2s[sn])
    #print(center_2)
    # print(np.exp(f.logLikelihood(1, Data, center_1)))
    # print(np.exp(f.logLikelihood(2, Data, center_2)))
    centers = [center_1, center_2]

    # initial covariances (diagonal)
    covariance_1 = np.multiply(0.01, [0.1, 0.01, 0.1])
    covariance_2 = np.multiply(0.01, [0.1, 0.01, 0.1, 0.0001, 0.01, 0.01, 0.1])#0.5

    #covariance_1s = np.multiply(1, [0.01, 0.01, 0.1])
    #covariance_2s = np.multiply(1, [0.01, 0.01, 0.1, 0.0001, 0.0001, 0.001, 0.001])#0.5
    #covariance_1 = np.outer(covariance_1s, covariance_1s)
    #covariance_2 = np.outer(covariance_2s, covariance_2s)
    #covariance_p = [covariance_1, covariance_2]

    # Use adaptiveMCMC to calculate initial covariances
    burns = 25
    iters = 100#25
    theta_1i = center_1
    theta_2i = center_2
    covariance_1p, states_1, means_1, c_1, NULL = f.AdaptiveMCMC(1, Data, theta_1i, priors, covariance_1, burns, iters)
    covariance_2p, states_2, means_2, c_2, NULL = f.AdaptiveMCMC(2, Data, theta_2i, priors, covariance_2, burns, iters)

    covariance_p = [covariance_1p, covariance_2p]


    # loop specific values

    print(states_1[:, -1])
    theta = states_1[:, -1]#[36., 0.133, 61.5]#, 0.0014, 0.0009, 1.26, 224.]
    m = 1
    pi = (f.logLikelihood(m, Data, f.unscale(m, theta), priors))
    #print(pi)

    ms = np.zeros(iterations, dtype=int)
    ms[0] = m
    states = []
    score = 0
    Dscore = 0
    Dtotal = 1
    J_2 = np.prod(center_2[0:3])
    J_1 = np.prod(center_1)
    J = np.abs([J_1/J_2, J_2/J_1])
    #print(J)

    #adaptive params
    t=[burns+iters,burns+iters]
    I = [np.identity(3), np.identity(7)] 
    s = [2.4**2 / 3, 2.4**2 / 7] # Arbitrary(ish), good value from Haario et al
    eps = 1e-12 # Needs to be smaller than the scale of parameter values
    means = [np.zeros((3, iters+burns+iterations)), np.zeros((7, iters+burns+iterations))]
    #print(means[0][:,0:2])
    #print(means_1)
    means[0][:, 0:burns+iters] = means_1
    means[1][:, 0:burns+iters] = means_2

    bests = [0, 0]
    bestt = [[], []]

    print('Running RJMCMC')

                
    mem_2 = states_2[:, -1]
    adaptive_score = [[], []]
    #inter_props = [0, 0]

    for i in range(iterations): # loop through RJMCMC steps
        
        #diagnostics
        #print(f'\rLikelihood: {np.exp(pi):.3f}', end='')
        cf = i/(iterations-1);
        print(f'Current: Likelihood {np.exp(pi):.4f}, M {m} | Progress: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')

        mProp = random.randint(1,2) # since all models are equally likelly, this has no presence in the acceptance step
        #thetaProp = f.RJCenteredProposal(m, mProp, theta, covariance_p[mProp-1], centers, mem_2, priors) #states_2)

        #priorRatio = np.exp(f.PriorRatio(m, mProp, f.unscale(m, theta), f.unscale(mProp, thetaProp), priors))
        
        #piProp = (f.logLikelihood(mProp, Data, f.unscale(mProp, thetaProp), priors))

        #print(piProp, pi, priorRatio, mProp)
        #scale = 1
        
        #if mProp == 2 and m == 1: 
        #    l = (theta - centers[m-1]) / centers[m-1]
        #    scale = 1/(np.max(l) - np.min(l))
        #elif mProp == 1 and m == 2:
        #    l = (thetaProp - centers[mProp-1]) / centers[mProp-1]
        #    scale = 1/(np.max(l[0:3]) - np.min(l[0:3]))

        #scale = 1
        thetaProp, piProp, acc = f.Propose(Data, m, mProp, theta, pi, covariance_p, centers, mem_2, priors, False)
        #if random.random() <= scale * np.exp(piProp-pi) * priorRatio * m_pi[mProp-1]/m_pi[m-1] * J[mProp-1]: # metropolis acceptance
        if random.random() <= acc: #*q!!!!!!!!!!!!# metropolis acceptance
            if m == mProp: adaptive_score[mProp - 1].append(1)

            theta = thetaProp
            m = mProp
            score += 1
            pi = piProp
            if mProp == 2: mem_2 = thetaProp

            if bests[mProp-1] < np.exp(piProp): 
                bests[mProp-1] = np.exp(piProp)
                bestt[mProp-1] = f.unscale(mProp, thetaProp)

        elif m == mProp: adaptive_score[mProp - 1].append(0)


        elif m != mProp and random.random() <= 0.5 and False: #Delayed rejection for Jump False: #
            Dtotal += 1

            thetaProp_2, piProp_2, acc_2 = f.Propose(Data, m, mProp, theta, pi, covariance_p, centers, mem_2, priors, True)

            pi_2 = f.logLikelihood(mProp, Data, f.unscale(mProp, thetaProp_2), priors)
            thetaProp_25, piProp_25, acc_25 = f.Propose(Data, mProp, m, thetaProp_2, pi_2, covariance_p, centers, mem_2, priors, False)
            
            if random.random() <= acc_2 * (1-acc_25)/(1 - acc) * 1/(2**3): #*q!!!!!!!!!!!!# delayed metropolis acceptance
                theta = thetaProp_2
                m = mProp
                score += 1
                Dscore += 1
                pi = piProp_2
                
            if mProp==2: mem_2 = thetaProp_2

            if bests[mProp-1] < np.exp(piProp_2): 
                bests[mProp-1] = np.exp(piProp_2)
                bestt[mProp-1] = f.unscale(mProp, thetaProp_2)

        #scale = 1
        states.append(theta)
        ms[i] = m


        tr = t[m-1]
        means[m-1][:, tr] = (means[m-1][:, tr-1]*tr + theta)/(tr + 1) # recursive mean (offsets indices starting at zero by one)    
        # update step (recursive covariance)

        covariance_p[m-1] = (tr - 1)/tr * covariance_p[m-1] + s[m-1]/tr * (tr*means[m-1][:, tr - 1]*np.transpose(means[m-1][:, tr - 1]) - (tr + 1)*means[m-1][:, tr]*np.transpose(means[m-1][:, tr]) + theta*np.transpose(theta)) #+ eps*I[m-1]



        t[m-1] += 1

    # performance diagnostics:
    print("\nIterations: "+str(iterations))
    print("Accepted Move Fraction: "+str(score/iterations))
    print("Accepted Delayed Move Fraction: "+str(Dscore/Dtotal))
    print("P(Singular): "+str(1-np.sum(ms-1)/iterations))
    print("P(Binary): "+str(np.sum(ms-1)/iterations))
    #print(states)

    return states, adaptive_score, ms, bestt, bests, centers


params = [sn, Data, priors, m_pi, iterations]

states, adaptive_score, ms, bestt, bests, centers = ParralelMain(params)
center_1, center_2 = centers
'''
p = 2
pool = Pool(p)

poolSol = pool.map(ParralelMain, np.tile(params, (p, 1)))

#print(poolSol[0][0])
#print(poolSol[1][0])
#print(poolSol[0][1])
#print(poolSol[1][1])


states = np.array(poolSol)[:, 0]
adaptive_history = np.array(poolSol)[:, 1]
#print(adaptive_history)
#print(states)

## AUTO CORR ANALYSIS TRUNCATION + PLOTS ##


# Plot the comparisons
#N=iterations
N = np.exp(np.linspace(np.log(1000), np.log(iterations), 10)).astype(int)
#N = np.linspace((100), iterations, 20).astype(int)

new = np.zeros((p, len(N)))
#newm = np.empty(len(N))
#newu = np.empty(len(N))
for chain in range(p):
    y = np.array(AC.scalarPolyProjection(states[chain]))
#y = states
#y= states

    for i, n in enumerate(N):
    #gw2010[i] = a.autocorr_gw2010(y[:, :n])
        new[chain][i] = MC.autocorr.integrated_time(y[:n], c=5, tol=50, quiet=True)#AC.autocorr_new(y[:n])#autocorr_new(y[:, :n])
    #newm[i] = MC.autocorr.integrated_time(jumpStates_2[:n, 4], c=5, tol=50, quiet=True)
    #newu[i] = MC.autocorr.integrated_time(states_u[:n], c=5, tol=50, quiet=True)


#plt.loglog(N, gw2010, "o-", label="G&W 2010")

plt.plot(N, np.average(new, 0), "o-", label="Average Scalar")
plt.plot(N, new[0][:], "o-", label="0")
plt.plot(N, new[1][:], "o-", label="1")
#plt.plot(N, newm, "--", label="M")
#plt.plot(N, newu, label="newu")
#plt.plot(np.linspace(1, iterations, num=iterations), y, "o-", label="new")

ylim = plt.gca().get_ylim()
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
#plt.axhline(true_tau, color="k", label="truth", zorder=-100)
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14)
plt.savefig('Plots/AutoCorr.png')
plt.clf()

#plt.plot(np.linspace(1, iterations, num=iterations), m, "o-", label="m")
plt.plot(np.linspace(1, iterations, num=iterations), y, "-", label="scalar")
plt.legend(fontsize=14)
plt.savefig('Plots/Temp.png')
plt.clf()


pltf.AdaptiveProgression(adaptive_history, ['Single', 'Binary'], p)

g=g
'''









## OTHER PLOT RESULTS ##

markerSize=75

states_2 = []
jumpStates_2 = []
h_ind = []
h=0
for i in range(iterations): # record all binary model states in the chain
    if ms[i] == 2: 
        states_2.append(f.unscale(2, states[i]))
        if ms[i-1] == 1: 
            jumpStates_2.append(f.unscale(2, states[i]))
            h_ind.append(len(states_2))


states_2=np.array(states_2)
jumpStates_2 = np.array(jumpStates_2)



states_1 = []
h_states_1 = []
for i in range(iterations): # record all single model states in the chain
    if ms[i] == 1: 
        states_1.append(f.unscale(1, states[i]))
        if ms[i-1] == 2: h_states_1.append(f.unscale(1, states[i]))
states_1=np.array(states_1)
h_states_1=np.array(h_states_1)



details = True


states_u = []
for i in range(iterations): # record all single model states in the chain
    states_u.append(states[i][1])
states_u=np.array(states_u)









#states_2df = pd.DataFrame(data = states_2, columns = labels)
#print(states_2df)

#grid = sns.PairGrid(data = states_2df, vars = labels, height = 7)
#grid = grid.map_upper(pltf.PPlotWalk)
#plt.savefig('Plots/RJ-binary-pplot.png')
#plt.clf()



pltf.LightcurveFitError(2, bestt[1], priors, Data, Model, epochs, error, details)

pltf.LightcurveFitError(1, bestt[0], priors, Data, Model, epochs, error, details)





pltf.PlotWalk(4, 5, states_2, f.unscale(2, center_2), theta_Model, labels, symbols, details)

pltf.PlotWalk(5, 6, states_2, f.unscale(2, center_2), theta_Model, labels, symbols, details)

pltf.TracePlot(5, states_2, jumpStates_2, h_ind, f.unscale(2, center_2), theta_Model, labels, symbols, details)



pltf.DistPlot(2, 4, states_2, f.unscale(2, center_2), theta_Model, labels, symbols, details)

pltf.DistPlot(2, 5, states_2, f.unscale(2, center_2), theta_Model, labels, symbols, details)


## SINGLE MODEL ##

pltf.DistPlot(1, 0, states_1, f.unscale(1, center_1), theta_Model, labels, symbols, details)
pltf.DistPlot(1, 1, states_1, f.unscale(1, center_1), theta_Model, labels, symbols, details)
pltf.DistPlot(1, 2, states_1, f.unscale(1, center_1), theta_Model, labels, symbols, details)


pltf.PlotWalk(2, 1, states_1, f.unscale(1, center_1), theta_Model, labels, symbols, details)



plt.plot(np.linspace(1, iterations, num = iterations), ms, linewidth=0.5)
plt.title('RJMCMC Model Trace')
plt.xlabel('Iterations')
plt.ylabel('Model Index')
plt.locator_params(axis="y", nbins=2)
plt.tight_layout()
plt.savefig('Plots/M-Trace.png')
plt.clf()





plt.grid()
plt.plot(np.linspace(1, iterations, num=iterations), AC.scalarPolyProjection(states), "-", label="scalar")
#plt.legend(fontsize=14)
plt.title('RJMCMC scalar indicator trace')
plt.xlabel('Iterations')
plt.ylabel('Scalar Indicator')
plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
plt.tight_layout()
plt.savefig('Plots/Temp.png')
plt.clf()


# Plot the comparisons
N = np.exp(np.linspace(np.log(1000), np.log(iterations), 10)).astype(int)

new = np.zeros(len(N))
y = np.array(AC.scalarPolyProjection(states))
for i, n in enumerate(N):
    new[i] = MC.autocorr.integrated_time(y[:n], c=5, tol=50, quiet=True)

#
plt.loglog(N, new, "o-")#, label=r"$\tau$")

ylim = plt.gca().get_ylim()
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")

plt.ylim(ylim)
#plt.gca().set_yticks([])
#plt.gca().set_xticks([])
plt.title('RJMCMC Bias')
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
#plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/AutoCorr.png')
plt.clf()

