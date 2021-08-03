from math import pi
from numpy.core.defchararray import array
from numpy.core.fromnumeric import mean
from numpy.core.function_base import linspace
import MulensModel as mm
import Functions as f
import Autocorrelation as AC
import emcee as MC
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2
import scipy
import copy


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


def PlotWalk(xi, yi, states, labels, symbols, letters, model, center, true):

    markerSize = 75
    plt.grid()
    plt.scatter(states[:, xi], states[:, yi], c = np.linspace(0.0, 1.0, len(states)), cmap = 'spring', alpha = 0.25, marker = "o")
    cbar = plt.colorbar(fraction = 0.046, pad = 0.04, ticks = [0, 1]) # empirical nice auto sizing
    ax = plt.gca()
    cbar.ax.set_yticklabels(['Initial\nStep', 'Final\nStep'], fontsize=9)
    cbar.ax.yaxis.set_label_position('right')
    plt.xlabel(labels[xi])
    plt.ylabel(labels[yi])
    plt.title('RJMCMC walk projected\n onto '+model+' ('+symbols[xi]+', '+symbols[yi]+') space')
    
    if isinstance(center, np.ndarray):
        #plt.scatter(center[xi], center[yi], marker = r'$\odot$', label = 'Centre', s = markerSize, c = 'black', alpha = 1)
        plt.legend()

    if isinstance(true, np.ndarray):
        #plt.scatter(true[xi], true[yi], marker = '*', label = 'True', s = markerSize, c = 'black', alpha = 1) # r'$\circledast$'
        plt.legend()
    

    plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
    plt.tight_layout()
    plt.savefig('Plots/Walks/RJ-'+model+letters[xi]+letters[yi]+'-Walk.png')
    plt.clf()
    
    return


def PPlotWalk(xs, ys, **kwargs):
    #Make sure to unscale centers

    markerSize = 75
    plt.scatter(xs, ys, c = np.linspace(0.0, 1.0, len(xs)), cmap = 'spring', alpha = 0.75, marker = "o")
    #cbar = plt.colorbar(fraction = 0.046, pad = 0.04, ticks = [0, 1]) # empirical nice auto sizing
    #ax = plt.gca()
    #cbar.ax.set_yticklabels(['Initial\nStep', 'Final\nStep'], fontsize=9)
    #cbar.ax.yaxis.set_label_position('right')
    #plt.xlabel(labels[xi])
    #plt.ylabel(labels[yi])
    #plt.title('RJMCMC walk\nprojected onto Binary ('+symbols[xi]+', '+symbols[yi]+') space')
    
    #if details == True:
    #    plt.scatter(center[yi], center[xi], marker = r'$\odot$', label = 'Centre', s = markerSize, c = 'black', alpha = 1)
    #    plt.scatter(true[yi], true[xi], marker = '*', label = 'True', s = markerSize, c = 'black', alpha = 1) # r'$\circledast$'
    #    plt.legend()
    
    plt.grid()
    #plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    #plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
    #plt.tight_layout()
    #plt.savefig('Plots/RJ-binary?-('+symbols[xi]+', '+symbols[yi]+')-Walk.png')
    #plt.clf()
    
    return




def TracePlot(yi, states, jumpStates, jump_i, labels, symbols, letters, model, center, true):
    #Make sure to unscale centers

    plt.grid()
    plt.plot(np.linspace(1, len(states), len(states)), states[:, yi], linewidth = 0.5)
    plt.xlabel('Binary Steps')
    plt.ylabel(labels[yi])
    plt.title('RJMCMC '+model+' model'+symbols[yi]+'Trace')

    if isinstance(jumpStates, np.ndarray):
        plt.scatter(jump_i, jumpStates[:, yi], alpha = 0.25, marker = "*", label = 'Jump')
        plt.legend()

    if isinstance(true, np.ndarray):
        plt.axhline(true[yi], label = 'True', color = 'red')
        plt.legend()

    if isinstance(center, np.ndarray):
        plt.axhline(center[yi], label = 'Centre', color = 'black')
        plt.legend()


    plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.tight_layout()
    plt.savefig('Plots/Trace/RJ-'+model+letters[yi]+'-Trace.png')
    plt.clf()

    return

#def contour():


def contourPlot(xi, yi, states, labels, symbols, letters, model, base, true, m, priors, Data, n_points):

    

    # extents
    
    yLower = np.min([np.min(states[:, yi]), base[yi]])
    yUpper = np.max([np.max(states[:, yi]), base[yi]])
    xLower = np.min([np.min(states[:, xi]), base[xi]])
    xUpper = np.max([np.max(states[:, xi]), base[xi]])

    yaxis = np.linspace(yLower, yUpper, n_points)
    xaxis = np.linspace(xLower, xUpper, n_points)
    density = np.zeros((n_points, n_points))
    x = -1
    y = -1

    for i in yaxis:
        x += 1
        y = -1
        for j in xaxis:
            y += 1
            theta = copy.deepcopy(base)

            theta[xi] = j
            theta[yi] = i

            theta = f.unscale(m, theta)
            #print(theta[4], xaxis, yaxis)
            density[x][y] = np.exp(f.logLikelihood(m, Data, theta, priors))

    density = np.sqrt(np.flip(density, 0)) # So lower bounds meet
    #density = np.flip(density, 1) # So lower bounds meet
    plt.imshow(density, interpolation='none', extent=[xLower, xUpper, yLower, yUpper,], aspect=(xUpper-xLower) / (yUpper-yLower))#, cmap = plt.cm.BuPu_r) #
    cbar = plt.colorbar(fraction = 0.046, pad = 0.04, ticks = [0, 1]) # empirical nice auto sizing
    ax = plt.gca()
    cbar.ax.set_yticklabels(['Initial\nStep', 'Final\nStep'], fontsize=9)
    cbar.ax.yaxis.set_label_position('right')


    #https://stats.stackexchange.com/questions/60011/how-to-find-the-level-curves-of-a-multivariate-normal

    mu = [np.mean(states[:, xi]), np.mean(states[:, yi])]
    K = np.cov([states[:, xi], states[:, yi]])
    angles = np.linspace(0, 2*np.pi, 360)
    R = [np.cos(angles), np.sin(angles)]
    R = np.transpose(np.array(R))

    for levels in [0.6, 0.9, 0.975]:

        rad = np.sqrt(chi2.isf(levels, 2))
        level_curve = rad*R.dot(scipy.linalg.sqrtm(K))
        plt.plot(level_curve[:, 0]+mu[0], level_curve[:, 1]+mu[1], color = 'White')


    markerSize = 75
    #plt.grid()
    #plt.scatter(states[:, xi], states[:, yi], c = np.linspace(0.0, 1.0, len(states)), cmap = 'spring', alpha = 0.25, marker = "o")
    #cbar = plt.colorbar(fraction = 0.046, pad = 0.04, ticks = [0, 1]) # empirical nice auto sizing
    #ax = plt.gca()
    #cbar.ax.set_yticklabels(['Initial\nStep', 'Final\nStep'], fontsize=9)
    #cbar.ax.yaxis.set_label_position('right')
    plt.xlabel(labels[xi])
    plt.ylabel(labels[yi])
    plt.title('RJMCMC walk projected\n onto '+model+' ('+symbols[xi]+', '+symbols[yi]+') space')
    
    #if isinstance(center, np.ndarray):
    #    plt.scatter(center[xi], center[yi], marker = r'$\odot$', label = 'Centre', s = markerSize, c = 'black', alpha = 1)
    #    plt.legend()

    if isinstance(true, np.ndarray):
        #print(true)
        plt.scatter(true[xi], true[yi], marker = '*', s = markerSize, c = 'red', alpha = 1) # r'$\circledast$'
        plt.legend()
    

    plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
    plt.tight_layout()
    plt.savefig('Plots/Walks/RJ-'+model+letters[xi]+letters[yi]+'-Density.png')
    plt.clf()

    return


def DistPlot(xi, states, labels, symbols, letters, m, model, center, true, priors, Data):
    # Make sure to unscale centers

    plt.grid()

    '''
    n_points = 100
    density = np.zeros((n_points, 1))
    points = np.linspace(np.min([np.min(states[:, xi]), true[xi]]), np.max([np.max(states[:, xi]), true[xi]]), n_points)#np.linspace(priors[xi].lb, priors[xi].rb, n_points)#
    theta = copy.deepcopy(true)
    for i in range(n_points):

        theta[xi] = points[i]
        #if xi == 4:
        #    theta[xi] = np.exp(theta[xi])
        theta = f.unscale(m, theta)
        #density[i] = np.exp(f.PriorRatio(0, 2, theta, theta, priors))*np.exp(f.logLikelihood(m, Data, theta, priors))

    #density = (density - np.min(density))/(np.max(density)-np.min(density))

    #plt.plot(points, density, label="True Density", color="Purple")
    '''
    plt.hist(states[:, xi], bins = 50, density = True)
    plt.xlabel(labels[xi])
    plt.ylabel('Probability Density')
    plt.title('RJMCMC '+model+' model ' + symbols[xi] + ' distribution')



    mu = np.average(states[:, xi])
    sd = np.std(states[:, xi])
    plt.axvline(mu, label = r'$\mu$', color = 'cyan')
    plt.axvspan(mu - sd, mu + sd, alpha = 0.25, color='cyan', label = r'$\sigma$')

    if isinstance(true, np.ndarray):
        plt.axvline(true[xi], label = 'True', color = 'red')
        plt.legend()

    if isinstance(center, np.ndarray):
        #plt.axvline(center[xi], label = 'Centre', color = 'black')
        plt.legend()

    plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))

    plt.tight_layout()
    plt.savefig('Plots/Dist/RJ-'+model+letters[xi]+'-Dist.png')
    plt.clf()

    return


def AdaptiveProgression(history, inters, covs, name):

    size = int(len(history)/25)#100#50
    
    if len(history) <= size:
        plt.scatter(0, 0)
        plt.savefig('Plots/Adaptive-RJMCMC-acceptance-progression-'+name+'.png')
        plt.clf()
        return

    
    #for l in range(2):
        #for chain in range(p):
    acc = []
    inter_acc = []
    trace = []
    bins = int(np.ceil(len(history) / size))
    for bin in range(bins - 1): # record the ratio of acceptance for each bin
        acc.append(np.sum(history[size*bin:size*(bin+1)]) / size)
        inter_acc.append(np.sum(inters[size*bin:size*(bin+1)]) / size)
            #print(chain, l, history[chain][l])
        #print(covs[:][:][size*bin:size*(bin+1)])
        #print('999999')
        trace.append(np.sum(np.trace(covs[:][:][size*bin:size*(bin+1)])) / size)



    normed_trace = (trace - np.min(trace))/(np.max(trace)-np.min(trace))


    rate_colour = 'purple'
    trace_colour = 'blue'

    a1 = plt.axes()
    a1.plot((np.linspace(0, bins - 1, num = bins - 1)), acc, c = rate_colour)

    a1.plot((np.linspace(0, len(inter_acc), num = len(inter_acc))), inter_acc, c = rate_colour, linestyle = 'dashed')
    

    a1.set_ylabel('Rate of accepted proposals')
    a1.set_ylim((0.0, 1.0))


    plt.grid()
    a1.set_xlabel(f'Binned iterations over time [{size} samples]')
    a2 = a1.twinx()

    a2.plot((np.linspace(0, bins - 1, num = bins - 1)), normed_trace, c = trace_colour)
    a2.set_ylabel(r'Average $\sum$ Tr$(K_{xx}$)')
    
    a2.set_ylim((0.0, 1.0))
    a2.set_yticks([0.0, 1.0])
    a2.set_yticklabels(['min', 'max'])

    


    a1.spines['left'].set_color(rate_colour)
    a2.spines['left'].set_color(rate_colour)

    a1.yaxis.label.set_color(rate_colour)
    a1.tick_params(axis = 'y', colors = rate_colour)

    a2.spines['right'].set_color(trace_colour)
    a1.spines['right'].set_color(trace_colour)

    a2.yaxis.label.set_color(trace_colour)
    a2.tick_params(axis = 'y', colors = trace_colour)


    plt.title('Adpt-RJMCMC '+name+' \nintra-model move timeline')
    #plt.legend()

    plt.tight_layout()
    plt.savefig('Plots/Adaptive-RJMCMC-acceptance-progression-'+name+'.png')
    plt.clf()

    return

def LightcurveFitError(m, FitTheta, priors, Data, TrueModel, t, error, details, name):

    if m == 1:
        FitModel = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], FitTheta)))
        FitModel.set_magnification_methods([0., 'point_source', 72.])

    if m == 2:
        FitModel = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], FitTheta)))
        FitModel.set_magnification_methods([0., 'VBBL', 72.])


    
    if details == True:
        t_inb = np.where(np.logical_and(0 <= t, t <= 72))
        error_inb = error[t_inb]
        plt.title('Best model ' + str(m) + r'$\chi^2$: ' + str(-2*(f.logLikelihood(m, Data, FitTheta, priors))))
        lower = TrueModel.magnification(t[t_inb]) - error_inb
        upper = TrueModel.magnification(t[t_inb]) + error_inb
        plt.fill_between(t[t_inb], lower, upper, alpha = 0.25, label = r'error $\sigma$')
        plt.scatter(t[t_inb], Data.flux, label = 'signal', color = 'grey', s=1)


    plt.xlabel('Time [days]')
    plt.ylabel('Magnification')

    #err = mpatches.Patch(label='Error', alpha=0.5)
    #plt.legend(handles=[err])


    TrueModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black', label = 'True', alpha = 1)

    FitModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red', label = 'Fit', linestyle = 'dashed', alpha=0.75)

    plt.legend()
    #plt.grid()
    plt.tight_layout()
    plt.savefig('Plots/' + name + '-Fit.png')
    plt.clf()

    return


def PlotLightcurve(m, Theta, label, color, alpha, caustics, ts):

    if m == 1:
        Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], Theta)))
        Model.set_magnification_methods([0., 'point_source', 72.])

    if m == 2:
        Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], Theta)))
        Model.set_magnification_methods([0., 'VBBL', 72.])

    if caustics:
        Model.plot_trajectory(t_start = ts[0], t_stop = ts[1], color = color, linewidth = 1, alpha = alpha, arrow_kwargs = {'width': 0.012})
        Model.plot_caustics(color = 'purple', s = 2, marker = '.')
    else:
        Model.plot_magnification(t_range = ts, subtract_2450000 = False, color = color, label = label, alpha = alpha)

    return

def Style():
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
    return