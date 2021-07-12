import MulensModel as mm
import Functions as f
import Autocorrelation as AC
import emcee as MC
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def PlotWalk(xi, yi, states, center, true, labels, symbols, details):
    #Make sure to unscale centers

    markerSize = 75
    plt.grid()
    plt.scatter(states[:, xi], states[:, yi], c = np.linspace(0.0, 1.0, len(states)), cmap = 'spring', alpha = 0.25, marker = "o")
    cbar = plt.colorbar(fraction = 0.046, pad = 0.04, ticks = [0, 1]) # empirical nice auto sizing
    ax = plt.gca()
    cbar.ax.set_yticklabels(['Initial\nStep', 'Final\nStep'], fontsize=9)
    cbar.ax.yaxis.set_label_position('right')
    plt.xlabel(labels[xi])
    plt.ylabel(labels[yi])
    plt.title('RJMCMC walk\nprojected onto Binary ('+symbols[xi]+', '+symbols[yi]+') space')
    
    if details == True:
        plt.scatter(center[xi], center[yi], marker = r'$\odot$', label = 'Centre', s = markerSize, c = 'black', alpha = 1)
        plt.scatter(true[xi], true[yi], marker = '*', label = 'True', s = markerSize, c = 'black', alpha = 1) # r'$\circledast$'
        plt.legend()
    

    plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
    plt.tight_layout()
    plt.savefig('Plots/RJ-binary?-('+symbols[xi]+', '+symbols[yi]+')-Walk.png')
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




def TracePlot(yi, states, jumpStates, jump_i, center, true, labels, symbols, details):
    #Make sure to unscale centers

    plt.grid()
    plt.plot(np.linspace(1, len(states), len(states)), states[:, yi], linewidth = 0.5)
    plt.xlabel('Binary Steps')
    plt.ylabel(labels[yi])
    plt.title('RJMCMC Binary model'+symbols[yi]+'Trace')

    if details == True:
        plt.scatter(jump_i, jumpStates[:, yi], alpha = 0.25, marker = "*", label = 'Jump')
        plt.axhline(true[yi], label = 'True', color = 'red')
        plt.axhline(center[yi], label = 'Centre', color = 'black')
        plt.legend()


    #plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.tight_layout()
    plt.savefig('Plots/RJ-Binary-' + symbols[yi] + '-Trace.png')
    plt.clf()

    return




def DistPlot(m, xi, states, center, true, labels, symbols, details):
    # Make sure to unscale centers

    plt.grid()
    plt.hist(states[:, xi], bins = 25, density = True)
    plt.xlabel(labels[xi])
    plt.ylabel('Probability Density')
    plt.title('RJMCMC Binary model ' + symbols[xi] + ' distribution')

    if details == True:
        plt.axvline(true[xi], label = 'True', color = 'red')
        plt.axvline(center[xi], label = 'Centre', color = 'black')
        plt.legend()

    plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))

    plt.tight_layout()
    plt.savefig('Plots/RJ-Binary-' + symbols[xi] + '-dist')
    plt.clf()

    return




def LightcurveFitError(m, FitTheta, priors, Data, TrueModel, t, error):

    if m == 1:
        FitModel = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], FitTheta)))
        FitModel.set_magnification_methods([0., 'point_source', 72.])

    if m == 2:
        FitModel = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], FitTheta)))
        FitModel.set_magnification_methods([0., 'VBBL', 72.])

    t_inb = np.where(np.logical_and(0 <= t, t <= 72))
    error_inb = error[t_inb]
    

    plt.title('Best model '+str(m)+' fit: '+str(np.exp(f.logLikelihood(m, Data, FitTheta, priors))))
    plt.xlabel('Time [days]')

    #err = mpatches.Patch(label='Error', alpha=0.5)
    #plt.legend(handles=[err])
    lower = TrueModel.magnification(t[t_inb]) - error_inb / 2
    upper = TrueModel.magnification(t[t_inb]) + error_inb / 2
    plt.fill_between(t[t_inb], lower, upper, alpha = 0.25, label = 'Error')

    TrueModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black', label = 'True')

    FitModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red', label = 'Fit', linestyle = 'dashed')

    plt.legend()
    #plt.grid()
    plt.tight_layout()
    plt.savefig('Plots/' + str(m) + '-Fit.png')
    plt.clf()

    return