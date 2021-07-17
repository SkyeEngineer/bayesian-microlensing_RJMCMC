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
        plt.scatter(center[xi], center[yi], marker = r'$\odot$', label = 'Centre', s = markerSize, c = 'black', alpha = 1)
        plt.legend()

    if isinstance(true, np.ndarray):
        plt.scatter(true[xi], true[yi], marker = '*', label = 'True', s = markerSize, c = 'black', alpha = 1) # r'$\circledast$'
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




def DistPlot(xi, states, labels, symbols, letters, model, center, true):
    # Make sure to unscale centers

    plt.grid()
    plt.hist(states[:, xi], bins = 25, density = True)
    plt.xlabel(labels[xi])
    plt.ylabel('Probability Density')
    plt.title('RJMCMC '+model+' model ' + symbols[xi] + ' distribution')

    if isinstance(true, np.ndarray):
        plt.axvline(true[xi], label = 'True', color = 'red')
        plt.legend()

    if isinstance(center, np.ndarray):
        plt.axvline(center[xi], label = 'Centre', color = 'black')
        plt.legend()

    plt.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))

    plt.tight_layout()
    plt.savefig('Plots/Dist/RJ-'+model+letters[xi]+'-Dist.png')
    plt.clf()

    return


def AdaptiveProgression(history, labels, p):

    size = 50#50
    
    
    for l in range(2):
        for chain in range(p):
            acc = []
            bins = int(np.ceil(len(history[chain][l]) / size))
            for bin in range(bins - 1): # record the ratio of acceptance for each bin
                acc.append(np.sum(history[chain][l][size*bin:size*(bin+1)]) / size)
            #print(chain, l, history[chain][l])

            plt.plot((np.linspace(1, bins - 1, num = bins - 1)), acc, label = labels[l] + str(chain), alpha = 0.5)

    plt.ylim((0.0, 1.0))
    plt.xlabel(f'Binned Iterations Over Time [Bins, n={size}]')
    plt.ylabel('Fraction of Accepted\nProposals '+r'[$Bins^{-1}$]')
    plt.title('Adaptive RJMCMC acceptance timeline')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('Plots/Adaptive-RJMCMC-acceptance-progression.png')
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
        plt.title('Best model '+str(m)+' fit: '+str(np.exp(f.logLikelihood(m, Data, FitTheta, priors))))
        lower = TrueModel.magnification(t[t_inb]) - error_inb / 2
        upper = TrueModel.magnification(t[t_inb]) + error_inb / 2
        plt.fill_between(t[t_inb], lower, upper, alpha = 0.25, label = 'Error')


    plt.xlabel('Time [days]')
    plt.ylabel('Magnification')

    #err = mpatches.Patch(label='Error', alpha=0.5)
    #plt.legend(handles=[err])


    TrueModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black', label = 'True')

    FitModel.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='red', label = 'Fit', linestyle = 'solid')

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