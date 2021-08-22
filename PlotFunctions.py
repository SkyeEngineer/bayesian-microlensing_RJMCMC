import math
from numpy.core.defchararray import array
from numpy.core.fromnumeric import mean, ndim
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
import corner
import matplotlib as mpl

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


plt.rc('axes.formatter', useoffset=False)
'''

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


def Adaptive_Progression(history, covs, name):

    size = int(len(history)/25)#100#50
    
    if len(history) <= size or size == 0:
        plt.scatter(0, 0)
        plt.savefig('Plots/ARJMH-acc-prog-'+name+'.png')
        plt.clf()
        return

    #print(covs[:][:][1:5])

    acc = []
    trace = []
    stable_trace = []
    covs = np.array(covs)
    bins = int(np.ceil(len(history) / size))
    for bin in range(bins - 1): # record the ratio of acceptance for each bin
        acc.append(np.sum(history[size*bin:size*(bin+1)]) / size)



        trace.append(np.sum(np.trace(covs[size*bin:size*(bin+1)], 0, 2)) / size)
        stable_trace.append(np.sum(np.trace(covs[size*bin:size*(bin+1)][1:4, 1:4], 0, 2)) / size)

    #covs = np.array(covs)
    #print(covs[0][:3, :3])
    #print(covs[0][:3][:3])
    #print(covs[0])
    
    #print(np.trace(covs[0:2][:3, :3]))
    #print(np.trace(covs[0:2], 0, 2))
    #print(np.sum(np.trace(covs[0:2], 0, 2)))
    #print(np.sum(np.sum(np.trace(covs[0:2], 1, 2))))
    #print(np.trace(covs[1]))
    #print(covs[-2, -1][:, :])

    #min = np.min([np.min(trace), np.min(stable_trace)])
    #max = np.max(trace)
    normed_trace = (trace - np.min(trace))/(np.max(trace)-np.min(trace))
    normed_stable_trace = (stable_trace - np.min(stable_trace))/(np.max(stable_trace)-np.min(stable_trace))


    rate_colour = 'purple'
    trace_colour = 'blue'

    a1 = plt.axes()
    a1.plot((np.linspace(0, bins - 1, num = bins - 1)), acc, c = rate_colour)

    a1.set_ylabel('Acceptance rate per bin')
    a1.set_ylim((0.0, 1.0))


    #plt.grid()
    a1.set_xlabel(f'Binned iterations over time')
    a2 = a1.twinx()

    a2.plot((np.linspace(0, bins - 1, num = bins - 1)), normed_trace, c = trace_colour)
    a2.plot((np.linspace(0, bins - 1, num = bins - 1)), normed_stable_trace, c = trace_colour, linestyle = 'dashed')
    a2.set_ylabel(r'Average $Tr(K_{xx})$ per bin')
    
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


    #plt.title('Adpt-RJMCMC '+name+' \nintra-model move timeline')

    plt.tight_layout()
    plt.savefig('Plots/ARJMH-acc-prog-'+name+'.png', bbox_inches="tight")
    plt.clf()

    return


def Light_Curve_Fit_Error(m, FitTheta, priors, Data, TrueModel, t, error, details, name):

    if m == 1:
        FitModel = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho'], FitTheta)))
        FitModel.set_magnification_methods([0., 'finite_source_uniform_Gould94', 72.])

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
    print('add flux to plot')

    plt.legend()
    #plt.grid()
    plt.tight_layout()
    plt.savefig('Plots/' + name + '-Fit.png', bbox_inches="tight")
    plt.clf()

    return


def Draw_Light_Curve_Noise_Error(data, ax):

    ax.axis('on')
    
    error = data.err_flux
    lower = data.flux - error
    upper = data.flux + error
    ax.scatter(data.time, data.flux, label = r'$\gamma$', color = 'black', s = 2)
    ax.fill_between(data.time, lower, upper, alpha = 0.5, label = r'$\pm\sigma$')
    


    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Magnification')
    #ax.xticks(ax.xticks())
    #ax.yticks(ax.yticks())



    ax.legend(fontsize = 14, handlelength=0.7)
    #plt.grid()
    #ax.tight_layout()

    return


def PlotLightcurve(m, theta, label, color, alpha, caustics, ts):

    if m == 0:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho'], theta[1:])))
        model.set_magnification_methods([0., 'finite_source_uniform_Gould94', 72.])

    if m == 1:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta[1:])))
        model.set_magnification_methods([0., 'VBBL', 72.])

    epochs = np.linspace(ts[0], ts[1], 720)

    if caustics:
        model.plot_trajectory(t_start = ts[0], t_stop = ts[1], color = color, linewidth = 1, alpha = alpha, arrow_kwargs = {'width': 0.012})
        model.plot_caustics(color = 'purple', s = 2, marker = '.')

    elif isinstance(label, str):
        
        A = (model.magnification(epochs) - 1.0) * theta[0] + 1.0
        plt.plot(epochs, A, color = color, label = label, alpha = alpha)

    else:

        A = (model.magnification(epochs) - 1.0) * theta[0] + 1.0
        plt.plot(epochs, A, color = color, alpha = alpha)

    return

def Style():
    plt.rcParams["font.family"] = "sans-serif"
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




def Contour_Plot(n_dim, n_points, states, covariance, true, center, m, priors, data, symbols, name, P):

    figure = corner.corner(states)

    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14

    # extract the axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))

    # params to evaluate 2d slices at
    if isinstance(true, np.ndarray):
        base = true
    else:
        base = center

    # Loop over the diagonal
    for i in range(n_dim):
        ax = axes[i, i]

        ax.cla()
        #ax.grid()

        # distribution plots
        ax.hist(states[:, i], bins = 50, density = True)

        mu = np.average(states[:, i])
        sd = np.std(states[:, i])
        ax.axvline(mu, label = r'$\mu$', color = 'black')
        ax.axvspan(mu - sd, mu + sd, alpha = 0.1, color = 'cyan', label = r'$\sigma$')

        if isinstance(true, np.ndarray):
            ax.axvline(base[i], label = r'$\theta$', color = 'red')

        #ax.xaxis.tick_top()
        #ax.yaxis.tick_right()
        ax.set_title(r'$\bar{\mu} = $'+f'{mu:.4}'+',\n'+r'$\bar{\sigma} = \pm$'+f'{sd:.4}')

        if i == 0: 
            ax.set_ylabel(symbols[i])
            #ax.tick_params(axis='y', labelrotation = 45)
            ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.get_xaxis().set_ticklabels([])
        elif i == n_dim - 1:
            ax.set_xlabel(symbols[i])
            ax.tick_params(axis='x', labelrotation = 45)
            ax.axes.get_yaxis().set_ticklabels([])
        else:
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])


        xLower = np.min([np.min(states[:, i]), base[i]])
        xUpper = np.max([np.max(states[:, i]), base[i]])
        xWidth = xUpper - xLower
        xUpper += xWidth/2
        xLower -= xWidth/2

        # limits within prior bounds
        if xUpper > priors[i].rb and i != 5:
            xUpper = priors[i].rb
        if xLower < priors[i].lb and i != 5:
            xLower = priors[i].lb

        if i == 5:
            xLower = np.log(10e-6)

        ax.set_xlim((xLower, xUpper))



    # loop over lower triangular 
    for yi in range(n_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            
            # posterior heat map
            # work out limits and double them
            yLower = np.min([np.min(states[:, yi]), base[yi]])
            yUpper = np.max([np.max(states[:, yi]), base[yi]])
            yWidth = yUpper - yLower
            yUpper += yWidth/2
            yLower -= yWidth/2

            # limits within prior bounds
            if yUpper > priors[yi].rb and yi != 5:
                yUpper = priors[yi].rb
            if yLower < priors[yi].lb and yi != 5:
                yLower = priors[yi].lb
            if yi == 5:
                yLower = np.log(10e-6)

            xLower = np.min([np.min(states[:, xi]), base[xi]])
            xUpper = np.max([np.max(states[:, xi]), base[xi]])
            xWidth = xUpper - xLower
            xUpper += xWidth/2
            xLower -= xWidth/2

            # limits within prior bounds
            if xUpper > priors[xi].rb and xi != 5:
                xUpper = priors[xi].rb
            if xLower < priors[xi].lb and xi != 5:
                xLower = priors[xi].lb

            if xi == 5:
                xLower = np.log(10e-6)

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

                    density[x][y] = np.exp(f.Log_Likelihood(m, theta, priors, data) + f.Log_Prior_Product(m, theta, priors))

            density = np.sqrt(np.flip(density, 0)) # so lower bounds meet. sqrt to get better definition between high vs low posterior 
            ax.imshow(density, interpolation = 'none', extent=[xLower, xUpper, yLower, yUpper], aspect = (xUpper-xLower) / (yUpper-yLower))


            # the fit normal distribution's contours
            # https://stats.stackexchange.com/questions/60011/how-to-find-the-level-curves-of-a-multivariate-normal

            mu = [np.mean(states[:, xi]), np.mean(states[:, yi])]
            row = np.array([xi, yi])
            col = np.array([xi, yi])
            K = covariance[row[:, np.newaxis], col]  #np.cov([states[:, xi], states[:, yi]])
            angles = np.linspace(0, 2*math.pi, 360)
            R = [np.cos(angles), np.sin(angles)]
            R = np.transpose(np.array(R))

            ylim = ax.get_ylim()
            xlim = ax.get_xlim()

            for level in [1 - 0.989, 1 - 0.865, 1 - 0.393]: # sigma levels

                rad = np.sqrt(chi2.isf(level, 2))
                level_curve = rad * R.dot(scipy.linalg.sqrtm(K))
                ax.plot(level_curve[:, 0] + mu[0], level_curve[:, 1] + mu[1], color = 'white')

            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

            # plot true values in scaled space if they exist
            if isinstance(true, np.ndarray):
                ax.scatter(base[xi], base[yi], marker = 'o', s = 75, c = 'red', alpha = 1)
                ax.axvline(base[xi], color = 'red')
                ax.axhline(base[yi], color = 'red')

            # labels if on edge
            if yi == n_dim - 1:
                ax.set_xlabel(symbols[xi])
                ax.tick_params(axis='x', labelrotation = 45)

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            # labels if on edge
            if xi == 0:
                ax.set_ylabel(symbols[yi])
                ax.tick_params(axis = 'y', labelrotation = 45)

            else:    
                ax.axes.get_yaxis().set_ticklabels([])


    # inset lightcurve
    axs = figure.get_axes()[4].get_gridspec()
    half = math.floor(n_dim/2)
    ax = figure.add_subplot(axs[:half, n_dim-half:n_dim])
    Draw_Light_Curve_Noise_Error(data, ax)
    #ax.tick_params(axis="x", direction="in", pad=-22)
    #ax.tick_params(axis="y", direction="in", pad=-22)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    # classification prob text
    ax = axes[0, 3]
    ax.text(0, 0, 'Classification:\n'+r'$\mathbb{P}(m=$'+str(m+1)+r'|$\gamma$)='+f'{P:.2f}', size=20, va='center', ha='center')

    # fake cbar
    #ax = axes[4, 6]
    #col_map = plt.get_cmap('viridis')
    #cbar = mpl.colorbar.ColorbarBase(ax, cmap=col_map, orientation = 'vertical')
    #cbar.set_ticks([0, 1])
    #cbar.set_ticklabels([r'$Min \pi$', r'$Max \pi$'])

    #cbar = figure.colorbar(fraction = 0.046, pad = 0.04, ticks = [0, 1]) # empirical nice auto sizing
    #cbar.ax.set_yticklabels([r'$Min \pi$', r'$Max \pi$'], fontsize=9)
    #cbar.ax.yaxis.set_label_position('')
    #ax.()

    figure.savefig('results/'+name+'.png', dpi=500, bbox_inches="tight") #tight layout destroys spacing
    figure.clf()

    return









def Double_Plot(ndim, states_1, states_2, symbols, name):


    # construct shape with corner
    figure = corner.corner(states_1)

    #ndim = f.D(0)

    # extract the axes
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]

        ax.cla()
        #ax.grid()
        #ax.plot(np.linspace(1, len(states_2), len(states_2)), states_2[:, i], linewidth = 0.5)

        ax.patch.set_alpha(0.0)
        ax.axis('off')
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        
    # loop over lower triangle
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            #ax.grid()
            ax.scatter(states_2[:, xi], states_2[:, yi], c = np.linspace(0.0, 1.0, len(states_2)), cmap = 'winter', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)
            ax.scatter(states_1[:, xi], states_1[:, yi], c = np.linspace(0.0, 1.0, len(states_1)), cmap = 'spring', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)
                
            if yi == ndim - 1:
                ax.set_xlabel(symbols[xi])
                ax.tick_params(axis = 'x', labelrotation = 45)

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            if xi == 0:
                ax.set_ylabel(symbols[yi])
                ax.tick_params(axis = 'y', labelrotation = 45)

            else:    
                ax.axes.get_yaxis().set_ticklabels([])

    #plt.tight_layout()
    figure.savefig('Plots/' + name + '.png', dpi = 500, bbox_inches="tight")
    figure.clf()

    return








def Walk_Plot(n_dim, single_states, binary_states, signals, data, symbols, name, sampled_params):


    

    figure = corner.corner(binary_states)

    ls = 20
    lr = 45
    n_lb = 3
    plt.rcParams['font.size'] = 12
    #plt.rcParams['axes.titlesize'] = 1
    plt.rcParams['axes.labelsize'] = 20

    # extract the axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))

    # loop diagonal
    for i in range(n_dim):
        ax = axes[i, i]

        ax.cla()
        #ax.grid()
        #ax.plot(np.linspace(1, len(binary_states), len(states)), states[:, i], linewidth = 0.25)
        ax.plot(np.linspace(1, len(signals[:, i]), len(signals[:, 1])), signals[:, i], linewidth = 0.25, color='black')
        


        if i == 0: 
            ax.set_ylabel(symbols[i])
            ax.yaxis.label.set_size(ls)
            ax.tick_params(axis='y', labelrotation = lr)
            #ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.get_xaxis().set_ticklabels([])
        elif i == n_dim - 1:
            ax.set_xlabel(symbols[i])
            ax.xaxis.label.set_size(ls)
            #ax.tick_params(axis='x', labelrotation = 45)
            ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.get_xaxis().set_ticklabels([])
        else:
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])

        ax.locator_params(nbins = n_lb) # 3 ticks max
        #ax.xaxis.tick_top()
        #ax.yaxis.tick_right()
        #ax.set_title(symbols[i])
        
    # loop lower triangular
    for yi in range(n_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            #ax.grid()
            ax.scatter(binary_states[:, xi], binary_states[:, yi], c = np.linspace(0.0, 1.0, len(binary_states)), cmap = 'spring', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)
                
            if yi == n_dim - 1:
                ax.set_xlabel(symbols[xi])
                ax.xaxis.label.set_size(ls)
                ax.tick_params(axis = 'x', labelrotation = lr)

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            if xi == 0:
                ax.set_ylabel(symbols[yi])
                ax.yaxis.label.set_size(ls)
                ax.tick_params(axis = 'y', labelrotation = lr)

            else:    
                ax.axes.get_yaxis().set_ticklabels([])

            ax.locator_params(nbins = n_lb) # 3 ticks max

            # add upper triangular plots
            if xi < f.D(0) and yi < f.D(0):
                axs = figure.get_axes()[4].get_gridspec()
                axt = figure.add_subplot(axs[xi, yi])
                axt.scatter(single_states[:, yi], single_states[:, xi], c = np.linspace(0.0, 1.0, len(single_states)), cmap = 'winter', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)

                

                #axt.xaxis.tick_top()
                #axt.xaxis.set_label_position("top")


                
                if yi == f.D(0)-1:
                    axt.set_ylabel(symbols[xi])
                    ax.yaxis.label.set_size(ls)
                    axt.yaxis.tick_right()
                    axt.yaxis.set_label_position("right")
                    axt.tick_params(axis = 'y', labelrotation = lr)
                else:
                    axt.axes.get_yaxis().set_ticklabels([])
                
                if xi == 0:
                    axt.set_xlabel(symbols[yi])
                    axt.tick_params(axis = 'x', labelrotation = lr)
                    ax.xaxis.label.set_size(ls)
                    axt.xaxis.tick_top()
                    axt.xaxis.set_label_position("top") 
                else:
                    axt.axes.get_xaxis().set_ticklabels([])

                axt.locator_params(nbins = n_lb) # 3 ticks max

    # inset plot of data
    #figure.axes([0.125, 0.7, 0.3, 0.2])
    #Draw_Light_Curve_Noise_Error(data)
    #ax = plt.gca()

    axs = figure.get_axes()[4].get_gridspec()
    #half = math.floor(n_dim/2)
    ax = figure.add_subplot(axs[:2, n_dim-2:n_dim])
    Draw_Light_Curve_Noise_Error(data, ax)
    #ax.tick_params(axis = "y", direction = "in", pad = -25)
    #ax.tick_params(axis = "x", direction = "in", pad = -15)
    #plt.axes(ax)
    #plt.scatter(data.time, data.flux, label = r'$\gamma$', color = 'black', s = 3)
    '''
    for params in sampled_params:
        #pltf.PlotLightcurve(chain_ms[i], f.unscale(np.array(chain_states[i])), False, 'red', 0.01, False, [0,72])
        #sampled_params[i].append([chain_ms[i], f.unscale(np.array(chain_states[i]))])
        
        m = params[0]
        theta = params[1:]

        if m == 0:
            model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta)))
            model.set_magnification_methods([0., 'point_source', 72.])

        if m == 1:
            model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta)))
            model.set_magnification_methods([0., 'VBBL', 72.])

    #    model.plot_magnification(t_range = [0, 72], color = 'red', alpha = 0.25)
    

    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='red', label=r'$\theta, N=1$', alpha = 0.25)   
    #line = Line2D([0], [0], label='manual line', color='k')
    handles.extend([patch])
    #plt.legend(handles=handles, fontsize = 10)


    #red_patch = mpatches.Patch()
    #plt.legend(handles=[red_patch])



    #ax.fill_between(data.time, lower, upper, alpha = 0.5, label = r'$\pm\sigma$')
    #ax.scatter(data.time, data.flux, label = r'$\gamma$', color = 'black', s = 3)

    #ax.tick_params(axis="y", direction="in", pad=-22)
    #ax.tick_params(axis="x", direction="in", pad=-22)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Magnification')
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position("top")
    #ax.yaxis.tick_right()
    #ax.yaxis.set_label_position("right")
    #plt.legend()
    '''


        

    #plt.tight_layout()
    figure.savefig('results/' + name + '.png', dpi = 500, bbox_inches="tight")
    figure.clf()

    return