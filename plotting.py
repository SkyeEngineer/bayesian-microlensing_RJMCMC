"""Plot posterior distributions of microlensing events.

Creates "pointilism" plots which show the discrete posterior,
and heatmap plots which show the true posterior.
"""

from cProfile import label
import MulensModel as mm

import math
from copy import deepcopy
import sampling
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chi2
import corner
import light_curve_simulation
import seaborn as sns



def magnification(m, theta, ts, caustics = None, label = None, color = None, alpha = None, ls = None, lw = None):
    """Plot the magnification produced by a microlensing event.

    Args:
        m: [int] Model index.
        theta: [state] Model parameters.
        ts: [list] Range to plot over.
        caustics: [optional, bool] Whether the plot should be of the caustic curves.
    """
    if m == 0:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta.truth[1:])))
    if m == 1:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'q', 's', 'alpha'], theta.truth[1:])))
    model.set_magnification_methods([ts[0], 'point_source', ts[1]])

    epochs = np.linspace(ts[0], ts[1], 720)

    if caustics is not None:
        model.plot_trajectory(t_start = ts[0], t_stop = ts[1], color = color, linewidth = 1, alpha = alpha, arrow_kwargs = {'width': caustics})
        model.plot_caustics(color = color, s = 1, marker = 'o', n_points = 10000)

    else:
        A = (model.magnification(epochs)-1)*theta.truth[0]+1
        plt.plot(epochs, A, color = color, label = label, alpha = alpha, ls = ls, lw = lw)

    return



def fitted_flux(m, theta, data, ts, label = None, color = None, alpha = None, ls = None, lw = None):
    """Plot the flux produced by fitted microlensing parameters.

    Args:
        m: [int] Model index.
        theta: [state] Model parameters.
        ts: [list] Range to plot over.
    """
    if m == 0:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta.truth[1:])))
    if m == 1:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'q', 's', 'alpha'], theta.truth[1:])))
    model.set_magnification_methods([ts[0], 'point_source', ts[1]])

    epochs = np.linspace(ts[0], ts[1], 720)

    a = model.magnification(epochs)
    # Fit proposed flux as least squares solution.
    #F = light_curve_simulation.least_squares_signal(a, data.flux)
    F = (a-1)*theta.truth[0]+1
    plt.plot(epochs, F, color = color, label = label, alpha = alpha, ls = ls, lw = lw)

    return


def style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['font.size'] = 15 #12
    plt.style.use('seaborn-bright')
    plt.rcParams["legend.edgecolor"] = '0'
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.title_fontsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["grid.linestyle"] = 'dashed' 
    plt.rcParams["grid.alpha"] = 0.25
    plt.rc('axes.formatter', useoffset=False)
    return





def broccoli(supset_states, subset_states, surrogate_supset_states, surrogate_subset_states, symbols, ranges, curves, event_params = None, name = '', dpi = 100):
    """Plot the joint posterior surface of two models.

    Args:
        supset_model: [model] Larger parameter space model.
        subset_model: [model] Smaller parameter space model.
        joint_model_chain: [chain] Generalised chain with states from both models.
        symbols: [list] Strings to label plots with.
    """

    # Fonts/visibility.
    label_size = 20
    #lr = 45 # label rotation
    n_ticks = 4 # tick labels

    # Model sizing.
    N_dim = 7#supset_model.D
    n_dim = 4#subset_model.D

    style()
    figure = corner.corner(supset_states.T) # Use corner for layout/sizing.

    # Fonts/visibility.
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.locator_params(axis='both', nbins=n_ticks)
    # Extract axes.
    axes = np.array(figure.axes).reshape((N_dim, N_dim))

    #print(supset_surrogate.samples.numpy())
    single_theta, binary_theta, data = curves

    # Loop diagonal.
    for i in range(N_dim):
        ax = axes[i, i]

        ax.cla()

        if i<3:
            ax.hist(np.concatenate((surrogate_supset_states[i, :], surrogate_subset_states[i, :])), bins = 10, density = True, color = 'tab:orange', alpha = 1.0, histtype='step', range=ranges[i])
            ax.hist(np.concatenate((supset_states[i, :], subset_states[i, :])), bins = 10, density = True, color = 'tab:blue', alpha = 1.0, histtype='step', range=ranges[i])
        
        else:
            ax.hist(surrogate_supset_states[i, :], bins = 10, density = True, color = 'tab:orange', alpha = 1.0, histtype='step', range=ranges[i])
            ax.hist(supset_states[i, :], bins = 10, density = True, color = 'tab:blue', alpha = 1.0, histtype='step', range=ranges[i])

        if event_params is not None:
            xlim = ax.get_xlim()                
            ax.axvline(event_params.scaled[i], color = 'black', ls = '-', lw = 1)
            if i<4:
                ax.axvline(single_theta.scaled[i], color = 'tab:green', ls = ':', lw = 1)
            ax.axvline(binary_theta.scaled[i], color = 'tab:purple', ls = '--', lw = 1)
            ax.set_xlim(xlim)

        if i == 0: # First diagonal tile.
            ax.set_ylabel(symbols[i])
            ax.yaxis.label.set_size(label_size)
            ax.tick_params(axis='y',  labelrotation = 45)
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])
            ax.set_title('a)', loc='left')

        elif i == N_dim - 1: # Last diagonal tile.
            ax.set_xlabel(symbols[i])
            ax.xaxis.label.set_size(label_size)
            ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.get_xaxis().set_ticklabels([])

        else:
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])

        #ax.locator_params(axis='both', nbins=n_ticks)

    # Loop lower triangular.
    for yi in range(N_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            
            #if yi<3 and xi<3:
            #    ax.scatter(subset_states[xi, :], subset_states[yi, :], c = 'white', alpha = 0.0, marker = ".", s = 75, linewidth = 0.0)
            #ax.scatter(supset_states[xi, :], supset_states[yi, :], c = np.linspace(0.0, 1.0, len(supset_states[yi, :])), cmap = plt.get_cmap('RdBu'), alpha = 0.05, marker = "o", s = 25, linewidth = 0.0)

            #xlim = ax.get_xlim()
            #ylim = ax.get_ylim()
            sns.kdeplot(x=surrogate_supset_states[xi, :], y=surrogate_supset_states[yi, :], ax=ax, levels=[0.393, 0.865, 0.989], color='tab:orange', bw_adjust=1.2, clip=[ranges[xi], ranges[yi]])
            sns.kdeplot(x=supset_states[xi, :], y=supset_states[yi, :], ax=ax, levels=[0.393, 0.865, 0.989], color='tab:blue', bw_adjust=1.2, clip=[ranges[xi], ranges[yi]])
            

            ax.scatter(event_params.scaled[xi], event_params.scaled[yi], color = 'black', alpha = 1.0, marker = "o", s = 50, linewidth = 0.0, zorder=9)
            ax.axvline(event_params.scaled[xi], color = 'black', ls = '-', lw = 1)
            ax.axhline(event_params.scaled[yi], color = 'black', ls = '-', lw = 1)
            ax.scatter(binary_theta.scaled[xi], binary_theta.scaled[yi], color = 'tab:purple', alpha = 1.0, marker = "o", s = 25, linewidth = 0.0, zorder=10)
            ax.set_xlim(ranges[xi])
            ax.set_ylim(ranges[yi])


            if yi == N_dim - 1: # Bottom row.
                ax.set_xlabel(symbols[xi])
                ax.xaxis.label.set_size(label_size)
                ax.tick_params(axis = 'x',  labelrotation = 45)
                plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation_mode="anchor")

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            if xi == 0: # First column.
                ax.set_ylabel(symbols[yi])
                ax.yaxis.label.set_size(label_size)
                ax.tick_params(axis = 'y',  labelrotation = 45)
                plt.setp(ax.yaxis.get_majorticklabels(), va="bottom", rotation_mode="anchor")

            else:    
                ax.axes.get_yaxis().set_ticklabels([])

            #ax.locator_params(nbins = n_lb) # 3 ticks max
            #ax.locator_params(axis='both', nbins=n_ticks)

            # Add upper triangular plots.
            if xi < n_dim and yi < n_dim:
                
                # Acquire axes and plot.
                axs = figure.get_axes()[4].get_gridspec()
                axt = figure.add_subplot(axs[xi, yi])

                #axt.scatter(subset_states[yi, :], subset_states[xi, :], c = np.linspace(0.0, 1.0, len(subset_states[yi, :])), cmap = plt.get_cmap('RdBu'), alpha = 0.05, marker = "o", s = 25, linewidth = 0.0)
                sns.kdeplot(x=surrogate_subset_states[yi, :], y=surrogate_subset_states[xi, :], ax=axt, levels=[0.393, 0.865, 0.989], color='tab:orange', bw_adjust=1.2, clip=[ranges[yi], ranges[xi]])
                sns.kdeplot(x=subset_states[yi, :], y=subset_states[xi, :], ax=axt, levels=[0.393, 0.865, 0.989], color='tab:blue', bw_adjust=1.2, clip=[ranges[yi], ranges[xi]])

                ax.scatter(single_theta.scaled[yi], single_theta.scaled[xi], color = 'tab:green', alpha = 1.0, marker = "o", s = 25, linewidth = 0.0, zorder=10)

                axt.set_xlim(ranges[yi])
                axt.set_ylim(ranges[xi])


                if yi == n_dim - 1: # Last column.
                    axt.set_ylabel(symbols[xi])
                    #ax.yaxis.label.set_size(label_size)
                    axt.yaxis.tick_right()
                    axt.yaxis.set_label_position("right")
                    axt.tick_params(axis = 'y',  labelrotation = 45)
                    plt.setp(axt.yaxis.get_majorticklabels(), va="top", rotation_mode="anchor")
                
                else:
                    axt.axes.get_yaxis().set_ticklabels([])
                
                if xi == 0: # First row.
                    axt.set_xlabel(symbols[yi])
                    axt.tick_params(axis = 'x',  labelrotation = 45)
                    #ax.xaxis.label.set_size(label_size)
                    axt.xaxis.tick_top()
                    axt.xaxis.set_label_position("top")
                    plt.setp(axt.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
                
                else:
                    axt.axes.get_xaxis().set_ticklabels([])

                #axt.locator_params(axis='both', nbins=n_ticks)
                #axt.locator_params(nbins = n_lb) # 3 ticks max
    #plt.locator_params(axis='both', nbins=n_ticks)

    # Inset light curve plot. 
    axes = figure.get_axes()[4].get_gridspec()
    inset_ax = figure.add_subplot(axes[:2, N_dim-2:])
    inset_ax.set_ylabel('flux')
    inset_ax.set_xlabel('time [days]')
    ts = [0, 72]
    #epochs = np.linspace(0, 72, 720)
    #lower = data.flux - 3*data.err_flux
    #upper = data.flux + 3*data.err_flux
    #inset_ax.fill_between(epochs, lower, upper, alpha = 1.0, label = r'$F \pm 3\sigma$', color = 'black', linewidth=0.0)

    #fitted_params = sampling.State(scaled = fit_mu)

    magnification(1, event_params, ts, label = 'truth', color = 'black', lw=1)
    fitted_flux(1, binary_theta, data, ts, label = 'binary fit', color='tab:purple', ls='--')
    fitted_flux(0, single_theta, data, ts, label = 'single fit', color='tab:green', ls=':')
    inset_ax.set_title('b)', loc='left')
    inset_ax.legend(fontsize = 13, handlelength=0.7, frameon = False)
    inset_ax.set_xlim([10, 20])

    figure.savefig('results/' + name + '-broccoli.png', bbox_inches = "tight", dpi = dpi, transparent=True)
    figure.clf()

    return