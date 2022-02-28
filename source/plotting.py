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
import matplotlib.ticker as ticker
import autocorrelation

def flux(m, theta, ts, caustics = None, label = None, color = None, alpha = None, ls = None, lw = None):
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
        if caustics > 0:
            model.plot_trajectory(t_start = ts[0], t_stop = ts[1], color = 'black', linewidth = 1, ls='-', alpha = alpha, arrow = False)
        model.plot_caustics(color = color, s = 0.75, marker = 'o', n_points = 5000, label = label)

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





def broccoli(joint_model_chain, supset_states, subset_states, surrogate_supset_states, surrogate_subset_states, symbols, ranges, curves, event_params = None, name = '', dpi = 100):
    """Plot the joint posterior surface of two models.

    Args:
        supset_model: [model] Larger parameter space model.
        subset_model: [model] Smaller parameter space model.
        joint_model_chain: [chain] Generalised chain with states from both models.
        symbols: [list] Strings to label plots with.
    """

    # Fonts/visibility.
    #lr = 45 # label rotation
    n_ticks = 5 # tick labels
    n_m_ticks = 5

    axis_title_size = 16
    axis_tick_size = 10

    # Model sizing.
    N_dim = 6 #supset_model.D
    n_dim = 3 #subset_model.D

    style()
    figure = corner.corner(supset_states.T) # Use corner for layout/sizing.
    figure.subplots_adjust(wspace = 0, hspace = 0)

    # Fonts/visibility.
    # plt.rcParams['font.size'] = 12
    # plt.rcParams['axes.titlesize'] = 20
    # plt.rcParams['axes.labelsize'] = 20

    # Extract axes.
    axes = np.array(figure.axes).reshape((N_dim, N_dim))

    #print(supset_surrogate.samples.numpy())
    single_theta, binary_theta, data = curves

    # Loop diagonal.
    for i in range(N_dim):
        ax = axes[i, i]

        ax.cla()

        nbins = 20

        if i < 3:
            ax.hist(np.concatenate((surrogate_supset_states[i, :], surrogate_subset_states[i, :])), bins = nbins, density = True, color = 'tab:orange', alpha = 1.0, histtype='step', range=ranges[i])
            ax.hist(np.concatenate((supset_states[i, :], subset_states[i, :])), bins = nbins, density = True, color = 'tab:blue', alpha = 1.0, histtype='step', range=ranges[i])
            ax.axvline(single_theta.scaled[i+1], color='tab:green', ls='-', lw=1)

        else:
            ax.hist(surrogate_supset_states[i, :], bins = nbins, density = True, color = 'tab:orange', alpha = 1.0, histtype='step', range=ranges[i])
            ax.hist(supset_states[i, :], bins = nbins, density = True, color = 'tab:blue', alpha = 1.0, histtype='step', range=ranges[i])

        if event_params is not None:             
            ax.axvline(event_params.scaled[i+1], color = 'black', ls = '-', lw = 2)
            ax.axvline(binary_theta.scaled[i+1], color='tab:purple', ls='-', lw=1)



        ax.set_xlim(ranges[i])

        ax.tick_params(which='both', top=False, direction='in')
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_m_ticks))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_m_ticks))
        ax.tick_params(which='major', length=8)
        ax.tick_params(which='minor', length=4)
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_ticks))

        if i == 0: # First diagonal tile.
            #ax.set_ylabel(symbols[i], fontsize = axis_title_size)
            #ax.tick_params(axis='y',  labelrotation = 45)
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])
            ax.set_title('(a)', loc='left', fontsize=20)

        elif i == N_dim - 1: # Last diagonal tile.
            ax.set_xlabel(symbols[i], fontsize = axis_title_size)
            ax.axes.get_yaxis().set_ticklabels([])
            ax.tick_params(axis = 'x',  labelrotation = 45, labelsize = axis_tick_size)
            plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation_mode="anchor")

        else:
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])

        ax.axes.get_yaxis().set_ticks([])

        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticks([])
        #ax.axes.get_xaxis().set_ticklabels([])
        #ax.axes.get_xaxis().set_ticks([])






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
            

            ax.scatter(event_params.scaled[xi+1], event_params.scaled[yi+1], color = 'black', alpha = 1.0, marker = "D", s = 50, linewidth = 1, zorder=9)
            #ax.axvline(event_params.scaled[xi], color = 'black', ls = '-', lw = 2)
            #ax.axhline(event_params.scaled[yi], color = 'black', ls = '-', lw = 2)
            ax.scatter(binary_theta.scaled[xi+1], binary_theta.scaled[yi+1], color = 'tab:purple', alpha = 1.0, marker = "8", s = 50, linewidth = 1, zorder=10)
            ax.set_xlim(ranges[xi])
            ax.set_ylim(ranges[yi])


            ax.tick_params(which='both', top=True, right=True, direction='in')
            ax.minorticks_on()
            ax.tick_params(which='major', length=8)
            ax.tick_params(which='minor', length=4)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_m_ticks))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_m_ticks))
            ax.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))
            ax.xaxis.set_major_locator(plt.MaxNLocator(n_ticks))

            if yi == N_dim - 1: # Bottom row.
                ax.set_xlabel(symbols[xi], fontsize = axis_title_size)
                ax.tick_params(axis = 'x',  labelrotation = 45, labelsize = axis_tick_size)
                plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation_mode="anchor")

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            if xi == 0: # First column.
                ax.set_ylabel(symbols[yi], fontsize = axis_title_size)
                ax.tick_params(axis = 'y',  labelrotation = 45, labelsize = axis_tick_size)
                plt.setp(ax.yaxis.get_majorticklabels(), va="bottom", rotation_mode="anchor")

            else:    
                ax.axes.get_yaxis().set_ticklabels([])



            # Add upper triangular plots.
            if xi < n_dim and yi < n_dim:
                
                # Acquire axes and plot.
                axs = figure.get_axes()[4].get_gridspec()
                axt = figure.add_subplot(axs[xi, yi])

                #axt.scatter(subset_states[yi, :], subset_states[xi, :], c = np.linspace(0.0, 1.0, len(subset_states[yi, :])), cmap = plt.get_cmap('RdBu'), alpha = 0.05, marker = "o", s = 25, linewidth = 0.0)
                
                sns.kdeplot(x=surrogate_subset_states[yi, :], y=surrogate_subset_states[xi, :], ax=axt, levels=[0.393, 0.865, 0.989], color='tab:orange', bw_adjust=1.2, clip=[ranges[yi], ranges[xi]])
                #sns.kdeplot(x=subset_states[yi, :], y=subset_states[xi, :], ax=axt, levels=[0.393, 0.865, 0.989], color='tab:blue', bw_adjust=1.2, clip=[ranges[yi], ranges[xi]])

                axt.scatter(x=single_theta.scaled[yi+1], y=single_theta.scaled[xi+1], color = 'tab:green', alpha = 1.0, marker = "8", s = 50, linewidth = 1, zorder=9)

                axt.set_xlim(ranges[yi])
                axt.set_ylim(ranges[xi])

                axt.tick_params(which='both', top=True, right=True, direction='in')
                axt.minorticks_on()
                axt.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_m_ticks))
                axt.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_m_ticks))
                axt.tick_params(which='major', length=8)
                axt.tick_params(which='minor', length=4)
                axt.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))
                axt.xaxis.set_major_locator(plt.MaxNLocator(n_ticks))

                if yi == n_dim - 1: # Last column.
                    axt.set_ylabel(symbols[xi], fontsize = axis_title_size)
                    axt.yaxis.tick_right()
                    axt.tick_params(which='both', bottom=True, left=True, labelsize = axis_tick_size)
                    axt.yaxis.set_label_position("right")
                    axt.tick_params(axis = 'y',  labelrotation = 45, labelsize = axis_tick_size)
                    plt.setp(axt.yaxis.get_majorticklabels(), va="top", rotation_mode="anchor")
                
                else:
                    axt.axes.get_yaxis().set_ticklabels([])
                
                if xi == 0: # First row.
                    axt.set_xlabel(symbols[yi], fontsize = axis_title_size)
                    axt.tick_params(axis = 'x',  labelrotation = 45, labelsize = axis_tick_size)
                    axt.xaxis.tick_top()
                    axt.tick_params(which='both', bottom=True, left=True, labelsize = axis_tick_size)
                    axt.xaxis.set_label_position("top")
                    plt.setp(axt.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
                
                else:
                    axt.axes.get_xaxis().set_ticklabels([])

    # Inset light curve plot.
    inset_curve = figure.add_axes([0.625, 0.745, 0.345, 0.225]) 
    inset_curve.set_ylabel('normalised flux', fontsize = 18)
    inset_curve.set_xlabel('time [days]', fontsize = 18)
    ts = [0, 72]

    flux(1, event_params, ts, label = 'truth', color = 'black', lw=2)
    flux(0, single_theta, ts, label = 'single MAP', color='tab:green', ls=':', lw=2)
    flux(1, binary_theta, ts, label = 'binary MAP', color='tab:purple', ls='--', lw=2)

    inset_curve.set_title('(b)', loc='left', fontsize=20)
    inset_curve.legend(fontsize = 16, handlelength=0.7, frameon = False, handletextpad=0.4)
    inset_curve.set_xlim([10, 20])
    #inset_curve.set_ylim([1.1, 5.9])#1
    inset_curve.set_ylim([1.1, 6.3])#4

    inset_curve.tick_params(which='both', top=True, right=True, direction='in', labelsize = 12)
    inset_curve.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    inset_curve.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    inset_curve.tick_params(which='major', length=8)
    inset_curve.tick_params(which='minor', length=4)

    # Autocorrelation time.
    inset_act = figure.add_axes([0.75, 0.475, 0.22, 0.15]) 
    inset_act.set_xlim([1000, 10000])
    #inset_act.set_ylim([10, 100])#1
    inset_act.set_ylim([1, 10])#4
    autocorrelation.plot_act(inset_act, joint_model_chain)
    inset_act.set_title('(c)', loc='left', fontsize=20)
    inset_act.tick_params(which='both', direction='in', labelsize = 12)
    inset_act.tick_params(which='major', length=10)
    inset_act.tick_params(which='minor', length=5, labelsize = 0, labelcolor = (0, 0, 0, 0))

    figure.savefig('results/' + name + '-broccoli.pdf', bbox_inches = "tight", dpi = dpi, transparent=True)
    figure.clf()

    return