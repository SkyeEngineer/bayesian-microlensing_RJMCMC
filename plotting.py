"""Plotting tools for microlensing distribution sampling."""


import math
from numpy.core.defchararray import array
from numpy.core.fromnumeric import mean, ndim
from numpy.core.function_base import linspace
from copy import deepcopy
import MulensModel as mm
import sampling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2
import scipy
import corner
import matplotlib as mpl
import light_curve_simulation


def adaption_contraction(model, iterations, name = '', dpi = 100):

    acc = model.acc
    covs = np.array(model.covariances)
    N = model.sampled.n

    size = int(N/25)
    
    if N <= size or size == 0:
        plt.scatter(0, 0)
        plt.savefig('results/'+name+'-acc-trace-prog.png')
        plt.clf()
        return

    acc_rate = []
    trace = []
    stable_trace = []
    bins = int(np.ceil(N/size))

    for bin in range(bins - 1): # record the average rate and trace for each bin

        acc_rate.append(np.sum(acc[size*bin:size*(bin+1)]) / size)
        trace.append(np.sum(np.trace(covs[size*bin:size*(bin+1)], 0, 2)) / size)
        stable_trace.append(np.sum(np.trace(covs[size*bin:size*(bin+1)][:3, :3], 0, 2)) / size)

    normed_trace = (trace - np.min(trace))/(np.max(trace)-np.min(trace))
    normed_stable_trace = (stable_trace - np.min(stable_trace))/(np.max(stable_trace)-np.min(stable_trace))

    rate_colour = 'purple'
    trace_colour = 'blue'

    a1 = plt.axes()
    a1.plot(int(iterations/(bins-1)) * (np.linspace(0, bins - 1, num = bins - 1)), acc_rate, c = rate_colour)

    a1.set_ylabel(r'Average $\alpha(\theta_i, \theta_j)$')
    a1.set_ylim((0.0, 1.0))

    a1.set_xlabel(f'Iterations')
    a2 = a1.twinx()

    a2.plot(int(iterations/(bins-1)) * (np.linspace(0, bins - 1, num = bins - 1)), normed_trace, c = trace_colour)
    if len(covs[0][:]) == 6:
        a2.plot(int(iterations/(bins-1)) * (np.linspace(0, bins - 1, num = bins - 1)), normed_stable_trace, c = trace_colour, linestyle = 'dashed')
    a2.set_ylabel(r'Average $Tr(C)$')
    
    a2.set_ylim((0.0, 1.0))
    a2.set_yticks([0.0, 1.0])
    a2.set_yticklabels(['Min', 'Max'])

    a1.spines['left'].set_color(rate_colour)
    a2.spines['left'].set_color(rate_colour)

    a1.yaxis.label.set_color(rate_colour)
    a1.tick_params(axis = 'y', colors = rate_colour)

    a2.spines['right'].set_color(trace_colour)
    a1.spines['right'].set_color(trace_colour)

    a2.yaxis.label.set_color(trace_colour)
    a2.tick_params(axis = 'y', colors = trace_colour)

    plt.savefig('results/'+name+'-acc-trace-prog.png', bbox_inches="tight", dpi=dpi)
    plt.clf()

    return




def amplification(m, theta, ts, label = None, color = None, alpha = None, caustic = None):

    if m == 0:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta.truth)))
    if m == 1:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'q', 's', 'alpha'], theta.truth)))
    model.set_magnification_methods([ts[0], 'point_source', ts[1]])

    epochs = np.linspace(ts[0], ts[1], 720)

    if caustic is not None:
        model.plot_trajectory(t_start = ts[0], t_stop = ts[1], color = color, linewidth = 1, alpha = alpha, arrow_kwargs = {'width': 0.006})
        model.plot_caustics(color = color, s = 1, marker = 'o', n_points = 10000)

    else:
        A = model.magnification(epochs)
        plt.plot(epochs, A, color = color, label = label, alpha = alpha)

    return

def fitted_flux(m, theta, data, ts, label = None, color = None, alpha = None):

    if m == 0:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta.truth)))
    if m == 1:
        model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'q', 's', 'alpha'], theta.truth)))
    model.set_magnification_methods([ts[0], 'point_source', ts[1]])

    epochs = np.linspace(ts[0], ts[1], 720)


    a = model.magnification(epochs)
    # Fit proposed flux as least squares solution.
    F = light_curve_simulation.least_squares_signal(a, data.flux)
    plt.plot(epochs, F, color = color, label = label, alpha = alpha)

    return

def style():
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



def adjust_viewing_axis(axis, ax, states, density_base, bounds, size):
    
    # adjust viweing axis
    Lower = np.min([np.min(states), density_base])
    Upper = np.max([np.max(states), density_base])
    Width = Upper - Lower
    Upper += (size * Width) / 2
    Lower -= (size * Width) / 2

    # check for scaling (hacky)
    if not(bounds.lb <= density_base <= bounds.rb):
        rb = np.log10(bounds.rb)
        lb = np.log10(bounds.lb)

    else: # no scaling
        rb = bounds.rb
        lb = bounds.lb


    # limits within prior bounds
    if Upper > rb:
        Upper = rb
    if Lower < lb:
        Lower = lb

    if axis == 'x': ax.set_xlim((Lower, Upper))
    if axis == 'y': ax.set_ylim((Lower, Upper))

    return Upper, Lower


def density_heatmaps(model, n_pixels, data, event_params, symbols, view_size = 1, name = '', dpi = 100):

    n_dim = model.D

    states = model.sampled.states_array(scaled=True)

    # build off corners spacing and other styling
    style()
    figure = corner.corner(states.T)

    # font visible
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14

    # extract the axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))

    # params to evaluate 2d slices at
    #if event_params is not None:
    #    density_base = event_params

    #else:
    density_base = model.center

    fit_mu = np.zeros((model.D))

    # Loop over the diagonal
    for i in range(n_dim):
        ax = axes[i, i]
        ax.cla()

        # distribution plots
        fit_mu[i] = np.average(states[i, :])
        sd = np.std(states[i, :], ddof = 1)
        ax.axvspan(fit_mu[i] - sd, fit_mu[i] + sd, alpha = 0.5, color = 'lime', label = r'$\bar{\mu}\pm\bar{\sigma}$')

        ax.hist(states[i, :], bins = 10, density = False, color = 'black', alpha = 1.0)

        if event_params is not None:
            ax.axvline(event_params.scaled[i], label = r'$\theta$', color = 'red')

        ax.set_title(r'$\bar{\mu} = $'+f'{fit_mu[i]:.4}'+',\n'+r'$\bar{\sigma} = \pm$'+f'{sd:.4}')

        if i == 0: # first
            ax.set_ylabel(symbols[i])
            ax.axes.get_yaxis().set_ticklabels([]) # dont view frequency axis
            ax.axes.get_xaxis().set_ticklabels([])

        elif i == n_dim - 1: # last
            ax.set_xlabel(symbols[i])
            ax.tick_params(axis = 'x', labelrotation = 45)
            ax.axes.get_yaxis().set_ticklabels([])
        
        else:
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])

        xUpper, xLower = adjust_viewing_axis('x', ax, states[i, :], event_params.scaled[i], model.priors[i], view_size)


    # loop over lower triangular 
    for yi in range(n_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            
            # posterior heat map sizing
            xUpper, xLower = adjust_viewing_axis('x', ax, states[xi, :], event_params.scaled[xi], model.priors[xi], view_size)
            yUpper, yLower = adjust_viewing_axis('y', ax, states[yi, :], event_params.scaled[yi], model.priors[yi], view_size)

            yaxis = np.linspace(yLower, yUpper, n_pixels)
            xaxis = np.linspace(xLower, xUpper, n_pixels)
            density = np.zeros((n_pixels, n_pixels))
            x = -1
            y = -1

            for i in yaxis:
                x += 1
                y = -1
                for j in xaxis:
                    y += 1

                    temp = deepcopy(density_base.scaled)

                    temp[xi] = j
                    temp[yi] = i

                    theta = sampling.State(scaled=temp)

                    density[x][y] = np.exp(model.log_likelihood(theta) + model.log_prior_density(theta))

            density = np.sqrt(np.flip(density, 0)) # so lower bounds meet. sqrt to get better definition between high vs low posterior 
            ax.imshow(density, interpolation = 'none', extent=[xLower, xUpper, yLower, yUpper], aspect = (xUpper-xLower) / (yUpper-yLower))

            # the fit normal distribution's contours
            # https://stats.stackexchange.com/questions/60011/how-to-find-the-level-curves-of-a-multivariate-normal

            mu = [np.mean(states[xi, :]), np.mean(states[yi, :])]
            row = np.array([xi, yi])
            col = np.array([xi, yi])
            K = model.covariance[row[:, np.newaxis], col] 
            angles = np.linspace(0, 2*math.pi - 2*math.pi/720, 720)
            R = [np.cos(angles), np.sin(angles)]
            R = np.transpose(np.array(R))

            # keep bounds before sigma contours
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()

            for level in [1 - 0.989, 1 - 0.865, 1 - 0.393]: # 1,2,3 sigma levels
                rad = np.sqrt(chi2.isf(level, 2))
                level_curve = rad * R.dot(scipy.linalg.sqrtm(K))
                ax.plot(level_curve[:, 0] + mu[0], level_curve[:, 1] + mu[1], color = 'lime')

            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

            # plot true values as crosshairs if they exist
            if event_params is not None:
                ax.scatter(event_params.scaled[xi], event_params.scaled[yi], marker = 'o', s = 75, c = 'red', alpha = 1)
                ax.axvline(event_params.scaled[xi], color = 'red')
                ax.axhline(event_params.scaled[yi], color = 'red')

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

    
    axes = figure.get_axes()[4].get_gridspec()
    inset_ax = figure.add_subplot(axes[:2, n_dim-3:])
    inset_ax.set_ylabel('Flux')
    inset_ax.set_xlabel('Time [days]')
    ts = [0, 72]
    epochs = np.linspace(0, 72, 720)
    inset_ax.plot(epochs, data.flux, color = 'black', label = 'y')

    #fitted_flux(model.m, event_params, data, ts, label = 'truth', color = 'red')
    
    fitted_params = sampling.State(scaled = fit_mu)
    fitted_flux(model.m, fitted_params, data, ts, label = 'fit', color = 'lime')

    figure.savefig('results/' + name + '-density-heatmap.png', bbox_inches = "tight", dpi = dpi) # tight layout destroys spacing
    figure.clf()

    return




def center_offsets_pointilism(supset_model, subset_model, symbols, name = '', dpi = 100):

    supset_offsets = (supset_model.sampled.states_array(scaled = True) - supset_model.center.scaled[:, np.newaxis])
    subset_offsets = (subset_model.sampled.states_array(scaled = True) - subset_model.center.scaled[:, np.newaxis])
    n_dim = subset_model.D

    style()
    # construct shape with corner
    figure = corner.corner(subset_offsets.T)

    # font/visibility
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14

    # extract the axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))


    # Loop over the diagonal to remove from plot
    for i in range(n_dim):
        ax = axes[i, i]
        ax.cla()
        ax.patch.set_alpha(0.0)
        ax.axis('off')
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        

    # loop over lower triangle
    for yi in range(n_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            
            # overlay points
            ax.scatter(subset_offsets[xi, :], subset_offsets[yi, :], c = np.linspace(0.0, 1.0, subset_model.sampled.n), cmap = 'winter', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)
            ax.scatter(supset_offsets[xi, :], supset_offsets[yi, :], c = np.linspace(0.0, 1.0, supset_model.sampled.n), cmap = 'spring', alpha = 0.15, marker = ".", s = 20, linewidth = 0.0)

            if yi == n_dim - 1: # last row
                ax.set_xlabel(symbols[xi])
                ax.tick_params(axis = 'x', labelrotation = 45)

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            if xi == 0: # first column
                ax.set_ylabel(symbols[yi])
                ax.tick_params(axis = 'y', labelrotation = 45)

            else:    
                ax.axes.get_yaxis().set_ticklabels([])

    figure.savefig('results/' + name + '-centered-pointilism.png', bbox_inches = "tight", dpi = dpi)
    figure.clf()

    return








def joint_samples_pointilism(supset_model, subset_model, joint_model_chain, symbols, name = '', dpi = 100):

    # fonts/visibility
    ls = 20 # label size
    lr = 45 # label rotation
    n_lb = 3 # tick labels

    # sizing
    N_dim = supset_model.D
    n_dim = subset_model.D

    # extract points
    supset_states = supset_model.sampled.states_array(scaled = True)
    subset_states = subset_model.sampled.states_array(scaled = True)

    style()
    figure = corner.corner(supset_states.T) # use corner for layout/sizing

    #     # fonts/visibility
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14

    # extract the axes
    axes = np.array(figure.axes).reshape((N_dim, N_dim))


    # loop diagonal
    for i in range(N_dim):
        ax = axes[i, i]

        ax.cla()
        ax.plot(np.linspace(1, joint_model_chain.n, joint_model_chain.n), joint_model_chain.states_array(scaled = True)[i, :], linewidth = 0.5, color='black')

        if i == 0: # first tile
            ax.set_ylabel(symbols[i])
            #ax.yaxis.label.set_size(ls)
            ax.tick_params(axis='y', labelrotation = lr)
            ax.axes.get_xaxis().set_ticklabels([])

        elif i == N_dim - 1: # last tile
            ax.set_xlabel(symbols[i])
            #ax.xaxis.label.set_size(ls)
            ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.get_xaxis().set_ticklabels([])

        else:
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])

        #ax.locator_params(nbins = n_lb) # 3 ticks max
        #ax.xaxis.tick_top()
        #ax.yaxis.tick_right()
        #ax.set_title(symbols[i])
        

    # loop lower triangular
    for yi in range(N_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.cla()
            ax.scatter(supset_states[xi, :], supset_states[yi, :], c = np.linspace(0.0, 1.0, supset_model.sampled.n), cmap = 'spring', alpha = 0.25, marker = ".", s = 25, linewidth = 0.0)
                
            if yi == N_dim - 1: # bottom row
                ax.set_xlabel(symbols[xi])
                #ax.xaxis.label.set_size(ls)
                ax.tick_params(axis = 'x', labelrotation = lr)

            else:    
                ax.axes.get_xaxis().set_ticklabels([])

            if xi == 0: # first column
                ax.set_ylabel(symbols[yi])
                #ax.yaxis.label.set_size(ls)
                ax.tick_params(axis = 'y', labelrotation = lr)

            else:    
                ax.axes.get_yaxis().set_ticklabels([])

            #ax.locator_params(nbins = n_lb) # 3 ticks max


            # add upper triangular plots
            if xi < n_dim and yi < n_dim:
                
                # acquire axes and plot
                axs = figure.get_axes()[4].get_gridspec()
                axt = figure.add_subplot(axs[xi, yi])
                axt.scatter(subset_states[yi, :], subset_states[xi, :], c = np.linspace(0.0, 1.0, subset_model.sampled.n), cmap = 'winter', alpha = 0.25, marker = ".", s = 25, linewidth = 0.0)
                
                if yi == n_dim - 1: # last column
                    axt.set_ylabel(symbols[xi])
                    ax.yaxis.label.set_size(ls)
                    axt.yaxis.tick_right()
                    axt.yaxis.set_label_position("right")
                    axt.tick_params(axis = 'y', labelrotation = lr)
                
                else:
                    axt.axes.get_yaxis().set_ticklabels([])
                
                if xi == 0: # first row
                    axt.set_xlabel(symbols[yi])
                    axt.tick_params(axis = 'x', labelrotation = lr)
                    ax.xaxis.label.set_size(ls)
                    axt.xaxis.tick_top()
                    axt.xaxis.set_label_position("top") 
                
                else:
                    axt.axes.get_xaxis().set_ticklabels([])

                #axt.locator_params(nbins = n_lb) # 3 ticks max

    figure.savefig('results/' + name + '-joint-pointilism.png', bbox_inches = "tight", dpi = dpi)
    figure.clf()

    return




