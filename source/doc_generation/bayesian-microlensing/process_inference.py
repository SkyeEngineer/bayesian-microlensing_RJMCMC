"""Proccess a stored posterior object."""


import autocorrelation as acf
import plotting as pltf
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import MulensModel as mm


if __name__ == "__main__":

    # Result file to process.
    n_suite = 3

    names = ['1/1', '2/2', '3/3', '4/4', '5/5']
    name = names[n_suite]
    file = open('results/'+name+'_stored_run.mcmc', 'rb') 
    object = pickle.load(file)
    joint_model_chain, MAPests, binary_states, single_states, binary_sp_states, single_sp_states, warm_up_iterations, symbols, event_params, data, name, dpi = object

    # MAP estimates.
    print(f'single MAP {MAPests[0].truth}')
    print(f'binary MAP {MAPests[1].truth}')
    binary_theta = MAPests[1]
    single_theta = MAPests[0]
    curves = deepcopy([single_theta, binary_theta, data])

    # PLotting info.
    ranges=[[0.45 -0.005, 0.55 +0.005], [14.9, 15.1 +0.15], [0.08 -0.005, 0.12 + 0.025], [9.0+0.05, 10.7], [-5, -1+0.1], [0.2, 5], [1, 360]] #1
    #ranges=[[0.45 -0.005, 0.55 +0.005], [15.0 -0.05, 15.1 +0.05], [0.1 -0.01, 0.13 + 0.005], [9.0 -0.0, 10.5 +0.25], [-2.25 -0.025, -1.25 +0.025], [0.2, 2.5], [40, 200]]
    #ranges=[[0.45 -0.005, 0.55 +0.005], [15.0 -0.06, 15.1 +0.05], [0.092, 0.13], [9.0 -0.1, 10.5 +0.25], [-2.35, -1.675], [0.5, 1.8], [55, 63]] #4
    symbols = [r'$f_s$', r'$t_0$', r'$u_0$', r'$t_E$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']

    # Joint posterior.
    pltf.broccoli(joint_model_chain, binary_states[1:, :], single_states[1:, :], binary_sp_states[1:, :], single_sp_states[1:, :], symbols[1:], ranges[1:], curves, event_params, name, dpi)

    # Autocorrelation time.
    import matplotlib.ticker as ticker
    plt.figure()
    act_ax = plt.gca()
    act_ax.set_xlim([1000, 10000])
    act_ax.set_ylim([10, 100])#1
    #act_ax.set_ylim([1, 10])#4
    acf.plot_act(act_ax, joint_model_chain)
    act_ax.set_title('(c)', loc='left', fontsize=20)
    act_ax.tick_params(which='both', direction='in', labelsize = 12)
    act_ax.tick_params(which='major', length=10)
    act_ax.tick_params(which='minor', length=5, labelsize = 0, labelcolor = (0, 0, 0, 0))

    plt.savefig('results/' + name + '-act.png', bbox_inches = "tight", dpi = dpi, transparent=True)
    plt.clf()

    # MAP light curves.
    plt.figure()
    inset_ax = plt.gca()
    # Inset light curve plot. 
    inset_ax.set_ylabel('normalised flux', fontsize = 18)
    inset_ax.set_xlabel('time [days]', fontsize = 18)
    ts = [0, 72]

    pltf.flux(1, event_params, ts, label = 'truth', color = 'black', lw=2)
    pltf.flux(0, single_theta, ts, label = 'single MAP', color='tab:green', ls=':', lw=2)
    pltf.flux(1, binary_theta, ts, label = 'binary MAP', color='tab:purple', ls='--', lw=2)

    inset_ax.set_title('(b)', loc='left', fontsize=20)
    inset_ax.legend(fontsize = 14, handlelength=0.7, frameon = False, handletextpad=0.4)
    inset_ax.set_xlim([10, 20])
    inset_ax.set_ylim([1.1, 5.9])#1
    #inset_ax.set_ylim([1.1, 6.3])#4

    inset_ax.tick_params(which='both', top=True, right=True, direction='in', labelsize = 12)
    inset_ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    inset_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    inset_ax.tick_params(which='major', length=8)
    inset_ax.tick_params(which='minor', length=4)

    plt.savefig('results/' + name + '-lc.png', bbox_inches = "tight", dpi = dpi, transparent=True)
    plt.clf()

    # Trace of model index.
    plt.plot(np.linspace(0, joint_model_chain.n, joint_model_chain.n), joint_model_chain.model_indices, linewidth = 0.25, color = 'purple')
    plt.xlabel('Samples')
    plt.ylabel(r'$m_i$')
    plt.locator_params(axis = "y", nbins = 2) # only two ticks
    plt.savefig('results/'+name+'-mtrace.png', bbox_inches = 'tight', dpi = dpi, transparent=True)
    plt.clf()

    # Marginal probabilities.
    for i in range(2):
        print(f"P(m{i}|y): {joint_model_chain.model_indices.count(i) / joint_model_chain.n} +- {np.std(np.array(joint_model_chain.model_indices))/(joint_model_chain.n**0.5):.6f}")

    # Chi2 values.
    model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], binary_theta.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])
    a = model.magnification(data.time) # The proposed magnification signal.
    y = data.flux # The observed flux signal.
    F = (a-1)*binary_theta.truth[0]+1
    sd = data.err_flux
    print(f'binary chi2 {np.sum(((y - F)/sd)**2):.4f}')

    model = mm.Model(dict(zip(["t_0", "u_0", "t_E"], single_theta.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])
    a = model.magnification(data.time) # The proposed magnification signal.
    y = data.flux # The observed flux signal.
    F = (a-1)*single_theta.truth[0]+1
    sd = data.err_flux
    print(f'single chi2 {np.sum(((y - F)/sd)**2):.4f}')

    model = mm.Model(dict(zip(["t_0", "u_0", "t_E", "q", "s", "alpha"], event_params.truth[1:])))
    model.set_magnification_methods([0., "point_source", 72.])
    a = model.magnification(data.time) # The proposed magnification signal.
    y = data.flux # The observed flux signal.
    F = (a-1)*event_params.truth[0]+1
    sd = data.err_flux
    print(f'true chi2 {np.sum(((y - F)/sd)**2):.4f}')