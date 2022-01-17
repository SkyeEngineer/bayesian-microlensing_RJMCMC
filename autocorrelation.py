"""Autocorrelation tools for convergence analysis.

See the detailed tutorial from:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import emcee as mc
import plotting as pltf


def attempt_truncation(joint_model_chain):
    """Truncate a burn in period
        
    Uses autocorrelation time heursistic to remove intial period below 50
    autocorrelation times, for a joint model space.

    Args:
        joint_model_chain: [chain] Collection of models parameter states.

    Returns:
        truncated: [int] Number of truncated states.
    """
    n_samples = joint_model_chain.n
    
    # Points to compute act.
    n_act = 10
    N = np.exp(np.linspace(np.log(int(n_samples/n_act)), np.log(n_samples), n_act)).astype(int)

    act_m_indices = np.zeros(len(N))
    m_indices_signal = np.array(joint_model_chain.model_indices)

    for i, n in enumerate(N):
        act_m_indices[i] = mc.autocorr.integrated_time(m_indices_signal[:n], c = 5, tol = 5, quiet = True)
        
        if i>0:
            if N[i-1] - N[i-1]/50 < act_m_indices[i] < N[i-1] + N[i-1]/50: # IACT stabilises.
                truncated = N[i]

                # Remove stored states and update count.
                joint_model_chain.model_indices = joint_model_chain.model_indices[truncated:]
                joint_model_chain.states = joint_model_chain.states[truncated:]
                joint_model_chain.n = joint_model_chain.n - truncated

                return truncated

    print("Integrated autocorrelation time did not converge.")
    return 0


def plot_act(joint_model_chain, symbols, name='', dpi=100):
    """Plot parameter autocorrelation times.
        
    Args:
        joint_model_chain: [chain] Collection of states from any model.
        symbols: [list] Variable name strings.
        name: [optional, string] File ouptut name.
        dpi: [optional, int] File output dpi.
    """
    pltf.style()

    n_samples = joint_model_chain.n
    states = joint_model_chain.states_array()

    # Points to compute IACT.
    n_act = 10
    N = np.exp(np.linspace(np.log(int(n_samples/n_act)), np.log(n_samples), n_act)).astype(int)



    # Loop through parameters.
    for p in range(joint_model_chain.states[-1].D):
        act_p = np.zeros(len(N))
        p_signal = np.array(states[p, :])

        for i, n in enumerate(N):
            act_p[i] = mc.autocorr.integrated_time(p_signal[:n], c = 5, tol = 5, quiet = True)

        plt.loglog(N, act_p, "o-", label = symbols[p], color = plt.get_cmap('tab10')(p/joint_model_chain.states[-1].D), linewidth = 2, markersize = 5)

    # Again for the model indices.
    act_m_indices = np.zeros(len(N))
    m_indices_signal = np.array(joint_model_chain.model_indices)
    for i, n in enumerate(N):
        act_m_indices[i] = mc.autocorr.integrated_time(m_indices_signal[:n], c = 5, tol = 5, quiet = True)
    plt.loglog(N, act_m_indices, "o-", label=r"$m$",  linewidth = 3, markersize = 7.5, color='black')

    # Plotting details.
    #ylim = plt.gca().get_ylim()
    #plt.plot(N, N / 50.0, "--k", label = r"$\tau = N/50$")
    #plt.ylim(ylim)
    plt.xlabel("Iterations")
    #plt.gca().set_xscale("log")
    #plt.xticks(ticks=[int(n_samples/2), n_samples])
    #plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.ylabel("Integrated autocorrelation time")
    plt.legend(fontsize = 12, frameon = False, handlelength=1.0, labelspacing=0.25)
    plt.savefig('results/'+name+'-act.png', dpi = dpi, bbox_inches = 'tight', transparent=True)
    plt.clf()

    return


