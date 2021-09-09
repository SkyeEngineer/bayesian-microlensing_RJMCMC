import numpy as np
import matplotlib.pyplot as plt
import emcee as mc
import plot_functions as pltf

# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

def plot_act(joint_model_chain, symbols, name='', dpi=100):
    
    pltf.style()

    n_samples = joint_model_chain.n
    states = joint_model_chain.states_array()

    # points to compute act
    n_act = 10
    N = np.exp(np.linspace(np.log(int(n_samples/n_act)), np.log(n_samples), n_act)).astype(int)

    # loop throguh parameters
    for p in range(joint_model_chain.states[-1].D):
        act_p = np.zeros(len(N))
        p_signal = np.array(states[p, :])

        for i, n in enumerate(N):
            act_p[i] = mc.autocorr.integrated_time(p_signal[:n], c = 5, tol = 5, quiet = True)

        plt.loglog(N, act_p, "o-", label = symbols[p], color = plt.cm.autumn(p/joint_model_chain.states[-1].D), linewidth = 2, markersize = 5)

    # again for m
    act_m_indices = np.zeros(len(N))
    m_indices_signal = np.array(joint_model_chain.model_indices)
    for i, n in enumerate(N):
        act_m_indices[i] = mc.autocorr.integrated_time(m_indices_signal[:n], c = 5, tol = 5, quiet = True)
    plt.loglog(N, act_m_indices, "o-b", label=r"$m_i$",  linewidth = 2, markersize = 5)

    # plot details
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label = r"$\tau = N/50$")
    plt.ylim(ylim)
    plt.xlabel(r"Samples, $N$")
    plt.ylabel(r"Autocorrelation time, $\tau$")
    plt.legend(fontsize = 7, frameon = False)
    plt.savefig('results/'+name+'-act.png', dpi = dpi, bbox_inches = 'tight')
    plt.clf()

    return


def attempt_truncation(Models, joint_model_chain, warm_up = 0):

    n_samples = joint_model_chain.n
    
    # points to compute act
    n_act = 10
    N = np.exp(np.linspace(np.log(int(n_samples/n_act)), np.log(n_samples), n_act)).astype(int)

    act_m_indices = np.zeros(len(N))
    m_indices_signal = np.array(joint_model_chain.model_indices)

    for i, n in enumerate(N):
        act_m_indices[i] = mc.autocorr.integrated_time(m_indices_signal[:n], c = 5, tol = 5, quiet = True)
        
        if act_m_indices[i] < N[i]/50: # stepwise interpolate truncation point
            truncated = N[i]

            for Model in Models: # drop warm_up and joint burn in
                Model.sampled.states = Model.sampled.states[warm_up + truncated:]
                Model.sampled.n = Model.sampled.n - warm_up - truncated

            # general chain doesn't have adapt_MH warm up
            joint_model_chain.model_indices = joint_model_chain.model_indices[truncated:]
            joint_model_chain.states = joint_model_chain.states[truncated:]
            joint_model_chain.n = joint_model_chain.n - truncated

            return truncated

    print("Did not have a low enough autocorrelation time")
    return 0