import numpy as np
import matplotlib.pyplot as plt
import emcee as mc

# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

def truncate_burn_in(Models, joint_model_chain, warm_up = 0):
    samples = joint_model_chain.n
    
    # points to compute act
    n_act = 50
    N = np.exp(np.linspace(np.log(int(samples/n_act)), np.log(samples), n_act)).astype(int)

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
            joint_model_chain.sampled.states = joint_model_chain.sampled.states[truncated:]
            joint_model_chain.sampled.n = joint_model_chain.sampled.n - truncated

            return truncated

    print("Did not have a low enough autocorrelation time")
    return 0



def plot_act(Models, joint_model_chain, warm_up = 0):
    samples = joint_model_chain.n
    
    # points to compute act
    n_act = 50
    N = np.exp(np.linspace(np.log(int(samples/n_act)), np.log(samples), n_act)).astype(int)

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
            joint_model_chain.sampled.states = joint_model_chain.sampled.states[truncated:]
            joint_model_chain.sampled.n = joint_model_chain.sampled.n - truncated

            return truncated

    print("Did not have a low enough autocorrelation time")
    return 0
