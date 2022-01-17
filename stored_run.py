
import sampling
import light_curve_simulation
import distributions
import autocorrelation as acf
import plotting as pltf
import random
import numpy as np
import matplotlib.pyplot as plt
import surrogate_posteriors
from copy import deepcopy
import time
import pickle

file = open('results/1/1_stored_run.mcmc', 'rb') 
object = pickle.load(file)
joint_model_chain, binary_states, single_states, binary_sp_states, single_sp_states, warm_up_iterations, symbols, event_params, data, name, dpi = object
#binary_states, single_states, warm_up_iterations, symbols, event_params, name, dpi = object


acf.plot_act(joint_model_chain, symbols, name, dpi)

binary_theta = np.zeros(7)
for i in range(7):
    y_a, x_a, _ = plt.hist(binary_states[i, :], bins=10)
    binary_theta[i] = np.mean([x_a[y_a.argmax()], x_a[y_a.argmax()+1]])
binary_theta = sampling.State(scaled=binary_theta)

single_theta = np.zeros(4)
for i in range(4):
    y_a, x_a, _ = plt.hist(single_states[i, :], bins=10)
    single_theta[i] = np.mean([x_a[y_a.argmax()], x_a[y_a.argmax()+1]])
single_theta = sampling.State(scaled=single_theta)
print(single_theta.truth)
print(binary_theta.truth)
print(event_params.truth)
#print(theta.truth)
#plt.clf()
#plt.plot(data.time, data.flux, color='black')
#pltf.fitted_flux(1, binary_theta, data, [0, 72], color='tab:purple', label='binary')
#pltf.fitted_flux(0, single_theta, data, [0, 72], color='gold', label='single')
#plt.savefig('results/'+name+'-curve.png', bbox_inches = 'tight', dpi = dpi, transparent=True)
#plt.clf()

curves = deepcopy([single_theta, binary_theta, data])

#pltf.joint_samples_pointilism_2(binary_states, single_states, warm_up_iterations, symbols, event_params, name, dpi)
#ranges=[[0.1, 1], [0, 72], [0, 1], [0, 100], [-6, 0], [0.2, 5], [0, 360]]
ranges=[[0.45, 0.55], [14.75 -0.05, 15.25 +0.05], [0.08 -0.005, 0.12 + 0.005], [8.5 -0.05, 11.0 +0.05], [-5 -0.15, -1 +0.15], [0.2 -0.15, 5 +0.15], [0 -5, 360 +5]]
pltf.broccoli(binary_states, single_states, binary_sp_states, single_sp_states, symbols, ranges, curves, event_params, name, 100)