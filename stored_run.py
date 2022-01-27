
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
import MulensModel as mm

file = open('results/2/2_stored_run.mcmc', 'rb') 
object = pickle.load(file)
joint_model_chain, MAPests, binary_states, single_states, binary_sp_states, single_sp_states, warm_up_iterations, symbols, event_params, data, name, dpi = object
#binary_states, single_states, warm_up_iterations, symbols, event_params, name, dpi = object

print(MAPests[0].truth)
print(MAPests[1].truth)


acf.plot_act(joint_model_chain, symbols, name, dpi)

binary_theta = MAPests[1]

single_theta = MAPests[0]

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
ranges=[[0.45 -0.005, 0.55 +0.005], [14.9 -0.05, 15.1 +0.05], [0.08 -0.005, 0.12 + 0.005], [9.0 -0.5, 11.0 +0.5], [-5 -0.5, -1 +0.5], [0.2-0.1, 5-0.1], [0, 360]]
symbols = [r'$f_s$', r'$t_0$', r'$u_0$', r'$t_E$', r'$log_{10}(q)$', r'$s$', r'$\alpha$']
pltf.broccoli(binary_states[1:, :], single_states[1:, :], binary_sp_states[1:, :], single_sp_states[1:, :], symbols[1:], ranges[1:], curves, event_params, name, 100)



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