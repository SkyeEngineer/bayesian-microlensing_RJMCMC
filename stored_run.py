
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

file = open('temp.obj', 'rb') 
object = pickle.load(file)
binary_states, single_states, warm_up_iterations, symbols, event_params, name, dpi = object

pltf.joint_samples_pointilism_2(binary_states, single_states, warm_up_iterations, symbols, event_params, name, 250)