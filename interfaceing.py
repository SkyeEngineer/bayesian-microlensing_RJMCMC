import matplotlib.pyplot as plt
import numpy as np
from sbi.utils.torchutils import BoxUniform

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import pickle
import sys

from sbi.inference import SNPE_C
from models.sbi_snpe_mod.snpe_c import SNPE_C

from sbi.utils.get_nn_models import posterior_nn

from sbi import utils
from sbi import analysis

from models.resnet_gru import ResnetGRU
#from models.apt_network import APTNetwork
from models.yule_net import YuleNet

import MulensModel as mm
import PlotFunctions as pltf
#from torch.utils.tensorboard import SummaryWriter

#from single_extract_features import extract_single
import os
import os.path
import shutil
from pathlib import Path

sn = 0

# Synthetic Event Parameters
theta_Models = [
    [36, 0.133, 31.5, 0.0096, 0.002, 1.27, 210.8], # strong binary
    [36, 0.133, 31.5, 0.0096, 0.00091, 1.3, 210.8], # weak binary 1
    [36, 0.133, 61.5, 0.0056, 0.0007, 1.3, 210.8], # weak binary 2
    [36, 0.133, 61.5, 0.0096, 0.0002, 4.9, 223.8], # indistiguishable from single
    [36, 0.133, 31.5]  # single
    ]
theta_Model = np.array(theta_Models[sn])

Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'alpha'], theta_Model)))
Model.set_magnification_methods([0., 'VBBL', 72.])

n_epochs = 720
epochs = np.linspace(0, 72, n_epochs + 1)[:n_epochs]
signal_data = Model.magnification(epochs)

#Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta)))
#Model.set_magnification_methods([0., 'point_source', 72.])

#Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('temp.jpg')
#plt.clf()



def get_posteriors(m):

    full_path = os.getcwd()
    out_path = (str(Path(full_path).parents[0]))

    if m == 1:
        with open(out_path+"/microlensing/output/single_25K_720.pkl", "rb") as handle: posterior = pickle.load(handle)

    if m == 2:
        with open(out_path+"/microlensing/output/binary_100K_720.pkl", "rb") as handle: posterior = pickle.load(handle)
    
    return posterior



def get_model_centers(posterior, signal_data):

    #print("\n", posterior)
    maxp = posterior.map(signal_data, num_iter = 50, num_init_samples = 50, show_progress_bars = False)

    maxp.numpy
    centers = np.float64(maxp)
    print(centers)

    return centers



def get_model_ensemble(posterior, signal_data, n):

    samples = posterior.sample((n, ), x = signal_data)
    log_prob_samples = np.array(posterior.log_prob(theta = samples, x = signal_data))

    return samples, log_prob_samples

#arr, l_arr = get_model_ensemble(get_posterior(2, signal_data), signal_data, 1)



#def library():

#    return