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

sn = 4

# Synthetic Event Parameters
theta_Models = [
    [36, 0.133, 61.5, 0.0096, 0.002, 1.27, 210.8], # strong binary
    [36, 0.133, 61.5, 0.0096, 0.00091, 1.3, 210.8], # weak binary 1
    [36, 0.133, 61.5, 0.0056, 0.0007, 1.3, 210.8], # weak binary 2
    [36, 0.133, 61.5, 0.0096, 0.0002, 4.9, 223.8], # indistiguishable from single
    [36, 0.133, 31.5]  # single
    ]
theta_Model = np.array(theta_Models[sn])

Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta_Model)))
Model.set_magnification_methods([0., 'point_source', 72.])

#Model = mm.Model(dict(zip(['t_0', 'u_0', 't_E'], theta)))
#Model.set_magnification_methods([0., 'point_source', 72.])

#Model.plot_magnification(t_range=[0, 72], subtract_2450000=False, color='black')
#plt.savefig('temp.jpg')
#plt.clf()

# Generate "Synthetic" Lightcurve
epochs = Model.set_times(n_epochs = 720)
signal_data = Model.magnification(epochs)

pltf.LightcurveFitError(1, [36.5, 0.133, 20], 0, signal_data, Model, Model.set_times(n_epochs = 720), 0, 0)



g=g

#cur_path = os.path.dirname(__file__)
d = os.getcwd()
print(d)
full_path = d
o=(str(Path(full_path).parents[0]))


with open(o+"/microlensing/output/posterior_noise_single6.pkl", "rb") as handle:
        posterior = pickle.load(handle)

x_o = signal_data #x[1, :]


#print(theta[1, :])

samples = posterior.sample((10000,), x=x_o)

print(posterior.map(x_o, num_iter = 100, num_init_samples = 100, show_progress_bars = False))