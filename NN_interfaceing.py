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

#import MulensModel as mm
#import PlotFunctions as pltf
#from torch.utils.tensorboard import SummaryWriter

#from single_extract_features import extract_single
import os
import os.path
import shutil
from pathlib import Path




def get_posteriors(m):

    full_path = os.getcwd()
    out_path = (str(Path(full_path).parents[0]))

    if m == 0:
        with open(out_path+"/microlensing/output/single_25K_720.pkl", "rb") as handle: posterior = pickle.load(handle)

    if m == 1:
        with open(out_path+"/microlensing/output/binary_100K_720.pkl", "rb") as handle: posterior = pickle.load(handle)
    
    return posterior



def get_model_centers(posterior, signal_data):

    maxp = posterior.map(signal_data, num_iter = 100, num_init_samples = 100, show_progress_bars = False)

    maxp.numpy
    centers = np.float64(maxp)
    
    print(centers)

    return np.array(centers)



def get_model_ensemble(posterior, signal_data, n):

    samples = posterior.sample((n, ), x = signal_data)
    log_prob_samples = np.array(posterior.log_prob(theta = samples, x = signal_data))

    return samples, log_prob_samples