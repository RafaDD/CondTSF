import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch
import time
from utils import *
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
from arch.mlp import *
from arch.lstm import *
from arch.cnn import *
from arch.dlinear import *


def get_network(model, channel=7):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    if model == 'DLinear':
        net = DLinear()
    elif model == 'LSTM':
        net = LSTM()
    elif model == 'CNN':
        net = CNN(channel)
    elif model == 'MLP':
        net = MLP()
    return net
