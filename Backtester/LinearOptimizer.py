from enum import Enum
import numpy as np
import cvxpy as cp
import math
from util import nearestPD

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import PortfolioClasses as pc
import LossFunctions as lf

from torch.utils.data import DataLoader

'''
Forward Pass:
    Calculate EW Portfolio - w_EW
    Calculate RP Portfolio - w_RP
    Perform a linear regression to learn alpha, where w_Opt = alpha*w_EW + (1-alpha)*w_RP
'''
class LinearEWAndRPOptimizer(nn.Module):
    def __init__(self):
        super(LinearEWAndRPOptimizer, self).__init__()

    def forward(self):
        raise NotImplementedError

    def net_train(self):
        raise NotImplementedError
