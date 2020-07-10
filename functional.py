"""Functional interface; some changes to standard pytorch functional for px_expander"""



import warnings
import math
from operator import mul
from functools import reduce
import sys

import torch
from torch.nn import _functions
from torch.nn.modules import utils
from torch.autograd import Variable


'''
Linear layer modified for PX gradients
'''

def linear(input, weight, bias=None, batch_size=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        if batch_size is None:
          return torch.addmm(bias, input, weight.t())
        else:
          print('fused op in functional.linear not implemented yet!')
          sys.exit(1)
          return torch.addmm(bias, input, weight.t())

    output = input.matmul(torch.transpose(weight,-2,-1))

    # not using bias at the moment
    if bias is not None:
        output += bias
    return output
