'''
Script originally for doing dp grads using parameter expansions
'''

import numpy as np
import torch
from torch.autograd import Variable

import sys

from utils import generate_proj_matrix_piece

# clip and accumulate clipped gradients
def acc_scaled_grads(model, C, cum_grads, use_cuda=False):

  batch_size = model.batch_proc_size

  g_norm = Variable(torch.zeros(batch_size),requires_grad=False)

  if use_cuda:
    g_norm = g_norm.cuda()

  for p in filter(lambda p: p.requires_grad, model.parameters() ):
    if p.grad is not None:
      g_norm += torch.sum( p.grad.view(batch_size,-1)**2, 1)
      
  g_norm = torch.sqrt(g_norm)
  
  # do clipping and accumulate
  for p, key in zip( filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys() ):
    if p is not None:
      cum_grads[key] += torch.sum( (p.grad/torch.clamp(g_norm.contiguous().view(-1,1,1)/C, min=1)), dim=0 )
      


# add noise and replace model grads with cumulative grads
def add_noise_with_cum_grads(model, C, neighbour_const, sigma, cum_grads, noise_tensors=None, use_cuda=False):

    batch_proc_size = model.batch_proc_size
    for p, key in zip( filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys() ):
        if p.grad is not None:

            if noise_tensors is None:
                noise = Variable( (sigma*neighbour_const*C)*torch.normal(mean=torch.zeros_like(p.grad[0]).data, \
                    std=1.0).expand(batch_proc_size,-1,-1) )

            else:
                noise = Variable(noise_tensors[key].expand(batch_proc_size,-1,-1))
        
            p.grad = ((cum_grads[key].expand(batch_proc_size,-1,-1) + noise)/model.batch_size)
            if use_cuda:
                p.grad = p.grad.cuda()
          



