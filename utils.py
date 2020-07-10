'''
Some general util functions used by master & clients
'''

import numpy as np
import torch

import torch.nn.functional as F
from torch import nn

import linear


# The convolutional part of the network for CIFAR
class Net1(nn.Module):
  def __init__(self):
    super(Net1, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))       
    return x


# FC layers when using conv layers (cifar data)
class FCNet(nn.Module):
  def __init__(self, batch_size, batch_proc_size, latent_dim, n_hidden_layers, output_dim, randomize_data):
    super(FCNet, self).__init__()

    self.batch_proc_size = batch_proc_size
    self.batch_size = batch_size
    self.latent_dim = latent_dim
    self.n_hidden_layers = n_hidden_layers
    self.output_dim = output_dim
    self.randomize_data = randomize_data
    self.relu = nn.ReLU()
    self.linears = nn.ModuleList([ linear.Linear(1600, latent_dim, bias=False, batch_size=batch_proc_size)])
    if n_hidden_layers > 0:
      for k in range(n_hidden_layers):
        self.linears.append( linear.Linear(latent_dim, latent_dim,bias=False,batch_size=batch_proc_size) )
    self.final_fc = linear.Linear(self.linears[-1].out_features, output_dim,bias=False, batch_size=batch_proc_size)
    
  def forward(self, x):
    x = torch.unsqueeze(x.view(-1, 1600),1)
    for k_linear in self.linears:
      x = self.relu(k_linear(x))
    x = self.final_fc(x)
    return nn.functional.log_softmax(x.view(-1,self.output_dim),dim=1)



# FC layers for MNIST
class FCNet_MNIST(nn.Module):
  def __init__(self, batch_size, batch_proc_size, latent_dim, n_hidden_layers, output_dim, randomize_data):
    super(FCNet_MNIST, self).__init__()

    self.batch_proc_size = batch_proc_size
    self.batch_size = batch_size
    self.latent_dim = latent_dim
    self.n_hidden_layers = n_hidden_layers
    self.output_dim = output_dim
    self.randomize_data = randomize_data    
    self.relu = nn.ReLU()    
    self.linears = nn.ModuleList([ linear.Linear(784, latent_dim, bias=False, batch_size=batch_proc_size)])
    if n_hidden_layers > 0:
      for k in range(n_hidden_layers):
        self.linears.append( linear.Linear(latent_dim, latent_dim,bias=False,batch_size=batch_proc_size) )
    self.final_fc = linear.Linear(self.linears[-1].out_features, output_dim,bias=False, batch_size=batch_proc_size)
    
  def forward(self, x):
    x = torch.unsqueeze(x.view(-1, 784),1)
    for k_linear in self.linears:
      x = self.relu(k_linear(x))
    x = self.final_fc(x)   
    return nn.functional.log_softmax(x.view(-1,self.output_dim),dim=1)



# bogus model class used with projected gradients
class BogusModel():
    def __init__(self, batch_size, batch_proc_size, param_vec):
        self.batch_size = batch_size
        self.batch_proc_size = batch_proc_size
        self.params = param_vec
        self.params.grad = torch.zeros_like(param_vec)
        #self.params.requires_grad = True
    
    def parameters(self):
        return [self.params]

# do projection in smaller pieces
def do_proj_in_steps(vector_for_proj, model, dim_reduction, max_proj_size, proj_type, dim_red_rng_state):
    
    # use shared randomness for generating proj matrices
    curr_rng_state = torch.get_rng_state()
    torch.set_rng_state(dim_red_rng_state)

    proj_grads = torch.zeros(dim_reduction)
    #print('vec dims:{}, single proj dims:{}, full proj dims:{}, proj grads dims:{}'.format(vectorized.size(), (model.total_params,max_proj_size ),(model.total_params, dim_reduction), proj_grads.size() ))

    for i_proj in range(dim_reduction//max_proj_size):
        proj_matrix = generate_proj_matrix_piece(model=model, dim_reduction=dim_reduction, max_proj_size=max_proj_size, proj_type=proj_type)
        #init_rng_state = False
        # do proj here
        if proj_type == 2:
            #vectorized = torch.sparse.mm(proj_matrix.t(),vectorized.t())
            sys.exit('Proj type 2 not immplemented yet!')
        elif proj_type in (1,3):
            #print('proj size:{}, vect size:{}'.format(proj_matrix.shape, vectorized.shape))
            #print(proj_grads[i_proj*max_proj_size:(i_proj+1)*max_proj_size].shape)
            proj_grads[i_proj*max_proj_size:(i_proj+1)*max_proj_size] = torch.mm(proj_matrix.t(),vector_for_proj.t()).view(max_proj_size)
    
    proj_matrix = None
    #print(proj_grads)

    # return to non-shared randomness
    dim_red_rng_state = torch.get_rng_state()
    torch.set_rng_state(curr_rng_state)

    return proj_grads, dim_red_rng_state


# function for generating shared projection matrices in pieces
def generate_proj_matrix_piece(model, dim_reduction, max_proj_size, proj_type=1):

    proj_matrix = None
    if proj_type == 1:
        proj_matrix = 1/np.sqrt(dim_reduction) * torch.randn(model.total_params, max_proj_size)

    return proj_matrix


# function for generating shared projection matrices
def generate_proj_matrix(model, dim_reduction, dim_red_rng_state, proj_const, proj_type=1):
    curr_rng_state = torch.get_rng_state()
    torch.set_rng_state(dim_red_rng_state)
    #print(torch.get_rng_state())
    
    if proj_type == 1:
        ## N(0,1) projection
        proj_matrix = (1/np.sqrt(dim_reduction))*torch.randn(model.total_params, dim_reduction)
    
    elif proj_type == 2:
        ## sparse projection
        # draw number of non-zeros
        non_zeros = int(torch.distributions.binomial.Binomial(total_count=(model.total_params*dim_reduction),probs=1/(2*proj_const)).sample([1]))
        # draw number of positives
        num_pos = int(torch.distributions.binomial.Binomial(total_count=non_zeros, probs=.5).sample([1]))
        # draw indices & values for non-zeros
        i = torch.randint(high=int(model.total_params), size=(1,non_zeros), dtype=torch.long)
        ii = torch.randint(high=int(dim_reduction), size=(1,non_zeros), dtype=torch.long)
        # take first num_pos as ones, rest as -1 (and add scaling)
        # do scaling later at master
        vals = torch.cat( (torch.ones(num_pos),torch.zeros(non_zeros-num_pos)-1)  )

        proj_matrix = torch.sparse.FloatTensor(torch.cat((i,ii),dim=0),vals,torch.Size([int(model.total_params),int(dim_reduction)]))#.coalesce()
    
    else:
        sys.exit('Unknown projection type!')
    
    dim_red_rng_state = torch.get_rng_state()
    torch.set_rng_state(curr_rng_state)
    return proj_matrix, dim_red_rng_state


