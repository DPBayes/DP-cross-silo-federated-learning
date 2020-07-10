'''
Script for running a distributed DP client
'''

import argparse
from collections import OrderedDict as od
import datetime
import numpy as np
import os
import pickle
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import sys

import linear
import px_expander

from client_comm_funs import *
from utils import BogusModel
from utils import FCNet_MNIST
from utils import FCNet
from utils import Net1
from utils import do_proj_in_steps


# distributed data sampler
class Joint_sampler():
    def __init__(self, trainset, shared_seed, client_data_indices, full_data_size, batch_size, batch_proc_size, use_cuda, use_joint_sampling, priv_seed, data_dims,poisson_sampling_fraction=0):

        # data_dims for cifar:
        # data_dims = (3,32,32)

        curr_prng_state = torch.get_rng_state()
        if use_cuda:
            curr_cuda_prng_state = torch.cuda.get_rng_state()
            if use_joint_sampling:
                torch.cuda.manual_seed(shared_seed)
            else:
                torch.cuda.manual_seed(priv_seed)

            self.joint_cuda_sampler_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(curr_cuda_prng_state)
        else:
            self.joint_cuda_sampler_state = None
        
        if use_joint_sampling:
            torch.manual_seed(shared_seed)
        else:
            torch.manual_seed(priv_seed)
        
        self.joint_sampler_state = torch.get_rng_state()
        torch.set_rng_state(curr_prng_state)

        self.data_indices = client_data_indices
        self.full_data_size = int(full_data_size)
        self.batch_size = batch_size
        self.batch_proc_size = batch_proc_size
        self.drawn_batch_sizes = []
        self.use_cuda=use_cuda
        self.trainset = trainset
        self.poisson_sampling_fraction = poisson_sampling_fraction
        
        if poisson_sampling_fraction > 0:
            self.Bin = torch.distributions.binomial.Binomial(total_count=len(self.data_indices), probs=torch.tensor([poisson_sampling_fraction]))


        
    def draw_batch(self):
        # Note: need to match between clients all eiter cpu or gpu; cpu & gpu random generators are not synced even when using common seed; currently assume all cpu
        if self.use_cuda:
            curr_cuda_prng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.joint_cuda_sampler_state)
        curr_prng_state = torch.get_rng_state()
        torch.set_rng_state(self.joint_sampler_state)

        if self.poisson_sampling_fraction == 0:
            full_batch = torch.randperm(n=self.full_data_size)[:self.batch_size] #.tolist()
            inds1 = torch.zeros(self.full_data_size,dtype=torch.uint8)
            inds1[self.data_indices] = 1
            inds2 = torch.zeros(self.full_data_size,dtype=torch.uint8)
            inds2[full_batch] = 1
            inds = (torch.arange(self.full_data_size))[inds1*inds2]
            inds -= self.data_indices[0] # since dataset currently starts from 0 for each client
            
        else:
            inds = torch.randperm(n=len(self.data_indices))[:int(self.Bin.sample())]
        

        self.drawn_batch_sizes.append(len(inds))
        # pad with zeros so batch_proc_size is ok
        #print('drawn batch:{}, batch proc:{}, tb % bp:{}'.format(self.drawn_batch_sizes[-1],self.batch_proc_size,self.drawn_batch_sizes[-1]%self.batch_proc_size ))

        if self.drawn_batch_sizes[-1] % self.batch_proc_size == 0:
            padded_size = self.drawn_batch_sizes[-1]
        else: 
            padded_size = self.drawn_batch_sizes[-1] + self.batch_proc_size - self.drawn_batch_sizes[-1] % self.batch_proc_size
        
        data = torch.zeros((padded_size, *data_dims ))
        target = torch.zeros((padded_size))-100 # set NLLLoss to ignore these values
        #print('data shape: {}'.format(data.shape))
        assert(data.shape[0] % self.batch_proc_size == 0 and target.shape[0] % self.batch_proc_size == 0)
        
        for i,ind in enumerate(inds):
            data[i,:,:,:], target[i] = self.trainset[ind][0], self.trainset[ind][1]
        self.joint_sampler_state = torch.get_rng_state()
        torch.set_rng_state(curr_prng_state)
        return data, target.long()


def calculate_grads(model1, fc_model, joint_sampler, loss_function, all_params, run_id, dim_red_rng_state, use_cuda):
    # calculate new grads
    if model1 is not None:
        model1.train()
    fc_model.train()
    loss_tot = 0
    
    # construct bogus model if doing projection
    if all_params['dim_reduction'] > 0:
        bogus_model = BogusModel(fc_model.batch_size, fc_model.batch_proc_size, torch.zeros(all_params['dim_reduction'],1).expand(1,-1,1))
    else:
        bogus_model = None

    data, target = joint_sampler.draw_batch()

    # in case client draws 0 samples
    if all_params['debug']:
        print('Client {} batch size {}'.format(run_id,target.shape[0]))
    
    if target.shape[0] == 0:
        # need to keep projection randomness in sync even when sending just zeros
        if all_params['dim_reduction'] > 0:
            vectorized = torch.zeros((1,fc_model.total_params))
            _, dim_red_rng_state = do_proj_in_steps(vector_for_proj=vectorized, model=fc_model, dim_reduction=all_params['dim_reduction'], max_proj_size=all_params['max_proj_size'], proj_type=all_params['proj_type'], dim_red_rng_state=dim_red_rng_state)

        return False, bogus_model, dim_red_rng_state

    
    data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
    
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    
    cum_grads = od()
    for i,p in enumerate(fc_model.parameters()):
        if p.requires_grad:
            if use_cuda:
                cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]),requires_grad=False).cuda()
            else:
                cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]),requires_grad=False)
    

    for i_batch in range(len(target)//joint_sampler.batch_proc_size):
           
        data_proc = data[i_batch*joint_sampler.batch_proc_size:(i_batch+1)*joint_sampler.batch_proc_size,:]
        target_proc = target[i_batch*joint_sampler.batch_proc_size:(i_batch+1)*joint_sampler.batch_proc_size]
        if model1 is not None:
            output1 = model1(data_proc)
            output2 = fc_model(output1)
        else:
            output2 = fc_model(data_proc)
      
        loss = loss_function(output2,target_proc)
        loss_tot += loss.data
        
        loss.backward()
        
        # accumulate (clipped) grads
        px_expander.acc_scaled_grads(model=fc_model,C=all_params['grad_norm_max'], cum_grads=cum_grads, use_cuda=use_cuda)
        
        fc_model.zero_grad()

    # construct bogus model & project if necessary
    if all_params['dim_reduction'] > 0:
        bogus_model = BogusModel(fc_model.batch_size, fc_model.batch_proc_size, torch.zeros(all_params['dim_reduction'],1).expand(1,-1,1))
        # vectorize grads & do projection
        vectorized = torch.zeros(1, fc_model.total_params)
        for i,key  in enumerate(cum_grads) :
            vectorized[:,fc_model.layer_summed[i]-fc_model.layer_params[i]:fc_model.layer_summed[i]] \
                        = cum_grads[key].view(-1,fc_model.layer_params[i])

        apu, dim_red_rng_state = do_proj_in_steps(vector_for_proj=vectorized, model=fc_model, dim_reduction=all_params['dim_reduction'], max_proj_size=all_params['max_proj_size'], proj_type=all_params['proj_type'], dim_red_rng_state=dim_red_rng_state)
        bogus_model.parameters()[0].grad[0,:,:] = apu.view((all_params['dim_reduction'],1))
        
    # add DP noise
    if all_params['dim_reduction'] > 0:
        # add dp noise in low dim
        if all_params['proj_type'] == 1: # no clip after proj, increase delta & noise instead
            # projection matrix includes constant
            bogus_model.parameters()[0].grad += (all_params['noise_sigma']*all_params['neighbour_const']*all_params['proj_sens'])*torch.randn(all_params['dim_reduction'],1).view(1,-1,1)
        
        elif all_params['proj_type'] == 2: 
            # Note: multiply proj. const. at master
            bogus_model.parameters()[0].grad += (all_params['noise_sigma']*all_params['neighbour_const']*all_params['proj_norm_max'])*torch.randn(all_params['dim_reduction'],1).view(1,-1,1)
        #noise_vec.view(1,-1,1)
        bogus_model.parameters()[0].grad /= bogus_model.batch_size
        
    else: # no projection
        px_expander.add_noise_with_cum_grads(model=fc_model, C=all_params['grad_norm_max'], neighbour_const=all_params['neighbour_const'], sigma=all_params['noise_sigma'], cum_grads=cum_grads, use_cuda=use_cuda)

    return True, bogus_model, dim_red_rng_state


parser = argparse.ArgumentParser(description='Distributed test')
parser.add_argument('--run_id', type=int, default=1, metavar='N', 
                    help='run_id')
parser.add_argument('--batch_size', type=int, default=-1, metavar='N',
                    help='batch size, overrides master cfg if given')
parser.add_argument('--batch_proc_size', type=int, default=1, metavar='N',
                    help='batch proc size for expander')
parser.add_argument('--conf_file', type=str, default=None, metavar='N', 
                    help='configuration filename')
parser.add_argument('--cluster_id', type=str, default=None, metavar='N', 
                    help='cluster id')
parser.add_argument('--cluster', type=str, default='ukko2', metavar='N', 
                    help='which cluster to run on (ukko2 or puhti)')
parser.add_argument('--run_locally', type=str, default='yes', metavar='N', 
                    help='run locally or on cluster')
parser.add_argument('--data', type=str, default='cifar', metavar='N', 
                    help='cifar or mnist')
parser.add_argument('--priv_seed', type=int, default=1, metavar='N', 
                    help='pytorch private seed for client')
args = parser.parse_args()

if args.run_locally == 'yes':
    run_locally = True
else:
    run_locally = False

if args.cluster_id is not None:
    cluster_id = args.cluster_id
else:
    cluster_id = '1'

if run_locally:
    folder_prefix = 'temp/'
else:
    # ukko2
    if cluster_id == '':
        folder_prefix = '/wrk/users/mixheikk/distributed_testing/temp/'
    else:
        if args.cluster == 'ukko2':
            folder_prefix = '/wrk/users/mixheikk/dist{}/temp/'.format(cluster_id)
        # csc puhti
        elif args.cluster == 'puhti':
            folder_prefix = '/scratch/project_2001003/dist{}/temp/'.format(cluster_id)
        else:
            sys.exit('Unknown cluster name {}'.format(args.cluster))

if args.conf_file is None:
    args.conf_file = folder_prefix+'config.pickle'

# read config file
while True:
    try:
        with open(args.conf_file,'rb') as f:
            all_params = pickle.load(f)
            print('Client {} all_params read.'.format(args.run_id))
            break
    except FileNotFoundError as err:
        pass
    time.sleep(.5)

if args.batch_size > 0:
    print('NOTE: Client {} batch size set with cmd-line option, not using joint sampler (nor Poisson sampling)!'.format(args.run_id))
    batch_size = args.batch_size
    use_joint_sampler = False
else:
    batch_size = all_params['batch_size']
    if all_params['poisson_sampling_fraction'] > 0:
        use_joint_sampler = False
    else:
        use_joint_sampler = True

if args.batch_proc_size > 0:
    batch_proc_size = args.batch_proc_size
else:
    batch_proc_size = 1

priv_seed = args.priv_seed*1314*args.run_id
np.random.seed(priv_seed)
torch.manual_seed(priv_seed)

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    #print('Client {} using cuda'.format(args.run_id))
    torch.cuda.manual_seed( priv_seed )
    print('Client {} NOT using cuda'.format(args.run_id))
    use_cuda = False
    if args.data == 'cifar':
        data_dir = '../data/CIFAR10/'
    else:
        data_dir = '../data/MNIST/'
else:
    print('Client {} not using cuda'.format(args.run_id))
    use_cuda = False
    if run_locally:
        if args.data == 'cifar':
            data_dir = '~/Documents/DL_datasets/CIFAR10/'
        else:
            data_dir = '~/Documents/DL_datasets/MNIST/'
    else:
        if args.data == 'cifar':
            data_dir = '../data/CIFAR10/'
        else:
            data_dir = '../data/MNIST/'


dataset_size = all_params['client_dataset_sizes'][args.run_id-1]
sample_start_index = int(np.sum( all_params['client_dataset_sizes'][:(args.run_id-1)]))
if all_params['debug']:
    print('Client {}: sample start: {}, data amount: {}'.format(args.run_id,sample_start_index, dataset_size))
data_indices = torch.arange(sample_start_index, sample_start_index + dataset_size,1)
#print(len(data_indices), data_indices)

if args.data == 'cifar':
    transform_train = transforms.Compose([
      #transforms.RandomCrop(32, padding=4),
      #transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    trainset = torch.utils.data.Subset(torchvision.datasets.CIFAR10(root=data_dir, train=True,
                            download=True, transform=transform_train), data_indices)
else:
    transform_train = transforms.Compose([
      #transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,)),
    ])
    trainset = torch.utils.data.Subset(torchvision.datasets.MNIST(root=data_dir, train=True,
                            download=True, transform=transform_train), data_indices)

if all_params['debug']:
    print('Client {} training data size: {}'.format( args.run_id,len(trainset)))

if args.data == 'cifar':
    data_dims = (3,32,32)
else:
    data_dims = (1,28,28)

if use_joint_sampler:
    joint_sampler = Joint_sampler(trainset, shared_seed=all_params['shared_seed'], client_data_indices=data_indices, full_data_size=all_params['total_dataset_size'], batch_size=batch_size, batch_proc_size=batch_proc_size, use_cuda=use_cuda, use_joint_sampling=True, priv_seed=None, data_dims=data_dims)
else:
    if all_params['poisson_sampling_fraction'] > 0:
        joint_sampler = Joint_sampler(trainset, shared_seed=all_params['shared_seed'], client_data_indices=torch.arange(0,dataset_size,1), full_data_size=dataset_size, batch_size=batch_size, batch_proc_size=batch_proc_size, use_cuda=use_cuda, use_joint_sampling=False, priv_seed=priv_seed, data_dims=data_dims, poisson_sampling_fraction=all_params['poisson_sampling_fraction'])

    else:
        joint_sampler = Joint_sampler(trainset, shared_seed=all_params['shared_seed'], client_data_indices=torch.arange(0,dataset_size,1), full_data_size=dataset_size, batch_size=batch_size, batch_proc_size=batch_proc_size, use_cuda=use_cuda, use_joint_sampling=False, priv_seed=priv_seed, data_dims=data_dims)


loss_function = nn.NLLLoss(size_average=False, ignore_index=-100) # ignore_index matches joint sampler

pairwise_secrets = get_pairwise_secrets(all_params, args.run_id)

# clear old kill pings
try:
    os.remove(all_params['kill_ping_file']+str(args.run_id))
    pass
except FileNotFoundError as err:
    pass

if args.data == 'cifar':
    # conv net
    model1 =  Net1()

    # fc net
    fc_model = FCNet(batch_size=batch_size, batch_proc_size=batch_proc_size, latent_dim=all_params['latent_dim'], n_hidden_layers=all_params['n_hidden_layers'], output_dim=all_params['output_dim'], randomize_data=all_params['randomize_data'])

    # Load the pre-trained convolutive layers
    tb_save = torch.load(all_params['conv_filename'], map_location='cpu')
    for ii,p in enumerate(model1.parameters()):
        #print('Client {} setting convolution params for layer {}, shape {}'.format(args.run_id, ii,p.shape))
        p.data = tb_save[ii].clone()
        p.requires_grad_(False)

else:
    model1 = None
    fc_model = FCNet_MNIST(batch_size=batch_size, batch_proc_size=batch_proc_size, latent_dim=all_params['latent_dim'], n_hidden_layers=all_params['n_hidden_layers'], output_dim=all_params['output_dim'], randomize_data=all_params['randomize_data'])


# load and expand fc_net weights
def load_weights(model, filename):
    w = torch.load(filename)
    for i,p in enumerate(zip(model.parameters(),w)):
      #print(i,p[0].shape, w[p[1]].shape)
      if p[0] is not None and w[p[1]] is not None:
        p[0].data.copy_( w[p[1]].data.clone().repeat(model.batch_proc_size,1,1) )

# Note: assume all parties CPU only for dim_red sampler
if all_params['dim_reduction'] > 0:
    if use_cuda:
        sys.exit('Can\'t currently do dim red with cuda! Aborting..')
    curr_rng_state = torch.get_rng_state()
    torch.manual_seed(all_params['dim_red_seed'])
    dim_red_rng_state = torch.get_rng_state() #, None]
    torch.set_rng_state(curr_rng_state)

    fc_model.layer_params = []
    fc_model.layer_summed = []
    for i,p in enumerate(fc_model.parameters()):
        fc_model.layer_params.append(int(p.data[0].numel()))
        fc_model.layer_summed.append(np.sum(fc_model.layer_params[:i+1]))
    fc_model.total_params = np.sum(fc_model.layer_params)
else:
    dim_red_rng_state = None

def check_for_weights(model, all_params, run_id):
    
    try:
        with open(all_params['weights_ping_file']+str(run_id),'r') as f:
            pass
        # remove read ping
        if not all_params['debug']:
            os.remove(all_params['weights_ping_file']+str(run_id)) 
        load_weights(model, all_params['sent_weights_file'])
        #print('Client {} weights updated'.format(run_id))
        return True
    
    except FileNotFoundError as err:
        #print(sys.exc_info())
        pass
    return False


all_crypto_times = []

# main loop: read current params, calculate grads, write encrypted grads back
params_updated = False
generate_noise = True

while True:

    check_kill_ping(all_params['kill_ping_file'], all_params, joint_sampler, all_crypto_times, args.run_id, all_params['print_crypto_times'])
    # generate noise for crypto
    if generate_noise:
        crypto_noise = do_noise_generation(all_params, args.run_id, pairwise_secrets, all_crypto_times)
        generate_noise = False

    params_updated = check_for_weights(fc_model, all_params, args.run_id)
    
    if params_updated:
        include_grads, bogus_model, dim_red_rng_state = calculate_grads(model1, fc_model, joint_sampler, loss_function, all_params, args.run_id, dim_red_rng_state, use_cuda)
        
        # send either full model grads or projected grads
        if bogus_model is None:
            model_for_crypto = fc_model
        else:
            model_for_crypto = bogus_model

        # fat clients with basic secret sharing
        if all_params['scheme_type'] == 0:
            send_grads_fat(model_for_crypto, all_params, args.run_id, include_grads, crypto_noise)
        # thin clients
        elif all_params['scheme_type'] == 1:
            send_grads_thin(model_for_crypto, all_params, args.run_id, include_grads, crypto_noise)
        # fat clients with pairwise crypto
        elif all_params['scheme_type'] == 2:
            send_grads_fat_pairwise(model_for_crypto, all_params, args.run_id, include_grads, crypto_noise, pairwise_secrets)
        else:
            sys.exit('Unknown scheme type!')
        fc_model.zero_grad()
        params_updated = False
        generate_noise = True
        generate_proj = True
        bogus_model = None
        
    time.sleep(all_params['loop_wait_time'])


