'''
Script for running distributed DP master node
'''

import copy
import datetime
import numpy as np
import os
import pickle
import sys
import time
import logging
from collections import OrderedDict as OD
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

import linear
import itertools
from types import SimpleNamespace
import px_expander

from master_comm_funs import * 
from utils import BogusModel
from utils import FCNet_MNIST
from utils import FCNet
from utils import Net1
from utils import generate_proj_matrix_piece

# measure time needed for running the whole training from start to finish
start = time.time()

parser = argparse.ArgumentParser(description='Distributed CIFAR test')
parser.add_argument('--run_id', type=int, default=0, metavar='N',
                    help='run_id')
parser.add_argument('--conf_file', type=str, default=None, metavar='N', 
                    help='configuration filename')
parser.add_argument('--n_clients', type=int, default=2, metavar='N',
                    help='number of client nodes')
parser.add_argument('--cluster_id', type=str, default=None, metavar='N', 
                    help='cluster id')
parser.add_argument('--cluster', type=str, default='ukko2', metavar='N', 
                    help='which cluster to run on (available ukko2 or puhti)')
parser.add_argument('--run_locally', type=str, default='yes', metavar='N', 
                    help='run locally or on cluster')
parser.add_argument('--data', type=str, default='cifar', metavar='N', 
                    help='cifar or mnist')
parser.add_argument('--priv_seed', type=int, default=1, metavar='N', 
                    help='prng seed for master')

# Note: number of clients needs to match with master.cfg when running on cluster
args = parser.parse_args()

if args.run_locally == 'yes':
    run_locally = True
else:
    run_locally = False

if args.cluster_id is not None:
    cluster_id = args.cluster_id
else:
    cluster_id = '1'

all_params = OD()
all_params['n_hidden_layers'] = 2
all_params['latent_dim'] = 10#384 # used for all hidden layers; test in the paper use 384 for CIFAR10, 536 for MNIST
all_params['output_dim'] = 10
# Note: Cifar data uses pre-trained conv layers before FCs, MNIST only FC layers

all_params['batch_size'] = 400 # this is used as a global batch size; can be overridden by cmd-line arguments for each client for LDP comparisons; Note: ignored if poisson_sampling_fraction is not 0
all_params['poisson_sampling_fraction'] = .008 # set to 0 for sampling without replacement

# DP accounting is done in separate script with Fourier accountant
all_params['grad_norm_max'] = 2.5 # don't use with sparse projection [Note: this is used for clipping grads; for DP noise the code uses neighbour_const*grad_norm_max, which gives unbounded/bounded DP]
all_params['noise_sigma'] = 2/np.sqrt(args.n_clients) # noise sigma that EACH party will add
all_params['neighbour_const'] = 2 # DP neighbouring relation def, 1=unbounded, 2=bounded DP

all_params['n_train_iters'] = 4#125 # number of training batches to run before calculating test set error (length of one training epoch)
all_params['n_epochs'] = 1#10

all_params['l_rate'] = 9e-4

all_params['dim_reduction'] = 10  # dim reduction by random projection to given dim, set to 0 for no dim reduction
all_params['dim_red_seed'] = 1606*args.priv_seed # joint seed for dim reduction projections
all_params['proj_type'] = 1 # 1=N(0,1), 2=sparse projection matrix; set to 1
all_params['max_proj_size'] = 10#50 # to save on memory, avoid storing entire proj matrices in memory, instead generate (d*this val) matrices several times, NOTE: dim_reduction needs to be a multiple of this value! Generating proj in pieces takes more time than doing it on one go
all_params['proj_const'] = 0 #  s in sparse proj paper, set to 0 for sqrt(dim orig grad), no effect on N(0,1), don't use this
all_params['proj_norm_max'] = None # don't use this; only for clipping projected grads, set to None with N(0,1) proj, NOTE: as with grad_norm_max, this is for clipping, actual noise sens. will be neighbour_const*this

all_params['delta_prime'] = 5e-6 # when using N(0,1) proj, instead of clipping the projected grads, increase delta: delta_total = delta + delta', this increases noise depending on proj dim


if run_locally:
    folder_prefix = 'temp/'
else:
    if cluster_id == '':
        # ukko2
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

all_params['conf_file'] = args.conf_file
all_params['conv_filename'] = 'conv_layers.pt' #pretrained convolutions
all_params['sent_weights_file'] = folder_prefix+'new_weights.msg' # master sends new weights
all_params['sent_grads_file'] = folder_prefix+'grad_msg_' # clients send encrypted grads
all_params['weights_ping_file'] = folder_prefix+'weights_ping_' # ping clients to read new weights
all_params['grads_ping_file'] = folder_prefix+'grads_ping_' # ping master to read new grads
all_params['kill_ping_file'] = folder_prefix+'kill_ping_' # ping clients to terminate
all_params['client_list'] = list(range(1,args.n_clients+1))
all_params['res_save_file'] = folder_prefix+'res_'

if args.data == 'cifar':
    #all_params['client_dataset_sizes'] = [5000]#, 10000, 10000, 10000, 10000]
    all_params['client_dataset_sizes'] = (np.zeros(args.n_clients,dtype=int)+50000//(args.n_clients)).tolist()
else:
    all_params['client_dataset_sizes'] = (np.zeros(args.n_clients,dtype=int)+60000//(args.n_clients)).tolist()
# client 0 uses samples 0:(datasize[0]-1), 1 datasize[0]:(datasize[1]-1) etc
# should have an entry for each client, and cannot exceed total number of samples

all_params['shared_seed'] = 16361*args.priv_seed # seed for generating distributed batches, shared by all clients
all_params['loop_wait_time'] = .01 # general sleep time in waiting for msgs
all_params['master_max_wait'] = 1000000 # max iters to wait for grad msgs

all_params['scheme_type'] = 2 # 1 = thin clients, 2 = fat clients (pairwise encryption)

# scheme 1 specific
all_params['n_computes'] = 2 # number of compute nodes for scheme 1, must be > 1
all_params['use_hash_in_scheme1'] = False # Note: not properly tested with current code
# if False use urandom numbers, else generate randomness with Blake2 seeded from urandom

# scheme 2 specific
#all_params['n_encrypted_scheme2'] = len(all_params['client_list']) # groupsize for pairwise encryption
all_params['n_encrypted_scheme2'] = 2
#   Note: has to be > 1 and divide number of parties

# both schemes
all_params['use_encryption'] = True # if False just send unencrypted gradients
all_params['fixed_point_int'] = 1*10**6
all_params['modulo'] = 2*10**9 # modulo for encryption; should be larger than fixed point, small-enough to be done with int32; expect problems if 2*modulo is too close to int32 max value, there's no checking in the code for this
all_params['offset'] = 5  # max abs value for grads, enforced by clamping before encryption

all_params['randomize_data'] = True

all_params['optimiser'] = 'Adam' # 'SGD' or 'Adam'
all_params['optimiser_kwargs'] = {}
# add optional arguments for optimiser here
if all_params['optimiser'] == 'SGD':
    all_params['optimiser_kwargs']['momentum'] = 0

all_params['lr_scheduler'] = False # use lr scheduler on plateau
all_params['scheduler_kwargs'] = {}
all_params['scheduler_kwargs']['factor'] = 0.2
all_params['scheduler_kwargs']['patience'] = 2
all_params['scheduler_kwargs']['verbose'] = True
all_params['scheduler_kwargs']['cooldown'] = 2
all_params['scheduler_kwargs']['min_lr'] = 4e-5


############################################################
# some debug options
all_params['debug'] = False # print grads, NOTE: using scheme=1, no enc fails

all_params['use_true_entropy'] = True # set to False to use PRNG with a fixed seed for encryption, doesn't currently work with scheme=2
all_params['print_crypto_times'] = False # print encryption times when finishing

############################################################
# END SETUP
############################################################



def run_train_loop(model1, fc_model, optimizer, all_params, epoch=None, run_times=None, dim_red_rng_state=None):

    for t in range(all_params['n_train_iters']):
        print('Master starting training iteration {} on epoch {}'.format(t,epoch))
        start_iter = time.time()

        if not all_params['debug']:
            # clear old grads
            for k in all_params['client_list']:
                try:
                    os.remove(all_params['sent_grads_file']+str(k))
                    pass
                except FileNotFoundError as err:
                    pass
        
        optimizer.zero_grad()
        
        # send current params to all parties and ping
        send_weights(fc_model, all_params)

        # wait for gradients
        waiting_dict = {key: 1 for key in all_params['client_list']}
        #print(waiting_dict, len(waiting_dict))
        
        all_msg = {}
        #print('Master waiting for grads on epoch {}'.format(epoch))
        for i_wait in range(all_params['master_max_wait']):
            
            time.sleep(all_params['loop_wait_time'])
            found_grads = []
            for k in waiting_dict.keys():
                try:
                    with open(all_params['grads_ping_file']+str(k),'r') as f:
                        pass
                    with open(all_params['sent_grads_file']+str(k), 'rb') as f:
                        all_msg[str(k)] = pickle.load(f)
                    #print('Read grad msg from client {}'.format(k))
                    if not all_params['debug']:
                        os.remove(all_params['grads_ping_file']+str(k))
                        os.remove(all_params['sent_grads_file']+str(k))
                    
                    found_grads.append(k)
                    
                except FileNotFoundError as err:
                    pass
            for k in found_grads:
                waiting_dict.pop(k)
            if len(waiting_dict) == 0:
                break
            
            if i_wait == all_params['master_max_wait']-1:
                kill_clients(all_params['client_list'], all_params['kill_ping_file'])
                sys.exit('Master didn\'t get all grads, waiting for {}! Aborting..'.format(waiting_dict.keys()))
        
        # use a bogus model for reading projected grad msgs
        if all_params['dim_reduction'] > 0:
            bogus_model = BogusModel(fc_model.batch_size, fc_model.batch_proc_size, torch.zeros(all_params['dim_reduction'],1).expand(1,-1,1))
            model_for_parsing = bogus_model
        else:
            model_for_parsing = fc_model
        
        # parse grads from messages
        if all_params['scheme_type'] == 0:
            parse_fat_grads(model_for_parsing, all_msg, all_params)
        elif all_params['scheme_type'] == 1:
            parse_thin_grads(model_for_parsing, all_msg, all_params)
        elif all_params['scheme_type'] == 2:
            parse_fat_grads_pairwise(model_for_parsing, all_msg, all_params)
        else:
            kill_clients(all_params['client_list'], all_params['kill_ping_file'])
            sys.exit('Unknown scheme type!')
        
        # project gradients back to high dim if necessary
        if all_params['dim_reduction'] > 0:
             
            param_vec = torch.zeros(fc_model.total_params)
            # use shared randomness for generating proj matrices
            curr_rng_state = torch.get_rng_state()
            torch.set_rng_state(dim_red_rng_state)
            
            for i_proj in range(all_params['dim_reduction']//all_params['max_proj_size']):
                proj_matrix = generate_proj_matrix_piece(model=fc_model, dim_reduction=all_params['dim_reduction'], max_proj_size=all_params['max_proj_size'], proj_type=1)                

                if all_params['proj_type'] == 1:
                    param_vec += torch.mm(proj_matrix, (bogus_model.parameters()[0].grad)[0,i_proj*all_params['max_proj_size']:(i_proj+1)*all_params['max_proj_size']].reshape(-1,1)).view(fc_model.total_params)
                elif all_params['proj_type'] == 2:
                    sys.exit('proj 2 not implemented on master!')
                else:
                    sys.exit('Unknown proj type: {}!'.format(all_params['proj_type']))
            proj_matrix = None

            # return to non-shared randomness
            dim_red_rng_state = torch.get_rng_state()
            torch.set_rng_state(curr_rng_state)

            for i,p in enumerate(fc_model.parameters()):
                if p.requires_grad:
                    p.grad = param_vec[fc_model.layer_summed[i]-fc_model.layer_params[i]:fc_model.layer_summed[i]].reshape(p.size()).detach()

        '''
        # check grads
        for i,p in enumerate(fc_model.parameters()):
            if p.requires_grad:
        #        print(p.grad)
                print(p.grad.norm())
        #        sys.exit()
        '''

        # take optimizer step
        optimizer.step()

        if run_times is not None:
            run_times.append(time.time()-start_iter)

    return dim_red_rng_state


def test(model1, fc_model, epoch, all_params, data_dims):
    # calculate model test accuracy

    if model1 is not None:
        model1.eval()
    fc_model.eval()    
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        if data.shape[0] != fc_model.batch_size:
            temp = fc_model.batch_size - data.shape[0]            
            data = torch.cat((data, torch.zeros((temp, *data_dims))-100),dim=0 )
            target = torch.cat((target, torch.zeros(temp,dtype=torch.long)-100),dim=0)
        
        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        
        for i_batch in range(fc_model.batch_size//fc_model.batch_proc_size):
            data_proc = data[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size,:]
            target_proc = target[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size]
            if use_cuda:
                data_proc = data_proc.cuda()
                target_proc = target_proc.cuda()
            if model1 is not None:
                output1 = model1(data_proc)
                output2 = fc_model(output1)
            else:
                output2 = fc_model(data_proc)
            
            test_loss += F.nll_loss(output2, target_proc, size_average=False).item()
            pred = output2.data.max(1, keepdim=True)[1]
            correct += pred.eq(target_proc.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    acc = correct.numpy() / len(test_loader.dataset)
    print('\nTest set full: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),100. * acc))    
    
    return test_loss, acc


################################
# some checks & setting variables based on the setup

all_params['total_dataset_size'] = np.sum(all_params['client_dataset_sizes'][:args.n_clients])
print('Total amount of data over all clients: {}'.format(all_params['total_dataset_size']))

if all_params['poisson_sampling_fraction'] > 0:
    print('Using Poisson sampling with sampling prob. {}.'.format(all_params['poisson_sampling_fraction']))

assert all_params['neighbour_const'] in (1,2)

if all_params['max_proj_size'] == 0:
    all_params['max_proj_size'] = all_params['dim_reduction']
assert all_params['dim_reduction'] == 0 or all_params['dim_reduction'] % all_params['max_proj_size'] == 0

if all_params['dim_reduction'] == 0:
    print('Given sigma: {}, grad norm max: {}, neighbourhood relation: {}. Total noise variance without colluders using SMC:{}'.format(all_params['noise_sigma'],all_params['grad_norm_max'], all_params['neighbour_const'], args.n_clients*(all_params['neighbour_const']*all_params['grad_norm_max'])**2*all_params['noise_sigma']**2))
else:
    # with N(0,1) proj, calculate increase to DP noise
    if all_params['proj_type'] == 1:
        from scipy.stats import gamma
        if all_params['proj_norm_max'] is not None:
            print('Setting proj_norm_max to None for N(0,1) proj')
            all_params['proj_norm_max'] = None
        all_params['proj_sens'] = np.sqrt( gamma.ppf(1-all_params['delta_prime'], a=all_params['dim_reduction']/2,loc=0,scale=(2*(all_params['neighbour_const']*all_params['grad_norm_max'])**2/all_params['dim_reduction'])) )
        print('Using normal projection: k={}, C={}, delta\'={}, so increased sensitivity={}'.format(all_params['dim_reduction'],all_params['grad_norm_max'],all_params['delta_prime'],all_params['proj_sens']) )
        print('Given sigma: {}, proj.norm max: {}, neighbourhood relation: {}. Total noise variance without colluders: {}'.format(all_params['noise_sigma'], all_params['proj_sens'], all_params['neighbour_const'], args.n_clients*(all_params['neighbour_const']*all_params['proj_sens'])**2*all_params['noise_sigma']**2))
   
    # with sparse proj. use clipping after projection
    elif all_params['proj_type'] == 2:

        print('Given sigma: {}, proj norm max: {}. Total noise variance without colluders: {}'.format(all_params['noise_sigma'],all_params['proj_norm_max'], args.n_clients*(2*all_params['proj_norm_max'])**2*all_params['noise_sigma']**2))

if all_params['scheme_type'] == 1:
    assert all_params['n_computes'] > 1
assert all_params['fixed_point_int']*all_params['offset'] < all_params['modulo']

if all_params['scheme_type'] == 2:
    if all_params['n_encrypted_scheme2'] > len(all_params['client_list']):
            print('Too many pairwise encryption pairs set, using all pairs')
            all_params['n_encrypted_scheme2'] = len(all_params['client_list'])
    assert all_params['n_encrypted_scheme2'] > 1 or all_params['use_encryption'] is False

    # check that encryption pairs can be determined simply
    if np.remainder(len(all_params['client_list']), all_params['n_encrypted_scheme2']) != 0:
        sys.exit('Cannot handle dividing {} parties into non-overlapping groups of size {}'.format(len(all_params['client_list']),all_params['n_encrypted_scheme2'] ))
    print('Scheme 2 encryption group size: {}'.format(all_params['n_encrypted_scheme2']))

np.random.seed(17*args.priv_seed)
torch.manual_seed(16*args.priv_seed)

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    #print('Using cuda')
    torch.cuda.manual_seed( 15*args.priv_seed )
    use_cuda = False # no cuda implemented
    print('Master NOT using cuda')
    data_dir = '../data/'
else:
    print('Not using cuda')
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

# currently not using GPUs
batch_proc_size = 1
assert use_cuda is False


if args.data == 'cifar':
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform_test)
    data_dims = (3,32,32)
else:
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,)),
    ])
    testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                           download=True, transform=transform_test)
    data_dims =(1,28,28)

test_loader = torch.utils.data.DataLoader(testset, batch_size=all_params['batch_size'], shuffle=all_params['randomize_data'], num_workers=2)



# conv layers & FCs with cifar
if args.data == 'cifar':
    # conv net
    model1 =  Net1()

    # fully connected net
    fc_model = FCNet(batch_size=all_params['batch_size'], batch_proc_size=batch_proc_size, latent_dim=all_params['latent_dim'], n_hidden_layers=all_params['n_hidden_layers'], output_dim=all_params['output_dim'], 
    randomize_data=all_params['randomize_data'])

    # Load the pre-trained convolutive layers
    tb_save = torch.load(all_params['conv_filename'], map_location='cpu')
    for ii,p in enumerate(model1.parameters()):
        if all_params['debug']:
            print('setting convolution params for layer {}, shape {}'.format(ii,p.shape))
        p.data = tb_save[ii].clone()
        p.requires_grad_(False)

# only fully connected layers with MNIST
else:
    model1 = None
    fc_model = FCNet_MNIST(batch_size=all_params['batch_size'], batch_proc_size=batch_proc_size, latent_dim=all_params['latent_dim'], n_hidden_layers=all_params['n_hidden_layers'], output_dim=all_params['output_dim'], randomize_data=all_params['randomize_data'])

# set expander weights
for i,p in enumerate(fc_model.parameters()):
    if p is not None:
        if all_params['debug']:
            print('FC layer {} shape: {}'.format(i,p.shape))
        p.data.copy_( p[0].data.clone().repeat(batch_proc_size,1,1) )


if use_cuda:
    if model1 is not None:
        model1 = model1.cuda()
    fc_model = fc_model.cuda()

loss_function = nn.NLLLoss(size_average=False, ignore_index=-100)

if all_params['optimiser'] == 'SGD':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, fc_model.parameters()), lr=all_params['l_rate'], **all_params['optimiser_kwargs'])
elif all_params['optimiser'] == 'Adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, fc_model.parameters()), lr=all_params['l_rate'], **all_params['optimiser_kwargs'])
else:
    sys.exit('Unknown optimiser!')

if all_params['lr_scheduler']:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **all_params['scheduler_kwargs'])

# Note: change if using parameter expander
assert batch_proc_size == 1
all_params['total_params'] = sum(p.numel() for p in fc_model.parameters() if p.requires_grad)

print('Number of clients={}, latent dim={}, scheme type={}, encrypt={}, total params={}'.format(len(all_params['client_list']),all_params['latent_dim'], all_params['scheme_type'],all_params['use_encryption'],all_params['total_params'] ))

# Note: assume all parties CPU only for dim_red sampler
if all_params['dim_reduction'] > 0:
    assert not use_cuda
    print('Using dim reduction to {}'.format(all_params['dim_reduction']))
    curr_rng_state = torch.get_rng_state()
    torch.manual_seed(all_params['dim_red_seed'])
    dim_red_rng_state = torch.get_rng_state()
    torch.set_rng_state(curr_rng_state)

    fc_model.layer_params = []
    fc_model.layer_summed = []
    fc_model.total_params = 0
    for i,p in enumerate(fc_model.parameters()):
        if batch_proc_size > 1:
            fc_model.layer_params.append(int(p.data[0].numel()))
            fc_model.total_params += int(p.data[0].numel())
            fc_model.layer_summed.append(fc_model.total_params)
        else:
            fc_model.layer_params.append(int(p.data.numel()))
            fc_model.total_params += int(p.data.numel())
            fc_model.layer_summed.append(fc_model.total_params)
    if all_params['proj_const'] == 0:
        all_params['proj_const'] = np.sqrt(fc_model.total_params)
else:
    dim_red_rng_state = None


# save all parameters
print('Master writing all_params..')
with open(all_params['conf_file'],'wb') as f:
    pickle.dump(all_params,f)


accs = []
run_times = []
# main loop
for epoch in range(1,all_params['n_epochs']+1):
  dim_red_rng_state = run_train_loop(model1, fc_model, optimizer, all_params, epoch=epoch, run_times=run_times,dim_red_rng_state=dim_red_rng_state)
  
  loss, acc = test(model1, fc_model, epoch, all_params, data_dims)
  accs.append(acc)
  if all_params['lr_scheduler']:
    lr_scheduler.step(loss)
  
  
# kill clients
kill_clients(all_params['client_list'], all_params['kill_ping_file'])

end = time.time()
all_params['total_runtime'] = end-start
all_params['run_times'] = run_times

# save final results
np.save(all_params['res_save_file'] + str(all_params['l_rate']) + '_final', accs)
with open(all_params['res_save_file'] + str(all_params['l_rate'])+'_params.pickle', 'wb') as f:
    pickle.dump(all_params,f)
#print(all_params['run_times'])
print('Master all done! Total training time: {}s'.format(np.round(all_params['total_runtime'],2)))
