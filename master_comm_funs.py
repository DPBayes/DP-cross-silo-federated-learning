'''
Communication functions used by the Master for distributed DP learning
'''

import numpy as np
import torch
import sys

def kill_clients(client_list, filename):
    # send kill ping to all Clients
    print('Master sending kill ping')
    for k in client_list:
        with open(filename+str(k), 'w') as f:
            pass

def parse_fat_grads(model, partial_sums, all_params):

    if all_params['use_encryption']:
        # decrypted grads are given by summing over the partial sums from all Clients
        init = True
        for k in partial_sums:
            for i,p in enumerate(model.parameters()):
                if p is not None:
                    if init:
                        p.grad = torch.zeros_like(p)
                    p.grad = torch.remainder(p.grad+((partial_sums[k][str(i)]).to(torch.float32)).expand_as(p), all_params['modulo'])
            init = False
        for p in model.parameters():
            if p is not None:
                p.grad = 1/all_params['fixed_point_int']*(p.grad -  len(all_params['client_list'])*all_params['offset'])

    else:
        for i,p in enumerate(model.parameters()):
            if p is not None:
                p.grad = torch.zeros_like(p)
                for client in all_params['client_list']:
                    p.grad[0] += all_msg[str(client)][str(i)]

    if all_params['debug']:
        true_models = []
        true_grads = {}
        for k in all_params['client_list']:
            true_models.append( torch.load(all_params['sent_grads_file']+str(k)+'_debug'))
        for i,p in enumerate(model.parameters()):
            if p is not None:
                true_grads[str(i)] = torch.zeros_like(p)
                for m in true_models:
                    true_grads[str(i)] += m[str(i)]
                
        max_err = 0
        for i,p in enumerate(model.parameters()):
            apu = torch.max(torch.abs(p.grad-true_grads[str(i)]))
            if apu > max_err:
                max_err = apu
            print(p.grad.shape)
            print(p.grad[0,:3,:3])
            print('should be:')
            print(true_grads[str(i)][0,:3,:3])
        print('max err in grads: {}'.format(max_err))
            #break
        kill_clients(all_params['client_list'], all_params['kill_ping_file'])
        sys.exit()

def parse_fat_grads_pairwise(model, all_msg, all_params):

    if all_params['use_encryption']:
        # decrypted grads are given by summing over all the contributions
        for i,p in enumerate(model.parameters()):
            if p is not None:
                grads = np.zeros_like(p.detach().numpy(), dtype='uint64')
                for k in all_msg.keys():
                    grads = np.remainder(grads+all_msg[k][str(i)].reshape(p.detach().numpy().shape), all_params['modulo'])
                grads = 1/all_params['fixed_point_int']*grads.astype('float64') - all_params['offset']*len(all_params['client_list'])
                p.grad = torch.tensor(grads, dtype=torch.float32)
    else:
        for i,p in enumerate(model.parameters()):
            if p is not None:
                p.grad = torch.zeros_like(p)
                for client in all_params['client_list']:
                    p.grad[0] += all_msg[str(client)][str(i)]

    if all_params['debug']:
        true_models = []
        true_grads = {}
        for k in all_params['client_list']:
            true_models.append( torch.load(all_params['sent_grads_file']+str(k)+'_debug'))
        for i,p in enumerate(model.parameters()):
            if p is not None:
                true_grads[str(i)] = torch.zeros_like(p)
                for m in true_models:
                    true_grads[str(i)] += m[str(i)]
                
        max_err = 0
        for i,p in enumerate(model.parameters()):
        #for i,p in enumerate(grads):
            apu = torch.max(torch.abs(p.grad[0]-true_grads[str(i)]))
            if apu > max_err:
                max_err = apu
            print(p.grad.shape)
            print(p.grad[0,:3,:3])
            print('should be:')
            print(true_grads[str(i)][0,:3,:3])
        print('max err in grads: {}'.format(max_err))
            #break
        kill_clients(all_params['client_list'], all_params['kill_ping_file'])
        sys.exit()

def parse_thin_grads(model, msg, all_params):    
    # simulate separate Compute nodes: first calculate partial sums for each Compute,
    # then sum over the partial sums broadcasted by each Compute
    
    partial_sums = {}
    if all_params['use_encryption']:

        for i_c in range(all_params['n_computes']):
            partial_sums[str(i_c)] = {}
            for i_param, p in enumerate(model.parameters()):
                if p is not None:
                    partial_sums[str(i_c)][str(i_param)] = np.zeros_like(p.detach().numpy(), dtype='uint64')
                    # add each msg 
                    for client in all_params['client_list']:
                        partial_sums[str(i_c)][str(i_param)] = np.remainder(partial_sums[str(i_c)][str(i_param)] + msg[str(client)][str(i_param)][i_c], all_params['modulo'])
        
        # decrypted grads are given by summing over the partial sums from all Computes
        init = True
        for i,p in enumerate(model.parameters()):
            if p is not None:
                for k in partial_sums:
                    if init:
                        apu = np.zeros_like(p.detach().numpy(),dtype='uint64')
                        init = False
                    apu = np.remainder(apu+partial_sums[k][str(i)], all_params['modulo'])
                p.grad = torch.as_tensor(1/all_params['fixed_point_int']*apu - len(all_params['client_list'])*all_params['offset'], dtype=torch.float32)
                init = True
    
    else: 
        for i,p in enumerate(model.parameters()):
            if p is not None:
                p.grad = torch.zeros_like(p)
                for client in all_params['client_list']:
                    p.grad[0] += msg[str(client)][str(i)]


    if all_params['debug']:
        true_models = []
        true_grads = {}
        for k in all_params['client_list']:
            true_models.append( torch.load(all_params['sent_grads_file']+str(k)+'_debug'))
        for i,p in enumerate(model.parameters()):
            if p is not None:
                true_grads[str(i)] = torch.zeros_like(p)
                for m in true_models:
                    true_grads[str(i)] += m[str(i)]
                
        max_err = 0
        for i,p in enumerate(model.parameters()):
            apu = torch.max(torch.abs(p.grad-true_grads[str(i)]))
            if apu > max_err:
                max_err = apu
            print(p.grad.shape)
            print(p.grad[0,:3,:3])
            print('should be:')
            print(true_grads[str(i)][0,:3,:3])
        print('max err in grads: {}'.format(max_err))
        kill_clients(all_params['client_list'], all_params['kill_ping_file'])
        sys.exit()

def send_weights(model, all_params):
    # send current params to all parties and ping
    torch.save(model.state_dict(), all_params['sent_weights_file'])
    for k in all_params['client_list']:
        with open(all_params['weights_ping_file']+str(k), 'w') as f:
            pass






