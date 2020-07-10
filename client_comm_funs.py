'''
Communication functions used by Clients for distributed DP learning
'''

from collections import OrderedDict as od
import hashlib
import numpy as np
import os
import pickle
import secrets
import time
import torch
import sys


def get_pairwise_secrets(all_params, run_id):
    # this can be done with any standard key-exchange; this is a toy implementation
    # which doesn't actually share any secrets; also uses simple fixing for determining the pairs
    
    #all_params['n_encrypted_scheme2']

    pairwise_secrets = od()
    pairwise_secrets['secrets'] = {}
    # use simple groups of first #group size etc
    group = np.ceil(run_id/all_params['n_encrypted_scheme2'])
    if run_id > 1:
        for k in range(run_id-1, 0,-1):
            if np.ceil(k/all_params['n_encrypted_scheme2']) == group:
                pairwise_secrets['secrets'][str(k)] = 'some_pairwise_secret'.encode('utf-8')
            else:
                break
    if run_id < len(all_params['client_list']):
        for k in range(run_id+1, len(all_params['client_list'])+1):
            if np.ceil(k/all_params['n_encrypted_scheme2']) == group:
                pairwise_secrets['secrets'][str(k)] = 'some_pairwise_secret'.encode('utf-8')
            else:
                break

    if all_params['debug']:
        print('all key-pairs for Client {}: {}'.format(run_id, pairwise_secrets))
    pairwise_secrets['round'] = 0
    return pairwise_secrets

def read_incoming_messages(all_params, run_id):
    # for message exchange using basic secret sharing
    # wait for messages
    waiting_dict = {key: 1 for key in all_params['client_list']}
    waiting_dict.pop(run_id)

    if all_params['debug']:
        print('Client {} waiting for messages..'.format(run_id))
    all_msg = {}
    for i_wait in range(all_params['master_max_wait']):
        
        time.sleep(all_params['loop_wait_time'])
        found_grads = []
        for k in waiting_dict.keys():
            try:
                with open(all_params['grads_ping_file']+str(k)+'-to-'+str(run_id),'r') as f:
                    pass
                all_msg[str(k)] = torch.load(all_params['sent_grads_file']+str(k)+'-to-'+str(run_id))
                #print('Read grad msg from client {}'.format(k))
                os.remove(all_params['grads_ping_file']+str(k)+'-to-'+str(run_id))
                os.remove(all_params['sent_grads_file']+str(k)+'-to-'+str(run_id))                
                found_grads.append(k)
                
            except FileNotFoundError as err:
                pass
        for k in found_grads:
            waiting_dict.pop(k)
        if len(waiting_dict) == 0:
            break
        
        if i_wait == all_params['master_max_wait']-1:
            sys.exit('Client {} didn\'t get all grads, waiting for {}! Aborting..'.format(run_id, waiting_dict.keys()))
    return all_msg

def parse_partial_grads(model, msg, all_params, run_id):
    # for parsing messages from other clients in basic secret sharing scheme
    partial_sums = {}
    for i_param, p in enumerate(model.parameters()):
        if p is not None:
            partial_sums[str(i_param)] = torch.zeros_like(p[0], dtype=torch.int64)
            # add each msg 
            for client in all_params['client_list']:
                if client == run_id: continue
                partial_sums[str(i_param)] = torch.remainder(partial_sums[str(i_param)] + msg[str(client)][str(i_param)].to(torch.int64), all_params['modulo'])
    return partial_sums
    


def send_grads_fat_pairwise(model, all_params, run_id, include_grads, all_rand, pairwise_secrets):
    # fat clients using pairwise keys
    
    true_grads = {}
    if all_params['debug'] or not all_params['use_encryption']:
         for i, p in enumerate(model.parameters()):
            if p is not None:
                if include_grads:
                    true_grads[str(i)] = p.grad[0].clone()
                else:
                    true_grads[str(i)] = torch.zeros_like(p[0])
    
    if all_params['use_encryption']:
        '''
        if all_params['debug']:
            print('Client {} generated {} random ints in {}s'.format(run_id, total_params*(len(all_params['client_list'])-1),rand_time))
        '''
        
        all_grads = {}
        rand_counter = 0
        for i,p in enumerate(model.parameters()):
            if p is not None:
                if include_grads:
                    pp = np.remainder( torch.round(all_params['fixed_point_int']*(torch.clamp(p.grad[0],min=-all_params['offset'],max=all_params['offset'])+all_params['offset'])).detach().numpy(),all_params['modulo'])#.astype('uint32')
                else:
                    pp = (np.zeros_like(p[0].detach().numpy(),dtype='uint64')+all_params['fixed_point_int']*all_params['offset'])

                if all_params['debug']:
                    apu, apu2, apu3 = [], [], []
                i_rand = rand_counter
                for k in pairwise_secrets['secrets']:
                    if int(k) < run_id:
                        pp = np.remainder(pp + np.remainder(all_rand[str(k)][i_rand:(i_rand+pp.size)] ,all_params['modulo']).reshape(pp.shape) , all_params['modulo'])
                    else:
                        pp = np.remainder(pp + np.remainder(all_params['modulo'] - all_rand[str(k)][i_rand:(i_rand+pp.size)],all_params['modulo']).reshape(pp.shape) , all_params['modulo'])
                    # test that noise actually cancels out
                    if all_params['debug']:
                        apu.append( np.remainder(np.zeros_like(pp,dtype='uint32') + np.remainder(all_rand[str(k)][i_rand:(i_rand+pp.size)],all_params['modulo']).reshape(pp.shape), all_params['modulo'],dtype='uint32'))
                        apu2.append( np.remainder( all_params['modulo'] - np.remainder(all_rand[str(k)][i_rand:(i_rand+pp.size)],all_params['modulo']).reshape(pp.shape), all_params['modulo'], dtype='uint32'))

                rand_counter += pp.size
                all_grads[str(i)] = pp.astype('uint32')

                
            if all_params['debug']:
                if all_params['n_encrypted_scheme2'] == 3:
                    print('remainders')
                    print(np.remainder(apu[0][:4,:4]+apu[1][:4,:4]+apu2[0][:4,:4]+apu[1][:4,:4]+apu2[0][:4,:4]+apu2[1][:4,:4] ,all_params['modulo']))
        
        with open(all_params['sent_grads_file']+str(run_id),'wb') as f:
            pickle.dump(all_grads,f)
    
    #print(true_grads)
    if all_params['debug']:
        torch.save(true_grads, all_params['sent_grads_file']+str(run_id)+'_debug')
        print('Client {} grad msg sent'.format(run_id))
    
    if not all_params['use_encryption']:
        with open(all_params['sent_grads_file']+str(run_id), 'wb') as f:
            pickle.dump(true_grads, f)

    # ping master
    with open(all_params['grads_ping_file']+str(run_id), 'w') as f:
        pass
    
    if all_params['debug']:
        print('true grads')
        print(true_grads['1'][:4,:4])
        sys.exit()

    # clear weight ping and zero used grads
    try:
        os.remove(all_params['weights_ping_file']+str(run_id))
    except FileNotFoundError as err:
        pass



def send_grads_thin(model, all_params, run_id, include_grads, all_rand):
    
    # send grads to Compute nodes
    true_grads = {}
    if all_params['debug'] or not all_params['use_encryption']:
         for i, p in enumerate(model.parameters()):
            if p is not None:
                if include_grads:
                    true_grads[str(i)] = p.grad[0].clone()
                else:
                    true_grads[str(i)] = torch.zeros_like(p[0])
    
    if all_params['use_encryption']:
        all_grads = {}
        all_noise = {}
        
        i_rand = 0
        for i,p in enumerate(model.parameters()):
            if p is not None:
                # generate all messages
                # 0 elem = true grads+noise, rest pure noise, final elem = -sum of all noise
                all_grads[str(i)] = []
                if include_grads:
                    all_grads[str(i)].append( np.remainder( (np.rint(all_params['fixed_point_int']*\
                        (torch.clamp(p.grad[0],min=-all_params['offset'],max=all_params['offset']).numpy()+all_params['offset']))).astype('uint32',copy=False), all_params['modulo']))
                    if all_params['debug']:
                        assert np.greater_equal(all_grads[str(i)][0], all_params['modulo']).sum() == 0
                        assert np.less(all_grads[str(i)][0], 0).sum() == 0
                else:
                    all_grads[str(i)].append( np.zeros_like(p[0].detach().numpy(),dtype='uint32')+all_params['fixed_point_int']*all_params['offset'])
                all_noise = np.zeros_like(p[0].detach().numpy(), dtype='uint32')
                for k in range(all_params['n_computes']):
                    if k > 0:
                        all_grads[str(i)].append(np.zeros_like(p[0].detach().numpy(),dtype='uint32'))
                    if k < all_params['n_computes']-1:
                        noise = all_rand[i_rand:(i_rand+p[0].numel())].reshape(p[0].detach().numpy().shape)
                        i_rand += p[0].numel()
                        all_grads[str(i)][k] = np.remainder( all_grads[str(i)][k]+noise, all_params['modulo'])
                        all_noise = np.remainder( all_noise + noise.copy(), all_params['modulo'])
                    else:
                        all_grads[str(i)][k] = all_params['modulo']-all_noise
        
        with open(all_params['sent_grads_file']+str(run_id), 'wb') as f:
            pickle.dump(all_grads,f)
    else:
        with open(all_params['sent_grads_file']+str(run_id), 'wb') as f:
            pickle.dump(true_grads, f)
    
    # check that noise cancels out on summation
    if all_params['debug']:
        max_err = 0
        for i,p in enumerate(model.parameters()):
            if include_grads: 
                apu2 = p.grad[0].detach().numpy()
            else:
                apu2 = torch.zeros_like(p[0]).detach().numpy()
            apu = np.max(np.abs(apu2 - \
                (1/all_params['fixed_point_int']*(np.remainder(np.stack(all_grads[str(i)],axis=0).sum(axis=0), all_params['modulo']).astype('float') ) \
                -all_params['offset'] ) ) )
            if apu > max_err:
                max_err = apu
            if p.grad is not None:
                print(p.grad.shape)
                print(p.grad[0][:3,:3])
                print( (1/all_params['fixed_point_int']*(np.remainder(np.stack(all_grads[str(i)],axis=0).sum(axis=0), all_params['modulo']).astype('float'))-all_params['offset']) [:3,:3])
        print('max err in decrypted grads: {}'.format(max_err))
        torch.save(true_grads,all_params['sent_grads_file']+str(run_id)+'_debug')
    

    # ping master
    with open(all_params['grads_ping_file']+str(run_id), 'w') as f:
        pass
    if all_params['debug']:
        sys.exit()

    # clear weight ping and zero used grads
    try:
        os.remove(all_params['weights_ping_file']+str(run_id))
    except FileNotFoundError as err:
        pass


def do_noise_generation(all_params, run_id, pairwise_secrets, all_crypto_times):
    
    if not all_params['use_encryption']:
        all_crypto_times.append(0)
        return None
    
    if all_params['dim_reduction'] > 0:
        total_params = all_params['dim_reduction']
    else:
        total_params = all_params['total_params']

    start = time.time()

    # scheme 1: thin clients
    if all_params['scheme_type'] == 1:
        # measure time needed for generating randomness for the encryption
        # use either urandom calls directly or seed blake2 hash with urandom
        if not all_params['use_hash_in_scheme1']:
            if all_params['use_true_entropy']:
                all_rand = np.asarray([secrets.randbelow(all_params['modulo']) for i in range(total_params*(all_params['n_computes']-1))],dtype='uint32')
            else:
                print('Using PRNG for encrypting, this is for debugging!')
                all_rand = np.random.randint(0,all_params['modulo']-1, size=total_params*(all_params['n_computes']-1),dtype='uint32')
        else:
            if all_params['use_true_entropy']:
                all_rand = np.remainder(np.asarray([int(hashlib.blake2b(str(i).encode('utf-8'),digest_size=4,key=secrets.token_bytes(8)).hexdigest(),16) for i in range(total_params*(all_params['n_computes']-1))],dtype='uint32'), all_params['modulo'] )
            else:
                print('Using PRNG seeding hash, this is for debugging!')
                all_rand = np.remainder(np.asarray([int(hashlib.blake2b(str(i).encode('utf-8'),digest_size=4,key=np.random.bytes(8)).hexdigest(),16) for i in range(total_params*(all_params['n_computes']-1))],dtype='uint32'), all_params['modulo'] )

        all_crypto_times.append(time.time()-start)
        return all_rand

    # scheme 2: fat clients with pairwise encryption
    elif all_params['scheme_type'] == 2:
        all_rand = {}
        
        for k in pairwise_secrets['secrets']:
            all_rand[str(k)] = np.remainder(np.asarray([int(hashlib.blake2b(str(i+pairwise_secrets['round']).encode('utf-8'),digest_size=4,key=pairwise_secrets['secrets'][str(k)]).hexdigest(),16) for i in range(total_params)],dtype='uint32'), all_params['modulo'] )

        all_crypto_times.append( time.time()-start)
        pairwise_secrets['round'] += total_params

        return all_rand

    sys.exit('Invalid scheme for crypto rand generation!')


def check_kill_ping(kill_ping_file, all_params, joint_sampler, all_crypto_times, run_id, print_res=False):
    try:
        with open(kill_ping_file+str(run_id), 'r') as f:
            pass
        print('Kill ping received on client {}, exiting..'.format(run_id))
        # NOTE: SAVE ESSENTIAL STUFF HERE
        res = {}
        res['all_crypto_times'] = all_crypto_times
        res['all_batches'] = joint_sampler.drawn_batch_sizes
        with open(all_params['res_save_file']+str(all_params['l_rate']) +'_' + str(run_id)+'.pickle' ,'wb') as f:
            pickle.dump(res, f)
        if print_res:
            print('res client {}:\n{}'.format(run_id, res))
        sys.exit()
        #raise RuntimeError('Kill ping received')
    except FileNotFoundError as err:
        #print(sys.exc_info())
        pass



