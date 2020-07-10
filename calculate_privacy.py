'''
Calculate privacy for distributed DP
set the parameters in this file and run this without args
'''

import sys

import numpy as np
import fourier_accountant as FA
from fourier_accountant import compute_eps_var

def parse_args(all_args):
    method = all_args.pop('method')
    if all_args['use_sigma_t']:
        var_q = True
        all_args.pop('ncomp')
        all_args.pop('sigma')
        all_args.pop('q')
        all_args.pop('use_sigma_t')
    else:
        var_q = False
        all_args.pop('use_sigma_t')
        all_args['q'] = all_args['q'][0]

    if all_args['target_eps'] is None and all_args['target_delta'] is not None:
        all_args.pop('target_eps')
        if var_q is True:
            return get_general_privacy_bound_var_eps(all_args, method=method)
        else:
            return get_general_privacy_bound_eps(all_args, method=method)
    elif all_args['target_eps'] is not None and all_args['target_delta'] is None:
        all_args.pop('target_delta')
        if var_q is True:
            sys.exit('Var q for delta not implemented yet!')
        else:
            return get_general_privacy_bound_delta(all_args, method=method)
    else:
        sys.exit('Invalid (eps,delta) combination: {},{}'.format(all_args['target_eps'],all_args['target_delta']))


def get_general_privacy_bound_eps(kwargs, method='unbounded'):
    if method == 'bounded':
        print('Using bounded dp definition')
        res = FA.compute_eps.get_epsilon_S(**kwargs)
        print('eps={}'.format( res ))
        return res
    else: 
        print('Using unbounded dp definition')
        print('eps={}'.format(FA.compute_eps.get_epsilon_R(**kwargs)) )


def get_general_privacy_bound_delta(kwargs, method='unbounded'):
    if method == 'bounded':
        print('Using bounded dp definition')
        print('delta={}'.format( FA.compute_delta.get_delta_S(**kwargs) ))
    else: 
        print('Using unbounded dp definition')
        print('delta={}'.format( FA.compute_delta.get_delta_R(**kwargs)))

def get_general_privacy_bound_var_eps(kwargs, method='unbounded'):
    if method == 'bounded':
        print('Using bounded dp definition')
    else:
        print('Using unbounded dp definition')
        print('eps={}'.format( compute_eps_var.get_eps_R(**kwargs) ))



if __name__ == '__main__':
    all_args = {}
    
    #######################################################
    # SETUP
    #######################################################

    all_args['use_sigma_t'] = 0 # set to 0
    all_args['target_delta'] = 1e-5 # 1e-5 for basic accuracy tests, 5e-6 for projection
    all_args['method'] = 'bounded' # 'bounded' or 'unbounded'
    all_args['L'] = 30 # range for Fourier accountant
    all_args['nx'] = 5e6 # number of grid points for Fourier accountant
    all_args['ncomp'] = 1250 # number of compositions, basic tests use 10*125 iters=1250 compositions
    all_args['sigma'] = .5 # noise level
    all_args['q'] = [0.008] # sampling fraction
    all_args['target_eps'] = None # tests fix delta & search eps

    ### To get the privacy costs reported in the paper ###
    # for basic accuracy tests: CIFAR10, 32 clients, 1250 comps, bounded DP, delta=1e-5 
    #   with SMC b=400 -> q=0.008
    #       sigma=.5, eps~=13.81
    #       sigma= 1, eps~=2.51
    #       sigma= 2, eps~=1.07
    #   with LDP b=76 -> q~=0.049 (1562 samples/client)
    #       sigma=1.34, eps~=13.91
    #       sigma=5.5 , eps~=2.58
    #       sigma=12  , eps~=1.08
    #
    # tests with projection: CIFAR10, 10 clients, 1250 comps, b=400, q=0.008, unbounded DP
    #   no proj baseline:
    #       * sigma=1, delta=1e-5 ->  eps~=1.6
    #   proj:           
    #       * sigma=1.03 , delta=5e-6 (leaves 5e-6 for delta' in proj.) ->  eps~=1.6
    #   Note: additionally need to upscale both sigma values by 10/9 in the actual test code due to all hbc clients, see Section on distributed noise addition
    #
    #######################################################

    parse_args(all_args)



