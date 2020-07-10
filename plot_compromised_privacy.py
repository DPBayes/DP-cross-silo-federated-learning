'''
Script for plotting privacy guarantees with compromised parties; not used in the paper
'''

import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np
from calculate_privacy import parse_args



all_args = {}
#######################################################
# SETUP
#######################################################
# general privacy bound for anyone (Thm 7)

all_args['L'] = 20 # range
all_args['nx'] = 5e6 # number of grid points


#all_args['q'] = 0.04 # trusted aggregator sampling fraction
method = 'bounded' # 'bounded' or 'unbounded'

# set eps/delta to the desired value, the other one to None
target_delta = 1e-5
target_eps = None


sigmas = [1.0, 3.0, 5.0] # trusted aggregator noise levels
ncomps = [10,100,1000,10000] # number of compositions
#scheme = 1 # 1=thin clients, 2=fat clients
n_parties = 10
data_per_party = 10000
batch_size = 500
compromised = [0,1,2,3]

filename = '../test_results/compromised_privacy_res.pickle'
from_scratch = True

save_to_disk = False
figure_name = '../test_results/compromised_privacy_res.jpg'

#######################################################

if from_scratch:
    res = np.zeros((len(sigmas),len(compromised),len(ncomps)))
    for i,k in enumerate(compromised):
        for i_sigma, sigma in enumerate(sigmas):
            all_args['sigma'] = np.sqrt((n_parties-k)/n_parties)*sigma
            all_args['q']= batch_size/((n_parties-k)*data_per_party)
            for i_nc, nc in enumerate(ncomps):
                all_args['ncomp'] = nc
                all_args['method'] = method
                all_args['target_delta'] = target_delta
                all_args['target_eps'] = target_eps
                res[i_sigma,i,i_nc] = parse_args(all_args)
    with open(filename, 'wb') as f:
        pickle.dump(res,f)

else:
    with open(filename,'rb') as f:
        res = pickle.load(f)

#print(res)

#lines = ['solid','dotted','dashed','dashdot']
lines= [(0,(1,5)),(0,(1,1)), (0,())]
cols = ['blue', 'green', 'orange', 'cyan','magenta']

for i_nc, nc in enumerate(compromised):
    for i_sigma, sigma in enumerate(sigmas):
        plt.plot(ncomps, res[i_sigma, i_nc,:], color=cols[i_nc], linestyle=lines[i_sigma], label='compromised={}, sigma={}'.format(nc,sigma))
plt.xscale('log')
plt.suptitle('Privacy guarantees with compromised parties')
plt.title('{}-DP, samples per party={}, $\delta$={}, b={}'.format(method, data_per_party, target_delta, batch_size))
plt.xlabel('Number of compositions')
plt.ylabel(r'$\epsilon$')
plt.legend()
#plt.tight_layout()
if save_to_disk:
    plt.savefig(figure_name, format='jpg', Bbox='tight')
else:
    plt.show()




