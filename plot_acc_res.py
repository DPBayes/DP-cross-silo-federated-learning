'''
Plotting script for model accuracy
'''

from collections import OrderedDict as od
from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys

all_res = od()
# bounded dp
eps = [13.8, 2.5, 1.1, 0] # these are calculated with calculate_privacy.py

#unbounded dp, cifar, b=400/76, 125*10
#eps = [4, 1.6, .55]

colors_full = ['#1abc9c','#16a085','#27ae60','#2ecc71',' #f1c40f']
colors_group = ['#c0392b' ,'#e74c3c', '#ec7063','#8e44ad']
lines = ['-','--','-.',':']

# folder prefix for results
# bounded dp
prefix = '../test_results/acc_tests/full_test5/'
# test names
all_tests = [
            'centr_sigma0.5',
            'centr_sigma1',
            'centr_sigma2',
            #'centr_sigma4',
            'smc_sigma0.5',
            'smc_sigma1',
            'smc_sigma2',
            #'smc_sigma4',
            'ldp_sigma1.34',
            'ldp_sigma5.5',
            'ldp_sigma12',
            #'ldp_sigma22',
            ]
# additional params in filenames
C='2.5'
lr = '0.0009'

n_repeats = 5
y = np.zeros((5,3,n_repeats))

# read results
for i_test, test in enumerate(all_tests):
    for repeat in range(1,n_repeats+1):
        all_res[test] = np.load( prefix+test+'_C'+C+'_'+str(repeat)+'/res_'+lr+'_final.npy' )
        #continue
        if 'centr' in test:
            ind = 0
        elif 'smc' in test:
            ind = 1
        elif 'ldp' in test:
            ind = 2
        y[ind, i_test%3, repeat-1] = np.amax(all_res[test])

# add comparison to Poisson sampling
prefix = '../test_results/acc_tests/bounded_poisson_test1/'
all_tests = [
            'centr_sigma0.5',
            'centr_sigma1',
            'centr_sigma2',
            'smc_sigma0.5',
            'smc_sigma1',
            'smc_sigma2',
            ]
for i_test, test in enumerate(all_tests):
    for repeat in range(1,n_repeats+1):
        if 'centr' in test and repeat == 5: continue
        #some poisson runs apparently crashed

        all_res[test] = np.load( prefix+test+'_C'+C+'_'+str(repeat)+'/res_'+lr+'_final.npy' )
        #continue
        if 'centr' in test:
            ind = 3
            #pass
        elif 'smc' in test:
            ind = 4
        elif 'ldp' in test:
            #ind = 2
            pass
        y[ind, i_test%3, repeat-1] = np.amax(all_res[test])


#plt.plot(eps[:3], y[0,:3,:], linestyle=lines[1], marker='*', label='Trusted DP', color='b', alpha=.7)
plt.errorbar(eps[:3], np.mean(y[0,:3,:],axis=1),yerr=np.vstack( ( np.mean(y[0,:3,:],axis=1) -np.min(y[0,:3,:],axis=1) , np.max(y[0,:3,:],axis=1)-np.mean(y[0,:3,:],axis=1)) ) , linestyle='-', marker='*', label='Trusted DP', color='grey', alpha=.7)
#plt.plot(eps[:3], y[1,:3], linestyle=lines[0], marker='*', label='DP-SMC', color=colors_full[0], alpha=.9)
plt.errorbar(eps[:3], np.mean(y[1,:3,:],axis=1),yerr =np.vstack( (np.mean(y[1,:3,:],axis=1)-np.min(y[1,:3,:],axis=1) , np.max(y[1,:3,:],axis=1)-np.mean(y[1,:3,:],axis=1)) ) , linestyle=lines[1], marker='*', label='DP-SMC', color='blue', alpha=.95)
#plt.plot(eps[:3], y[2,:3] ,linestyle=lines[0], marker='*', label='LDP', color=colors_group[1], alpha=.9)
plt.errorbar(eps[:3], np.mean(y[2,:3,:],axis=1),yerr =np.vstack( (np.mean(y[2,:3,:],axis=1)-np.min(y[2,:3,:],axis=1) , np.max(y[2,:3,:],axis=1)-np.mean(y[2,:3,:],axis=1)) ) , linestyle=lines[1], marker='*', label='LDP', color=colors_group[1], alpha=.95)

# skip trusted poisson to keep plot more readable
#plt.errorbar(eps[:3], np.mean(y[3,:3,:4],axis=1),yerr =np.vstack( (np.mean(y[3,:3,:4],axis=1)-np.min(y[3,:3,:4],axis=1) , np.max(y[3,:3,:4],axis=1)-np.mean(y[3,:3,:4],axis=1)) ) , linestyle=lines[1], marker='*', label='Trusted Poisson', color='grey', alpha=.95)

plt.errorbar(eps[:3], np.mean(y[4,:3,:],axis=1),yerr =np.vstack( (np.mean(y[4,:3,:],axis=1)-np.min(y[4,:3,:],axis=1) , np.max(y[4,:3,:],axis=1)-np.mean(y[4,:3,:],axis=1)) ) , linestyle=lines[1], marker='*', label='DP-SMC Poisson', color='green', alpha=.95)

plt.suptitle('CIFAR10 classification accuracy, 32 clients')
plt.title('\nBounded DP $\delta=1e-5$')
plt.xlabel('$\epsilon$')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(alpha=.3)
plt.tight_layout()
#plt.savefig('../test_results/CIFAR_fat_32cl_acc_low_priv_bounded_poisson.pdf')
plt.show()

