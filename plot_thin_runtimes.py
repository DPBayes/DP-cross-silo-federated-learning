'''
Script for plotting thin client runtimes
'''

from collections import OrderedDict as od
from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys


colors_full = ['#1abc9c','#16a085','#27ae60','#2ecc71',' #f1c40f']
colors_group = ['#c0392b' ,'#e74c3c', '#ec7063','#8e44ad']
lines = ['-','--','-.',':']

all_colors = ['blue','green','magenta','cyan','lime','red','orange','black']

n_parties = 100
n_computes = [0,2,4,8]

plot_fold_inc = 1


prefix = '../test_results/runtime_tests/bounded_swor_thin_536_test1/'

all_tests = [
            'no-crypto',
            'compute2',
            'compute4',
            'compute8',
            ]
lr = '0.0009'
n_repeats = 5

res = {}
res['crypto_times'] = od()
res['runtimes'] = od()
res['runtimes']['total'] = od()
res['runtimes']['sum'] = od()
res['runtimes']['min*iters'] = od()
res['runtimes']['median*iters'] = od()


for i_test,test in enumerate(all_tests):
    res['crypto_times'][test] = np.zeros(n_repeats)-1
    res['runtimes']['total'][test] = np.zeros(n_repeats)
    res['runtimes']['sum'][test] = np.zeros(n_repeats)
    res['runtimes']['min*iters'][test] = np.zeros(n_repeats)

    for repeat in range(n_repeats):
 
        with open(prefix+test+'_'+str(repeat+1)+'/res_'+lr+'_params.pickle', 'rb') as f:
            params = pickle.load(f)
        # check total runtime vs sum overiters & iters*min iter time
        res['runtimes']['total'][test][repeat] = params['total_runtime']
        res['runtimes']['sum'][test][repeat] = np.sum(params['run_times'])
        res['runtimes']['min*iters'][test][repeat] = params['n_train_iters']*(np.amin(params['run_times']))

        for i in range(1,n_parties+1):
            # crypto times = only time for encryptions
            with open(prefix+test+'_'+str(repeat+1)+'/res_'+lr+'_'+str(i)+'.pickle', 'rb') as f:
                apu = np.amin( pickle.load(f)['all_crypto_times'] )
            if res['crypto_times'][test][repeat] < 0 or apu < res['crypto_times'][test][repeat]:
                res['crypto_times'][test][repeat] = apu

        #sys.exit()


# plot total running time fold increases
if plot_fold_inc:
    
    ylim = (.5,20)
    
    measure = 'total'
    #measure = 'median*iters'
    
    comparison = np.median(res['runtimes'][measure][all_tests[0]])
    

    plt.plot(n_computes[1:] , [np.median(i)/comparison for i in list(res['runtimes'][measure].values())[1:]], color=colors_full[0],linestyle=lines[0], alpha=.9, label='1m', marker='*' )
    plt.hlines(1,xmin=n_computes[1],xmax=n_computes[-1], linestyle=lines[-1], label='No encryption' )
    plt.grid(alpha=.3)
    plt.ylabel('Runtime, encrypted/non-encrypted')
    plt.xlabel('Number of compute nodes')
    plt.legend(title='Model:', loc='upper left')
    plt.title('Fold increase in running time with thin clients')
    #plt.ylim(ylim)
    #plt.yscale('log')
    #plt.savefig('../test_results/thin_100cl_5run_medians_fold_increase.pdf', bbox_inches='tight')
    plt.show()

