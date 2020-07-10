'''
Script for plotting model accuracy when using projection
'''

from collections import OrderedDict as od
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys

n_parties = 10
all_res = od()

colors_full = ['#1abc9c','#16a085','#27ae60','#2ecc71',' #f1c40f']
colors_group = ['#c0392b' ,'#e74c3c', '#ec7063','#8e44ad']
lines = ['-','--','-.',':']

prefix = '../test_results/proj_tests/acc/unbounded_poisson_proj1/'

all_tests = [
            'sigma1.03_C2.5_no-proj',
            'sigma1.14_C2.5_k50',
            'sigma1.14_C2.5_k100',
            'sigma1.14_C2.5_k200',
            'sigma1.14_C2.5_k400'
            ]
n_repeats = 5

lr = '0.0009'

# Note: read runtimes from separate runtime tests; need to match to the settings above!
use_separate_runtimes = True
runtime_prefix = '../test_results/proj_tests/runtimes/unbounded_poisson_test1/'
runtime_tests = [
        'sigma1.03_C2.5_compute8_no-proj_',
        'sigma1.14_C2.5_compute8_k50_',
        'sigma1.14_C2.5_compute8_k100_',
        'sigma1.14_C2.5_compute8_k200_',
        'sigma1.14_C2.5_compute8_k400_'
        ]
r_lr = '0.0009'

res = {}
res['crypto_times'] = od()
res['runtimes'] = od()
res['runtimes']['total'] = od()
res['runtimes']['sum'] = od()
res['runtimes']['min*iters'] = od()
res['runtimes']['median*iters'] = od()

res['runtimes']['runtime_tests'] = od()
res['runtimes']['runtime_tests']['total'] = od()

y = np.zeros((len(all_tests), n_repeats))
for i_test, test in enumerate(all_tests):
    all_res[test] = od()
    res['crypto_times'][test] = np.zeros(n_repeats) #od() #None
    
    # check total runtime vs sum overiters & iters*min iter time
    res['runtimes']['total'][test] = np.zeros(n_repeats)
    res['runtimes']['sum'][test] = np.zeros(n_repeats)
    res['runtimes']['min*iters'][test] = np.zeros(n_repeats)

    for i_repeat in range(n_repeats):
        all_res[test][str(i_repeat)] = np.load( prefix+test+'_'+str(i_repeat+1)+'/res_'+lr+'_final.npy' )
        #print(all_res[test][str(i_repeat)])
        #sys.exit()
        with open(prefix+test+'_'+str(i_repeat+1)+'/res_'+lr+'_params.pickle', 'rb') as f:
            params = pickle.load(f)
        for i in range(1,n_parties+1):
            # crypto times = only time for encryptions
            with open(prefix+test+'_'+str(i_repeat+1)+'/res_'+lr+'_'+str(i)+'.pickle', 'rb') as f:
                apu = np.amin( pickle.load(f)['all_crypto_times'] )
            if res['crypto_times'][test][i_repeat] is None or apu < res['crypto_times'][test][i_repeat]:
                res['crypto_times'][test][i_repeat] = apu
                
        # check total runtime vs sum overiters & iters*min iter time
        res['runtimes']['total'][test][i_repeat] = params['total_runtime']
        res['runtimes']['sum'][test][i_repeat] = np.sum(params['run_times'])
        res['runtimes']['min*iters'][test][i_repeat] = params['n_train_iters']*(np.amin(params['run_times']))

        y[i_test, i_repeat] = np.amax(all_res[test][str(i_repeat)])

#print(y.shape,y)
#sys.exit()


if use_separate_runtimes:
    for i_test, test in enumerate(runtime_tests):
        res['runtimes']['runtime_tests']['total'][test] = np.zeros(n_repeats)
        for repeat in range(n_repeats):
            with open(runtime_prefix+test+str(repeat+1)+'/res_'+r_lr+'_params.pickle', 'rb') as f:
                params = pickle.load(f)

            # check total runtime vs sum overiters & iters*min iter time
            res['runtimes']['runtime_tests']['total'][test][repeat] = params['total_runtime']
            #res['runtimes']['sum'][test] = np.sum(params['run_times'])
            #res['runtimes']['min*iters'][test] = params['n_train_iters']*(np.amin(params['run_times']))


# plot running time vs accuracy
if 1:
    x = [50,100,200,400]

    measure = 'total'
    #measure = 'median*iters'
    if not use_separate_runtimes:
        comparison = np.median( res['runtimes'][measure][all_tests[0]] )
        #print(comparison)
        #sys.exit()
    else:
        comparison = np.median(res['runtimes']['runtime_tests'][measure][runtime_tests[0]])
    
    # fold change vs accuracy
    if not use_separate_runtimes:
        # plot no-proj on time scale
        apu = np.array([np.median(i)/comparison for i in list(res['runtimes'][measure].values())])
    else:
        # plot no-proj on time scale
        apu = np.array([np.median(i)/comparison for i in list(res['runtimes']['runtime_tests'][measure].values())])
    
    plt.errorbar( apu[1:] , np.mean(y,axis=1)[1:], yerr= \
        np.vstack( (np.mean(y,axis=1)[1:]-np.min(y,axis=1)[1:] , np.max(y,axis=1)[1:]-np.mean(y,axis=1)[1:] ) ) \
            , color=colors_full[0],linestyle=lines[0], alpha=.9, label='Projected', marker='*' )
    
    # add non-projected as a separate dot with time scale
    plt.errorbar( apu[0] , np.mean(y,axis=1)[0], yerr= \
        np.vstack( (np.mean(y,axis=1)[0]-np.min(y,axis=1)[0] , np.max(y,axis=1)[0]-np.mean(y,axis=1)[0] ) ) \
            , color='red',linestyle=lines[0], alpha=.9, label='No projection', marker='*' )

    plt.grid(alpha=.3)
    plt.ylim(.53,.65)
    plt.xlabel('Runtime, projected/non-projected')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    for i, (i_y,i_apu) in enumerate( zip(np.mean(y,axis=1),apu )):
        #print(i,i_y,i_apu)
        # add a separate note on non-projected
        if i == 0:
            # doesn't work without separate runtimes
            plt.text(i_apu-1e-1, i_y-8e-3,'$d\simeq{}$'.format('9 E+5'))
        else:
            if not use_separate_runtimes:
                plt.text(i_apu-1*i*(2e-2), i_y-.01,'$k={}$'.format(x[i-1]))
            else:
                #plt.text(i_apu-1.4*i*(2e-3), i_y-.01,'k={}'.format(x[i]))
                plt.text(i_apu+(7e-3), i_y-2e-3,'$k={}$'.format(x[i-1]))

    plt.suptitle('CIFAR10 Classification accuracy using projection')
    plt.title('\n$(1.7,1e-5)$-unbounded DP')
    plt.tight_layout()
    #plt.savefig('../test_results/CIFAR_proj_acc_fold_8computes_k400_with_no-proj.pdf')
    plt.show()

