'''
Script for plotting fat client runtimes
'''

from collections import OrderedDict as od
from matplotlib import pyplot as plt
import numpy as np
import pickle
from pstats import Stats
import sys

colors_full = ['#1abc9c','#16a085','#27ae60','#2ecc71',' #f1c40f']
colors_group = ['#c0392b' ,'#e74c3c', '#ec7063','#8e44ad']
lines = ['-','--','-.',':']

# Note: profiler not used (& not currently working)
read_profiler = False

plot_profiler_fold = 0
plot_fold_inc = 1
plot_enc_times = 0

n_clients = [32,16,2]
crypto_groups = [[0,2,16,32],[16],[2]]

prefix = '../test_results/runtime_tests/bounded_swor_fat_536_test1/'

all_tests = [[
            '32CG0', # no crypto
            '32CG2',
            '32CG16',
            '32CG32'
            ],[
            '16CG16'
            ],[
            '2CG2'
            ]]
lr = '0.0009'
n_repeats = 5

# Note: not used currentlV
# read cprofiler info & write to regular .txt
if read_profiler:
    for i_test,test in enumerate(all_tests):
        for i in range(1,n_clients+1):
            filename = prefix+all_tests[i_test]+'/runtime_'+str(i)
            with open(filename+'.txt','w') as f:
                p = Stats(filename, stream=f).strip_dirs().sort_stats('tottime')
                p.print_stats()
    
res = {}
res['crypto_times'] = od()
res['runtimes'] = od()
res['runtimes']['total'] = od()
res['runtimes']['sum'] = od()
res['runtimes']['min*iters'] = od()
res['runtimes']['median*iters'] = od()
res['runtimes']['profiler'] = od()


for i_group, group in enumerate(all_tests):
    for i_test, test in enumerate(group):
        res['crypto_times'][test] = np.zeros(n_repeats) - 1 

        res['runtimes']['total'][test] = np.zeros(n_repeats)
        res['runtimes']['sum'][test] = np.zeros(n_repeats)
        res['runtimes']['min*iters'][test] = np.zeros(n_repeats)
        res['runtimes']['median*iters'][test] = np.zeros(n_repeats)

        for repeat in range(n_repeats):
 
            with open(prefix+test+'_'+str(repeat+1)+'/res_'+lr+'_params.pickle', 'rb') as f:
                params = pickle.load(f)
            
            #print(  params['n_train_iters']*(np.median(params['run_times'])) )
            #sys.exit()
            # check total runtime vs sum overiters & iters*min iter time
            res['runtimes']['total'][test][repeat] = params['total_runtime']
            res['runtimes']['sum'][test][repeat] = np.sum(params['run_times'])
            res['runtimes']['min*iters'][test][repeat] = params['n_train_iters']*(np.amin(params['run_times']))
            res['runtimes']['median*iters'][test][repeat] = params['n_train_iters']*(np.median(params['run_times']))

            
            for i in range(1,n_clients[i_group]+1):
                # crypto times = only time for encryptions
                with open(prefix+test+'_'+str(repeat+1)+'/res_'+lr+'_'+str(i)+'.pickle', 'rb') as f:
                    apu = np.amin( pickle.load(f)['all_crypto_times'] )
                if res['crypto_times'][test][repeat] < 0 or apu < res['crypto_times'][test][repeat]:
                    res['crypto_times'][test][repeat] = apu




colors_full = ['#1abc9c','#16a085','#27ae60','#2ecc71',' #f1c40f']
colors_group = ['#c0392b' ,'#e74c3c', '#ec7063','#8e44ad']
lines = ['-','--','-.',':']

if plot_fold_inc:
    
    ylim = (.5,101)
    plot_group = ['32CG32','16CG16','2CG2']
    
    measure = 'total'
    #measure = 'median*iters'

    # compare medians over repeats
    comparison = np.min(res['runtimes'][measure][all_tests[0][0]])

    #print(comparison)
    #sys.exit()
    y = np.zeros(len(plot_group))
    for i_test, test in enumerate(plot_group):
        y[i_test] = np.median(res['runtimes'][measure][test])/comparison
    
    legend_handles = []
    legend_handles.append(plt.plot([], [], ' ', label="Model 1m:")[0])
    legend_handles.append( plt.plot(n_clients, y, color=colors_full[0],linestyle=lines[0], alpha=.9, label='Full pairwise', marker='*') )
    
    legend_handles.append( plt.hlines( np.median(res['runtimes'][measure]['32CG16'])/comparison,xmin=crypto_groups[1][0], xmax=32 , color=colors_full[0],linestyle=lines[1], alpha=.4, label='16') )
    legend_handles.append( plt.hlines( np.median(res['runtimes'][measure]['32CG2'])/comparison,xmin=crypto_groups[2][0], xmax=32, color=colors_full[0],linestyle=lines[2], alpha=.4, label='2' ) )
    
    # add stars to horizontal lines
    plt.plot(32,np.median(res['runtimes'][measure]['32CG16'])/comparison ,marker='*', color=colors_full[0])
    plt.plot(32,np.median(res['runtimes'][measure]['32CG2'])/comparison ,marker='*', color=colors_full[0])

    # add horizontal lines for no encryption
    plt.hlines(1, n_clients[0], n_clients[-1], color='black', linestyle=':', label='No encryption')

    plt.grid(alpha=.3)
    plt.legend(title='Encryption group size:')
    plt.xlabel('Number of clients')
    plt.ylabel('Runtime, encrypted/non-encrypted')

    plt.title('Fold increase in running time with fat clients')
    #plt.ylim(ylim)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    #plt.savefig('../test_results/fat_32cl_5run_medians_fold_increase.pdf')

