'''
Script for plotting amplification factors for different sampling without replacement & Poisson sampling
'''

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import hypergeom as hg

n = 50000 # total number of samples
deltas = [1e-15,1e-10,1e-5]
batches = [n//100, 5*n//100] # total batch size
n_points = 100
x = np.linspace(n//100,n//2,n_points)
slacked_y = np.zeros((len(batches),len(deltas),n_points))
worst_case_y = np.zeros((len(batches), n_points))
for b_i, b in enumerate(batches):
    for i, delta in enumerate(deltas):
        for ii in range(n_points):
            if ii in (0,50,100):
                print('delta={}, floor(x)={}, q={}'.format(delta,np.floor(x[ii]),hg.ppf( q=delta, M=n,n=x[ii], N=b)))
            slacked_y[b_i,i,ii] = (b-hg.ppf( q=delta, M=n,n=x[ii], N=b))/(n-np.floor(x[ii]))
            if i == 0:
                worst_case_y[b_i,ii] = b/(n-np.floor(x[ii]))
fig, axs = plt.subplots(2)
for b_i, b in enumerate(batches):
    axs[b_i].plot(x,worst_case_y[b_i,:], label='with 0 slack (worst-case)', color='black', linestyle=':',alpha=.7)
    for i,delta in enumerate(deltas):
        axs[b_i].plot(x,slacked_y[b_i,i,:], label=r'with {} slack in $\delta$'.format(delta),alpha=.9)
    # add corresponding poisson sampling fraction for comparison
    axs[b_i].plot(x,np.ones(len(x))*b/n, label='Poisson sampling', alpha=.7, color='darkgrey', linestyle='-.')

    #axs[b_i].set_ylabel('sampling fraction for amplification (smaller=better)')
    #axs[b_i].set_title('Dataset size={}, batch size={}'.format(n,b))
    axs[b_i].set_ylabel('Sampling fraction')
    if b_i == 1:
        axs[b_i].set_xlabel('Samples controlled by malicious parties')
    axs[b_i].legend(title='Dataset size={}, batch size={}'.format(n,b),loc='upper left')
#plt.ylabel('Sampling fraction')
plt.suptitle('Sampling fractions for privacy amplification')
#plt.legend()
#plt.tight_layout()
#plt.savefig('../test_results/SWOR_sampling_fracs.pdf', bbox_inches = "tight")
plt.show()



