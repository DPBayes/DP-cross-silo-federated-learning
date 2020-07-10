'''
Script for plotting the extra nosie scaling factor for the distributed noise addition; not used in the paper
'''

import sys

from matplotlib import pyplot as plt
import numpy as np

n = 100 # number of plotting points

T_list = [0, 5, 10]
N = np.linspace(0,100,n)

for i,T in enumerate(T_list):
    y = N/(N-T-1)
    y[:(T+1)] = None
    plt.plot(N,y, label='T={}'.format(T))

plt.xlabel('Total number of parties')
plt.ylabel('Scaling factor')
plt.legend(title='Number of malicious parties')
plt.grid(alpha=.3)
plt.ylim(1,5)
plt.xlim(0,100)

plt.show()
#plt.savefig('../test_results/scaling_factor_plot.pdf')






