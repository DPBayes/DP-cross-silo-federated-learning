'''
Script for plotting hypergeometric distribution; not used in the paper
'''

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import hypergeom as hg


n = 50000 # total number of samples
n_Ts = [n//2,n//5,n//10, n//100] # number of samples controlled by malicious parties
b = n//100 # total batch size

print('Total samples: {}, malicious data: {}, batch size: {}'.format(n,n_Ts,b))

x = np.arange(0,b)
for i_line, n_T in enumerate(n_Ts):
    plt.plot( hg.pmf(k=x,M=n,n=n_T, N=b ), label=n_T )
#plt.ylim(0,1)
plt.ylabel('Probability')
plt.xlabel('Malicious samples in batch')
plt.legend(title='Malicious dataset size')
plt.title('Number of malicious samples, dataset size={}, batch size={}'.format(n,b))
plt.show()







