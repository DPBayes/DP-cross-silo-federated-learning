'''
Test how long single urandom call takes

Generating one int in [0,1e8-1] takes about 1.6689300537109375e-06
'''

import secrets
import time

max_iter = int(1e7)
max_int = int(4294967295)

min_time = 1e8

for i in range(max_iter):
    if i % 1e6 == 0: print(i)
    start = time.time()
    secrets.randbelow(max_int)
    end = time.time()
    if end-start < min_time:
        min_time = end-start

print('min call time: {}'.format(min_time))




