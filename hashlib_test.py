'''
Test how long generating 32-bit int by using 4-byte blake2 hash takes

Generating one (fixed) 32-bit int takes about 7.152557373046875e-07 seconds
'''

import hashlib
import time

max_iter = int(1e7)
max_int = int(1e8)

min_time = 1e8

for i in range(max_iter):
    if i % 1e6 == 0: print(i)
    start = time.time()
    int(hashlib.blake2b(str('1').encode('utf-8'),digest_size=4,key=b'some_secret_key').hexdigest(),16)
    end = time.time()
    if end-start < min_time:
        min_time = end-start

print('min call time: {}'.format(min_time))

