# Testing parallel jobs (from job lib) to speed up modelling

from time import sleep
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory

import numpy as np

pl = Parallel(n_jobs=2, verbose=5, max_nbytes=1e6)

def foo(i, ones):
	print ("executing", i)
	sleep(np.random.rand())
	return i

result = pl(delayed(has_shareable_memory)(foo(i,np.ones(int(i)))) for i in [1e2, 1e4, 1e6])

print result

