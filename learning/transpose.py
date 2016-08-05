# From https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

chunksize = 100

# read 12320 genes
reader = pd.read_csv('training.csv', header=None, chunksize=chunksize, engine='c', iterator=True)

#print next(reader)
chunks = list()

# concat with progress
for i in range(0,124):
	print "Records %i..." % (i*chunksize)
	chunks.append(next(reader))

df = pd.concat(chunks, ignore_index=True)

print "Transposing..."

df = df.transpose()

print "Saving..."

df.to_csv('ttt-1000.csv', index=False, header=False)

print "Saved data:"
print df.shape
# print df
	#print chunk.shape
	#print chunk
#print test_data.shape
#print test_data.head()
