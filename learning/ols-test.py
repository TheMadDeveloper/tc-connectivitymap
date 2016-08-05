# From https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib

chunksize = 100

print "Reading landmarks..."

# read landmarks
df = pd.read_csv('../exampleTestData/landmarks.csv', header=None, engine='c')

print df.shape

print "Loading model..."

rgs = joblib.load('models/test_ols.pkl')
rgs = rgs[180:] # temp dyslexia fix

print "Predicting..."

df = df.transpose()

genes = list()
for i in range(0,len(rgs)):
	genes.append(rgs[i].predict(df.iloc[:,:]))

preds = pd.DataFrame(genes)

preds.to_csv('predictions.csv', index=False, header=False)
