# From https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
from sklearn import linear_model, svm
import warnings
import os


# Read in transposed training data. The file is very large (100,000 lines of 12320 numbers) so we may only read part of the file
# and will be read in chunks of the specified size. Returns a DataFrame of `chunk_count * chunksize` samples (rows)
# of 12320 genes (columns), the first 970 of which are the landmark genes
def read_tt(chunk_count, chunksize):

	reader = pd.read_csv('transposed.training.12320.csv', header=None, chunksize=chunksize, engine='c', iterator=True)
	
	chunks = list()

	# read one chunk at a time with progress to stdout
	for i in range(0,chunk_count):
		print('Samples {}...'.format(i*chunksize))
		chunks.append(next(reader))

	# concatenate chunks
	df = pd.concat(chunks, ignore_index=True)

	print("Loaded:", df.shape)

	return df

# Deprecated (non functioning) method for visually plotting a set of genes against each other (for visualizing possible
# correlations, linear or otherwise)
def browseplots(gene, loffset, rows, cols):
	fig, axs = plt.subplots(rows, cols, sharey=True)

	for i in range(0,rows):
		for j in range(0,cols):
			axs[i][j].tick_params(labelbottom=False, labelleft=False)
			df.plot(kind='scatter', alpha=.1, s=5, x=i*rows+j+loffset, y=gene, ax=axs[i][j])

	plt.show()

# Calculate a linear regression model for each of the 
# - 	data: the training data (first 970 columns should be landmark genes)
# - 	jobs: the number of parallel processes to use for calculation (5 or 6 will max processor)	
def ols(data, jobs):
	warnings.simplefilter('ignore')
	pl = Parallel(n_jobs=jobs, max_nbytes=1e6)

	landmarks = data.iloc[:,0:970]

	results = pl(delayed(ols_p)(i, data, landmarks) for i in range(970,12320))

	return results

# Actual linear regression calculation job using ordinary least squares (y = b0 + b1x + ... + b790x)
# -		i: the gene we are predicting
# -		data: the training data
# -		landmarks: the landmark genes (first 970 columns of data)
def ols_p(i, data, landmarks):
	print("Modeling: ", i)
	target = data.iloc[:,i]
	model = linear_model.LinearRegression()
	model.fit(landmarks, target)
	return model

def lasso(data, count):
	mlist = list()

	landmarks = data.iloc[:-100,0:970]
	landmarks_test = data.iloc[-100:,0:970]
	#for i in range(970,12320):
	for i in range(970,970+count):
		print("Modeling: ", i)
		target = data.iloc[:-100,i]
		target_test = data.iloc[-100:,i]
		alphas = np.logspace(-4, -1, 6)
		model = linear_model.Lasso()
		scores = [model.set_params(alpha=alpha, max_iter=1000,precompute=True, warm_start=True).fit(landmarks, target).score(landmarks_test, target_test) for alpha in alphas]
		best_alpha = alphas[scores.index(max(scores))]
		model.alpha = best_alpha
		model.fit(landmarks, target)

		mlist.append(model)

	return mlist

def svr(data, count):
	mlist = list()

	landmarks = data.iloc[:-100,0:970]
	landmarks_test = data.iloc[-100:,0:970]

	for i in range(970,970+count):
		print("Modeling: ", i)
		target = data.iloc[:-100,i]
		target_test = data.iloc[-100:,i]

		svr = svm.SVR()
		svr.fit(landmarks, target)

		mlist.append(svr)

	return mlist

def plot_svr(data, lm, ti, clf):
    landmarks = data.loc[:,lm]
    target = data.loc[:,ti]

    #data.plot(kind='scatter', s=5, x=lm, y=ti)
    
    plt.clf()
    plt.scatter(landmarks, target, zorder=10, cmap=plt.cm.Paired)

    #landmarks = data.iloc[:-100,0:970]
    #landmarks_test = data.iloc[-100:,0:970]

    # Circle out the test data
    #plt.scatter(data.iloc[:, lm], data.iloc[:, compg], s=80, facecolors='none', zorder=10)

    plt.axis('tight')
    x_min = landmarks.min()
    x_max = landmarks.max()
    y_min = target.min()
    y_max = target.max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    plt.show()
    return XX
    #Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(XX.shape)
    # plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    # plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
    #             levels=[-.5, 0, .5])

    # plt.title(kernel)

    

def predict(landmark_path, models):
	print('Reading {}...'.format(landmark_path))

	# read landmarks
	df = pd.read_csv(landmark_path, header=None, engine='c').transpose()

	print("Predicting...")

	genes = list()
	for i in range(0,len(models)):
		if i % 100 == 0:
			print('Processing {}'.format(i))
		genes.append(models[i].predict(df.iloc[:,:]))

	return pd.DataFrame(genes)

if __name__ == '__main__':
	os.environ["JOBLIB_START_METHOD"] = "forkserver"

	samples = read_tt(10,10)
	models = ols(samples, 6)

	preds = predict("../scoring/landmarks.csv", models)

	print("Predictions:", preds.shape)

	preds.to_csv('ols.sample.x100.csv', index=False, header=False)

# df.plot(kind='scatter', s=3, x=0, y=970, ax=axs[0][0], figsize=(8, 8))
# df.plot(kind='scatter', x=1, y=970, ax=axs[0][1])
# df.plot(kind='scatter', x=2, y=970, ax=axs[0][2])
# df.plot(kind='scatter', x=3, y=970, ax=axs[1][0])
# df.plot(kind='scatter', x=4, y=970, ax=axs[1][1])
# df.plot(kind='scatter', x=5, y=970, ax=axs[1][2])
