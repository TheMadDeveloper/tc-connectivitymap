###########################################################################################
# Another linear regression/machine learning walkthrough.
#
# From https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb
#

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# read adv into a advFrame
adv = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
#print
#print adv.head()
#print
#print adv.shape

# visualize the relationship between the features and the response using scatterplots
#fig, axs = plt.subplots(1, 3, sharey=True)
#adv.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
#adv.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
#adv.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])

#print
#print adv.columns


#plt.show()
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
#regr.fit(adv.loc[:,['TV']],adv.loc[:,['Sales']])
regr.fit(adv.iloc[:,0:3],adv.loc[:,['Sales']])

#print
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


# The mean square error
#print("Residual sum of squares: %.2f"
 #     % np.mean((regr.predict(adv['TV']) - adv['Sales']) ** 2))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(adv['TV'], adv['Sales']))

# Plot the regression
X_new = pd.DataFrame({'TV': [adv.TV.min(), adv.TV.max()]})
# make predictions for those x values and store them
preds = regr.predict(X_new)

# first, plot the observed data
adv.plot(kind='scatter', x='TV', y='Sales')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)

#plt.show()