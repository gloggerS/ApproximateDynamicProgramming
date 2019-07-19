from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library

#%%
import numpy as np
import pandas as pd# define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

#%%
# SKLEARN
from sklearn import linear_model

X = df
y = target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)
print(predictions[0:5])


#%%
# STATSMODEL
## Without a constant

import statsmodels.api as sm

X = df["RM"]
y = target["MEDV"]



#%%
X = df
y = target["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()
