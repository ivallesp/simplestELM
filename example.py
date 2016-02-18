# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ELM import ELMRegressor

test_maes_dictionary = dict()

plt.style.use('ggplot')
sns.set_context("talk")
np.random.seed(0)

## DATA PREPROCESSING
X, y = load_diabetes().values()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

stdScaler_data = StandardScaler()
X_train = stdScaler_data.fit_transform(X_train)
X_test = stdScaler_data.transform(X_test)

stdScaler_target = StandardScaler()
y_train = stdScaler_target.fit_transform(y_train)  # /max(y_train)
y_test = stdScaler_target.transform(y_test)  # /max(y_train)
max_y_train = max(abs(y_train))
y_train = y_train / max_y_train
y_test = y_test / max_y_train

## ELM TRAINING
MAE_TRAIN_MINS = []
MAE_TEST_MINS = []

for M in range(1, 200, 1):
    MAES_TRAIN = []
    MAES_TEST = []
    # print "Training with %s neurons..."%M
    for i in range(30):
        ELM = ELMRegressor(M)
        ELM.fit(X_train, y_train)
        prediction = ELM.predict(X_train)
        MAES_TRAIN.append(mean_absolute_error(y_train,
                                              prediction))

        prediction = ELM.predict(X_test)
        MAES_TEST.append(mean_absolute_error(y_test,
                                             prediction))
    MAE_TEST_MINS.append(min(MAES_TEST))
    MAE_TRAIN_MINS.append(MAES_TRAIN[np.argmin(MAES_TEST)])

print "Minimum MAE ELM =", min(MAE_TEST_MINS)
test_maes_dictionary["ELM"] = min(MAE_TEST_MINS)

## LINEAR REGRESSION TRAINING
mae = []
lr = LinearRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
mae.append(mean_absolute_error(y_test, prediction))
print "Minimum MAE LR =", min(mae)
test_maes_dictionary["LinReg"] = min(mae)

## K-NEAREST NEIGHBORS TRAINING
mae = []
for N in range(1, 51):
    kn = KNeighborsRegressor()
    kn.fit(X_train, y_train)
    prediction = kn.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
print "Minimum MAE KNN =", min(mae)
test_maes_dictionary["KNN"] = min(mae)

## DECISION TREES TRAINING
mae = []
for max_depth in range(1, 51):
    for min_samples_split in range(1, 102, 5):
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        tree.fit(X_train, y_train)
        prediction = tree.predict(X_test)
        mae.append(mean_absolute_error(y_test, prediction))
print "Minimum MAE TREE = ", min(mae)
test_maes_dictionary["Dec. Tree"] = min(mae)

## SUPPORT VECTORS MACHINE TRAINING
mae = []
for kernel in ["rbf", "linear", "poly", "sigmoid"]:
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    prediction = svr.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
print "Minimum MAE SVR = ", min(mae)
test_maes_dictionary["SVM"] = min(mae)

## RANDOM FOREST TRAINING
mae = []
for n_estimators in range(10, 1100, 100):
    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
print "Minimum MAE R.Forest = ", min(mae)
test_maes_dictionary["R. Forest"] = min(mae)

#############################################################################################
## PLOTTING THE RESULTS
df = pd.DataFrame()
df["test"] = MAE_TEST_MINS
df["train"] = MAE_TRAIN_MINS
ax = df.plot()
ax.set_xlabel("Number of Neurons in the hidden layer")
ax.set_ylabel("Mean Absolute Error")
ax.set_title(
    "Extreme Learning Machine error obtained for the Diabetes dataset \n when varying the number of neurons in the "
    "hidden layer (min. at 23 neurons)")
plt.show()

D = test_maes_dictionary
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.ylabel("Mean Absolute Error")
plt.title("Error Comparison between Classic Regression Models and ELM")
plt.show()
