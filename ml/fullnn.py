#created may 28, 2020
#last edited by athena flint

import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.callbacks
from keras.callbacks import Callback
import sklearn.metrics as sklm
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt 

inputs = np.loadtxt("6_places_4_param_unnormalized_145.csv", delimiter = ",")
outputs = np.loadtxt("FeaturesMullikenEditedCropped.csv", delimiter =",")
energy = np.loadtxt("energies.csv", delimiter =",")
labels= np.loadtxt("labelscropped.csv", delimiter =",",dtype=str)
labels=labels[1:-1]
#print(labels)
outputs=outputs[np.argsort(outputs[:, 0])]
ids = outputs[:,0]
output = np.append(outputs,energy,axis=1)
output=output[:,1:-1]
FullInp = inputs[:,1:]

inp_train, inp_test, out_train, out_test = train_test_split(FullInp, output, test_size=0.8)
scaler = StandardScaler()
trainX = inp_train
trainY = out_train
testX = inp_test
testY = out_test
trans_trainX = pd.DataFrame(scaler.fit_transform(trainX))
trans_testX = pd.DataFrame(scaler.fit_transform(testX))

trans_trainY = np.array([])
for q in range(0, 15):
	labstrt = 0+q
	labend = 1+q
	mean = np.mean(trainY, axis = 0)
	mean = np.array(mean)
	mean_reshape = np.reshape(mean, (1,15))
	std = np.std(trainY, axis =0)
	std = np.array(std)
	std_reshape = np.reshape(std, (1,15))
	labnorm = (((trainY[:, labstrt:labend]) - float(mean_reshape[:, labstrt:labend]))/float(std_reshape[:, labstrt:labend]))
	if trans_trainY.shape[0]==0:
		trans_trainY=labnorm
	else:
		trans_trainY=np.append(trans_trainY,labnorm, axis = 1)
trans_testY = np.array([])
for q in range(0, 15):
	labstrt = 0+q
	labend = 1+q
	mean = np.mean(testY, axis = 0)
	mean = np.array(mean)
	mean_reshape = np.reshape(mean, (1,15))
	std = np.std(testY, axis =0)
	std = np.array(std)
	std_reshape = np.reshape(std, (1,15))
	labnorm = (((testY[:, labstrt:labend]) - float(mean_reshape[:, labstrt:labend]))/float(std_reshape[:, labstrt:labend]))
	if trans_testY.shape[0]==0:
		trans_testY=labnorm
	else:
		trans_testY=np.append(trans_testY,labnorm, axis = 1)

trainfile = np.append(trans_trainX, trans_trainY, axis = 1)
testfile = np.append(trans_testX, trans_testY, axis = 1)

testlen = 24
trainlen = 42

trainset = trainfile[:,0:trainlen] 
testset = testfile[:,0:testlen]

mollength = 145
moleculeID = np.reshape(ids, (mollength,1))
moleculeID = moleculeID.astype(int)
testID = np.reshape(testset[:,0], (116,1))

#collects the test results of individual runs
testIndividPreds = [[None]*16]*mollength 
#collects the average test results
testAverage = [[None]*15]*mollength 
#check if 29 and 28 are correct for this

def ml(trainset, testset, moleculeID):
	global testIndividPreds, testAverage
	trainFeatures = trainset[:, 0:testlen]
	trainLabels = trainset[:,-15:]
	#testing set with labels
	X = testset[:, 0:testlen]

	model = Sequential()
	model.add(Dense(20, input_dim = (testlen-1), activation = 'tanh'))
	model.add(Dense(18, activation = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")))
	model.add(Dense(18, activation = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")))
	model.add(Dense(16, activation = 'tanh'))
	model.add(Dense(15, activation = 'linear'))
	#opt = keras.optimizers.Adam(learning_rate = 0.0003)
	model.compile(loss = 'mse', optimizer = 'adam')
	#we want to exclude the first column (the molecule name) of trainFeatures but how does the testlen-1 know which column to toss?
	#maybe it's not the testlen-1 that tosses the molecule name, it's 1:testlen?

	model.fit(trainFeatures[:,1:testlen], trainLabels, epochs = 220, batch_size = 8, validation_split = .1, verbose = 0)
	testPrediction = model.predict(X[:,1:testlen])
	
	if (np.any(testAverage)==True):
		testAverage = np.append(testAverage, testPrediction, axis = 0) 
	else:
		testAverage = testPrediction
		

	#add the molecule names
	testPrediction2 = np.concatenate([testID, testPrediction], axis = 1)
	#the individual runs have the molecule names
	if (np.any(testIndividPreds)==True):
		testIndividPreds = np.append(testIndividPreds, testPrediction2,axis = 0)
	else:
		testIndividPreds = testPrediction2

i = 0
#train/test 30 times
n = 30
print('\n ML')
while i < n:
	ml(trainset, testset, moleculeID)
	i += 1

testIndividPreds = testIndividPreds.reshape((n,116,16))

#the average of all runs is taken
testAverage = testAverage.reshape((n,116,15))
mltestmean = np.mean(testAverage, axis = 0)
#the names of the molecules are added on (note that same order is kept)
mltest = np.append(testID, mltestmean, axis = 1)
#mltest = np.append(mltest, testfile[:, testlen:].reshape(116,28), axis = 1)
#standard deviation
mlteststd = np.append(testID, np.std(testAverage, axis = 0), axis = 1)

#mean square error
mse = np.sum(np.square(testfile[:,testlen:].reshape(116,15)-mltestmean),axis=0)/145
print(mse)
mltestingvalue = np.mean(np.array(mse), axis = 0)
print(mltestingvalue)

#outputs correllation graphs for each of the 15 properties
for q in range(0,15):
	labstrt = 0+q
	labend = 1+q
	prediction  = np.array(mltestmean[:, labstrt:labend])
	actual = np.array(trans_testY[:, labstrt:labend])
	#creating linear regression between actual and presicted property values per label
	slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(prediction,actual)
	#graph creation
	title=("%s : RSquared = %f" % (labels[labstrt],r_value**2))
	plt.title(title)
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.scatter(prediction,actual)
	plt.show()
