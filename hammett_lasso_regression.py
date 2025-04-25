######
#lasso regression on hammett nn data
#created may 21, 2020
#last edited by athena flint
######

import numpy as np
import random
import keras.callbacks
from keras.callbacks import Callback
import sklearn.metrics as sklm
from sklearn import linear_model
from sklearn.linear_model import Ridge, RidgeCV
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings, sys
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
import seaborn as sns
import pandas as pd 

if not sys.warnoptions:
	warnings.simplefilter("ignore")

#Bringing in data
inputs = np.loadtxt("6_places_4_param_unnormalized_145.csv", delimiter = ",")
outputs = np.loadtxt("FeaturesMullikenEditedLasso.csv", delimiter =",")
energy = np.loadtxt("energies.csv", delimiter =",")
labels= np.loadtxt("labelslasso.csv", delimiter =",",dtype=str)
labels=labels[1:-2]
outputs=outputs[np.argsort(outputs[:, 0])]
ids = outputs[:,0]
output = np.append(outputs,energy,axis=1)
output=output[:,1:-1]

scaler = StandardScaler
FullInp = inputs[:,1:]

#collects the test results of individual runs
lastestInd = np.array([])
#collects the average test results
lastestAvg = np.array([])

def las(inp_train, out_train, inp_test, out_test, ids, l):
	global lastestInd, lastestAvg
	n = 0
	#training set without labels
	trainX = inp_train
	trainY = out_train[:,0]
	FullInp = inputs[:,1:]

	scaler = StandardScaler()
	#lasmod = linear_model.Lasso(alpha=0.000001, max_iter = 100000)
	norm_trainX = pd.DataFrame(scaler.fit_transform(trainX))
	lasmod = linear_model.LassoCV(max_iter = 10000)
	lasmod.fit(norm_trainX,trainY)
	lastrainrsquared = lasmod.score(norm_trainX, trainY)
	norm_FullInp = pd.DataFrame(scaler.transform(FullInp))
	lasrsquared=lasmod.score(norm_FullInp,norm_out[:,l])
	#print(lasmod.alpha_)
	#print(lasmod.coef_,lasmod.intercept_)
	#print(lasrsquared)
	#predicts energies
	if ((lasmod.score(norm_FullInp,norm_out[:,l]) < (0.8*lastrainrsquared)) or (lasmod.score(FullInp,norm_out[:,l]) > (1.2*lastrainrsquared))):
		n=1
		return n
	laspred = lasmod.predict(norm_FullInp)
	laspred=np.reshape(laspred,(laspred.shape[0],1))
	#add the molecule names
	#laspred2 = np.concatenate([mols, laspred.reshape(mollength,1)], axis = 1)
	#the individual runs have the molecule names
	if lastestInd.shape[0]==0:
		lastestInd=np.array([lasrsquared])
	else:
		lastestInd=np.append(lastestInd,np.array([lasrsquared]),0)

	if lastestAvg.shape[0]==0:
		lastestAvg=laspred
	else:
		lastestAvg=np.append(lastestAvg,laspred,1)
	return n

rsquaredavg=0

#Used to move over range of properties selecting one at a time
for q in range(0,28):
	labstrt=0+q
	labend=1+q
	mean_reshape = np.mean(np.reshape(output[:,labstrt:labend], (1,145)), axis = 1)
	labmean = np.array(mean_reshape)
	labmean = float(labmean[0])
	stdev_reshape = np.std(np.reshape(output[:,labstrt:labend], (1,145)), axis = 1)
	labstdev = np.array(stdev_reshape)
	labstdev = float(labstdev[0])
	labtrans = ((output[:, labstrt:labend]) - labmean) / labstdev
	norm_out = np.append(labtrans,np.reshape(ids,(ids.shape[0],1)),1)
	for l in range(0,(labend-labstrt)):# used to select output value
		for k in range(17,18): #used for test size selection
			for j in range(0,20): #How many linear regressions to do.
				inp_train, inp_test, out_train, out_test = train_test_split(FullInp, norm_out, test_size=(0.05*k))
				las(inp_train, out_train, inp_test, out_test, ids, l)

			lastestAvg = (lastestAvg * labstdev) + labmean
			norm_out = np.append(output[:, labstrt:labend], np.reshape(ids,(ids.shape[0],1)),1)

			if lastestAvg.shape[0] != 0:
			#Get average for prediction 
				avg = np.mean(lastestAvg,1)
				rsquaredavg = np.mean(lastestInd,0)
				eng=np.reshape(norm_out[:,l],(norm_out[:,l].shape[0],1))
				if (l+labstrt)<26:
					title=("%s : Rsquared %f" % ((labels[l+labstrt]),rsquaredavg))
				elif (l+labstrt)==26:
					title=("Scaled SCF Energy : Rsquared %f" % (rsquaredavg))
				elif (l+labstrt)==27:
					title=("Scaled Gibbs Free Energy : Rsquared %f" % (rsquaredavg))
				plt.title(title)
				plt.xlabel("Predicted")
				plt.ylabel("Actual")
				plt.scatter(avg,eng)
				plt.show()
			
			#for i in range(0,avg.shape[0]):
			#        print(avg[i],norm_out[i,0],norm_out[i,2])
			if (l+labstrt)<26:
				print("test_size = %f , r squared = %f Output Feature: %s" % ((k*0.05),rsquaredavg,labels[l+labstrt]))
			elif (l+labstrt)==26:
				print("test_size = %f , r squared = %f Output Feature: Scaled SCF Energy" % ((k*0.05),rsquaredavg))
			elif (l+labstrt)==27:
				print("test_size = %f , r squared = %f Output Feature: Scaled Gibbs Free Energy" % ((k*0.05),rsquaredavg))
			#collects the test results of individual runs
			lastestInd = np.array([])
			#collects the average test results
			lastestAvg = np.array([])
			avg = 0
			rsquaredavg=0
