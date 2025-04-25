#####
#testing data leakage/normalization with lassoCV
#created jun 8 2020
#last edited by athena flint
#####

import numpy as np
import random
import statistics
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils import normalize
import keras.callbacks
from keras.callbacks import Callback
import sklearn.metrics as sklm
from sklearn import linear_model, kernel_ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
import pandas as pd
import seaborn as sns

#Bringing in data
inputs = np.loadtxt("6_places_4_param_unnormalized_145.csv", delimiter = ",")
outputs = np.loadtxt("FeaturesMullikenEdited.csv", delimiter =",")
energy = np.loadtxt("energies.csv", delimiter =",")
labels= np.loadtxt("labels.csv", delimiter =",",dtype=str)
labels=labels[1:-2]
outputs=outputs[np.argsort(outputs[:, 0])]
ids = outputs[:,0]
output = np.append(outputs,energy,axis=1)
output=output[:,1:-1]
#Combinations of inputs
Tol_inp = np.concatenate([[inputs[:,1],inputs[:,5],inputs[:,9],inputs[:,13],inputs[:,17],inputs[:,21]]],1)
Tol_inp=np.transpose(Tol_inp)
Lev_inp = np.concatenate([[inputs[:,2],inputs[:,6],inputs[:,10],inputs[:,14],inputs[:,18],inputs[:,22]]],1)
Lev_inp=np.transpose(Lev_inp)
Ham_inp = np.concatenate([[inputs[:,3],inputs[:,4],inputs[:,7],inputs[:,8],inputs[:,11],inputs[:,12],inputs[:,15],inputs[:,16],inputs[:,19],inputs[:,20],inputs[:,23],inputs[:,24]]],1)
Ham_inp=np.transpose(Ham_inp)
Ham_p_inp = np.concatenate([[inputs[:,4],inputs[:,8],inputs[:,12],inputs[:,16],inputs[:,20],inputs[:,24]]],1)
Ham_p_inp=np.transpose(Ham_p_inp)
Ham_m_inp = np.concatenate([[inputs[:,3],inputs[:,7],inputs[:,11],inputs[:,15],inputs[:,19],inputs[:,23]]],1)
Ham_m_inp=np.transpose(Ham_m_inp)
HT_inp = np.concatenate([[inputs[:,1],inputs[:,3],inputs[:,4],inputs[:,5],inputs[:,7],inputs[:,8],inputs[:,9],inputs[:,11],inputs[:,12],inputs[:,13],inputs[:,15],inputs[:,16],inputs[:,17],inputs[:,19],inputs[:,20],inputs[:,21],inputs[:,23],inputs[:,24]]],1)
HT_inp=np.transpose(HT_inp)
HL_inp = np.concatenate([[inputs[:,2],inputs[:,3],inputs[:,4],inputs[:,6],inputs[:,7],inputs[:,8],inputs[:,10],inputs[:,11],inputs[:,12],inputs[:,14],inputs[:,15],inputs[:,16],inputs[:,18],inputs[:,19],inputs[:,20],inputs[:,22],inputs[:,23],inputs[:,24]]],1)
HL_inp=np.transpose(HL_inp)
TL_inp = np.concatenate([[inputs[:,1],inputs[:,2],inputs[:,5],inputs[:,6],inputs[:,9],inputs[:,10],inputs[:,13],inputs[:,14],inputs[:,17],inputs[:,18],inputs[:,21],inputs[:,22]]],1)
TL_inp=np.transpose(TL_inp)

#Normalization
#for column in range(0, 35)
	#labstrt=0+q
	#labend=1+q
	#labstdev = np.std(np.reshape(output[:, labstrt:labend]),(1,145))
	#labmean = np.mean(np.reshape(output[:, labstrt:labend]),(1,145))
	#labtrans = ((output[:,labstrt:labend]) - labmean) / labstdev


#scaler = StandardScaler()
norm_out = np.append(output,np.reshape(ids,(ids.shape[0],1)),1)
#norm_out1 = np.append(normalize(energy[:,:-1],axis=0),np.reshape(ids,(ids.shape[0],1)),1)
#print(np.append(output[:,-3:-1],energy[:,:-1],1))

#print(norm_out)
#print(norm_out)
#print(str(output[0,24])+","+str(output[0,-3]))
#plt.scatter(output[:,24],output[:,-3])
#plt.show()
#print(outputs.shape)
#Select the dataset. For full dataset use inputs[:,1:] to remove ids
FullInp=inputs[:,1:]
#FullInp=Ham_inp
# alldat=np.append(FullInp,norm_out,axis=1)
# alldat=np.transpose(alldat)
# # #print(alldat)
# with open("correlationMat.csv",'w+') as temp:
#     corrmat=np.corrcoef(alldat)
#     np.savetxt("correlationMat1.csv",corrmat,delimiter=',')
#     print(corrmat.shape)
#     for i in range(corrmat.shape[0]):
#         temp.write(str(corrmat[i,:])+"\n")

# f, ax = plt.subplots(figsize=(10,10))
# cmap = sns.diverging_palette(360, 360,sep=1, as_cmap=True)

# #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#  #           square=True, linewidths=.5, cbar_kws={"shrink": .5})
# sns.heatmap(corrmat,  cmap=cmap, center=0,
#             square=True, linewidths=.25, cbar_kws={"shrink": .5})

# plt.show()

#print(FullInp)

#print("%d inp train, %d inp test, %d out train, %d out test" % (inp_train.shape[1], inp_test.shape[1], out_train.shape[1], out_test.shape[1]))
#print(norm_out)

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
	#testX = inp_test
	#testY = out_test[:,0]
	FullInp = inputs[:,1:]

	scaler = StandardScaler()
	#lasmod = linear_model.Lasso(alpha=0.000001, max_iter = 100000)
	norm_trainX = pd.DataFrame(scaler.fit_transform(trainX))
	lasmod = linear_model.RidgeCV()
	lasmod.fit(norm_trainX,trainY)
	lastrainrsquared = lasmod.score(norm_trainX, trainY)
	norm_FullInp = pd.DataFrame(scaler.transform(FullInp))
	lasrsquared=lasmod.score(norm_FullInp,norm_out[:,l])
	#print(lasmod.alpha_)
	#print(lasmod.coef_,lasmod.intercept_)
	#print(lasrsquared)
	#predicts energies
	if ((lasmod.score(norm_FullInp,norm_out[:,l]) < (0.9*lastrainrsquared)) or (lasmod.score(norm_FullInp,norm_out[:,l]) > (1.1*lastrainrsquared))):
		n=1
		return n
	laspred = lasmod.predict(norm_FullInp)
	#print(laspred)
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
for q in range(0, 35)
	labstrt=0+q
	labend=1+q
	#labstdev = np.std(np.reshape(output[:, labstrt:labend]),(1,145))
	#labmean = np.mean(np.reshape(output[:, labstrt:labend]),(1,145))
	#labtrans = ((output[:,labstrt:labend]) - labmean) / labstdev
	norm_out = np.append(norm_out,np.reshape(ids,(ids.shape[0],1)),1)
	for l in range(0,(labend-labstrt)):# used to select output value
		for k in range(17,18): #used for test size selection
			for j in range(0,10): #How many linear regressions to do.
				inp_train, inp_test, out_train, out_test = train_test_split(FullInp, norm_out, test_size=(0.05*k))
				las(inp_train, out_train, inp_test, out_test, ids, l)

			if lastestAvg.shape[0] != 0:
			#Get average for prediction 
				avg = np.mean(lastestAvg,1)
				rsquaredavg = np.mean(lastestInd,0)
				eng=np.reshape(norm_out[:,l],(norm_out[:,l].shape[0],1))
				if (l+labstrt)<33:
					title=("%s : Rsquared %f" % ((labels[l+labstrt]),rsquaredavg))
				elif (l+labstrt)==33:
					title=("Scaled SCF Energy : Rsquared %f" % (rsquaredavg))
				elif (l+labstrt)==34:
					title=("Scaled Gibbs Free Energy : Rsquared %f" % (rsquaredavg))
				plt.title(title)
				plt.xlabel("Predicted")
				plt.ylabel("Actual")
				plt.scatter(avg,eng)
				plt.show()
			
			#for i in range(0,avg.shape[0]):
			#        print(avg[i],norm_out[i,0],norm_out[i,2])
			if (l+labstrt)<33:
				print("test_size = %f , r squared = %f Output Feature: %s" % ((k*0.05),rsquaredavg,labels[l+labstrt]))
			elif (l+labstrt)==33:
				print("test_size = %f , r squared = %f Output Feature: Scaled SCF Energy" % ((k*0.05),rsquaredavg))
			elif (l+labstrt)==34:
				print("test_size = %f , r squared = %f Output Feature: Scaled Gibbs Free Energy" % ((k*0.05),rsquaredavg))
			#collects the test results of individual runs
			lastestInd = np.array([])
			#collects the average test results
			lastestAvg = np.array([])
			avg = 0
			rsquaredavg=0
