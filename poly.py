import os
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression as LR
import numpy as np
from scipy import optimize

insert_extra_data = 1

#trimmed 0.1: 102.5; 0.005: 184.1 from linear bend
#trimmed 0.1: 932.8; 0.005: 656.0; 0.0075, 846.0 from antisymmetric
#negate #trimmed 0.1: 495.0 from symmetric

steps=np.array([0.020,0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.100])
#freqs=np.array([894.7,896.1,897.4,898.7,900.2,901.9,903.7,905.7,908.0,910.4,913.0,915.8,918.8,922.0,925.4]) #antisymmetric
freqs=np.array([478.8,479.2,479.7,480.2,480.8,481.5,482.3,483.2,484.1,485.2,486.3,487.5,488.9,490.3,491.8,495.0]) #symmetric
#freqs=np.array([91.3,91.4,91.7,92,92.5,93,93.6,94.3,95,95.7,96.6,97.4,98.4,99.3,100.4]) #linear bend

exsteps=np.array([0.005,0.0075,0.010,0.0125,0.015])
#aux=np.array([879.4,888.6,892.0]) #antisymmetric
aux=np.array([475.5,478.0,478.3,478.4,478.5]) #symmetric
#aux=np.array([108.8,96.2,93.0,91.9]) #linear bend

if insert_extra_data == 1:
	steps = np.append(exsteps,steps,axis=0)
	freqs = np.append(aux,freqs,axis=0)

print("\n")
deg=2
while deg <= 20:
	print("For a polynomial of degree " + str(deg) + ":")
	poly = PF(degree=deg, include_bias=False)
	poly_features = poly.fit_transform(steps.reshape(-1,1))
	poly_reg_model = LR()
	mod=poly_reg_model.fit(poly_features, freqs)
	r2 = mod.score(poly_features, freqs)
	print("R2: " + str(r2))
	#print(poly_reg_model.intercept_, poly_reg_model.coef_)
	num = steps.shape[0]
	var = 1
	adj = 1 - (((1 - r2) * (num - 1)) / (num - var - 1))
	print("Adjusted R2: " + str(adj))


	chunks = np.array([])
	for i in range(0,deg):
		factor = i + 1
		chunk = (factor * poly_reg_model.coef_[i])
		if chunks.shape[0]==0:
			chunks = np.array([chunk])
		else:
			chunks = np.append(chunks,np.array([chunk]))
	#print(chunks)

	leng=chunks.shape[0]
	def f(x):
		func=0
		for i in range(0,leng):
			func += float(chunks[i]) * x**i
		return func

	#locate left bracket with opposing sign
	abra = 0.0
	bsign = np.sign(f(0.3))
	if np.sign(f(abra)) == bsign:
		while np.sign(f(abra)) == bsign:
			abra += 0.05
			if abra == 0.3:
				print("No reasonable root available.")
				break
	if abra != 0.3:
		sol = optimize.root_scalar(f, bracket=[abra,0.3], method='brentq')
		print("Step size: " + str(sol.root))

	print("\n")
	deg+=2


