"""
Scikit-Learn-based classification of remotely-sensed imagery using ground
reflectance spectra as training dataset.

Remotely-sensed imagery are accepted as xarray DataArrays opened using the 
dask backend.

Author: Andrew Tedstone, 30 April 2018 (a.j.tedstone@bristol.ac.uk)

Based on jmcook1186/IceSurfClassifiers/machine_black_bloom_UAV.py 3c19a1a

"""

import pandas as pd
import xarray as xr
import sklearn_xarray
import numpy as np

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from timeit import default_timer as tic


def load_hcrf_data(hcrf_file, wvls, hcrf_classes=None):
	"""
	Load HCRF data corresponding to requested multispectral bands

	Arguments:
	hcrf_file : path to file containing HCRF data (columns=spectra, 
		rows=wavelengths)
	hcrf_classes : path to file containing HCRF classes 
		(columns=spectrum id, label, numeric_label; rows=spectra)
	wvls : pd.DataFrame with columns 'low' and 'high' in units nanometres
		if requesting central wavelength then set low and high the same.

	returns: pd.DataFrame spectra

	"""

	hcrf_master = pd.read_csv(hcrf_file, index_col=0)
	hcrf_master.index = np.arange(350,2500,1)

	# Create column for each multispectral band
	store = []
	for key, wvl in wvls.iterrows():
		r = hcrf_master.loc[wvl.low:wvl.high]
		if len(r.shape) > 1:
			# Segment of spectrum requested
			r = r.mean()
			midpoint = int(wvl.low + ((wvl.high - wvl.low) / 2))
			r.name = 'R' + str(midpoint)
		else:
			# Single wavelength requested
			r.name = 'R' + (wvl.low)
		store.append(r)

	# Concatenate to df
	df = pd.concat(store, axis=1)

	# Add surface classifications if requested
	# (these are needed for scikit-learn-based label classification)
	print(hcrf_classes)
	if hcrf_classes is not None:

		if type(hcrf_classes) is str:
			hcrf_class = pd.read_csv(hcrf_classes, index_col=0)
		else:
			hcrf_class = hcrf_classes
		# Remove errant leading/trailing spaces
		hcrf_class.index = pd.Index([v.strip() for v in hcrf_class.index])
		df = df.join(hcrf_class)

		# Remove (1) unclassified spectra and (2) classified spectra missing associated data
		df = df[~pd.isnull(df.numeric_label)]

	# Add 'unknown' option to training set
	d = {}
	for col in df.filter(like='R').columns:
		d[col] = 0
	if hcrf_classes is not None:
		d['label'] = 'UNKNOWN'
		d['numeric_label'] = 0
	spectra = pd.concat([df, pd.DataFrame(d, index=['unknown'])], axis=0)
	spectra = spectra[pd.notnull(spectra)]
	spectra = spectra.dropna()

	return spectra



def train_test_split(spectra, test_size=0.2):
	""" Split spectra into training and testing data sets

	Arguments:
	spectra : pd.DataFrame of spectra (each spectra = row, columns = bands)
	test_size

	returns training and testing datasets
	"""

	# Split into test and train datasets
	features = spectra.drop(labels=['label', 'numeric_label'], axis=1)
	labels = spectra.filter(items=['numeric_label'])
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(features, labels, 
	    test_size=test_size)

	# Convert training and test datasets to DataArrays
	X_train_xr = xr.DataArray(X_train, dims=('samples','b'), coords={'b':features.columns})
	Y_train_xr = xr.DataArray(Y_train, dims=('samples','numeric_label'))

	return X_train_xr, Y_train_xr, X_test, Y_test



def train_RF(X_train_xr, Y_train_xr):
	"""
	Train a Random Forest Classifier (wrapped for xarray)

	"""

	clf_RF = sklearn_xarray.wrap(
	    RandomForestClassifier(n_estimators=1000, max_leaf_nodes=16, n_jobs=-1), 
	    sample_dim='samples', reshapes='b')

	clf_RF.fit(X_train_xr, Y_train_xr)

	return clf_RF



def classify_dataset(ds, clf):
	"""
	Classify a dataset using a trained classifier

	Arguments:
	ds : xarray dataset with coordinates b, y, x (where b will be reduced)
	clf : a trained classifier

	Returns:
	xr.DataArray of classified data

	"""

	### Prediction phase
	stacked = ds.stack(allpoints=['y','x'])
	# DataArray 'matrix' needs to have exactly the same layout/labels as the training DataArray.
	stackedT = stacked.T
	stackedT = stackedT.rename({'allpoints':'samples'})

	t1 = tic()
	pred = clf.predict(stackedT).compute()
	print('Time taken (seconds): ', tic()-t1)

	# Unstack back to x,y grid
	predu = pred.unstack(dim='samples')

	return predu