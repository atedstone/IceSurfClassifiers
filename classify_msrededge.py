""" 
Classifier driver script for MicaSense RedEdge sensor

Author: Andrew Tedstone (a.j.tedstone@bristol.ac.uk)

"""
import pandas as pd
import xarray as xr
import xarray_classify

# Load ground reflectance spectra corresponding to red-edge sensor
rededge = pd.read_excel('/home/at15963/Dropbox/work/black_and_bloom/multispectral-sensors-comparison.xlsx',
	sheet_name='RedEdge')
wvl_centers = rededge.central
wvls = pd.DataFrame({'low':wvl_centers, 'high':wvl_centers})
HCRF_file = '/scratch/field_spectra/HCRF_master.csv'
HCRF_classes = '/home/at15963/Dropbox/work/data/field_spectra_classes.csv'
spectra = xarray_classify.load_hcrf_data(HCRF_file, HCRF_classes, wvls)

# Split dataset and train classifier
train_X, train_Y, test_X, test_Y = xarray_classify.train_test_split(spectra)
clf_RF = xarray_classify.train_RF(train_X, train_Y)

# Classify data
uav20 = xr.open_dataset('uav_20170720_refl_5cm.nc',chunks={'x':1000,'y':1000}) 
b_ix = pd.Index([1,2,3,4,5],name='b') 
concat = xr.concat([uav20.Band1, uav20.Band2, uav20.Band3, uav20.Band4, uav20.Band5], b_ix)
# Just look at a subset area in this case (slice)
predicted = xarray_classify.classify_dataset(concat.isel(x=slice(3000,3500),y=slice(3000,3500)), clf_RF)

"""
https://gist.github.com/jakevdp/8a992f606899ac24b711
# use plt.cm.get_cmap(cmap, N) to get an N-bin version of cmap
plt.scatter(iris.data[:, 0], iris.data[:, 1], s=30,
            c=iris.target, cmap=plt.cm.get_cmap('Greens', 3))

# This function formatter will replace integers with target names
formatter = plt.FuncFormatter(lambda val, loc: iris.target_names[val])

# We must be sure to specify the ticks matching our target names
plt.colorbar(ticks=[0, 1, 2], format=formatter);

# Set the clim so that labels are centered on each block
plt.clim(-0.5, 2.5)
"""

# Lookup table of text label vs numeric label
labs = spectra.filter(items=['numeric_label','label'])
labs.groupby('label').mean()