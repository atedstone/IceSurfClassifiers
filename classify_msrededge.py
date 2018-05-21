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
HCRF_file = '/scratch/field_spectra/HCRF_master_machine_snicar.csv'
HCRF_classes = '/home/at15963/Dropbox/work/data/field_spectra_classes.csv'
spectra = xarray_classify.load_hcrf_data(HCRF_file, wvls, hcrf_classes=HCRF_classes)

# Split dataset and train classifier
train_X, train_Y, test_X, test_Y = xarray_classify.train_test_split(spectra)
clf_RF = xarray_classify.train_RF(train_X, train_Y)

xarray_classify.test_performance(clf_RF, spectra, train_X, train_Y, test_X, test_Y)

# Save classifier
xarray_classify.save_classifier(clf_RF, '/scratch/UAV/clf_RF_RedEdge.pkl')


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
# labs = spectra.filter(items=['numeric_label','label'])
# labs.groupby('label').mean()

# figure(),predicted.plot(cmap=plt.cm.get_cmap('viridis', 6))

# next up - pickle classifier so it can be re-used between images

# def calculate_albedo(band):
#     # create albedo array by applying Knap (1999) narrowband - broadband conversion
#     albedo_array = np.array([0.726*(arrays[1]-0.18) - 0.322*(arrays[1]-0.18)**2 - 0.015*(arrays[4]-0.2) + 0.581*(arrays[4]-0.2)])
