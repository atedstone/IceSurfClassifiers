import pandas as pd
import xarray as xr
import xarray_classify
import joblib
from osgeo import gdal, osr
import numpy as np
import georaster

# Stats for classifier in use
#Accuracy =  0.952380952381 F1_Score =  0.949801013523 Recall =  0.952380952381 Precision =  0.950284090909
""" For RedEdge_snicarsnow.pkl
Accuracy =  0.940217391304 F1_Score =  0.934766392673 Recall =  0.940217391304 Precision =  0.942895709008

Feature Importances
(relative importance of each feature (wavelength) for prediction)

R475 0.233517596955
R560 0.260483344995
R668 0.20433026048
R717 0.152076480941
R840 0.149592316629
"""
clf_RF = joblib.load('/scratch/UAV/L3/classifiers/clf_RF_Sentinel2_snicarsnow_20190305_170648.pkl')
#clf_RF = joblib.load('/home/at15963/S2/clf_RF_Sentinel2_20m_snicarsnow.pkl')

fn_path = '/home/at15963/projects/uav/data/S2/'
#fn_path = '/scratch/atlantis1/at15963/L0data/S2/'
images = {
'20170720':'S2A_MSIL2A_20170720T145921_N0205_R125_T22WEV_20170720T150244.SAFE_20m/S2A_MSIL2A_20170720T145921_N0205_R125_T22WEV_20170720T150244_20m.data/'
#'20170721':'S2B_MSIL2A_20170721T151909_N0205_R068_T22WEV_20170721T152003.SAFE_20m/S2B_MSIL2A_20170721T151909_N0205_R068_T22WEV_20170721T152003_20m.data/'
}

setup = False

"""
R490 = B2
R560 = B3
R665 = B4
R705 = B5
R740 = B6
R783 = B7
R842 = B8
R865 = B8a
R1610 = B11
R2190 = B12
"""



for image in images:

	print(image)

	# concat = xr.Dataset({
	# 	'R490':xr.open_rasterio(fn_path+images[image]+'/MTD_MSIL2A.data/B2.img'),
	# 	'R560':xr.open_rasterio(fn_path+images[image]+'/MTD_MSIL2A.data/B3.img'),
	# 	'R665':xr.open_rasterio(fn_path+images[image]+'/MTD_MSIL2A.data/B4.img'),
	# 	'R842':xr.open_rasterio(fn_path+images[image]+'/MTD_MSIL2A.data/B8.img'),
	# 	})

	# Order is critically important - has to match order of columns that pandas decided in classify script
	concat = xr.concat([
		xr.open_rasterio(fn_path+images[image]+'B11.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B12.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B2.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B3.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B4.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B5.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B6.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B7.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B8.img', chunks={'x':1000, 'y':1000}),
		xr.open_rasterio(fn_path+images[image]+'B8A.img', chunks={'x':1000, 'y':1000}),
		], pd.Index(['B11','B12','B2','B3','B4','B5','B6','B7','B8','B8A'],name='b'))
		#], pd.Index(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'],name='b'))


	#concat = xr.concat([uav.Band1, uav.Band2, uav.Band3, uav.Band4, uav.Band5], b_ix)
	# Mask nodata areas
	#concat = concat.where(concat.sum(dim='b') > 0)

	concat = concat / 10000
	predicted = xarray_classify.classify_dataset(concat.squeeze(), clf_RF)

	# Just look at a subset area in this case (slice)
	#predicted = xarray_classify.classify_dataset(concat.isel(x=slice(3000,3500),y=slice(3000,3500)).squeeze(), clf_RF)

	# Calculate albedo
	#uav = uav.isel(x=slice(3000,3500),y=slice(3000,3500))
	#albedo = 0.726*(uav['Band2']-0.18) - 0.322*(uav['Band2']-0.18)**2 - 0.015*(uav['Band4']-0.2) + 0.581*(uav['Band4']-0.2)
	
	#albedo = 0.726*uav['Band2'] - 0.322*uav['Band2']**2 - 0.015*uav['Band4'] + 0.581*uav['Band4']

	"""
	Joe:
	data[0] = B02
	data[1] = B03
	data[2] = B04
	data[3] = B05
	data[4] = B06
	data[5] = B07
	data[6] = B08
	data[7] = B8A
	data[8] = B11
	data[9] = B12
	"""

	albedo = (0.356*concat.sel(b='B2') + 0.13*concat.sel(b='B4') + 0.373*concat.sel(b='B8') + 0.085*concat.sel(b='B8A') + 0.072*concat.sel(b='B11') -0.0018 )

	# Save data

	if not setup:
		# Define projection
		srs = osr.SpatialReference()
		srs.ImportFromProj4('+init=epsg:32622')
		crs = xr.DataArray(0, encoding={'dtype':np.dtype('int8')})
		crs.attrs['projected_crs_name'] = srs.GetAttrValue('projcs')
		crs.attrs['grid_mapping_name'] = 'universal_transverse_mercator'
		crs.attrs['scale_factor_at_central_origin'] = srs.GetProjParm('scale_factor')
		crs.attrs['standard_parallel'] = srs.GetProjParm('latitude_of_origin')
		crs.attrs['straight_vertical_longitude_from_pole'] = srs.GetProjParm('central_meridian')
		crs.attrs['false_easting'] = srs.GetProjParm('false_easting')
		crs.attrs['false_northing'] = srs.GetProjParm('false_northing')
		crs.attrs['latitude_of_projection_origin'] = srs.GetProjParm('latitude_of_origin')

		## Create associated lat/lon coordinates DataArrays
		uav_gr = georaster.SingleBandRaster(fn_path+images['20170720']+'B2.img',
			load_data=False)
		grid_lon, grid_lat = uav_gr.coordinates(latlon=True)
		#uav = xr.open_dataset(fn_path + images['20160720'],chunks={'x':1000,'y':1000}) 
		coords_geo = {'y': concat['y'], 'x': concat['x']}
		#uav = None

		lon_da = xr.DataArray(grid_lon, coords=coords_geo, dims=['y', 'x'], 
			encoding={'_FillValue': -9999., 'dtype':'int16', 'scale_factor':0.000000001})
		lon_da.attrs['grid_mapping'] = 'universal_transverse_mercator'
		lon_da.attrs['units'] = 'degrees'
		lon_da.attrs['standard_name'] = 'longitude'

		lat_da = xr.DataArray(grid_lat, coords=coords_geo, dims=['y', 'x'], 
			encoding={'_FillValue': -9999., 'dtype':'int16', 'scale_factor':0.000000001})
		lat_da.attrs['grid_mapping'] = 'universal_transverse_mercator'
		lat_da.attrs['units'] = 'degrees'
		lat_da.attrs['standard_name'] = 'latitude'

		setup = True

	predicted.encoding = {'dtype':'int16', 'zlib':True, '_FillValue':-9999}
	predicted.name = 'Surface Class'
	predicted.attrs['long_name'] = 'Surface classification using Random Forests 6-class classifier'
	predicted.attrs['units'] = 'None'
	predicted.attrs['key'] = 'Unknown:0; Water:1; Snow:2; Clean Ice:3; Light Algae on Ice:4; Heavy Algae on Ice:5; Cryoconite:6' ## FIX !!
	predicted.attrs['grid_mapping'] = 'universal_transverse_mercator'

	albedo.encoding = {'dtype':'int16', 'scale_factor':0.01, 'zlib':True, '_FillValue':-9999}
	albedo.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
	albedo.attrs['units'] = 'dimensionless'
	albedo.attrs['grid_mapping'] = 'universal_transverse_mercator'

	ds = xr.Dataset({'classified':predicted,
		'albedo':albedo,
		'universal_transverse_mercator':crs,
		'lon':lon_da,
		'lat':lat_da})

	ds = ds.squeeze().transpose('y','x')

	# Main metadata
	ds.attrs['Conventions'] = 'CF-1.4'
	#ds.attrs['history'] = ''
	ds.attrs['institution'] = 'University of Bristol (Andrew Tedstone)'
	ds.attrs['title'] = 'Greenland ice sheet Sentinel 2 L2A imagery converted to classification and albedo maps'

	# Additional geo-referencing
	ds.attrs['nx'] = len(ds.x)
	ds.attrs['ny'] = len(ds.y)
	ds.attrs['xmin'] = float(ds.x.min())
	ds.attrs['ymax'] = float(ds.y.max())
	ds.attrs['spacing'] = 0.05

	# NC conventions metadata for dimensions variables
	ds.x.attrs['units'] = 'meters'
	ds.x.attrs['standard_name'] = 'projection_x_coordinate'
	ds.x.attrs['point_spacing'] = 'even'
	ds.x.attrs['axis'] = 'x'

	ds.y.attrs['units'] = 'meters'
	ds.y.attrs['standard_name'] = 'projection_y_coordinate'
	ds.y.attrs['point_spacing'] = 'even'
	ds.y.attrs['axis'] = 'y'

	save_fn = 'S2_20170720_class_clf20190305_170648.nc'
	ds.to_netcdf('%s%s' %(fn_path, save_fn), format='NetCDF4')

	ds = None

	alb_gr = uav_gr
	alb_gr.r = np.flipud(albedo.values)
	save_fn = 'S2_20170720_albedo_clf20190305_170648.tif'
	alb_gr.save_geotiff('%s%s' %(fn_path, save_fn))

	clas_gr = uav_gr
	clas_gr.r = np.flipud(predicted.values)
	save_fn = 'S2_20170720_classified_clf20190305_170648.tif'
	clas_gr.save_geotiff('%s%s' %(fn_path, save_fn))	

	alb_gr = None
	clas_gr = None