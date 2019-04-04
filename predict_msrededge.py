import pandas as pd
import xarray as xr
import xarray_classify
import joblib
from osgeo import gdal, osr
import numpy as np
import georaster

from timeit import default_timer as tic


clf_id = '20190130_171930'
clf_RF = joblib.load('/scratch/UAV/L3/classifiers/clf_RF_RedEdge_snicarsnow_' + clf_id + '.pkl')

output_path = '/scratch/UAV/L3/'

fn_path = '/scratch/UAV/uav2017_commongrid_bandcorrect/'
flights = {
'20170715':'uav_20170715_refl_5cm_commongrid_epsg32622_bilinear.nc',
'20170720':'uav_20170720_refl_5cm_v2_epsg32622native_commongrid.nc',
'20170721':'uav_20170721_refl_5cm_epsg32622native_commongrid.nc',
'20170722':'uav_20170722_refl_5cm_commongrid_epsg32622_bilinear.nc',
'20170723':'uav_20170723_refl_5cm_commongrid_epsg32622_bilinear.nc',
'20170724':'uav_20170724_refl_5cm_commongrid_epsg32622_bilinear.nc'
}


# fn_path = '/scratch/UAV/photoscan_outputs_2018/'
# flights = {
# 	'20180724_PM':'uav_20180724_PM_refl.nc'
# }

setup = False



for flight in flights:

	print(flight)

	# Classify data
	uav = xr.open_dataset(fn_path + flights[flight],chunks={'x':2000,'y':2000}) 

	# Correct using ground-UAV comparisons from compare_hcrf_uav.py
	uav['Band1'] -= 0.17
	uav['Band2'] -= 0.18
	uav['Band3'] -= 0.15
	uav['Band4'] -= 0.16
	uav['Band5'] -= 0.1

	b_ix = pd.Index([1,2,3,4,5],name='b') 

	concat = xr.concat([uav.Band1, uav.Band2, uav.Band3, uav.Band4, uav.Band5], b_ix)
	# Mask nodata areas
	#concat = concat.where(concat.sum(dim='b') > 0)

	predicted = xarray_classify.classify_dataset(concat, clf_RF)

	# Just look at a subset area in this case (slice)
	#predicted = xarray_classify.classify_dataset(concat.isel(x=slice(3000,3500),y=slice(3000,3500)), clf_RF)

	# Calculate albedo
	#uav = uav.isel(x=slice(3000,3500),y=slice(3000,3500))
	#albedo = 0.726*(uav['Band2']-0.18) - 0.322*(uav['Band2']-0.18)**2 - 0.015*(uav['Band4']-0.2) + 0.581*(uav['Band4']-0.2)
	t1 = tic()
	albedo = 0.726*uav['Band2'] - 0.322*uav['Band2']**2 - 0.015*uav['Band4'] + 0.581*uav['Band4']
	print('xarray albedo (seconds): ', tic()-t1)	

	# Save outputs

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
		uav_gr = georaster.SingleBandRaster('NETCDF:"%s%s":Band1' %(fn_path, flights[flight]),
			load_data=False)
		grid_lon, grid_lat = uav_gr.coordinates(latlon=True)
		uav = xr.open_dataset(fn_path + flights[flight],chunks={'x':1000,'y':1000}) 
		coords_geo = {'y': uav['y'], 'x': uav['x']}
		uav = None

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

	albedo.encoding = {'dtype':'int16', 'scale_factor':0.001, 'zlib':True, '_FillValue':-9999}
	albedo.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
	albedo.attrs['units'] = 'dimensionless'
	albedo.attrs['grid_mapping'] = 'universal_transverse_mercator'

	t2 = tic()
	ds = xr.Dataset({'classified':predicted,
		'albedo':albedo,
		'universal_transverse_mercator':crs,
		'lon':lon_da,
		'lat':lat_da})
	print('Dataset generation (seconds):', tic()-t2)

	ds = ds.transpose('y','x')

	# Main metadata
	ds.attrs['Conventions'] = 'CF-1.4'
	#ds.attrs['history'] = ''
	ds.attrs['institution'] = 'University of Bristol (Andrew Tedstone)'
	ds.attrs['title'] = 'Greenland ice sheet UAV (MicaSense RedEdge) imagery converted to classification and albedo maps'

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

	save_fn = flights[flight][:-3] + '_class_clf' + clf_id + '.nc'
	t3 = tic()
	ds.to_netcdf('%s%s' %(output_path, save_fn), format='NetCDF4')
	print('netcdf output (seconds):',tic()-t3)

	ds = None

	t4 = tic()
	alb_gr = uav_gr
	alb_gr.r = np.flipud(albedo.values)
	save_fn = flights[flight][:-3] + '_albedo_clf' + clf_id + '.tif'
	alb_gr.save_geotiff('%s%s' %(output_path, save_fn))

	clas_gr = uav_gr
	clas_gr.r = np.flipud(predicted.values)
	save_fn = flights[flight][:-3] + '_cla_clf' + clf_id + '.tif'
	clas_gr.save_geotiff('%s%s' %(output_path, save_fn))	
	print('Geotiff output (seconds):', tic()-t4)

	alb_gr = None
	clas_gr = None

	uav = None
	concat = None
	albedo = None
	predicted = None