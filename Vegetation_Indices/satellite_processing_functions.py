import numpy as np
import shapely
import xarray as xr
import geopandas as gpd
from glob import glob
import pandas as pd
import netCDF4 as nc
from shapely import Polygon

#### AUXILIARY FUNCTIONS
def convert_to_string(arr, fmt):
    """
    Converts an integer representing a year, month, or day (5, for instance) into its string representation. 
    Since many of the files need '05' rather than '5', we need to use the zfill function to alter the string
    representation to contain these leading zeroes.

    Parameters
    ----------
    arr : array
        An array containing integer representations of years, months, or days
    
    fmt : string
        'year', 'month', or 'day'
    
    Returns
    -------
    string_arr
        string representation of the inputted array (arr)
    
    """
    string_arr = np.empty(len(arr), dtype=object) 
    
    if fmt == 'year':
        for i in range(len(arr)):
            string_arr[i] = str(arr[i])
    else:
        for i in range(len(arr)):
            string_arr[i] = str(str(arr[i]).zfill(2))
    return string_arr;

#### CREATING GRIDS
def new_grid(lon_min, lat_min, lon_max, lat_max, cell_size_lat, cell_size_lon):
    """
    Creates desired grid from lat_min, lat_max, lon_min, lon_max, and a cell_size.

    Returns
    -------
    grid_cells
        List of shapely geometries representing the new grid boxes.
    """
    # Creates grid
    lat_axis = np.arange(lat_min, lat_max, cell_size_lat)
    long_axis = np.arange(lon_min, lon_max, cell_size_lon)
    grid_cells = [] # Where the regridded cells will be stored

    for x0 in long_axis:
        for y0 in lat_axis:
            # Bounds for each of the boxes
            x1 = x0-cell_size_lon
            y1 = y0+cell_size_lat
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    
    return grid_cells

def global_MODIS_grid(nx, ny, step):
    """
    Creates meshgrid corresponding to points in the global MODIS grid.

    Parameters
    ----------
    nx : int
        Number of points in x direction (longitude)
    
    ny : int
        Number of points in y direction (latitude)

    step : float
        "Step" (in number of degrees) between consecutive points (e.g. (7200, 3600) translates to 0.025 degree spacing.
        
    Returns
    -------
    lat, lon
        latitude and longitude meshgrid.
    
    """
    y0 = -90 + step
    y1 = 90 - step
    x0 = -180 + step
    x1 = 180 - step
    
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y1, y0, ny) # Because the top left corner starts at maximum latitude
    lon, lat = np.meshgrid(x, y)
    lat = lat.reshape(-1,1) # No transpose because the data is in (lat, lon)
    lon = lon.reshape(-1,1)
    return lon, lat

def save_averaging_kernel(file_name, product_ds, data_type, bounds, threshold):
    """
    Helper function that saves the averaging kernel, pressure dimensions, and a priori profile from the TROPOMI dataset.

    Parameters
    ----------
    file_name : String
        The name of the TROPOMI L2 data file
    
    product_ds : xarray Dataset
        Dataset of the PRODUCT (main) TROPOMI product

    data_type : String
        Name of the TROPOMI dataset (e.g. CH4, SO2)

    bounds : array of [lon_min, lat_min, lon_max, lat_ax]
        Geographic bounds
    
    Threshold : float
        QA threshold

    Returns
    -------
    avg_kernel : DataFrame
        Dataframe containing the averaging kernel, a priori profile, and pressures
    """
    lat_lon = product_ds[['latitude', 'longitude', 'qa_value']];

    if data_type == 'CH4':
        averaging_kernel_group = xr.open_dataset(file_name, group = 'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS');
        averaging_kernel = averaging_kernel_group[['column_averaging_kernel']]
    
        input_data_group = xr.open_dataset(file_name, group = 'PRODUCT/SUPPORT_DATA/INPUT_DATA')
        input_data = input_data_group[['methane_profile_apriori', 'surface_pressure', 'pressure_interval']]

    elif data_type == 'HCHO':
        averaging_kernel_group = xr.open_dataset(file_name, group = 'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS');
        averaging_kernel = averaging_kernel_group[['averaging_kernel', 'formaldehyde_profile_apriori']]
    
        input_data_group = xr.open_dataset(file_name, group = 'PRODUCT/SUPPORT_DATA/INPUT_DATA')
        input_data = input_data_group[['tm5_constant_a', 'tm5_constant_b']]

    avg_kernel = xr.merge([lat_lon, averaging_kernel, input_data], join = 'left');

    input_data.close();
    averaging_kernel.close();

    avg_kernel = avg_kernel.to_dataframe();
    
    if bounds != 0:
        avg_kernel = avg_kernel[(avg_kernel.latitude >= bounds[1]) & (avg_kernel.latitude <= bounds[3]) & (avg_kernel.longitude >= bounds[0]) & (avg_kernel.longitude <= bounds[2])];

    avg_kernel = avg_kernel[avg_kernel.qa_value > threshold]

    if data_type == 'CH4':
        avg_kernel = avg_kernel.drop(['latitude', 'longitude', 'qa_value'], axis = 1).dropna(subset = ['column_averaging_kernel'], axis = 0);
    elif data_type == 'HCHO':
        avg_kernel = avg_kernel.drop(['latitude', 'longitude', 'qa_value'], axis = 1).dropna(subset = ['averaging_kernel'], axis = 0);

    return avg_kernel;

def save_pixel_borders(file_name, product_ds, data_type, bounds, threshold):
    """
    Helper function that saves pixel borders from the TROPOMI dataset.

    Parameters
    ----------
    file_name : String
        The name of the TROPOMI L2 data file
    
    product_ds : xarray Dataset
        Dataset of the PRODUCT (main) TROPOMI product

    data_type : String
        Name of the TROPOMI dataset (e.g. CH4, SO2)

    bounds : array of [lon_min, lat_min, lon_max, lat_ax]
        Geographic bounds
    
    Threshold : float
        QA threshold

    Returns
    -------
    pivot_geolocations : DataFrame
        Dataframe containing the four corners of the TROPOMI pixel
    """
    supporting_data = xr.open_dataset(file_name, group = 'PRODUCT/SUPPORT_DATA/GEOLOCATIONS') # Supporting data contains all the geolocations
        
    # We merge before, because sometimes the metadata is messed up (time was 0)
    if (data_type == 'CH4'):
        merged_dataset = xr.merge([product_ds[['latitude', 'longitude', 'qa_value','methane_mixing_ratio']], supporting_data], join = 'left').to_dataframe();
        merged_dataset = merged_dataset.dropna(subset = ['methane_mixing_ratio'], axis = 0);
    elif data_type == 'SO2':
        merged_dataset = xr.merge([product_ds[['latitude', 'longitude', 'qa_value','sulfurdioxide_total_vertical_column']], supporting_data], join = 'left').to_dataframe()
        merged_dataset = merged_dataset.dropna(subset = ['sulfurdioxide_total_vertical_column'], axis = 0);
        merged_dataset = merged_dataset[merged_dataset.sulfurdioxide_total_vertical_column >= -0.001] # Filtering for outliers; see User Guide for negative values
    elif data_type == 'NO2':
        merged_dataset = xr.merge([product_ds[['latitude', 'longitude', 'qa_value', 'nitrogendioxide_tropospheric_column']], supporting_data], join = 'left').to_dataframe()
        merged_dataset = merged_dataset.dropna(subset = ['nitrogendioxide_tropospheric_column'], axis = 0);
    elif data_type == 'HCHO':
        merged_dataset = xr.merge([product_ds[['latitude', 'longitude', 'qa_value', 'formaldehyde_tropospheric_vertical_column']], supporting_data], join = 'left').to_dataframe()
        merged_dataset = merged_dataset.dropna(subset = ['formaldehyde_tropospheric_vertical_column'], axis = 0);
    elif data_type == 'SIF':
        merged_dataset = xr.merge([product_ds[['latitude', 'longitude', 'SIF_Corr_743']], supporting_data], join = 'left').to_dataframe()
        merged_dataset = merged_dataset.dropna(subset = ['SIF_Corr_743'], axis = 0);

    if data_type != 'SIF':
        merged_dataset = merged_dataset[merged_dataset.qa_value > threshold];
        
    if bounds != 0:
        merged_dataset = merged_dataset[(merged_dataset.latitude >= bounds[1]) & (merged_dataset.latitude <= bounds[3]) & (merged_dataset.longitude >= bounds[0]) & (merged_dataset.longitude <= bounds[2])];
    
    supporting_data.close()
    
    # Pivot on corner!
    if data_type == 'SIF':
        pivot_geolocations = merged_dataset[['latitude_bounds', 'longitude_bounds']].reset_index("ncorner")
        pivot_geolocations = pivot_geolocations.pivot(columns="ncorner").droplevel(0, axis = 1) 
    else:
        pivot_geolocations = merged_dataset[['latitude_bounds', 'longitude_bounds']].reset_index("corner")
        pivot_geolocations = pivot_geolocations.pivot(columns="corner").droplevel(0, axis = 1) 
    pivot_geolocations.columns = ['lat_bound_1', 'lat_bound_2', 'lat_bound_3', 'lat_bound_4', 'lon_bound_1', 'lon_bound_2', 'lon_bound_3', 'lon_bound_4']
    return pivot_geolocations;

def read_scattered_TROPOMI_data(file_name, data_type, threshold = 0.5, bounds = 0, geolocation = False, averaging_kernel = False):
    """
    Reads in L2 TROPOMI datasets and outputs it as a GeoDataFrame.

    Parameters
    ----------
    file_name : string
        One filename referring to SIF dataset of interest.

    data_type : string
        Specifies what type of dataset this is

    threshold : float
        Threshold for QA thresholding (DEFAULT PARAMETER is 0.5)

    bounds : float array in the following format
        [lon_min, lat_min, lon_max, lat_max]. If you do not want any bounds and want the whole swath, don't 
        include this parameter (DEFAULT PARAMETER is 0)

    geolocation : boolean
        If true, will output the TROPOMI footprints (pixels) as geometry. If false, will output (x,y) centroid geometries.
    
    Returns
    -------
    gdf
        A GeoDataFrame (from geopandas) dataframe containing the scattered data with specified geometries.
    """
    original_data = xr.open_dataset(file_name, group = 'PRODUCT') # Contains the actual remote sensing product

    if data_type == 'CH4':
        df = original_data[['delta_time', 'time_utc', 'qa_value', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']].to_dataframe() # Takes the variables you are interested in
        df['time_utc'] = pd.to_datetime(df['time_utc']);
        
    elif data_type == 'SO2':
        df = original_data[['latitude', 'longitude', 'delta_time', 'time_utc', 'qa_value', 'sulfurdioxide_total_vertical_column', 'sulfurdioxide_total_vertical_column_precision']].to_dataframe() # Takes the variables you are interested in
        df['time_utc'] = pd.to_datetime(df['delta_time']);
        df = df[df.sulfurdioxide_total_vertical_column >= -0.001] # Filtering for outliers; see User Guide for negative values

    elif data_type == 'NO2':
        threshold = 0.75 # Required for tropospheric NO2 columns -- different from the rest!
        if averaging_kernel:
            df = original_data[['latitude', 'longitude', 'delta_time', 'time_utc', 'qa_value', 'nitrogendioxide_tropospheric_column', 'nitrogendioxide_tropospheric_column_precision', 'nitrogendioxide_tropospheric_column_precision_kernel', 'averaging_kernel']].to_dataframe() # Takes the variables you are interested in 
        else:
            df = original_data[['latitude', 'longitude', 'delta_time', 'time_utc', 'qa_value', 'nitrogendioxide_tropospheric_column', 'nitrogendioxide_tropospheric_column_precision']].to_dataframe() # Takes the variables you are interested in
        df['time_utc'] = pd.to_datetime(df['delta_time']);

    elif data_type == 'HCHO':
        df = original_data[['latitude', 'longitude', 'delta_time', 'time_utc', 'qa_value', 'formaldehyde_tropospheric_vertical_column', 'formaldehyde_tropospheric_vertical_column_precision']].to_dataframe() # Takes the variables you are interested in 
        df['time_utc'] = pd.to_datetime(df['delta_time']);

    elif data_type == 'SIF':
        df = original_data[['latitude', 'longitude', 'delta_time', 'SIF_743', 'SIF_Corr_743', 'SIF_ERROR_743', 'SIF_735', 'SIF_Corr_735', 'SIF_ERROR_735']].to_dataframe() # Takes the variables you are interested in 
        df['time'] = pd.to_datetime(df['delta_time']);
    
    if (bounds != 0):
        df = df[(df.latitude >= bounds[1]) & (df.latitude <= bounds[3]) & (df.longitude >= bounds[0]) & (df.longitude <= bounds[2])];

    if data_type != 'SIF': # SIF is already filtered by QA (L2B)
        df = df[df.qa_value > threshold] # Quality-filtering
    original_data.close()

    if len(df) == 0:
        return df; # Saves compute power if there is no data that fulfills quality threshold and geographic bounds.

    if data_type == 'SO2':
        # SO2 layer height data
        df_layer_height = xr.open_dataset(file_name, group = 'PRODUCT/SO2_LAYER_HEIGHT');
        df_layer_height = df_layer_height[['sulfurdioxide_layer_height', 'sulfurdioxide_total_vertical_column_layer_height', 'qa_value_layer_height']].to_dataframe();
        df_layer_height = df_layer_height.dropna(subset = 'sulfurdioxide_layer_height', axis = 0)
        df_layer_height = df_layer_height[df_layer_height.qa_value_layer_height > 0.5];
        df = pd.merge(df, df_layer_height, left_index = True, right_index = True, how = 'left');

    #### Start options
    if averaging_kernel:
        if (data_type != 'NO2') & (data_type != 'SIF'): # Either do not have averaging kernel (SIF) or different place to get them (NO2)
            avg_kernel = save_averaging_kernel(file_name, original_data, data_type, bounds, threshold);
            df = pd.merge(df, avg_kernel, left_index = True, right_index = True);

    if geolocation:
        pivot_geolocations = save_pixel_borders(file_name, original_data, data_type, bounds, threshold);
        df = pd.merge(df, pivot_geolocations, left_index = True, right_index = True, how = 'left');        
        df['geometry'] = df.apply(lambda x: Polygon(zip([x.lon_bound_1, x.lon_bound_2, x.lon_bound_3, x.lon_bound_4], [x.lat_bound_1, x.lat_bound_2, x.lat_bound_3, x.lat_bound_4])), axis = 1); # Create polygons representing each TROPOMI footprint
        gdf = gpd.GeoDataFrame(df, geometry = 'geometry');
    else:
        gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['longitude'], df['latitude'])) # Create a GeoDataFrame
    
    return gdf;


#### REGRIDDING DATASETS
def regrid_scattered_data(gdf, new_grid, data_type):
    """
    Grids the scattered data into a the grid created by new_grid().
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing all of the scattered (unprocessed) SIF data, with the geometries
        being (lon, lat) Points

    new_grid : GeoDataFrame
        GeoDataFrame containing the new grid to regrid everything onto

    data_type : string
        Specifies what type of dataset this is.

    Returns
    -------
    cell
        GeoDataFrame containing the mean and standard deviation of desired values at each grid box.
    """
    
    grid_cells = new_grid
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry']) # Convert to a GeoPandas dataframe

    merged = gpd.sjoin(gdf, cell, how = 'left', predicate = 'intersects', lsuffix = '_L') # Merge the regridded boxes with the original dataframe
    
    dissolve = merged.dissolve(by = "index_right", aggfunc = "mean") # Take the mean of all values.
    dissolve_std = merged.dissolve(by = "index_right", aggfunc = "std") # Take the mean of all values.
    dissolve_count = merged.dissolve(by = "index_right", aggfunc = "count") # Take the mean of all values.

    if (data_type == 'SIF'):
        cell.loc[dissolve.index, 'SIF_743'] = dissolve.SIF_Corr_743.values # Cell now contains the mean of each grid.
        cell.loc[dissolve.index, 'SIF_735'] = dissolve.SIF_Corr_735.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'SIF_743_STD'] = dissolve_std.SIF_Corr_743.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_std.index, 'SIF_735_STD'] = dissolve_std.SIF_Corr_735.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_count.index, 'SIF_743_number_of_measurements'] = dissolve_count.SIF_Corr_743.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_count.index, 'SIF_735_number_of_measurements'] = dissolve_count.SIF_Corr_735.values # Cell now contains the STDEV of each grid.
        
        cell.loc[dissolve.index, 'Raw_SIF_743'] = dissolve.SIF_743.values # Cell now contains the mean of each grid.
        cell.loc[dissolve.index, 'Raw_SIF_735'] = dissolve.SIF_735.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'Raw_SIF_743_STD'] = dissolve_std.SIF_743.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_std.index, 'Raw_SIF_735_STD'] = dissolve_std.SIF_735.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_count.index, 'Raw_SIF_743_number_of_measurements'] = dissolve_count.SIF_743.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_count.index, 'Raw_SIF_735_number_of_measurements'] = dissolve_count.SIF_735.values # Cell now contains the STDEV of each grid.

    elif (data_type == 'VIIRS_downscaling'):
        cell.loc[dissolve.index, 'NIRv'] = dissolve.NIRv.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'NIRv_std'] = dissolve_std.NIRv.values # Cell now contains the mean of each grid.
        cell.loc[dissolve.index, 'EVI2'] = dissolve.EVI2.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'EVI2_std'] = dissolve_std.EVI2.values # Cell now contains the mean of each grid.
        cell.loc[dissolve.index, 'NIR'] = dissolve.NIR.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'NIR_std'] = dissolve_std.NIR.values # Cell now contains the mean of each grid.
        cell.loc[dissolve.index, 'NDVI'] = dissolve.NDVI.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'NDVI_std'] = dissolve_std.NDVI.values # Cell now contains the mean of each grid.

    elif (data_type == 'CH4'):
        cell.loc[dissolve.index, 'ch4'] = dissolve.methane_mixing_ratio_bias_corrected.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'ch4_std'] = dissolve_std.methane_mixing_ratio_bias_corrected.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_count.index, 'ch4_number_of_measurements'] = dissolve_count.methane_mixing_ratio_bias_corrected.values # Cell now contains the STDEV of each grid.

    elif (data_type == 'SO2'):
        cell.loc[dissolve.index, 'so2'] = dissolve.sulfurdioxide_total_vertical_column.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'so2_std'] = dissolve_std.sulfurdioxide_total_vertical_column.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_count.index, 'so2_number_of_measurements'] = dissolve_count.sulfurdioxide_total_vertical_column.values # Cell now contains the STDEV of each grid.
        
    elif (data_type == 'NO2'):
        cell.loc[dissolve.index, 'tropospheric_no2'] = dissolve.nitrogendioxide_tropospheric_column.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'tropospheric_no2_std'] = dissolve_std.nitrogendioxide_tropospheric_column.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_count.index, 'no2_number_of_measurements'] = dissolve_count.nitrogendioxide_tropospheric_column.values # Cell now contains the STDEV of each grid.

    elif (data_type == 'H2O'):
        cell.loc[dissolve.index, 'H2O_column'] = dissolve.h2o_column.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'H2O_column_std'] = dissolve_std.h2o_column.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve.index, 'HDO_column'] = dissolve.hdo_column.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_std.index, 'HDO_column_std'] = dissolve_std.hdo_column.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve.index, 'delta_d'] = dissolve.deltad.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_std.index, 'delta_d_std'] = dissolve_std.deltad.values # Cell now contains the STDEV of each grid.

    elif (data_type == 'HCHO'):
        cell.loc[dissolve.index, 'HCHO_column'] = dissolve.formaldehyde_tropospheric_vertical_column.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'HCHO_column_std'] = dissolve_std.formaldehyde_tropospheric_vertical_column.values # Cell now contains the STDEV of each grid.
    
    cell['lon'] = cell.geometry.centroid.x
    cell['lat'] = cell.geometry.centroid.y
    
    cell.drop(columns = ['geometry'], inplace = True)
    return cell
    

def read_scattered_TROPOMI_HDO(file_name, bounds = 0):
    """
    Reads in L2 TROPOMI H2O/HDO datasets and outputs it as a GeoDataFrame.

    Parameters
    ----------
    file_name : string
        One filename referring to SIF dataset of interest.

    bounds : float array in the following format
        [lon_min, lat_min, lon_max, lat_max]. If you do not want any bounds and want the whole swath, don't 
        include this parameter (DEFAULT PARAMETER is 0)

    geolocation : boolean
        If true, will output the TROPOMI footprints (pixels) as geometry. If false, will output (x,y) centroid geometries.
    
    Returns
    -------
    gdf
        A GeoDataFrame (from geopandas) dataframe containing the scattered data with specified geometries.
    """
    original_data = xr.open_dataset(file_name, group = 'target_product') # Contains the actual remote sensing product
    location_data = xr.open_dataset(file_name, group = 'instrument') # Contains the actual remote sensing product
    
    df = original_data[['deltad', 'deltad_precision', 'h2o_column', 'h2o_column_precision', 'hdo_column', 'hdo_column_precision']].to_dataframe() # Takes the variables you are interested in # ['averaging_kernel']
    df_location = location_data[['latitude_center', 'longitude_center']].to_dataframe()
    
    df = pd.merge(df, df_location, left_index = True, right_index = True, how = 'left')

    original_data.close()
    location_data.close()
    
    if (bounds != 0):
        df = df[(df.latitude_center >= bounds[1]) & (df.latitude_center <= bounds[3]) & (df.longitude_center >= bounds[0]) & (df.longitude_center <= bounds[2])];
    
    if len(df) == 0:
        return df; # Saves compute power if there is no data that fulfills quality threshold and geographic bounds.
        
    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['longitude_center'], df['latitude_center'])) # Create a GeoDataFrame
    return gdf;

def read_viirs_data(fn, geolocation_bounds):
    """
    Reads in a VIIRS Vegetation Indices (16 day) tile with a sinusoidal grid. Filters using the pixel reliability metric (<= 2, which is "acceptable").

    Parameters
    ----------
    fn : string
        One filename referring to VIIRS tile dataset of interest.

    geolocation_bounds : float array in the following format
        [lon_min, lat_min, lon_max, lat_max]. If you do not want any bounds and want the whole swath, don't 
        include this parameter (DEFAULT PARAMETER is 0)

    Returns
    -------
    tile_data
        A GeoDataFrame (from geopandas) dataframe containing the VIIRS Vegetation Indices with scattered latitude/longitudes.
    """
    df_i = pd.DataFrame()
    lon_min, lat_min, lon_max, lat_max = geolocation_bounds; # Unpack geolocation_bounds 

    # Should be 2400 on both axes if 500 m resolution
    nx = 2400 # df_i.XDim.shape[0]
    ny = 2400 # df_i.YDim.shape[0]
    
    lon, lat = sinusoidal_grid(fn, nx, ny) # Create grid, then generate latitudes and longitudes
    locations = pd.DataFrame(zip(lat, lon), columns = ['lat', 'lon'])

    if (len(locations[(locations.lon >= lon_min) & (locations.lon <= lon_max) & (locations.lat >= lat_min) & (locations.lat <= lat_max)]) == 0):
        print("Skipped this tile because data doesn't intersect bounds.")
        return df_i;

    # Now that we know there is some overlap with our bounds, read in the data!
    version = int(re.search('\.00(.+?).', fn).group(1)) # Tells us what version of VIIRS we're using -- either v001 or v002
    if version == 2:
        df_i = xr.open_dataset(fn, group = 'HDFEOS/GRIDS/VIIRS_Grid_16Day_VI_500m/Data Fields', mask_and_scale = True)
    elif version == 1:
        df_i = xr.open_dataset(fn, group = 'HDFEOS/GRIDS/NPP_Grid_16Day_VI_500m/Data Fields', mask_and_scale = True)

    df_i = df_i[['500 m 16 days EVI2', '500 m 16 days NDVI', '500 m 16 days NIR reflectance', '500 m 16 days VI Quality', '500 m 16 days pixel reliability']].to_dataframe().reset_index()
    df_i.columns = ['Phony_Dim_0', 'Phony_Dim_1', 'EVI2', 'NDVI', "NIR", 'vi_quality',  'pixel_reliability']
    df_i = df_i.drop(['Phony_Dim_0', 'Phony_Dim_1'], axis = 1)

    # Only run the following if mask_and_scale doesn't work.
    # df_i['EVI2'] = df_i['EVI2'].where(df_i.EVI2 > -13000, other=np.nan).copy()
    # df_i['NIR'] = df_i['NIR'].where(df_i.NIR > -13000, other=np.nan).copy()
    # df_i['NDVI'] = df_i['NDVI'].where(df_i.NDVI > -13000, other=np.nan).copy()

    # df_i['EVI2'] = df_i['EVI2'] / 10000. / 10000.
    # df_i['NDVI'] = df_i['NDVI'] / 10000. / 10000.
    # df_i['NIR'] = df_i['NIR'] / 10000. / 10000.

    df_i['NIR'] = df_i['NIR'].where((df_i.pixel_reliability <= 2) & (df_i.pixel_reliability >= 0), other=np.nan).copy()
    df_i['EVI2'] = df_i['EVI2'].where((df_i.pixel_reliability <= 2) & (df_i.pixel_reliability >= 0), other=np.nan).copy()
    df_i['NDVI'] = df_i['NDVI'].where((df_i.pixel_reliability <= 2) & (df_i.pixel_reliability >= 0), other=np.nan).copy()

    tile_data = pd.concat([pd.DataFrame(lon), pd.DataFrame(lat), df_i], axis = 1) # Add latitude and longitude into the DataFrame
    tile_data.columns = ['lon', 'lat', 'EVI2', 'NDVI', 'NIR', 'vi_quality', 'pixel_reliability']
    tile_data = tile_data[(tile_data.lon >= lon_min) & (tile_data.lon <= lon_max) & (tile_data.lat >= lat_min) & (tile_data.lat <= lat_max)].copy()
    tile_data['NIRv'] = tile_data['NIR'] * tile_data['NDVI'];
    
    # Get metadata!
    metadata = nc.Dataset(fn) # Gives us metadata
    start_date = metadata.RangeBeginningDate
    end_date = metadata.RangeEndingDate
    start_time = metadata.RangeBeginningTime
    end_time = metadata.RangeEndingTime
    # north_bound = metadata.NorthBoundingCoord
    # south_bound = metadata.SouthBoundingCoord
    # east_bound = metadata.EastBoundingCoord
    # west_bound = metadata.WestBoundingCoord   
    metadata.close()

    start_datetime = pd.to_datetime(start_date + ' ' + start_time, format = '%Y-%m-%d %H:%M:%S.%f')
    end_datetime = pd.to_datetime(end_date + ' ' + end_time, format = '%Y-%m-%d %H:%M:%S.%f')
    
    tile_data['start_time'] = start_datetime;
    tile_data['end_time'] = end_datetime;
    return tile_data;