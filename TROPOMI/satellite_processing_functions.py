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

    avg_kernel = xr.merge([lat_lon, averaging_kernel, input_data], join = 'left');

    input_data.close();
    averaging_kernel.close();

    avg_kernel = avg_kernel.to_dataframe();
    
    if bounds != 0:
        avg_kernel = avg_kernel[(avg_kernel.latitude >= bounds[1]) & (avg_kernel.latitude <= bounds[3]) & (avg_kernel.longitude >= bounds[0]) & (avg_kernel.longitude <= bounds[2])];

    avg_kernel = avg_kernel[avg_kernel.qa_value > threshold]
    avg_kernel = avg_kernel.drop(['latitude', 'longitude', 'qa_value'], axis = 1).dropna(subset = ['column_averaging_kernel'], axis = 0);
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
        merged_dataset = xr.merge([product_ds, supporting_data], join = 'left').drop(['layer', 'level']).to_dataframe();
        merged_dataset = merged_dataset.dropna(subset = ['methane_mixing_ratio'], axis = 0);
    elif data_type == 'SO2':
        merged_dataset = xr.merge([product_ds, supporting_data], join = 'left').drop('layer').to_dataframe()
        merged_dataset = merged_dataset.dropna(subset = ['sulfurdioxide_total_vertical_column'], axis = 0);
        merged_dataset = merged_dataset[merged_dataset.sulfurdioxide_total_vertical_column >= -0.001] # Filtering for outliers; see User Guide for negative values
    elif data_type == 'NO2':
        merged_dataset = xr.merge([product_ds, supporting_data], join = 'left').to_dataframe()
        merged_dataset = merged_dataset.dropna(subset = ['nitrogendioxide_tropospheric_column'], axis = 0);
    merged_dataset = merged_dataset[merged_dataset.qa_value > threshold];
        
    if bounds != 0:
        merged_dataset = merged_dataset[(merged_dataset.latitude >= bounds[1]) & (merged_dataset.latitude <= bounds[3]) & (merged_dataset.longitude >= bounds[0]) & (merged_dataset.longitude <= bounds[2])];
    
    supporting_data.close()
    
    # Pivot on corner!
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
        df = original_data[['latitude', 'longitude', 'delta_time', 'time_utc', 'qa_value', 'nitrogendioxide_tropospheric_column', 'nitrogendioxide_tropospheric_column_precision']].to_dataframe() # Takes the variables you are interested in # ['averaging_kernel']
        df['time_utc'] = pd.to_datetime(df['delta_time']);
    
    if (bounds != 0):
        df = df[(df.latitude >= bounds[1]) & (df.latitude <= bounds[3]) & (df.longitude >= bounds[0]) & (df.longitude <= bounds[2])];
    df = df[df.qa_value > threshold] # Quality-filtering
    original_data.close()

    if len(df) == 0:
        return df; # Saves compute power if there is no data that fulfills quality threshold and geographic bounds.

    if data_type == 'SO2':
        # SO2 layer height data
        df_layer_height = xr.open_dataset(file_name, group = 'PRODUCT/SO2_LAYER_HEIGHT');
        df_layer_height = df_layer_height[['sulfurdioxide_layer_height', 'sulfurdioxide_total_vertical_column_layer_height', 'qa_value_layer_height']].to_dataframe();
        df_layer_height = df_layer_height[df_layer_height.qa_value_layer_height > 0.5];
        df_layer_height = df_layer_height.dropna(subset = 'sulfurdioxide_layer_height', axis = 0)
        df = pd.merge(df, df_layer_height, left_index = True, right_index = True, how = 'left');

    #### Start options
    if averaging_kernel:
        if data_type != 'NO2':
            avg_kernel = save_averaging_kernel(file_name, original_data, data_type, bounds, threshold);
            df = pd.merge(df, avg_kernel, left_index = True, right_index = True);

    if geolocation:
        pivot_geolocations = save_pixel_borders(file_name, original_data, data_type, bounds, threshold);
        df = pd.merge(df, pivot_geolocations, left_index = True, right_index = True, how = 'right');        
        df['geometry'] = df.apply(lambda x: Polygon(zip([x.lon_bound_1, x.lon_bound_2, x.lon_bound_3, x.lon_bound_4], [x.lat_bound_1, x.lat_bound_2, x.lat_bound_3, x.lat_bound_4])), axis = 1); # Create polygons representing each TROPOMI footprint
        gdf = gpd.GeoDataFrame(df, geometry = 'geometry');
    else:
        gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['longitude'], df['latitude'])) # Create a GeoDataFrame
    return gdf;

def read_scattered_data(file_name, new_grid, time, data_type):
    
    """
    Reads in scattered data, regrids it, and creates an xarray Dataset from this data. One file 
    should be read in at a time.

    Parameters
    ----------
    arr : string
        One filename referring to SIF dataset of interest.

    new_grid : GeoDataFrame
        Grid consisting of shapely objects; run new_grid!

    time : string
        String containing string representation of the time dataset was taken.
    
    data_type : string
        Specifies what type of dataset this is.

    Returns
    -------
    df
        A Xarray Dataset containing the gridded data.
    """
    
    df = pd.DataFrame([])

    if (data_type == 'CH4'):
        file = file.groups['PRODUCT'] # Read in data

        d = {'lat': np.ma.MaskedArray.flatten(file.variables['latitude'][:]), 
            'lon': np.ma.MaskedArray.flatten(file.variables['longitude'][:]), 
            'methane_mixing_ratio': np.ma.MaskedArray.flatten(file.variables['methane_mixing_ratio'][:]),
            'methane_mixing_ratio_precision': np.ma.MaskedArray.flatten(file.variables['methane_mixing_ratio_precision'][:]), 
            'methane_mixing_ratio_corrected': np.ma.MaskedArray.flatten(file.variables['methane_mixing_ratio_bias_corrected'][:]),
            'QA': np.ma.MaskedArray.flatten(file.variables['qa_value'][:])}
        d = pd.DataFrame(d)
        d = d[d.QA >= 0.5] # This is the threshold QA value!

    gdf = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d['lon'], d['lat']))
    df = regrid_scattered_data(gdf, new_grid, data_type) # Regrid data

    df['time'] = time
    df['time'] = pd.to_datetime(df['time'])

    df['lon'] = df.geometry.centroid.x
    df['lat'] = df.geometry.centroid.y
    df.drop(columns = ['geometry'], inplace = True)
    df = df.set_index(['time', 'lat', 'lon']).to_xarray()
    return df
    
def read_scattered_data_old(file_name, new_grid, time, data_type):
    
    """
    Reads in scattered data, regrids it, and creates an xarray Dataset from this data. One file 
    should be read in at a time.

    Parameters
    ----------
    arr : string
        One filename referring to SIF dataset of interest.

    new_grid : GeoDataFrame
        Grid consisting of shapely objects; run new_grid!

    time : string
        String containing string representation of the time dataset was taken.
    
    data_type : string
        Specifies what type of dataset this is.

    Returns
    -------
    df
        A Xarray Dataset containing the gridded data.
    """
    
    df = pd.DataFrame([])
    file = nc.Dataset(file_name)

    if (data_type == 'SIF'):
        file = file.groups['PRODUCT'] # Read in data

        d = {'lat': np.ma.MaskedArray.flatten(file.variables['latitude'][:]), 
            'lon': np.ma.MaskedArray.flatten(file.variables['longitude'][:]), 
            'SIF_743': np.ma.MaskedArray.flatten(file.variables['SIF_Corr_743'][:]),
            'SIF_743_Error': np.ma.MaskedArray.flatten(file.variables['SIF_ERROR_743'][:]), 
            'SIF_735': np.ma.MaskedArray.flatten(file.variables['SIF_Corr_735'][:]),
            'SIF_735_Error': np.ma.MaskedArray.flatten(file.variables['SIF_ERROR_735'][:]),
            'Time': np.ma.MaskedArray.flatten(file.variables['delta_time'][:])}
    
    elif (data_type == 'HCHO'):
        file = file.groups['PRODUCT'] # Read in data

        d = {'lat': np.ma.MaskedArray.flatten(file.variables['latitude'][:]), 
            'lon': np.ma.MaskedArray.flatten(file.variables['longitude'][:]), 
            'HCHO': np.ma.MaskedArray.flatten(file.variables['formaldehyde_tropospheric_vertical_column'][:]),
            'HCHO_prec': np.ma.MaskedArray.flatten(file.variables['formaldehyde_tropospheric_vertical_column_precision'][:]), 
            'QA': np.ma.MaskedArray.flatten(file.variables['qa_value'][:]),
            'Time': np.ma.MaskedArray.flatten(file.variables['delta_time'][:])}
        d = pd.DataFrame(d)
        d = d[d.QA >= 0.5] # This is the threshold QA value!

    elif (data_type == 'CH4'):
        file = file.groups['PRODUCT'] # Read in data

        d = {'lat': np.ma.MaskedArray.flatten(file.variables['latitude'][:]), 
            'lon': np.ma.MaskedArray.flatten(file.variables['longitude'][:]), 
            'methane_mixing_ratio': np.ma.MaskedArray.flatten(file.variables['methane_mixing_ratio'][:]),
            'methane_mixing_ratio_precision': np.ma.MaskedArray.flatten(file.variables['methane_mixing_ratio_precision'][:]), 
            'methane_mixing_ratio_corrected': np.ma.MaskedArray.flatten(file.variables['methane_mixing_ratio_bias_corrected'][:]),
            'QA': np.ma.MaskedArray.flatten(file.variables['qa_value'][:])}
        d = pd.DataFrame(d)
        d = d[d.QA >= 0.5] # This is the threshold QA value!

    gdf = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d['lon'], d['lat']))
    df = regrid_scattered_data(gdf, new_grid, data_type) # Regrid data

    df['time'] = time
    df['time'] = pd.to_datetime(df['time'])

    df['lon'] = df.geometry.centroid.x
    df['lat'] = df.geometry.centroid.y
    df.drop(columns = ['geometry'], inplace = True)
    df = df.set_index(['time', 'lat', 'lon']).to_xarray()
    return df

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
        cell.loc[dissolve.index, 'SIF_743'] = dissolve.SIF_743.values # Cell now contains the mean of each grid.
        cell.loc[dissolve.index, 'SIF_735'] = dissolve.SIF_735.values # Cell now contains the mean of each grid.
    
        cell.loc[dissolve_std.index, 'SIF_743_STD'] = dissolve_std.SIF_743.values # Cell now contains the STDEV of each grid.
        cell.loc[dissolve_std.index, 'SIF_735_STD'] = dissolve_std.SIF_735.values # Cell now contains the STDEV of each grid.

    elif (data_type == 'HCHO'):
        cell.loc[dissolve.index, 'HCHO_avg'] = dissolve.HCHO.values # Cell now contains the mean of each grid.
        cell.loc[dissolve_std.index, 'HCHO_std'] = dissolve_std.HCHO.values # Cell now contains the STDEV of each grid.
    
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