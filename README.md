# Satellite Remote Sensing in Python

Here is a repository of all of my satellite/remote sensing processing scripts that I've written over the past couple of years! :-)

---

### What scripts are available?
Currently, there are scripts for:

- **TROPOMI**: Methane, Water/HDO, Sulfur Dioxide, Formaldehyde, and Nitrogen Dioxide
- **Vegetation Indices**: TROPOSIF, SIF (downscaled), and VIIRS Vegetation Indices (both individual tiles and global grid). 
- **Other**: MOPITT, maybe OMI, maybe GEMS, maybe MODIS Ocean Color?

### Wishlist Before I Graduate

- MODIS CCI

---


### Where did I get the data?

- For TROPOMI, I got all my data from EarthData Search, as the old dhusget script no longer works (RIP).
- EarthData for VIIRS tiles
- L2B TROPOSIF was obtained from the [TROPOSIF portal](https://ftp.sron.nl/open-access-data-2/TROPOMI/tropomi/sif/v2.1/l2b/).

---

### What should I know before running these scripts?

For reproductibility, I ran all of these scripts on **Python 3.11.6**. To install my conda environment, run the following command:

`conda env create -f environment.yml`

To summarize the yml file, make sure you have these following libraries:

- Numpy, scipy, matplotlib
- Xarray
- NetCDF4
- Geopandas + Shapely

If you have any questions or concerns about the provided scripts, please contact me at jyyoon@uw.edu. Any updates to the satellite L2 files may break the script, so use at your own risk. 
