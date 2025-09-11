#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the gridded NORA3-based predictors into one netcdf file
"""

#%% imports
import sys
import glob
import xarray as xr
from dask.diagnostics import ProgressBar
import dask

#%% set split_large_chunks globally
dask.config.set({'array.slicing.split_large_chunks': True})


#%% define the function
def merge_preds(grid_path, out_path, fn):

    #% glob the files
    fn_list = glob.glob(grid_path + "*.nc")


    with ProgressBar():

        #% load the files
        ds = xr.open_mfdataset(fn_list, combine="by_coords", data_vars="minimal")
        # print(ds.chunksizes)

        # sys.exit()

        #% add a description
        desc = """Nameing convention: A 1 at the end of the parameter names always indicates that the parameter represents the
        daily value for the respective day. A 2 indicates that the data is from before the respective day (e.g., dtr2) or it
        indicates that the data are aggregated over two days (i.e., e.g. the mean over the respecive day and the day before, as
        in t2; although note that we only have t3 and t7 to unclutter the data). Some parameter descriptions follow:
            t1, t3, t7               1-, 3-day mean temperature, including the current day.
            w1, w3                   1-, 3-, 7-day mean wind speed, including the current day.
            r1, r3, r7               1-, 3-, 7-day sum of rain (= when 1h-temperature > 0 degC), including the current day.
            s1, s3, s7               1-, 3-, 7-day sum of snow (= when 1h-temperature <= 0 degC), including the current day.
            rh1, rh3, rh7            1-, 3-, 7-day mean relative humidity, including the current day.
            nlw1, nlw3, nlw7         1-, 3-day mean surface net long-wave radiation, including the current day.
            nsw1, nsw3, nsw7         1-, 3-day mean surface net short-wave radiation, including the current day.
            tmax1, tmax3, tmax7      1-, 3-, 7-day max temperature, including the current day.
            tmin1, tmin3, tmin7      1-, 3-, 7-day min temperature, including the current day.
            wmax1, wmax3             1-, 3-day max wind speed, including the current day.
            wmin1, wmin3, wmin7      1-, 3-, 7-day min wind speed, including the current day.
            wdir1, wdir3             1-, 3-day mean wind direction, including the current day.
            dwdir1, dwdir2, dwdir3   Std of wind direction on the day, the day before, and 2 days before.
            dwdird2, dwdird3         Std of wind direction over two and three days.
            dws1, dws2, dws3         Maximum wind speed range on the day, the day before, and 2 days before.
            dwsd2, dwsd3             Maximum wind speed range over two and three days.
            wdrift(_3), wdrift3(_3)  s1*w1, (s3*w3), s1*w1**3, (s3*w3**3)
            SWE1, SWE3, SWE7         1-, 3-day mean snow water equivalent (seNorge), including the current day. (Units: mm)
            SD1, SD3, SD7            1-, 3-day mean snow density (seNorge), including the current day. (Units: kg/L)
            SDP1, SDP3, SDP7         1-, 3-day mean snow depth (seNorge), including the current day. (Units: mm)
            MR1, MR3, MR7            1-, 3-day mean melt-refreeze (seNorge), including the current day. (Units: mm/d)

        Units are:
            temperature     K
            precipitation   mm
            wind speed      m/s
            wind direction  degrees
            radiation       Wm/m**2
            RH              kg/kg
            seNorge         see list above
            """

        ds.attrs['description'] = desc


        #% store the data
        ds.to_netcdf(out_path + f"{fn}_Gridded_Predictors.nc")
    # end with

# end def


