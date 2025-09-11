#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the gridded seNorge predictors.
"""

#%% imports
import os
import xarray as xr


#%% define the function
def gen_sn_preds(sn_path, out_path, fn, roll=[3, 7], syr_in=1970, eyr_in=2024, syr_out=2016, eyr_out=2024,
                 exist_ok=False):

    pred_exist = False
    if os.path.exists(out_path + f"seNorge_{fn}_Features.nc"):
        pred_exist = True
    # end if

    if pred_exist & (not exist_ok):
        print("\nseNorge predictors exist. Continuing with next...\n")
        return
    # end if

    #% load the necessary files
    sn_nc = xr.open_dataset(sn_path + f"seNorge_NorthNorway_{fn}_{syr_in}_{eyr_out}.nc")

    #% load the dataset
    swe = sn_nc["SWE"][sn_nc.time.dt.year >= syr_out]
    swe = swe.rename("SWE1")
    sdp = sn_nc["SnowDepth"][sn_nc.time.dt.year >= syr_out]
    sdp = sdp.rename("SDP1")
    sd = sn_nc["SnowDens"][sn_nc.time.dt.year >= syr_out]
    sd = sd.rename("SD1")
    mr = sn_nc["Melt_Refreeze"][sn_nc.time.dt.year >= syr_out]
    mr = mr.rename("MR1")
    print("Data loaded. Proceeding to calculations...\n")

    # % calculate the 3 and 7-day means
    ndays_l = roll

    swe_x = {}
    sdp_x = {}
    sd_x = {}
    mr_x = {}

    for ndays in ndays_l:
        swe_x[ndays] = swe.rolling(time=ndays, center=False).mean()
        swe_x[ndays] = swe_x[ndays].rename(f"SWE{ndays}")
        sdp_x[ndays] = sdp.rolling(time=ndays, center=False).mean()
        sdp_x[ndays] = sdp_x[ndays].rename(f"SDP{ndays}")
        sd_x[ndays] = sd.rolling(time=ndays, center=False).mean()
        sd_x[ndays] = sd_x[ndays].rename(f"SD{ndays}")
        mr_x[ndays] = mr.rolling(time=ndays, center=False).mean()
        mr_x[ndays] = mr_x[ndays].rename(f"MR{ndays}")
        print(f"\n{ndays}-days calculation started. Storing...\n")
    # end for ndays

    #% merge all the data arrays
    ds = xr.merge([swe, sdp, sd, mr, swe, swe_x[3], swe_x[7], sdp_x[3], sdp_x[7], sd_x[3], sd_x[7], mr_x[3], mr_x[7]])

    #% store the dataset as netcdf file
    ds.to_netcdf(out_path + f"seNorge_{fn}_Features.nc")

# end def