#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the relative humidity from NorCP ps, huss, and ta data.

--> Use the data regridded to NORA3 grid [?]
"""

#%% imports
import os
import glob
import numpy as np
import xarray as xr
from ava_functions.Relative_Humidity import rel_hum
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Progressbar import print_progress_bar


#%% set parameters
model = "EC-Earth"
scen = "historical"
period = ""
reggr = "Regridded_to_NORA3"


#%% set paths
data_path = f"{path_par}/IMPETUS/NorCP/{model.upper()}_{scen}"
out_path = data_path + f"_hur/{period}/{reggr}/"


#%% generate the output path
os.makedirs(out_path, exist_ok=True)


#%% load the file-name lists
spec_hum_l = glob.glob(data_path + f"_huss/{period}/{reggr}/*.nc")
air_pres_l = glob.glob(data_path + f"_ps/{period}/{reggr}/*.nc")
air_tem_l = glob.glob(data_path + f"_tas/{period}/{reggr}/*.nc")


#%% execute the relative humidity calculations
rel_hum_l = []
l = len(spec_hum_l)
print_progress_bar(0, total=l)
i = 0
for spec_hum_fn, air_pres_fn, air_tem_fn in zip(spec_hum_l, air_pres_l, air_tem_l):

    ds = xr.open_dataset(spec_hum_fn)

    spec_hum = np.array(ds.huss)
    air_pres = np.array(xr.open_dataset(air_pres_fn).ps)
    air_tem = np.array(xr.open_dataset(air_tem_fn).tas)

    ds["hur"] = (ds.huss.dims, rel_hum(spec_hum=spec_hum, air_pres=air_pres, air_tem=air_tem,
                                       method="Magnus", e_fac=1/100))

    ds.drop_vars(['huss']).to_netcdf(out_path + "hur" + spec_hum_fn.split("/")[-1][4:])

    i += 1
    print_progress_bar(iteration=i, total=l)

# end for



