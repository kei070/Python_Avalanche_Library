#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the SNOWPACK-derived stability indices over time.
"""

#%% imports
import numpy as np
import pandas as pd
import pylab as pl

from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set parameters
reg_code = 3009


#%% set paths
ncp_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Timeseries/Daily/"


#%% load files
df_dict = {}

for model in ["EC-Earth"]:
    for scen in ["rcp45", "rcp85"]:
        for period in ["MC", "LC"]:
            sim = f"{model}_{scen}_{period}"

            fn = f"{regions[reg_code]}_{sim}_SNOWPACK_Stability_TimeseriesDaily_ElevBandMin_Flat.csv"

            df_dict[sim] = pd.read_csv(ncp_path + f"/{sim}/" + fn, index_col=0, parse_dates=True)
        # end for period
    # end for scen
# end for model


#%% plot some indices
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.plot(df_dict["EC-Earth_rcp45_LC"]["Sk38_100"].replace(6, np.nan))

pl.show()
pl.close()


#%% plot snowdepth
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.plot(df_dict["EC-Earth_rcp45_MC"]["snow_depth"], c="black")
ax00.plot(df_dict["EC-Earth_rcp45_LC"]["snow_depth"], c="black")

ax00.plot(df_dict["EC-Earth_rcp85_MC"]["snow_depth"], c="red")
ax00.plot(df_dict["EC-Earth_rcp85_LC"]["snow_depth"], c="red")

ax00.set_ylabel("Snowdepth in cm")

pl.show()
pl.close()
