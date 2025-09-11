#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the SNOWPACK-derived stability indices over time.
"""

#%% imports
import numpy as np
import pandas as pd
import pylab as pl

from ava_functions.DatetimeSimple import date_dt
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set parameters
reg_codes = [3009, 3010, 3011, 3012, 3013]


#%% set paths
ncp_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Timeseries/Daily/"


#%% load files
df_dict = {}

for reg_code in reg_codes:
    df_dict[reg_code] = {}
    for model in ["EC-Earth"]:
        for scen in ["rcp45", "rcp85"]:
            for period in ["MC", "LC"]:
                sim = f"{model}_{scen}_{period}"

                fn = f"{regions[reg_code]}_{sim}_SNOWPACK_Stability_TimeseriesDaily_ElevBandMin_Flat.csv"

                df_dict[reg_code][sim] = pd.read_csv(ncp_path + f"/{sim}/" + fn, index_col=0, parse_dates=True)
            # end for period
        # end for scen
    # end for model
# end for reg_code


#%% plot some indices
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.plot(df_dict[reg_code]["EC-Earth_rcp45_LC"]["Sk38_100"].replace(6, np.nan))

pl.show()
pl.close()


#%% plot snowdepth
reg_code = 3009

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.errorbar(2050, df_dict[reg_code]["EC-Earth_rcp45_MC"]["snow_depth_emax"].mean(),
              yerr=df_dict[reg_code]["EC-Earth_rcp45_MC"]["snow_depth_emax"].std(), c="black")
ax00.scatter(2050, df_dict[reg_code]["EC-Earth_rcp45_MC"]["snow_depth_emax"].mean(), c="black")
ax00.errorbar(2090, df_dict[reg_code]["EC-Earth_rcp45_LC"]["snow_depth_emax"].mean(),
              yerr=df_dict[reg_code]["EC-Earth_rcp45_LC"]["snow_depth_emax"].std(), c="black")
ax00.scatter(2090, df_dict[reg_code]["EC-Earth_rcp45_LC"]["snow_depth_emax"].mean(), c="black")

ax00.errorbar(2050, df_dict[reg_code]["EC-Earth_rcp85_MC"]["snow_depth_emax"].mean(),
              yerr=df_dict[reg_code]["EC-Earth_rcp85_MC"]["snow_depth_emax"].std(), c="red")
ax00.scatter(2050, df_dict[reg_code]["EC-Earth_rcp85_MC"]["snow_depth_emax"].mean(), c="red")
ax00.errorbar(2090, df_dict[reg_code]["EC-Earth_rcp85_LC"]["snow_depth_emax"].mean(),
              yerr=df_dict[reg_code]["EC-Earth_rcp85_LC"]["snow_depth_emax"].std(), c="red")
ax00.scatter(2090, df_dict[reg_code]["EC-Earth_rcp85_LC"]["snow_depth_emax"].mean(), c="red")

ax00.set_ylabel("Snowdepth in cm")

ax00.set_title("SNOWPACK-derived snow depth")

pl.show()
pl.close()


#%% plot snowdepth -- multi region
reg_colors = {3009:"red", 3010:"green", 3011:"blue", 3012:"orange", 3013:"black"}

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.scatter(2050, df_dict[reg_code]["EC-Earth_rcp45_MC"]["snow_depth_emax"].mean(), c=reg_colors[reg_code],
    #              marker="x")
    # ax00.scatter(2090, df_dict[reg_code]["EC-Earth_rcp45_LC"]["snow_depth_emax"].mean(), c=reg_colors[reg_code],
    #              marker="x")

    ax00.scatter(2050, df_dict[reg_code]["EC-Earth_rcp85_MC"]["snow_depth_emax"].mean(), edgecolor=reg_colors[reg_code],
                 facecolor="none", s=80)
    ax00.scatter(2090, df_dict[reg_code]["EC-Earth_rcp85_LC"]["snow_depth_emax"].mean(), edgecolor=reg_colors[reg_code],
                 facecolor="none", s=80)

# end for reg_code

ax00.set_ylim(0, 160)

ax00.set_ylabel("Snowdepth in cm")

ax00.set_title("SNOWPACK-derived snow depth")

pl.show()
pl.close()


#%% snowdepth
param = "snow_depth_emax"

y_lim = (0, 510)

fig = pl.figure(figsize=(8, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)
ax21 = fig.add_subplot(326)

reg_code = 3009
ax00.plot(df_dict[reg_code]["EC-Earth_rcp85_MC"][param], c=reg_colors[reg_code])
ax00.plot(df_dict[reg_code]["EC-Earth_rcp85_LC"][param], c=reg_colors[reg_code])
ax00.text(date_dt(2070), 450, s=regions[reg_code], horizontalalignment="center")
ax00.set_ylim(y_lim)
ax00.set_xticklabels([])

reg_code = 3010
ax01.plot(df_dict[reg_code]["EC-Earth_rcp85_MC"][param], c=reg_colors[reg_code])
ax01.plot(df_dict[reg_code]["EC-Earth_rcp85_LC"]["snow_depth_emax"], c=reg_colors[reg_code])
ax01.text(date_dt(2070), 450, s=regions[reg_code], horizontalalignment="center")
ax01.set_ylim(y_lim)
ax01.set_xticklabels([])
ax01.set_yticklabels([])

reg_code = 3011
ax10.plot(df_dict[reg_code]["EC-Earth_rcp85_MC"][param], c=reg_colors[reg_code])
ax10.plot(df_dict[reg_code]["EC-Earth_rcp85_LC"][param], c=reg_colors[reg_code])
ax10.text(date_dt(2070), 450, s=regions[reg_code], horizontalalignment="center")
ax10.set_ylim(y_lim)
ax10.set_xticklabels([])

reg_code = 3012
ax11.plot(df_dict[reg_code]["EC-Earth_rcp85_MC"][param], c=reg_colors[reg_code])
ax11.plot(df_dict[reg_code]["EC-Earth_rcp85_LC"][param], c=reg_colors[reg_code])
ax11.text(date_dt(2070), 450, s=regions[reg_code], horizontalalignment="center")
ax11.set_ylim(y_lim)
ax11.set_yticklabels([])

reg_code = 3013
ax20.plot(df_dict[reg_code]["EC-Earth_rcp85_MC"][param], c=reg_colors[reg_code])
ax20.plot(df_dict[reg_code]["EC-Earth_rcp85_LC"][param], c=reg_colors[reg_code])
ax20.text(date_dt(2070), 450, s=regions[reg_code], horizontalalignment="center")
ax20.set_ylim(y_lim)

ax00.set_ylabel("Snowdepth in cm")
ax10.set_ylabel("Snowdepth in cm")
ax20.set_ylabel("Snowdepth in cm")

ax21.axis('off')

fig.suptitle("SNOWPACK-derived snow depth")

fig.subplots_adjust(top=0.95, hspace=0.1, wspace=0.1)

pl.show()
pl.close()
