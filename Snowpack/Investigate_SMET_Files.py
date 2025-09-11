#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate SNOWPACK .smet files.
"""

#%% imports
import pylab as pl
import pandas as pd

from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set paths and names
data_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Output/Flat/Between0_300m/EC-Earth_rcp45_LC/2080_2100/"
f_name = "Tromsoe_0_300m_Flat.smet"

in_data_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Input/Elev_Mean_SMET/Between0_300m/EC-EARTH_rcp45_LC/"
in_f_name = "Tromsoe_0_300m.smet"


#%% column names
cols = ["timestamp", "Qs", "Ql", "Qg", "TSG", "Qg0", "Qr", "OLWR", "ILWR", "LWR_net", "OSWR", "ISWR", "Qw", "pAlbedo",
        "mAlbedo", "ISWR_h", "ISWR_dir", "ISWR_diff",
        "TA", "TSS_mod", "TSS_meas", "T_bottom",
        "RH", "VW", "VW_drift",
        "DW", "MS_Snow", "HS_mod", "HS_meas", "SWE", "MS_Water", "MS_Wind", "MS_Rain", "MS_SN_Runoff", "MS_Soil_Runoff",
        "MS_Sublimation", "MS_Evap", "TS0", "TS1", "TS2", "TS3", "TS4", "Sclass1", "Sclass2", "zSd", "Sd", "zSn", "Sn",
        "zSs", "Ss", "zS4", "S4", "zS5", "S5"]

in_cols = ["timestamp", "TA", "PSUM", "VW", "ILWR", "ISWR", "RH", "TSG"]


#%% load the file
df = pd.read_csv(data_path + f_name, header=None, index_col=0, names=cols, skiprows=18,
                 encoding="utf-8", sep='\s+', na_values="-999", date_format='%Y-%m-%dT%H:%M:%S')
df_in = pd.read_csv(in_data_path + in_f_name, header=None, index_col=0, names=in_cols, skiprows=18,
                    encoding="utf-8", sep='\s+', na_values="-999", date_format='%Y-%m-%dT%H:%M:%S')


#%% plot
var = "TSS_mod"
var_in = "TA"
fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(111)
ax00_1 = ax00.twinx()
p00, = ax00.plot(df[var].iloc[-1500:], label=f"{var} output", c="black")
p00_1, = ax00_1.plot(df_in[var_in].iloc[-9250:-8800]-273.15, label=f"{var_in} input", c="gray")
ax00.legend(handles=[p00, p00_1])
ax00.set_title(f"{var} & {var_in}")
ax00.set_ylabel(var)
ax00_1.set_ylabel(var_in)

pl.show()
pl.close()
