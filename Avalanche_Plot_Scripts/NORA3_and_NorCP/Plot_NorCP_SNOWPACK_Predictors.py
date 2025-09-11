#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the NorCP and SNOWPACK predictors.
"""

#%% import
import pandas as pd
import numpy as np
import pylab as pl
from joblib import load, dump

from ava_functions.Data_Loading import load_snowpack_stab_ncp, load_agg_feats_ncp

from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Count_DangerLevels import count_dl_days
from ava_functions.Assign_Winter_Year import assign_winter_year


#%% set parameters
model_ty = "RF"
sea = "full"
balancing = "external"
balance_meth = "SMOTE"
ndlev = 2
# a_p = "wind_slab"
agg_type = "mean"
reg_code = 3011
reg_codes = [reg_code] #  list(regions.keys())

a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% prepare a suffix for the model name based on the data balancing
bal_suff = ""
if balancing == "internal":
    bal_suff = "_internal"
elif balancing == "external":
    bal_suff = f"_{balance_meth}"
# end if elif


#%% set paths
snowp_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Timeseries/Daily/"
norcp_path = f"{path_par}/IMPETUS/NorCP/Avalanche_Region_Predictors/"
model_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/{agg_str}/Elev_Agg/"
out_path = f"{path_par}/IMPETUS/NorCP/ML_Predictions/"
pl_path = f"{path_par}/IMPETUS/NorCP/Plots/TimeSeries_Annual_ADF/"


#%% set dictionaries
sno_stab = {}
feats_ncp = {}
feats = {}


#%% loop over the scenarios
for scen in ["rcp45", "rcp85"]:

    #% load the data
    sno_stab[scen] = pd.concat([load_snowpack_stab_ncp(path=path_par, model="EC-Earth", scen=scen, period="MC",
                                                       reg_codes=reg_code),
                                load_snowpack_stab_ncp(path=path_par, model="EC-Earth", scen=scen, period="LC",
                                                       reg_codes=reg_code)])
    feats_ncp[scen] = pd.concat([load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model="EC-Earth", scen=scen,
                                                    period="MC"),
                                 load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model="EC-Earth", scen=scen,
                                                    period="LC")])


    #% merge the dataframes
    feats[scen] = []
    for reg_code in reg_codes:
        feats[scen].append(feats_ncp[scen][feats_ncp[scen].reg_code == reg_code].\
                                                              merge(sno_stab[scen][sno_stab[scen].reg_code == reg_code],
                                                                    how="inner", left_index=True, right_index=True))
    # end for reg_code

    feats[scen] = pd.concat(feats[scen], axis=0).drop(["reg_code_x", "reg_code_y"], axis=1)

    for a_p in ["y"]:

        #% set the avalanche problem string
        a_p_str = "general" if a_p == "y" else a_p

        #% load the ML model
        mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}{bal_suff}_{a_p_str}.joblib"
        model = load(f"{model_path}/{mod_name}")

        #% test if model features are in the data
        for f in feats[scen].columns:
            if f not in model.feature_names_in_:
                print(f"{f} not in model features")
        # end for f
        for f in model.feature_names_in_:
            if f not in feats[scen].columns:
                print(f"{f} not in NorCP features")
        # end for f


        #% make sure the features are in the same order as in the model
        feats[scen] = feats[scen][model.feature_names_in_]
    # end for a_p
# end for scen


#%% extra treatment for historical and evaluation since they do not have MC and LC
for scen in ["historical", "evaluation"]:

    if scen == "historical":
        model = "EC-Earth"
        sta_yr = 1985
        end_yr = 2005
    elif scen == "evaluation":
        model = "ERAINT"
        sta_yr = 1998
        end_yr = 2018
    # end if elif


    #% load the data
    sno_stab[scen] = load_snowpack_stab_ncp(path=path_par, model=model, scen=scen, period="", reg_codes=reg_code)
    feats_ncp[scen] = load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model=model, scen=scen, period="")


    #% merge the dataframes
    feats[scen] = []
    for reg_code in reg_codes:
        feats[scen].append(feats_ncp[scen][feats_ncp[scen].reg_code == reg_code].\
                                                              merge(sno_stab[scen][sno_stab[scen].reg_code == reg_code],
                                                                    how="inner", left_index=True, right_index=True))
    # end for reg_code

    feats[scen] = pd.concat(feats[scen], axis=0).drop(["reg_code_x", "reg_code_y"], axis=1)

    for a_p in ["y"]:

        #% set the avalanche problem string
        a_p_str = "general" if a_p == "y" else a_p

        #% load the ML model
        mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}{bal_suff}_{a_p_str}.joblib"
        model = load(f"{model_path}/{mod_name}")

        #% test if model features are in the data
        for f in feats[scen].columns:
            if f not in model.feature_names_in_:
                print(f"{f} not in model features")
        # end for f
        for f in model.feature_names_in_:
            if f not in feats[scen].columns:
                print(f"{f} not in NorCP features")
        # end for f


        #% make sure the features are in the same order as in the model
        feats[scen] = feats[scen][model.feature_names_in_]
    # end for a_p

# end for scen


#%% add winter year for aggreation
scen = "historical"
feats[scen]["winter_year"] = assign_winter_year(feats[scen].index, start_month=10)
scen = "rcp45"
feats[scen]["winter_year"] = assign_winter_year(feats[scen].index, start_month=10)
scen = "rcp85"
feats[scen]["winter_year"] = assign_winter_year(feats[scen].index, start_month=10)


#%% plot some data - seasonal means
var = "snow_depth_emax"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.plot(feats["historical"].groupby("winter_year")[var].mean(), c="black", marker="o", markersize=ms,
          linewidth=lwd)
ax00.plot(feats["rcp85"].groupby("winter_year")[var].mean().iloc[:21], c="black", marker="o", markersize=ms,
          linewidth=lwd)
ax00.plot(feats["rcp85"].groupby("winter_year")[var].mean().iloc[21:], c="black", marker="o", markersize=ms,
          linewidth=lwd)

ax00.set_title(var + " " + str(reg_code))

pl.show()
pl.close()


#%% plot some data
"""
var = "NSW3"

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.plot(feats["historical"][var][feats["historical"].index.year == 2040], c="black")
ax00.plot(feats["rcp85"][var][feats["rcp85"].index.year == 2040], c="black")

ax00.set_title(var)

pl.show()
pl.close()
"""