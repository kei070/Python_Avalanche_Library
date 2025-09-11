#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the gridded predictors for NorCP.
"""


#%% imports
import os
import sys
import numpy as np
from Gridded_Predictors.Func_Gen_Gridded_Predictors import gen_daily_pred, gen_pred_roll
from Gridded_Predictors.Gen_Gridded_Precip_Predictors import gen_precip_daily, gen_precip_roll
from Gridded_Predictors.Gen_Gridded_WindDir_Predictors import gen_wdir_daily, gen_wdir_roll, gen_wdir_shift
from Gridded_Predictors.Gen_Gridded_WindDrift_Predictors import gen_wdrift
from Gridded_Predictors.Gen_Gridded_seNorge_Predictors import gen_sn_preds
from Gridded_Predictors.Gen_Gridded_Change_Predictors import gen_change_preds
from Gridded_Predictors.Merge_Gridded_Predictors import merge_preds

from Lists_and_Dictionaries.Paths import path_par
from Lists_and_Dictionaries.Variable_Name_NORA3_NorCP import varn_norcp as varn


#%% set up parameters
dsource = "NorCP"
model = "EC-Earth"
scen = "rcp45"
period = "MC"
exist_ok = True


#%% set the year according to the period
sta_yr = 1985
end_yr = 2005

if period == "MC":
    sta_yr = 2040
    end_yr = 2060
elif period == "LC":
    sta_yr = 2080
    end_yr = 2100
# end if elif
yrs = np.arange(sta_yr, end_yr+1)


#%% adjust the period
period1 = period
if period in ["MC", "LC"]:
    period1 = f"_{period}"
# end if


#%% set up the file name
fn = f"{dsource}_{model}_{scen}{period1}"

rg_p = "/Regridded_to_NORA3/"


#%% set up some paths and file names
data_path = f"/{path_par}/IMPETUS/{dsource}/{model.upper()}_{scen}_"
sn_path =  f"/{path_par}/IMPETUS/seNorge/Data/"
grid_path = f"/{path_par}/IMPETUS/{dsource}/Avalanche_Predictors/Gridded/{model}_{scen}{period1}/Individual/"
out_path = f"/{path_par}/IMPETUS/{dsource}/Avalanche_Predictors/Gridded/{model}_{scen}{period1}/"


#%% generate the paths for the outputs
os.makedirs(out_path, exist_ok=True)
os.makedirs(grid_path, exist_ok=True)


#%% execute the scripts
print("\nGenerating the temperature predictors (1, 3, 7 day for min, mean, max)...\n")
gen_daily_pred(data_path=f"{data_path}tas/{period}/{rg_p}/", out_path=grid_path, fn=fn, agg="all", var_name="t2m",
               var_dic=varn, yrs=yrs, exist_ok=exist_ok)
gen_pred_roll(data_path=grid_path, var_name="t2m", fn=fn, roll=[3, 7], agg="all", exist_ok=exist_ok)


print("\nGenerating the precipitation predictors (1, 3, 7 day sums for rain and snow)...\n")
gen_precip_daily(data_path=f"{data_path}Rain_Snow/{period}/{rg_p}/", out_path=grid_path, fn=fn, var_dic=varn, yrs=yrs,
                 exist_ok=exist_ok)
gen_precip_roll(data_path=grid_path, fn=fn, var_dic=varn, roll=[3, 7], exist_ok=exist_ok)


print("\nGenerating the radiation predictors (1, 3, 7 day means for LW and SW)...\n")
gen_daily_pred(data_path=f"{data_path}rlns/{period}/{rg_p}/", out_path=grid_path, fn=fn, agg="mean", var_name="nlw",
               var_dic=varn, yrs=yrs, exist_ok=exist_ok)
gen_pred_roll(data_path=grid_path, var_name="nlw", fn=fn, roll=[3, 7], agg="mean")
gen_daily_pred(data_path=f"{data_path}rsns/{period}/{rg_p}/", out_path=grid_path, fn=fn, agg="mean", var_name="nsw",
               var_dic=varn, yrs=yrs, exist_ok=exist_ok)
gen_pred_roll(data_path=grid_path, var_name="nsw", fn=fn, roll=[3, 7], agg="mean", exist_ok=exist_ok)


print("\nGenerating the RH predictors (1, 3, 7 day means)...\n")
gen_daily_pred(data_path=f"{data_path}hurs/{period}/{rg_p}/", out_path=grid_path, fn=fn, agg="mean",
               var_name="rh", var_dic=varn, yrs=yrs, exist_ok=exist_ok)
gen_pred_roll(data_path=grid_path, var_name="rh", fn=fn, roll=[3, 7], agg="mean", exist_ok=exist_ok)


print("\nGenerating the wind-direction predictors (1, 2, 3-day mean, std)...\n")
gen_wdir_daily(data_path=f"{data_path}Wind_Speed_Dir/{period}/{rg_p}/", out_path=grid_path, fn=fn, agg="mean",
               var_dic=varn, yrs=yrs, exist_ok=exist_ok)
gen_wdir_shift(data_path=grid_path, fn=fn, shift=[2, 3], var_dic=varn, exist_ok=exist_ok)
gen_wdir_roll(data_path=grid_path, fn=fn, roll=[2, 3], var_dic=varn, exist_ok=exist_ok)


print("\nGenerating the wind-speed predictors (1, 3 day min, mean, max)...\n")
gen_daily_pred(data_path=f"{data_path}Wind_Speed_Dir/{period}/{rg_p}/", out_path=grid_path, fn=fn, agg="all",
               var_name="w10m",
               var_dic=varn, yrs=yrs, exist_ok=exist_ok)
gen_pred_roll(data_path=grid_path, var_name="w10m", fn=fn, roll=[3], agg="all", exist_ok=exist_ok)


print("\nGenerating the wind-drift predictors (1, 3 day)...\n")
gen_wdrift(grid_path=grid_path, fn=fn, exist_ok=exist_ok)


print("\nGenerating the seNorge predictors (1, 3, 7 day)...\n")
gen_sn_preds(sn_path=sn_path, out_path=grid_path, fn=fn, syr_in=yrs[0], eyr_in=yrs[-1], syr_out=yrs[0], eyr_out=yrs[-1],
             exist_ok=exist_ok)


print("\nGenerating the temperature and wind change predictors (1, 3, 7 day)...\n")
gen_change_preds(grid_path=grid_path, fn=fn, exist_ok=exist_ok)


print("\nMerging all predictors into one nc file...\n")
merge_preds(grid_path=grid_path, out_path=out_path, fn=fn)
