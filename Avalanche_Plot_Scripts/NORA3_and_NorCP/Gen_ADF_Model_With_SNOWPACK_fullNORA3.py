#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply the model to NorCP data.
"""

#%% import
import sys
import pandas as pd
import numpy as np
import pylab as pl
from joblib import load, dump

from ava_functions.Data_Loading import load_snowpack_stab, load_agg_feats_no3

from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Count_DangerLevels import count_dl_days


#%% set parameters
model_ty = "RF"
sea = "full"
balancing = "external"
ndlev = 2
# a_p = "wind_slab"
agg_type = "mean"
reg_code = int(sys.argv[1])
reg_codes = [reg_code] #  list(regions.keys())

use_best = True

slope_angle = 0  # "agg"
slope_azi = 0  # "agg"
h_low = -1
h_hi = -1

a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

# n_best_d = {"wind_slab":30, "pwl_slab":30, "wet":35, "y":30}
n_best_d = {"wind_slab":20, "pwl_slab":50, "wet":50, "y":15}

class_weight = {"wind_slab":{0:1, 1:1}, "pwl_slab":{0:1, 1:1}, "wet":{0:1, 1:1}, "y":{0:1, 1:1}}

balance_meth_d = {"wind_slab":"SMOTE", "pwl_slab":"SMOTE", "wet":"SMOTE", "y":"SMOTE"}

with_snowpack = True


#%% set up the elevation string
elev_dir = "/Elev_Agg/"
elev_n = ""
if ((slope_angle == "agg") | (slope_azi == "agg")):
    elev_dir = "/ElevSlope_Agg/"
if ((h_low > -1) & (h_hi > -1)):
    elev_dir = f"/Between{h_low}_and_{h_hi}m/"#%% generate the output path
os.makedirs()


    elev_n = f"_Between{h_low}_and_{h_hi}m"
# end if


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% add a string to the file names depending on the inclusion of SNOWPACK-derived data
if with_snowpack:
    sp_str = ""
else:
    sp_str = "_wo_SNOWP"
# end if else


#%% set paths
# snowp_path = f"{path_par}/IMPETUS/NORA3/Snowpack/Timeseries/Daily/{elev_dir}/"
nora3_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/"
model_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/{agg_str}/{elev_dir}/"
out_path = f"{path_par}/IMPETUS/NORA3/ML_Predictions/"


#%% loop over the scenarios

#% load the data
sno_stab = load_snowpack_stab(path=path_par, reg_codes=reg_code, slope_angle=slope_angle, slope_azi=slope_azi)
feats_no3 = load_agg_feats_no3(path=path_par, reg_codes=reg_code)

adf = {}
adf_roll = {}
adf_mean = {}
adf_std = {}
for a_p in ["wind_slab", "wet", "pwl_slab", "y"]:

    # get the class weights for the individual AP
    cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight[a_p].values()])}"

    #% merge the dataframes
    feats = []
    for reg_code in reg_codes:
        feats.append(feats_no3[feats_no3.reg_code == reg_code].\
                     merge(sno_stab[sno_stab.reg_code == reg_code],
                           how="inner", left_index=True, right_index=True))
    # end for reg_code

    feats = pd.concat(feats, axis=0).drop(["reg_code_x", "reg_code_y"], axis=1)

    #% set the avalanche problem string
    a_p_str = "general" if a_p == "y" else a_p

    """
    #% load the ML model
    mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_{balance_meth_d[a_p]}_{a_p_str}.joblib"
    model = load(f"{model_path}/{mod_name}")
    """

    #% generate a suffix for the number of features
    nbest_suff = ""
    if use_best:
        nbest_suff = f"_{n_best_d[a_p]:02}best"
    # end if

    #% set up the model name
    mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_{balance_meth_d[a_p]}" +\
                                                                       f"{nbest_suff}_{cw_str}_{a_p_str}{sp_str}.joblib"
    print(f"\nLoading {mod_name}\n")

    #% load the ML model
    model = load(f"{model_path}/{mod_name}")

    """
    #% test if model features are in the data
    for f in feats.columns:
        if f not in model.feature_names_in_:
            print(f"{f} not in model features")
    # end for f
    for f in model.feature_names_in_:
        if f not in feats.columns:
            print(f"{f} not in NorCP features")
    # end for f
    """

    #% make sure the features are in the same order as in the model
    feats = feats[model.feature_names_in_]

    #% predict
    pred_avd = model.predict(feats)

    #% combine prediction with date
    pred_avd = pd.DataFrame({a_p:pred_avd}, index=feats.index)[a_p]

    #% convert to adf
    adf[a_p] = count_dl_days(df=pred_avd, sta_yr=1970, end_yr=2024)

    # rolling means
    adf_roll[a_p] = {}
    adf_roll[a_p]["full"] = adf[a_p]["full"][1].DL.rolling(window=7, center=True).mean()
    adf_roll[a_p]["winter"] = adf[a_p]["winter"][1].DL.rolling(window=7, center=True).mean()
    adf_roll[a_p]["spring"] = adf[a_p]["spring"][1].DL.rolling(window=7, center=True).mean()

    adf_mean[a_p] = {}
    adf_mean[a_p]["full"] = np.mean(adf[a_p]["full"][1])
    adf_mean[a_p]["winter"] = np.mean(adf[a_p]["winter"][1])
    adf_mean[a_p]["spring"] = np.mean(adf[a_p]["spring"][1])

    adf_std[a_p] = {}
    adf_std[a_p]["full"] = np.std(adf[a_p]["full"][1].DL, axis=0)
    adf_std[a_p]["winter"] = np.std(adf[a_p]["winter"][1].DL, axis=0)
    adf_std[a_p]["spring"] = np.std(adf[a_p]["spring"][1].DL, axis=0)

# end for a_p


#%% plot
sea = "full"
a_p = "wind_slab"
y_max00 = np.nanmax([adf[a_p][sea][1]])
y_min00 = np.nanmin([adf[a_p][sea][1]])
d_y00 = (y_max00 - y_min00) * 0.075
a_p = "pwl_slab"
y_max01 = np.nanmax([adf[a_p][sea][1]])
y_min01 = np.nanmin([adf[a_p][sea][1]])
d_y01 = (y_max01 - y_min01) * 0.075
a_p = "wet"
y_max02 = np.nanmax([adf[a_p][sea][1]])
y_min02 = np.nanmin([adf[a_p][sea][1]])
d_y02 = (y_max02 - y_min02) * 0.075
a_p = "y"
y_max03 = np.nanmax([adf[a_p][sea][1]])
y_min03 = np.nanmin([adf[a_p][sea][1]])
d_y03 = (y_max03 - y_min03) * 0.075


fig = pl.figure(figsize=(8, 10))
ax00 = fig.add_subplot(411)
ax01 = fig.add_subplot(412)
ax02 = fig.add_subplot(413)
ax03 = fig.add_subplot(414)

a_p = "wind_slab"
ax00.plot(adf[a_p][sea][1], c="black", linewidth=0.5)
ax00.plot(adf_roll[a_p][sea], c="black", linewidth=1.5)
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days")
ax00.text(1972, y_max00-d_y00, a_ps[a_p], horizontalalignment="center")

a_p = "pwl_slab"
ax01.plot(adf[a_p][sea][1], c="black", linewidth=0.5)
ax01.plot(adf_roll[a_p][sea], c="black", linewidth=1.5)
ax01.set_ylabel("ADF in days")
ax01.set_xticklabels([])
ax01.text(1972, y_max01-d_y01, a_ps[a_p], horizontalalignment="center")

a_p = "wet"
ax02.plot(adf[a_p][sea][1], c="black", linewidth=0.5)
ax02.plot(adf_roll[a_p][sea], c="black", linewidth=1.5)
ax02.set_xticklabels([])
ax02.set_ylabel("ADF in days")
ax02.text(1972, y_max02-d_y02, a_ps[a_p], horizontalalignment="center")

a_p = "y"
ax03.plot(adf[a_p][sea][1], c="black", linewidth=0.5)
ax03.plot(adf_roll[a_p][sea], c="black", linewidth=1.5)
ax03.set_xlabel("Year")
ax03.set_ylabel("ADF in days")
ax03.text(1972, y_max03-d_y03, a_ps[a_p], horizontalalignment="center")

fig.suptitle(regions_pl[reg_code] + f" ({reg_code}) $-$ {sea}")
fig.subplots_adjust(top=0.94, hspace=0.1)

pl.show()
pl.close()


#%% store the output
adf_out = {"adf":adf, "adf_roll":adf_roll, "adf_mean":adf_mean, "adf_std":adf_std}
dump(adf_out, out_path + f"ADF_{reg_code}_{regions[reg_code]}_NORA3.joblib")