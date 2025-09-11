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

from ava_functions.Data_Loading import load_snowpack_stab_ncp, load_agg_feats_ncp

from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Count_DangerLevels import count_dl_days


#%% set parameters
model_ncp = "EC-Earth"  # sys.argv[2]

model_ty = "RF"
sea = "full"
ndlev = 2
balancing = "external"

agg_type = "mean"
reg_code = 3009 # int(sys.argv[1])
reg_codes = [reg_code] #  list(regions.keys())

use_best = True

a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

# n_best_d = {"wind_slab":30, "pwl_slab":30, "wet":35, "y":30}
n_best_d = {"wind_slab":20, "pwl_slab":50, "wet":50, "y":15}

class_weight = {"wind_slab":{0:1, 1:1}, "pwl_slab":{0:1, 1:1}, "wet":{0:1, 1:1}, "y":{0:1, 1:1}}

balance_meth_d = {"wind_slab":"SMOTE", "pwl_slab":"SMOTE", "wet":"SMOTE", "y":"SMOTE"}

slope_angle = 0  # "agg"
slope_azi = 0  # "agg"
h_low = -1
h_hi = -1

with_snowpack = True


#%% set up the elevation string
elev_dir = "/Elev_Agg/"
elev_n = ""
if ((slope_angle == "agg") | (slope_azi == "agg")):
    elev_dir = "/ElevSlope_Agg/"
if ((h_low > -1) & (h_hi > -1)):
    elev_dir = f"/Between{h_low}_and_{h_hi}m/"
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


#%% prepare a suffix for the model name based on the data balancing
"""
bal_suff = ""
if balancing == "internal":
    bal_suff = "_internal"
elif balancing == "external":
    bal_suff = f"_{balance_meth}"
# end if elif
"""

#%% set paths
# snowp_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Timeseries/Daily/"
norcp_path = f"{path_par}/IMPETUS/NorCP/Avalanche_Region_Predictors/"
model_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/{agg_str}/{elev_dir}/"
out_path = f"{path_par}/IMPETUS/NorCP/ML_Predictions/"
pl_path = f"{path_par}/IMPETUS/NorCP/Plots/TimeSeries_Annual_ADF/"


#%% set dictionaries
sno_stab = {}
feats_ncp = {}
pred_avd = {}
adf = {}
adf_mean = {}
adf_std = {}


#%% scenarios
scens = {"EC-Earth":["rcp45", "rcp85"], "GFDL-CM3":["rcp85"]}


#%% loop over the scenarios
for scen in scens[model_ncp]:

    #% load the data
    sno_stab[scen] = pd.concat([load_snowpack_stab_ncp(path=path_par, model=model_ncp, scen=scen, period="MC",
                                                       reg_codes=reg_code),
                                load_snowpack_stab_ncp(path=path_par, model=model_ncp, scen=scen, period="LC",
                                                       reg_codes=reg_code)])
    feats_ncp[scen] = pd.concat([load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model=model_ncp, scen=scen,
                                                    period="MC"),
                                 load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model=model_ncp, scen=scen,
                                                    period="LC")])

    adf[scen + "_MC"] = {}
    adf[scen + "_LC"] = {}
    adf_mean[scen + "_MC"] = {}
    adf_mean[scen + "_LC"] = {}
    adf_std[scen + "_MC"] = {}
    adf_std[scen + "_LC"] = {}
    for a_p in ["wind_slab", "wet", "pwl_slab", "y"]:

        # get the class weights for the individual AP
        cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight[a_p].values()])}"

        #% merge the dataframes
        feats = {}
        feats[scen] = []
        for reg_code in reg_codes:
            feats[scen].append(feats_ncp[scen][feats_ncp[scen].reg_code == reg_code].\
                                                              merge(sno_stab[scen][sno_stab[scen].reg_code == reg_code],
                                                                    how="inner", left_index=True, right_index=True))
        # end for reg_code

        feats[scen] = pd.concat(feats[scen], axis=0).drop(["reg_code_x", "reg_code_y"], axis=1)


        #% set the avalanche problem string
        a_p_str = "general" if a_p == "y" else a_p

        #% generate a suffix for the number of features
        nbest_suff = ""
        if use_best:
            nbest_suff = f"_{n_best_d[a_p]:02}best"
        # end if

        #% set up the model name
        mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_{balance_meth_d[a_p]}" +\
                                                                       f"{nbest_suff}_{cw_str}_{a_p_str}{sp_str}.joblib"

        #% load the ML model
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


        #% predict
        pred_avd[scen] = model.predict(feats[scen])


        #% combine prediction with date
        pred_avd[scen] = pd.DataFrame({a_p:pred_avd[scen]}, index=feats[scen].index)[a_p]


        #% convert to adf
        adf[scen + "_MC"][a_p] = count_dl_days(df=pred_avd[scen], sta_yr=2040, end_yr=2060)
        adf[scen + "_LC"][a_p] = count_dl_days(df=pred_avd[scen], sta_yr=2080, end_yr=2100)

        adf_mean[scen + "_MC"][a_p] = {}
        adf_mean[scen + "_LC"][a_p] = {}
        adf_mean[scen + "_MC"][a_p]["full"] = np.mean(adf[scen + "_MC"][a_p]["full"][1])
        adf_mean[scen + "_LC"][a_p]["full"] = np.mean(adf[scen + "_LC"][a_p]["full"][1])
        adf_mean[scen + "_MC"][a_p]["winter"] = np.mean(adf[scen + "_MC"][a_p]["winter"][1])
        adf_mean[scen + "_LC"][a_p]["winter"] = np.mean(adf[scen + "_LC"][a_p]["winter"][1])
        adf_mean[scen + "_MC"][a_p]["spring"] = np.mean(adf[scen + "_MC"][a_p]["spring"][1])
        adf_mean[scen + "_LC"][a_p]["spring"] = np.mean(adf[scen + "_LC"][a_p]["spring"][1])

        adf_std[scen + "_MC"][a_p] = {}
        adf_std[scen + "_LC"][a_p] = {}
        adf_std[scen + "_MC"][a_p]["full"] = np.std(adf[scen + "_MC"][a_p]["full"][1].DL, axis=0)
        adf_std[scen + "_LC"][a_p]["full"] = np.std(adf[scen + "_LC"][a_p]["full"][1].DL, axis=0)
        adf_std[scen + "_MC"][a_p]["winter"] = np.std(adf[scen + "_MC"][a_p]["winter"][1].DL, axis=0)
        adf_std[scen + "_LC"][a_p]["winter"] = np.std(adf[scen + "_LC"][a_p]["winter"][1].DL, axis=0)
        adf_std[scen + "_MC"][a_p]["spring"] = np.std(adf[scen + "_MC"][a_p]["spring"][1].DL, axis=0)
        adf_std[scen + "_LC"][a_p]["spring"] = np.std(adf[scen + "_LC"][a_p]["spring"][1].DL, axis=0)

    # end for a_p

# end for scen


#%% extra treatment for historical and evaluation since they do not have MC and LC
for scen in ["historical"]:  #, "evaluation"]:

    if scen == "historical":
        model = model_ncp
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

    adf[scen] = {}
    adf_mean[scen] = {}
    adf_std[scen] = {}
    for a_p in ["wind_slab", "wet", "pwl_slab", "y"]:

        # get the class weights for the individual AP
        cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight[a_p].values()])}"

        #% merge the dataframes
        feats = {}
        feats[scen] = []
        for reg_code in reg_codes:
            feats[scen].append(feats_ncp[scen][feats_ncp[scen].reg_code == reg_code].\
                                                              merge(sno_stab[scen][sno_stab[scen].reg_code == reg_code],
                                                                    how="inner", left_index=True, right_index=True))
        # end for reg_code

        feats[scen] = pd.concat(feats[scen], axis=0).drop(["reg_code_x", "reg_code_y"], axis=1)


        #% set the avalanche problem string
        a_p_str = "general" if a_p == "y" else a_p

        #% generate a suffix for the number of features
        nbest_suff = ""
        if use_best:
            nbest_suff = f"_{n_best_d[a_p]:02}best"
        # end if

        #% set up the model name
        mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_{balance_meth_d[a_p]}" +\
                                                                       f"{nbest_suff}_{cw_str}_{a_p_str}{sp_str}.joblib"
        #% load the ML model
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


        #% predict
        pred_avd[scen] = model.predict(feats[scen])


        #% combine prediction with date
        pred_avd[scen] = pd.DataFrame({a_p:pred_avd[scen]}, index=feats[scen].index)[a_p]


        #% convert to adf
        adf[scen][a_p] = count_dl_days(df=pred_avd[scen], sta_yr=sta_yr, end_yr=end_yr)

        adf_mean[scen][a_p] = {}
        adf_mean[scen][a_p]["full"] = np.mean(adf[scen][a_p]["full"][1])
        adf_mean[scen][a_p]["winter"] = np.mean(adf[scen][a_p]["winter"][1])
        adf_mean[scen][a_p]["spring"] = np.mean(adf[scen][a_p]["spring"][1])


        adf_std[scen][a_p] = {}
        adf_std[scen][a_p]["full"] = np.std(adf[scen][a_p]["full"][1].DL, axis=0)
        adf_std[scen][a_p]["winter"] = np.std(adf[scen][a_p]["winter"][1].DL, axis=0)
        adf_std[scen][a_p]["spring"] = np.std(adf[scen][a_p]["spring"][1].DL, axis=0)

    # end for a_p

# end for scen


#%% store the output
adf_out = {"adf":adf, "adf_mean":adf_mean, "adf_std":adf_std}
dump(adf_out, out_path + f"ADF_{model_ncp}_{reg_code}_{regions[reg_code]}.joblib")
sys.exit()


#%% plot
sea = "full"
a_p = "wind_slab"
y_max00 = np.nanmax([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
y_min00 = np.nanmin([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
d_y00 = (y_max00 - y_min00) * 0.075
a_p = "pwl_slab"
y_max01 = np.nanmax([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
y_min01 = np.nanmin([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
d_y01 = (y_max01 - y_min01) * 0.075
a_p = "wet"
y_max02 = np.nanmax([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
y_min02 = np.nanmin([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
d_y02 = (y_max02 - y_min02) * 0.075
a_p = "y"
y_max03 = np.nanmax([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
y_min03 = np.nanmin([adf["historical"][a_p][sea][1], #adf["evaluation"][a_p][sea][1],
#                     adf["rcp45_MC"][a_p][sea][1], adf["rcp45_LC"][a_p][sea][1],
                     adf["rcp85_MC"][a_p][sea][1], adf["rcp85_LC"][a_p][sea][1]])
d_y03 = (y_max03 - y_min03) * 0.075


fig = pl.figure(figsize=(8, 9))
ax00 = fig.add_subplot(411)
ax01 = fig.add_subplot(412)
ax02 = fig.add_subplot(413)
ax03 = fig.add_subplot(414)

a_p = "wind_slab"
ax00.plot(adf["historical"][a_p][sea][1], label="historical", c="black")
#ax00.plot(adf["evaluation"][a_p][sea][1], label="evaluation", c="gray")
#ax00.plot(adf["rcp45_MC"][a_p][sea][1], label="RCP4.5", c="blue")
#ax00.plot(adf["rcp45_LC"][a_p][sea][1], c="blue")
ax00.plot(adf["rcp85_MC"][a_p][sea][1], label="RCP8.5", c="red")
ax00.plot(adf["rcp85_LC"][a_p][sea][1], c="red")
ax00.axhline(y=0, c="black", linewidth=0.5)
ax00.legend(loc=(0.225, 0.7), ncols=2)
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days")
ax00.text(2070, y_max00-d_y00, a_ps[a_p], horizontalalignment="center")

a_p = "pwl_slab"
ax01.plot(adf["historical"][a_p][sea][1], label="historical", c="black")
#ax01.plot(adf["evaluation"][a_p][sea][1], label="evaluation", c="gray")
#ax01.plot(adf["rcp45_MC"][a_p][sea][1], label="RCP4.5", c="blue")
#ax01.plot(adf["rcp45_LC"][a_p][sea][1], c="blue")
ax01.plot(adf["rcp85_MC"][a_p][sea][1], label="RCP8.5", c="red")
ax01.plot(adf["rcp85_LC"][a_p][sea][1], c="red")
ax01.axhline(y=0, c="black", linewidth=0.5)
ax01.set_ylabel("ADF in days")
ax01.set_xticklabels([])
ax01.text(2070, y_max01-d_y01, a_ps[a_p], horizontalalignment="center")

a_p = "wet"
ax02.plot(adf["historical"][a_p][sea][1], label="historical", c="black")
#ax02.plot(adf["evaluation"][a_p][sea][1], label="evaluation", c="gray")
#ax02.plot(adf["rcp45_MC"][a_p][sea][1], label="RCP4.5", c="blue")
#ax02.plot(adf["rcp45_LC"][a_p][sea][1], c="blue")
ax02.plot(adf["rcp85_MC"][a_p][sea][1], label="RCP8.5", c="red")
ax02.plot(adf["rcp85_LC"][a_p][sea][1], c="red")
ax02.axhline(y=0, c="black", linewidth=0.5)
ax02.set_xticklabels([])
ax02.set_ylabel("ADF in days")
ax02.text(2070, y_max02-d_y02, a_ps[a_p], horizontalalignment="center")

a_p = "y"
ax03.plot(adf["historical"][a_p][sea][1], label="historical", c="black")
#ax03.plot(adf["evaluation"][a_p][sea][1], label="evaluation", c="gray")
#ax03.plot(adf["rcp45_MC"][a_p][sea][1], label="RCP4.5", c="blue")
#ax03.plot(adf["rcp45_LC"][a_p][sea][1], c="blue")
ax03.plot(adf["rcp85_MC"][a_p][sea][1], label="RCP8.5", c="red")
ax03.plot(adf["rcp85_LC"][a_p][sea][1], c="red")
ax03.axhline(y=0, c="black", linewidth=0.5)
ax03.set_xlabel("Year")
ax03.set_ylabel("ADF in days")
ax03.text(2070, y_max03-d_y03, a_ps[a_p], horizontalalignment="center")

fig.suptitle(regions_pl[reg_code] + f" ({reg_code})")
fig.subplots_adjust(top=0.95, hspace=0.1)

# pl.savefig(pl_path + f"ADF_Annual_{regions[reg_code]}.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% print the table
a_p = "wind_slab"
sea = "full"
print(f"""
      AP = {a_p}

      historical {adf_mean['historical'][a_p][sea]:5.2f}+-{adf_std['historical'][a_p][sea]:4.2f}
      evaluation {adf_mean['evaluation'][a_p][sea]:5.2f}+-{adf_std['evaluation'][a_p][sea]:4.2f}

               MC           LC
      RCP4.5   {adf_mean['rcp45_MC'][a_p][sea]:5.2f}+-{adf_std['rcp45_MC'][a_p][sea]:4.2f}  {adf_mean['rcp45_LC'][a_p][sea]:5.2f}+-{adf_std['rcp45_LC'][a_p][sea]:5.2f}
      RCP8.5   {adf_mean['rcp85_MC'][a_p][sea]:5.2f}+-{adf_std['rcp85_MC'][a_p][sea]:4.2f}  {adf_mean['rcp85_LC'][a_p][sea]:5.2f}+-{adf_std['rcp85_LC'][a_p][sea]:5.2f}
      """)
a_p = "pwl_slab"
print(f"""
      AP = {a_p}

      historical {adf_mean['historical'][a_p][sea]:5.2f}+-{adf_std['historical'][a_p][sea]:4.2f}
      evaluation {adf_mean['evaluation'][a_p][sea]:5.2f}+-{adf_std['evaluation'][a_p][sea]:4.2f}

               MC           LC
      RCP4.5   {adf_mean['rcp45_MC'][a_p][sea]:5.2f}+-{adf_std['rcp45_MC'][a_p][sea]:4.2f}  {adf_mean['rcp45_LC'][a_p][sea]:5.2f}+-{adf_std['rcp45_LC'][a_p][sea]:5.2f}
      RCP8.5   {adf_mean['rcp85_MC'][a_p][sea]:5.2f}+-{adf_std['rcp85_MC'][a_p][sea]:4.2f}  {adf_mean['rcp85_LC'][a_p][sea]:5.2f}+-{adf_std['rcp85_LC'][a_p][sea]:5.2f}
      """)
a_p = "wet"
print(f"""
      AP = {a_p}

      historical {adf_mean['historical'][a_p][sea]:5.2f}+-{adf_std['historical'][a_p][sea]:4.2f}
      evaluation {adf_mean['evaluation'][a_p][sea]:5.2f}+-{adf_std['evaluation'][a_p][sea]:4.2f}

               MC           LC
      RCP4.5   {adf_mean['rcp45_MC'][a_p][sea]:5.2f}+-{adf_std['rcp45_MC'][a_p][sea]:4.2f}  {adf_mean['rcp45_LC'][a_p][sea]:5.2f}+-{adf_std['rcp45_LC'][a_p][sea]:5.2f}
      RCP8.5   {adf_mean['rcp85_MC'][a_p][sea]:5.2f}+-{adf_std['rcp85_MC'][a_p][sea]:4.2f}  {adf_mean['rcp85_LC'][a_p][sea]:5.2f}+-{adf_std['rcp85_LC'][a_p][sea]:5.2f}
      """)
a_p = "y"
print(f"""
      AP = {a_p}

      historical {adf_mean['historical'][a_p][sea]:5.2f}+-{adf_std['historical'][a_p][sea]:4.2f}
      evaluation {adf_mean['evaluation'][a_p][sea]:5.2f}+-{adf_std['evaluation'][a_p][sea]:4.2f}

               MC           LC
      RCP4.5   {adf_mean['rcp45_MC'][a_p][sea]:5.2f}+-{adf_std['rcp45_MC'][a_p][sea]:4.2f}  {adf_mean['rcp45_LC'][a_p][sea]:5.2f}+-{adf_std['rcp45_LC'][a_p][sea]:5.2f}
      RCP8.5   {adf_mean['rcp85_MC'][a_p][sea]:5.2f}+-{adf_std['rcp85_MC'][a_p][sea]:4.2f}  {adf_mean['rcp85_LC'][a_p][sea]:5.2f}+-{adf_std['rcp85_LC'][a_p][sea]:5.2f}
      """)
