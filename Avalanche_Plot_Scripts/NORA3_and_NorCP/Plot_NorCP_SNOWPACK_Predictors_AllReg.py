#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the NorCP and SNOWPACK predictors.
"""

#%% import
import pandas as pd
import numpy as np
import pylab as pl
from joblib import load, dump, Parallel, delayed
from timeit import default_timer as timer

from ava_functions.Data_Loading import load_snowpack_stab_ncp, load_agg_feats_ncp

from ava_functions.MonteCarlo_P_test_With_Plot import monte_carlo_p, monte_carlo_p2
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Count_DangerLevels import count_dl_days
from ava_functions.Assign_Winter_Year import assign_winter_year
from ava_functions.DatetimeSimple import date_dt
from ava_functions.Plot_Functions import heat_map
from ava_functions.Lists_and_Dictionaries.Features import feats_all


#%% set parameters
model_ty = "RF"
sea = "full"
balancing = "external"
balance_meth = "SMOTE"
ndlev = 2
# a_p = "wind_slab"
agg_type = "mean"
reg_codes = list(regions.keys())

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
pl_path = f"{path_par}/IMPETUS/NorCP/Plots/TimeSeries_Feats/"


#%% set dictionaries
sno_stab = {}
feats_ncp = {}
feats = {}


#%% loop over the scenarios
for reg_code in reg_codes:
    sno_stab[reg_code] = {}
    feats_ncp[reg_code] = {}
    feats[reg_code] = {}

    for scen in ["rcp45", "rcp85"]:

        #% load the data
        sno_stab[reg_code][scen] = pd.concat([load_snowpack_stab_ncp(path=path_par, model="EC-Earth", scen=scen,
                                                                     period="MC", reg_codes=reg_code),
                                              load_snowpack_stab_ncp(path=path_par, model="EC-Earth", scen=scen,
                                                                     period="LC", reg_codes=reg_code)])
        feats_ncp[reg_code][scen] = pd.concat([load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model="EC-Earth",
                                                                  scen=scen, period="MC"),
                                               load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model="EC-Earth",
                                                                  scen=scen, period="LC")])


        #% merge the dataframes
        feats[reg_code][scen] = []

        feats[reg_code][scen].append(feats_ncp[reg_code][scen][feats_ncp[reg_code][scen].reg_code == reg_code].\
                                          merge(sno_stab[reg_code][scen][sno_stab[reg_code][scen].reg_code == reg_code],
                                                how="inner", left_index=True, right_index=True))

        feats[reg_code][scen] = pd.concat(feats[reg_code][scen], axis=0).drop(["reg_code_x", "reg_code_y"], axis=1)

        for a_p in ["y"]:

            #% set the avalanche problem string
            a_p_str = "general" if a_p == "y" else a_p

            #% load the ML model
            mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}{bal_suff}_{a_p_str}.joblib"
            model = load(f"{model_path}/{mod_name}")

            #% test if model features are in the data
            for f in feats[reg_code][scen].columns:
                if f not in model.feature_names_in_:
                    print(f"{f} not in model features")
            # end for f
            for f in model.feature_names_in_:
                if f not in feats[reg_code][scen].columns:
                    print(f"{f} not in NorCP features")
            # end for f


            #% make sure the features are in the same order as in the model
            feats[reg_code][scen] = feats[reg_code][scen][model.feature_names_in_]
        # end for a_p
    # end for scen
# end reg_code


#%% extra treatment for historical and evaluation since they do not have MC and LC
for reg_code in reg_codes:

    for scen in ["historical"]:  # , "evaluation"]:

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
        sno_stab[reg_code][scen] = load_snowpack_stab_ncp(path=path_par, model=model, scen=scen, period="",
                                                          reg_codes=reg_code)
        feats_ncp[reg_code][scen] = load_agg_feats_ncp(path=path_par, reg_codes=reg_code, model=model,
                                                                 scen=scen, period="")


        #% merge the dataframes
        feats[reg_code][scen] = []
        feats[reg_code][scen].append(feats_ncp[reg_code][scen][feats_ncp[reg_code][scen].reg_code == reg_code].\
                                          merge(sno_stab[reg_code][scen][sno_stab[reg_code][scen].reg_code == reg_code],
                                                how="inner", left_index=True, right_index=True))
        # end for reg_code

        feats[reg_code][scen] = pd.concat(feats[reg_code][scen], axis=0).drop(["reg_code_x", "reg_code_y"], axis=1)

        for a_p in ["y"]:

            #% set the avalanche problem string
            a_p_str = "general" if a_p == "y" else a_p

            #% load the ML model
            mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}{bal_suff}_{a_p_str}.joblib"
            model = load(f"{model_path}/{mod_name}")

            #% test if model features are in the data
            for f in feats[reg_code][scen].columns:
                if f not in model.feature_names_in_:
                    print(f"{f} not in model features")
            # end for f
            for f in model.feature_names_in_:
                if f not in feats[reg_code][scen].columns:
                    print(f"{f} not in NorCP features")
            # end for f


            #% make sure the features are in the same order as in the model
            feats[reg_code][scen] = feats[reg_code][scen][model.feature_names_in_]
        # end for a_p

    # end for scen
# end for reg_code


#%% reduce the data to between December and May (including those months)
for reg_code in reg_codes:
    for scen in ["historical", "rcp45", "rcp85"]:

        winter_inds = feats[reg_code][scen].index.month == 0

        for mon in [12, 1, 2, 3, 4, 5]:
            winter_inds = winter_inds | (feats[reg_code][scen].index.month == mon)
        # end for mon

        feats[reg_code][scen] = feats[reg_code][scen][winter_inds]

    # end for scen
# end for reg_code


#%% add winter year for aggreation
for reg_code in reg_codes:
    scen = "historical"
    feats[reg_code][scen]["winter_year"] = assign_winter_year(feats[reg_code][scen].index, start_month=10)
    scen = "rcp45"
    feats[reg_code][scen]["winter_year"] = assign_winter_year(feats[reg_code][scen].index, start_month=10)
    scen = "rcp85"
    feats[reg_code][scen]["winter_year"] = assign_winter_year(feats[reg_code][scen].index, start_month=10)
# end for reg_code


#%% caclulate annual means
feats_an = {}
for reg_code in reg_codes:
    feats_an[reg_code] = {}
    for scen in ["historical", "rcp45", "rcp85"]:
        feats_an[reg_code][scen] = feats[reg_code][scen].groupby(feats[reg_code][scen]["winter_year"]).mean()
    # end for scen
# end for reg_code


#%% calculate the period means
feats_an2 = {}
feats_mean = {}
feats_std = {}
for reg_code in reg_codes:
    feats_an2[reg_code] = {}
    feats_mean[reg_code] = {}
    feats_std[reg_code] = {}
    for scen in ["historical", "rcp45", "rcp85"]:
        if scen == "historical":
            feats_an2[reg_code]["historical"] = feats_an[reg_code]["historical"]
            feats_mean[reg_code]["historical"] = feats_an[reg_code]["historical"].mean()
            feats_std[reg_code]["historical"] = feats_an[reg_code]["historical"].std()
        else:
            feats_an2[reg_code][scen + "_MC"] = feats_an[reg_code][scen][feats_an[reg_code][scen].index < 2061]
            feats_an2[reg_code][scen + "_LC"] = feats_an[reg_code][scen][feats_an[reg_code][scen].index > 2079]

            feats_mean[reg_code][scen + "_MC"] = feats_an[reg_code][scen][feats_an[reg_code][scen].index < 2061].mean()
            feats_mean[reg_code][scen + "_LC"] = feats_an[reg_code][scen][feats_an[reg_code][scen].index > 2079].mean()
            feats_std[reg_code][scen + "_MC"] = feats_an[reg_code][scen][feats_an[reg_code][scen].index < 2061].std()
            feats_std[reg_code][scen + "_LC"] = feats_an[reg_code][scen][feats_an[reg_code][scen].index > 2079].std()
        # end if else
    # end for scen
# end for reg_code


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "snow_depth_emin"
scen = "rcp45"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code][scen].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code][scen].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("Snow depth in cm")
ax00.set_title("Snow depth from SNOWPACK")

pl.show()
pl.close()


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "RTA_100"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("RTA")
ax00.set_title("RTA (=SSI [?])")

pl.show()
pl.close()


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "wdrift3_3_emax"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("wdrift3_3")
ax00.set_title("wdrift3_3")

pl.show()
pl.close()


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "s7_emax"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("s7 im mm")
ax00.set_title("s7")

pl.show()
pl.close()


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "t_min_emin"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("t_min im K")
ax00.set_title("t_min")

pl.show()
pl.close()


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "pdd_emax"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("PDD in K")
ax00.set_title("PDD")

pl.show()
pl.close()


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "NSW_emax"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("NSW in Wm$^{-2}$")
ax00.set_title("NSW")

pl.show()
pl.close()


#%% plot some data - seasonal means
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}

var = "r1_emax"

ms = 2
lwd = 0.5
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for reg_code in reg_codes:
    # ax00.plot(feats[reg_code]["historical"].groupby("winter_year")[var].mean(), marker="o", markersize=ms,
    #           linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[:20], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code])
    ax00.plot(feats[reg_code]["rcp85"].groupby("winter_year")[var].mean().iloc[21:-1], marker="o",
              markersize=ms, linewidth=lwd, c=colors[reg_code], label=regions_pl[reg_code])
# for reg_code

ax00.legend()
ax00.set_ylabel("r1 in mm")
ax00.set_title("r1")

pl.show()
pl.close()


#%% plot averages over the periods -- plots are supposed to be similar to the ADF plots
years = [1995, 2050, 2090]
scens45 = ["historical", "rcp45_MC", "rcp45_LC"]
scens85 = ["historical", "rcp85_MC", "rcp85_LC"]

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

var = "s7_emin"

reg_code = 3009
ax00.errorbar(years, [feats_mean[reg_code][scen][var] for scen in scens45],
              yerr=[feats_std[reg_code][scen][var] for scen in scens45])

pl.show()
pl.close()


#%% plot the variables
var1 = "s7_emin"
var1_pl = feats_all[var1]
var1_u = "\nin mm"
var2 = "RTA_2_d3"
var2_pl = feats_all[var2]
var2_u = ""
var3 = "t_max_emax"
var3_pl = feats_all[var3]
var3_u = "\nin K"
var4 = "lwc_max_d2"
var4_pl = feats_all[var4]
var4_u = "\nin %"
var5 = "snow_depth_emin"
var5_pl = feats_all[var5]
var5_u = "\nin mm"
# var6 = "snow_depth_emin"
# var6_u = "\nin mm"

# colors = {3009:"black", 3010:"black", 3011:"gray", 3012:"gray", 3013:"black"}
# stys = {3009:"-", 3010:"--", 3011:"-", 3012:"--", 3013:"-."}
# marks = {3009:"o", 3010:"o", 3011:"o", 3012:"o", 3013:"o"}

colors = {3009:"black", 3010:"gray", 3011:"red", 3012:"orange", 3013:"blue"}
stys = {3009:"-", 3010:"-", 3011:"--", 3012:"--", 3013:"-"}
marks = {3009:"o", 3010:"o", 3011:"o", 3012:"o", 3013:"o"}
dx_err = {3009:-4, 3010:-2, 3011:0, 3012:2, 3013:4}

lw = 1.25

yfac = 0.1
tick_size = 12
title_size = 14
label_size = 13

y_min1, y_max1 = None, None  # None
y_min2, y_max2 = None, None
y_min3, y_max3 = 268, 279  # None
y_min4, y_max4 = None, None  # None
y_min5, y_max5 = y_min1, y_max1  # None
y_min6, y_max6 = y_min2, y_max2  # None
y_min7, y_max7 = y_min3, y_max3  # None
y_min8, y_max8 = y_min4, y_max4  # None

fig = pl.figure(figsize=(11, 11))
ax00 = fig.add_subplot(5,2,1)
ax01 = fig.add_subplot(5,2,3)
ax02 = fig.add_subplot(5,2,5)
ax03 = fig.add_subplot(5,2,7)
ax04 = fig.add_subplot(5,2,9)

ax10 = fig.add_subplot(5,2,2)
ax11 = fig.add_subplot(5,2,4)
ax12 = fig.add_subplot(5,2,6)
ax13 = fig.add_subplot(5,2,8)
ax14 = fig.add_subplot(5,2,10)

years = {'historical':1995, 'evaluation':2011, 'rcp45_MC':2050, 'rcp45_LC':2090, 'rcp85_MC':2050, 'rcp85_LC':2090}
scens45 = ['historical', 'rcp45_MC', 'rcp45_LC']
scens85 = ['historical', 'rcp85_MC', 'rcp85_LC']

for reg_code in regions.keys():
    ax00.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var1] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var1] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax00.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var1] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax10.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var1] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var1] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax10.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var1] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax00.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var1] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var1] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax10.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var1] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var1] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
# end for reg_code

for reg_code in regions.keys():
    ax01.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var2] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax01.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var2] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax11.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var2] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax11.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var2] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax01.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var2] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax11.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var2] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
for reg_code in regions.keys():
    ax02.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var3] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax02.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var3] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax12.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var3] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax12.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var3] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax02.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var3] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax12.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var3] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
for reg_code in regions.keys():
    ax03.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var4] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax03.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var4] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax13.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var4] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax13.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var4] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax03.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var4] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax13.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var4] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
for reg_code in regions.keys():
    ax04.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var5] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var5] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax04.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var5] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax14.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var5] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var5] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax14.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var5] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax04.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var5] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var5] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax14.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var5] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var5] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
"""
for reg_code in regions.keys():
    ax04.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var6] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var6] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax14.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var6] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var6] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
"""
# end fors

p00, = ax00.plot([], [], c=colors[3009], linestyle=stys[3009], label=regions_pl[3009], linewidth=lw)
p01, = ax00.plot([], [], c=colors[3010], linestyle=stys[3010], label=regions_pl[3010], linewidth=lw)
p02, = ax00.plot([], [], c=colors[3011], linestyle=stys[3011], label=regions_pl[3011], linewidth=lw)
p03, = ax00.plot([], [], c=colors[3012], linestyle=stys[3012], label=regions_pl[3012], linewidth=lw)
p04, = ax00.plot([], [], c=colors[3013], linestyle=stys[3013], label=regions_pl[3013], linewidth=lw)
l00 = ax12.legend(handles=[p00, p01, p04], ncols=3, loc="lower right",
                  handletextpad=0.35, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.75)
l01 = ax02.legend(handles=[p03, p02], ncols=3, loc="lower right",
                  handletextpad=0.35, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.75)
# ax02.add_artist(l00)

ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)
ax03.tick_params(axis='both', which='major', labelsize=tick_size)
ax04.tick_params(axis='both', which='major', labelsize=tick_size)

ax10.tick_params(axis='both', which='major', labelsize=tick_size)
ax11.tick_params(axis='both', which='major', labelsize=tick_size)
ax12.tick_params(axis='both', which='major', labelsize=tick_size)
ax13.tick_params(axis='both', which='major', labelsize=tick_size)
ax14.tick_params(axis='both', which='major', labelsize=tick_size)

ax00.set_title("Historical & RCP4.5")
ax10.set_title("Historical & RCP8.5")

fig.suptitle("EC-Earth", fontsize=title_size)
fig.subplots_adjust(top=0.95, hspace=0.1, wspace=0.025)

ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax02.set_xticklabels([])
ax03.set_xticklabels([])

ax10.set_xticklabels([])
ax11.set_xticklabels([])
ax12.set_xticklabels([])
ax13.set_xticklabels([])

ax10.set_yticklabels([])
ax11.set_yticklabels([])
ax12.set_yticklabels([])
ax13.set_yticklabels([])
ax14.set_yticklabels([])

ax00.set_ylim(y_min1, y_max1)
ax01.set_ylim(y_min2, y_max2)
ax02.set_ylim(y_min3, y_max3)
ax03.set_ylim(y_min4, y_max4)
ax10.set_ylim(y_min5, y_max5)
ax11.set_ylim(y_min6, y_max6)
ax12.set_ylim(y_min7, y_max7)
ax13.set_ylim(y_min8, y_max8)

y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
x_text = 2005
ax00.text(x_text, y_max-dylim*yfac, "(a) " + var1_pl, fontdict={"fontsize":13})

y_min, y_max = ax01.get_ylim()
dylim = y_max - y_min
ax01.text(x_text, y_max-dylim*yfac, "(c) " + var2_pl, fontdict={"fontsize":13})

y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(x_text, y_max-dylim*yfac, "(e) " + var3_pl, fontdict={"fontsize":13})

y_min, y_max = ax03.get_ylim()
dylim = y_max - y_min
ax03.text(x_text, y_max-dylim*yfac, "(g) " + var4_pl, fontdict={"fontsize":13})

y_min, y_max = ax04.get_ylim()
dylim = y_max - y_min
ax04.text(x_text, y_max-dylim*yfac, "(i) " + var5_pl, fontdict={"fontsize":13})


y_min, y_max = ax10.get_ylim()
dylim = y_max - y_min
ax10.text(x_text, y_max-dylim*yfac, "(b) " + var1_pl, fontdict={"fontsize":13})

y_min, y_max = ax11.get_ylim()
dylim = y_max - y_min
ax11.text(x_text, y_max-dylim*yfac, "(d) " + var2_pl, fontdict={"fontsize":13})

ax12.set_xlabel("Year", fontsize=label_size)

y_min, y_max = ax12.get_ylim()
dylim = y_max - y_min
ax12.text(x_text, y_max-dylim*yfac, "(f) " + var3_pl, fontdict={"fontsize":13})

y_min, y_max = ax13.get_ylim()
dylim = y_max - y_min
ax13.text(x_text, y_max-dylim*yfac, "(h) " + var4_pl, fontdict={"fontsize":13})

y_min, y_max = ax14.get_ylim()
dylim = y_max - y_min
ax14.text(x_text, y_max-dylim*yfac, "(j) " + var5_pl, fontdict={"fontsize":13})

ax00.set_ylabel(f"{var1_pl}{var1_u}", fontsize=label_size)
ax01.set_ylabel(f"{var2_pl}{var2_u}", fontsize=label_size)
ax02.set_ylabel(f"{var3_pl}{var3_u}", fontsize=label_size)
ax03.set_ylabel(f"{var4_pl}{var4_u}", fontsize=label_size)
ax04.set_ylabel(f"{var5_pl}{var5_u}", fontsize=label_size)

ax04.set_xlabel("Year", fontsize=label_size)
ax14.set_xlabel("Year", fontsize=label_size)

pl.savefig(pl_path + f"TimeSeries_Feats_NorCP_EC-Earth_{sea}Season.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot the variables congruent with the APs

var1 = "RTA_2"
var1_pl = feats_all[var1]
var1_u = ""
var2 = "snow_depth_d1_emin"
var2_pl = feats_all[var2]
var2_u = "\nin mm"
var3 = "lwc_max"
var3_pl = feats_all[var3]
var3_u = "\nin %"
var4 = "ws_max_emax"
var4_pl = feats_all[var4]
var4_u = "\nin m/s"


# colors = {3009:"black", 3010:"black", 3011:"gray", 3012:"gray", 3013:"black"}
# stys = {3009:"-", 3010:"--", 3011:"-", 3012:"--", 3013:"-."}
# marks = {3009:"o", 3010:"o", 3011:"o", 3012:"o", 3013:"o"}

colors = {3009:"black", 3010:"gray", 3011:"red", 3012:"orange", 3013:"blue"}
stys = {3009:"-", 3010:"-", 3011:"--", 3012:"--", 3013:"-"}
marks = {3009:"o", 3010:"o", 3011:"o", 3012:"o", 3013:"o"}
dx_err = {3009:-4, 3010:-2, 3011:0, 3012:2, 3013:4}

lw = 1.25

yfac = 0.1
tick_size = 12
title_size = 14
label_size = 13


y_min1, y_max1 = None, None  # None
y_min2, y_max2 = None, None  # None
y_min3, y_max3 = None, None  # None
y_min4, y_max4 = None, None  # None
y_min5, y_max5 = y_min1, y_max1  # None
y_min6, y_max6 = y_min2, y_max2  # None
y_min7, y_max7 = y_min3, y_max3  # None
y_min8, y_max8 = y_min4, y_max4  # None


fig = pl.figure(figsize=(9, 9))
ax00 = fig.add_subplot(421)
ax01 = fig.add_subplot(423)
ax02 = fig.add_subplot(425)
ax03 = fig.add_subplot(427)

ax10 = fig.add_subplot(422)
ax11 = fig.add_subplot(424)
ax12 = fig.add_subplot(426)
ax13 = fig.add_subplot(428)

years = {'historical':1995, 'evaluation':2011, 'rcp45_MC':2050, 'rcp45_LC':2090, 'rcp85_MC':2050, 'rcp85_LC':2090}
scens45 = ['historical', 'rcp45_MC', 'rcp45_LC']
scens85 = ['historical', 'rcp85_MC', 'rcp85_LC']

for reg_code in regions.keys():
    ax00.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var1] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var1] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax00.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var1] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax10.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var1] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var1] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax10.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var1] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    #ax10.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var1] for scen in scens85],
    #              yerr=[feats_std[reg_code][scen][var1] for scen in scens85], capsize=2.5,
    #              marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code

for reg_code in regions.keys():
    ax01.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var2] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax01.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var2] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax11.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var2] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax11.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var2] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax01.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var2] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax11.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var2] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var2] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
for reg_code in regions.keys():
    ax02.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var3] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax02.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var3] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax12.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var3] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax12.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var3] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax02.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var3] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax12.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var3] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var3] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
for reg_code in regions.keys():
    ax03.errorbar(x=np.array([years[k] for k in scens45]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var4] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens45], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax03.plot([years[k] for k in scens45], [feats_mean[reg_code][scen][var4] for scen in scens45],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

    ax13.errorbar(x=np.array([years[k] for k in scens85]) + dx_err[reg_code],
                  y=[feats_mean[reg_code][scen][var4] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens85], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax13.plot([years[k] for k in scens85], [feats_mean[reg_code][scen][var4] for scen in scens85],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
    ax03.errorbar(x=[years[k] for k in scens45], y=[feats_mean[reg_code][scen][var4] for scen in scens45],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens45], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax13.errorbar(x=[years[k] for k in scens85], y=[feats_mean[reg_code][scen][var4] for scen in scens85],
                  yerr=[feats_std[reg_code][scen][var4] for scen in scens85], capsize=2.5,
                  marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    """
# end for reg_code

p00, = ax00.plot([], [], c=colors[3009], linestyle=stys[3009], label=regions_pl[3009], linewidth=lw, marker=marks[3009])
p01, = ax00.plot([], [], c=colors[3010], linestyle=stys[3010], label=regions_pl[3010], linewidth=lw, marker=marks[3010])
p02, = ax00.plot([], [], c=colors[3011], linestyle=stys[3011], label=regions_pl[3011], linewidth=lw, marker=marks[3011])
p03, = ax00.plot([], [], c=colors[3012], linestyle=stys[3012], label=regions_pl[3012], linewidth=lw, marker=marks[3012])
p04, = ax00.plot([], [], c=colors[3013], linestyle=stys[3013], label=regions_pl[3013], linewidth=lw, marker=marks[3013])
l00 = ax11.legend(handles=[p00, p01, p04], ncols=1, loc="lower left",
                  handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)  # (0.31, 0.825))
l01 = ax01.legend(handles=[p03, p02], ncols=1, loc="lower left",
                  handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)  #(0.585, 0.64))
# ax02.add_artist(l00)

ax00.set_xticklabels([])

ax10.tick_params(axis='both', which='major', labelsize=tick_size)
ax11.tick_params(axis='both', which='major', labelsize=tick_size)
ax12.tick_params(axis='both', which='major', labelsize=tick_size)
ax13.tick_params(axis='both', which='major', labelsize=tick_size)
ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)
ax03.tick_params(axis='both', which='major', labelsize=tick_size)

ax00.set_title("Historical & RCP4.5")
ax10.set_title("Historical & RCP8.5")

fig.suptitle("EC-Earth", fontsize=title_size)
fig.subplots_adjust(top=0.95, hspace=0.1, wspace=0.015)

ax01.set_xticklabels([])
ax02.set_xticklabels([])
ax10.set_xticklabels([])
ax10.set_yticklabels([])
ax11.set_xticklabels([])
ax11.set_yticklabels([])
ax12.set_xticklabels([])
ax12.set_yticklabels([])
ax13.set_yticklabels([])

ax00.set_ylim(y_min1, y_max1)
ax01.set_ylim(y_min2, y_max2)
ax02.set_ylim(y_min3, y_max3)
ax03.set_ylim(y_min4, y_max4)
ax10.set_ylim(y_min5, y_max5)
ax11.set_ylim(y_min6, y_max6)
ax12.set_ylim(y_min7, y_max7)
ax13.set_ylim(y_min8, y_max8)

y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
font_s = 12.5
ax00.text(2000, y_max-dylim*yfac, "(a) " + var1_pl, fontdict={"fontsize":font_s})

y_min, y_max = ax01.get_ylim()
dylim = y_max - y_min
ax01.text(2000, y_max-dylim*yfac, "(c) " + var2_pl, fontdict={"fontsize":font_s})

y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(2000, y_max-dylim*yfac, "(e) " + var3_pl, fontdict={"fontsize":font_s})

y_min, y_max = ax03.get_ylim()
dylim = y_max - y_min
ax03.text(2000, y_max-dylim*yfac, "(g) " + var4_pl, fontdict={"fontsize":font_s})

y_min, y_max = ax10.get_ylim()
dylim = y_max - y_min
ax10.text(2000, y_max-dylim*yfac, "(b) " + var1_pl, fontdict={"fontsize":font_s})


y_min, y_max = ax11.get_ylim()
dylim = y_max - y_min
ax11.text(2000, y_max-dylim*yfac, "(d) " + var2_pl, fontdict={"fontsize":font_s})

ax12.set_xlabel("Year", fontsize=label_size)

y_min, y_max = ax12.get_ylim()
dylim = y_max - y_min
ax12.text(2000, y_max-dylim*yfac, "(f) " + var3_pl, fontdict={"fontsize":font_s})

y_min, y_max = ax13.get_ylim()
dylim = y_max - y_min
ax13.text(2000, y_max-dylim*yfac, "(h) " + var4_pl, fontdict={"fontsize":font_s})

ax00.set_ylabel(f"{var1_pl}{var1_u}", fontsize=label_size)
ax01.set_ylabel(f"{var2_pl}{var2_u}", fontsize=label_size)
ax02.set_ylabel(f"{var3_pl}{var3_u}", fontsize=label_size)
ax03.set_ylabel(f"{var4_pl}{var4_u}", fontsize=label_size)

ax03.set_xlabel("Year", fontsize=label_size)
ax13.set_xlabel("Year", fontsize=label_size)

pl.savefig(pl_path + f"TimeSeries2_Feats_NorCP_EC-Earth_{sea}Season.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% create a mask for the upper triangle, including the diagonal
d_var = np.zeros((5, 5))
mask = np.triu(np.ones_like(d_var, dtype=bool))

# also mask those cells which are uninteresting
gray_mask = np.zeros((5, 5))
gray_mask[mask] = 1
gray_mask[4, 1] = 2
gray_mask[3, 2] = 2
gray_mask = gray_mask.flatten()[gray_mask.flatten() != 1]


#%% set up a heat map with the changes
"""
feat_n_l = ["s7_emax", "RTA_2_d3", "t_max_emax", "lwc_max_d2", "snow_depth_emax"]
d_var_dict = {}
p_var_dict = {}
for reg_code in regions.keys():

    d_var_dict[reg_code] = {}
    p_var_dict[reg_code] = {}

    for feat_n in feat_n_l:

        print(f"\n{regions_pl[reg_code]} {feat_n}\n")

        d_var = np.zeros((5, 5))
        p_var = np.zeros((5, 5))

        for i, ik in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):
            for j, jk in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):

                d_var[i, j] = feats_mean[reg_code][ik][feat_n] - feats_mean[reg_code][jk][feat_n]

                p_var[i, j] = monte_carlo_p2(feats_an2[reg_code][ik][feat_n],
                                             feats_an2[reg_code][jk][feat_n], num_permutations=100000)

            # end for j, jk
        # end for i, ik

        d_var_dict[reg_code][feat_n] = np.ma.masked_array(d_var, mask=mask)
        p_var_dict[reg_code][feat_n] = np.ma.masked_array(p_var, mask=mask)

    # end for feat_n
# end for reg_code
"""

#%% set up the function for parallelising
feat_n_l = ["s7_emin", "RTA_2_d3", "t_max_emax", "lwc_max_d2", "snow_depth_emax", "ws_max_emax", "Sn38_100_d3",
            "lwc_max"]
def monte(reg_code):
    d_var_dict = {}
    p_var_dict = {}

    for feat_n in feat_n_l:

        # print(f"\n{regions_pl[reg_code]} {feat_n}\n")

        d_var = np.zeros((5, 5))
        p_var = np.zeros((5, 5))

        for i, ik in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):
            for j, jk in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):

                d_var[i, j] = feats_mean[reg_code][ik][feat_n] - feats_mean[reg_code][jk][feat_n]

                p_var[i, j] = monte_carlo_p2(feats_an2[reg_code][ik][feat_n],
                                             feats_an2[reg_code][jk][feat_n], num_permutations=100000)

            # end for j, jk
        # end for i, ik

        d_var_dict[feat_n] = np.ma.masked_array(d_var, mask=mask)
        p_var_dict[feat_n] = np.ma.masked_array(p_var, mask=mask)

    # end for feat_n

    return d_var_dict, p_var_dict
# end def

start_ti = timer()
monte_res = Parallel(n_jobs=-1)(delayed(monte)(reg_code) for reg_code in regions.keys())
end_ti = timer()
elapsed_ti = end_ti - start_ti
print(f"Model training time: {elapsed_ti:.2f} seconds.")

d_var_dict = {k:monte_res[i][0] for i, k in enumerate(regions.keys())}
p_var_dict = {k:monte_res[i][1] for i, k in enumerate(regions.keys())}


#%% plot the heat map for all regions
ticks = ["historical", "45-MC", "45-LC", "85-MC", "85-LC"]

# vmins = {"s7_emin":-10, "RTA_2_d3":-0.2, "t_max_emax":-5.5, "lwc_max_d2":-0.7, "snow_depth_emax":-135,
#          "ws_max_emax":-5, "Sn38_100_d3":-0.2}
vmaxs = {"s7_emin":10, "RTA_2_d3":0.2, "t_max_emax":5.5, "lwc_max_d2":0.7, "snow_depth_emax":135,
         "ws_max_emax":0.5, "Sn38_100_d3":0.1, "lwc_max":2}
vmins = {k:-vmaxs[k] for k in vmaxs.keys()}

for feat_n in feat_n_l:

    feat_n_pl = feats_all[feat_n]

    annot_size = 16
    tick_size = 14
    title_size = 17

    gray_mask_color = "lightgray"

    fig = pl.figure(figsize=(15, 9))
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax10 = fig.add_subplot(233)
    ax11 = fig.add_subplot(234)
    ax20 = fig.add_subplot(235)
    ax21 = fig.add_subplot(236)

    reg_code = 3009
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_var_dict[reg_code][feat_n], p_vals=p_var_dict[reg_code][feat_n], ax=ax00, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=vmins[feat_n], vmax=vmaxs[feat_n],
             rects=None, annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax00.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax00.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    reg_code = 3010
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_var_dict[reg_code][feat_n], p_vals=p_var_dict[reg_code][feat_n], ax=ax01, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=vmins[feat_n], vmax=vmaxs[feat_n],
             rects=None, annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax01.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax01.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    reg_code = 3011
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_var_dict[reg_code][feat_n], p_vals=p_var_dict[reg_code][feat_n], ax=ax10, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=vmins[feat_n], vmax=vmaxs[feat_n],
             rects=None, annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax10.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax10.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))


    reg_code = 3012
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_var_dict[reg_code][feat_n], p_vals=p_var_dict[reg_code][feat_n], ax=ax11, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=vmins[feat_n], vmax=vmaxs[feat_n],
             rects=None, annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax11.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax11.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    reg_code = 3013
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_var_dict[reg_code][feat_n], p_vals=p_var_dict[reg_code][feat_n], ax=ax20, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=vmins[feat_n], vmax=vmaxs[feat_n],
             rects=None, annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax20.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax20.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    ax21.text(-0.1, 0.6, "model: EC-Earth", fontdict={'weight':'bold', "fontsize":15})
    ax21.text(-0.1, 0.5, f"feature: {feat_n_pl}", fontdict={'weight':'bold', "fontsize":15})
    ax21.text(-0.1, 0.4, "Read: [value in cell] = [vertical] $-$ [horizontal]", fontdict={"fontsize":13})
    ax21.text(-0.1, 0.3, "bold values indicate $p$ < 0.05", fontdict={"fontsize":13})
    ax21.axis('off')

    pl.savefig(pl_path + f"NorCP_EC-Earth_{feat_n}.png", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()
# end for a_p
