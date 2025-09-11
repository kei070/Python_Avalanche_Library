#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict ADF based on the EURO-CORDEX data.
"""

#%% import
import sys
import pandas as pd
import numpy as np
import pylab as pl
from joblib import load, dump

from ava_functions.Data_Loading import load_snowpack_stab_ncp, load_agg_feats_ncp
from ava_functions.MonteCarlo_P_test_With_Plot import monte_carlo_p, monte_carlo_p2
from ava_functions.Lists_and_Dictionaries.Paths import path_par, path_par3
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Count_DangerLevels import count_dl_days
from ava_functions.Plot_Functions import heat_map


#%% set parameters
mod = "CNRM_RCA"

model_ty = "RF"
sea = "full"
ndlev = 2
balancing = "external"

agg_type = "mean"
# reg_codes = [reg_code] #  list(regions.keys())

use_best = False

a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

# n_best_d = {"wind_slab":30, "pwl_slab":30, "wet":35, "y":30}
n_best_d = {"wind_slab":20, "pwl_slab":50, "wet":50, "y":15}

class_weight = {"wind_slab":{0:1, 1:1}, "pwl_slab":{0:1, 1:1}, "wet":{0:1, 1:1}, "y":{0:1, 1:1}}

balance_meth_d = {"wind_slab":"SMOTE", "pwl_slab":"SMOTE", "wet":"SMOTE", "y":"SMOTE"}

slope_angle = 0  # "agg"
slope_azi = 0  # "agg"
h_low = -1
h_hi = -1

with_snowpack = False

# set the years depending on the scenario
sta_yrs = {"hist":1971, "rcp45":2006, "rcp85":2006}
end_yrs = {"hist":2000, "rcp45":2100, "rcp85":2100}


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
eu_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Avalanche_Region_Predictors/"
model_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/EURO-CORDEX/{agg_str}/{elev_dir}/"
out_path = f"{path_par}/IMPETUS/EURO-CORDEX/ML_Predictions/"
pl_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Plots/ADF_Change/"


#%% set dictionaries
sno_stab = {}
feats_ncp = {}
pred_avd = {}
adf = {}
adf_mean = {}
adf_std = {}


#%% loop over the scenarios
scens = ["hist", "rcp45", "rcp85"]
# for reg_code in regions.keys():
for reg_code in regions.keys():
    adf[reg_code] = {}
    for scen in scens:

        adf[reg_code][scen] = {}
        for a_p in ["wind_slab", "wet", "pwl_slab", "y"]:

            # get the class weights for the individual AP
            cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight[a_p].values()])}"

            #% merge the dataframes
            feats = {}
            feats[scen] = []

            # set the name
            eu_name = f"{regions[reg_code]}_{mod}_{scen}_ElevAgg_Predictors_MultiCellMean.csv"

            #% load the data
            # feats[scen].append(pd.read_csv(eu_path + f"{scen}/{mod}/" + eu_name, index_col=0, parse_dates=True))
            # feats[scen] = pd.concat(feats[scen], axis=0)
            feats[scen] = pd.read_csv(eu_path + f"{scen}/{mod}/" + eu_name, index_col=0, parse_dates=True)

            #% set the avalanche problem string
            a_p_str = "danger_level" if a_p == "y" else a_p

            #% generate a suffix for the number of features
            nbest_suff = ""
            if use_best:
                nbest_suff = f"_{n_best_d[a_p]:02}best"
            # end if

            #% set up the model name
            mod_name = f"{model_ty}_EU_{ndlev}DL_AllReg_{agg_str}_{sea}_{balance_meth_d[a_p]}" +\
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
            adf[reg_code][scen][a_p] = count_dl_days(df=pred_avd[scen], sta_yr=sta_yrs[scen], end_yr=end_yrs[scen])

        # end for a_p

    # end for scen
# end for reg_code


#%% calculate the means over the periods corresponding (as far as possible) to the NorCP periods
periods = {"hist":{"hist":[1985, 2000]},
           "rcp45":{"MC":[2040, 2060], "LC":[2080, 2099]},
           "rcp85":{"MC":[2040, 2060], "LC":[2080, 2099]}}

for reg_code in regions.keys():

    adf_mean[reg_code] = {}
    adf_std[reg_code] = {}

    for scen in scens:

        adf_mean[reg_code][scen] = {}
        adf_std[reg_code][scen] = {}
        for a_p in ["wind_slab", "wet", "pwl_slab", "y"]:
            adf_mean[reg_code][scen][a_p] = {}
            adf_mean[reg_code][scen][a_p]["full"] = {}
            adf_mean[reg_code][scen][a_p]["winter"] = {}
            adf_mean[reg_code][scen][a_p]["spring"] = {}

            adf_std[reg_code][scen][a_p] = {}
            adf_std[reg_code][scen][a_p]["full"] = {}
            adf_std[reg_code][scen][a_p]["winter"] = {}
            adf_std[reg_code][scen][a_p]["spring"] = {}

            for period in periods[scen]:

                # get the start and end of the respective periods
                sta_yr = periods[scen][period][0]
                end_yr = periods[scen][period][1]

                temp = np.mean(adf[reg_code][scen][a_p]["full"][1].loc[sta_yr:end_yr])
                print(f"{period}, start: {sta_yr}, end: {end_yr}, mean: {temp}")

                adf_mean[reg_code][scen][a_p]["full"][period] = np.mean(adf[reg_code][scen][a_p]["full"][1].\
                                                                                                     loc[sta_yr:end_yr])
                adf_mean[reg_code][scen][a_p]["winter"][period] = np.mean(adf[reg_code][scen][a_p]["winter"][1].\
                                                                                                     loc[sta_yr:end_yr])
                adf_mean[reg_code][scen][a_p]["spring"][period] = np.mean(adf[reg_code][scen][a_p]["spring"][1].\
                                                                                                     loc[sta_yr:end_yr])

                adf_std[reg_code][scen][a_p]["full"][period] = np.std(adf[reg_code][scen][a_p]["full"][1].\
                                                                                          loc[sta_yr:end_yr].DL, axis=0)
                adf_std[reg_code][scen][a_p]["winter"][period] = np.std(adf[reg_code][scen][a_p]["winter"][1].\
                                                                                          loc[sta_yr:end_yr].DL, axis=0)
                adf_std[reg_code][scen][a_p]["spring"][period] = np.std(adf[reg_code][scen][a_p]["spring"][1].\
                                                                                          loc[sta_yr:end_yr].DL, axis=0)

                # end for period
            # end for period_par
        # end for a_p
    # end for scen
# end for reg_code


#%% plot annual values
fig = pl.figure(figsize=(8, 4))
ax00 = fig.add_subplot(111)

ax00.plot(adf[3009]["hist"]["wind_slab"]["full"][1], c="black", label="hist")
ax00.plot(adf[3009]["rcp45"]["wind_slab"]["full"][1], c="blue", label="rcp45")
ax00.plot(adf[3009]["rcp85"]["wind_slab"]["full"][1], c="red", label="rcp85")

ax00.legend()

pl.show()
pl.close()


#%% put the period means together
r45, r45_std = {}, {}
r85, r85_std = {}, {}

for reg_code in regions.keys():
    r45[reg_code] = {}
    r85[reg_code] = {}
    r45_std[reg_code] = {}
    r85_std[reg_code] = {}
    for a_p in a_ps:

        r45[reg_code][a_p] = [adf_mean[reg_code]["hist"][a_p]["full"]["hist"],
                              adf_mean[reg_code]["rcp45"][a_p]["full"]["MC"],
                              adf_mean[reg_code]["rcp45"][a_p]["full"]["LC"]]
        r85[reg_code][a_p] = [adf_mean[reg_code]["hist"][a_p]["full"]["hist"],
                              adf_mean[reg_code]["rcp85"][a_p]["full"]["MC"],
                              adf_mean[reg_code]["rcp85"][a_p]["full"]["LC"]]
        r45_std[reg_code][a_p] = [adf_std[reg_code]["hist"][a_p]["full"]["hist"],
                                  adf_std[reg_code]["rcp45"][a_p]["full"]["MC"],
                                  adf_std[reg_code]["rcp45"][a_p]["full"]["LC"]]
        r85_std[reg_code][a_p] = [adf_std[reg_code]["hist"][a_p]["full"]["hist"],
                                  adf_std[reg_code]["rcp85"][a_p]["full"]["MC"],
                                  adf_std[reg_code]["rcp85"][a_p]["full"]["LC"]]
# end for reg_code


#%% plot the period means
colors = {3009:"black", 3010:"black", 3011:"gray", 3012:"gray", 3013:"black"}
stys = {3009:"-", 3010:"--", 3011:"--", 3012:"-", 3013:"-."}
marks = {3009:"o", 3010:"o", 3011:"o", 3012:"o", 3013:"o"}
ax_labels = {"wind_slab":["(a)", "(b)"], "pwl_slab":["(c)", "(d)"], "wet":["(e)", "(f)"], "y":["(g)", "(h)"]}

colors = {3009:"black", 3010:"gray", 3011:"red", 3012:"orange", 3013:"blue"}
stys = {3009:"-", 3010:"-", 3011:"--", 3012:"--", 3013:"-"}
marks = {3009:"o", 3010:"o", 3011:"o", 3012:"o", 3013:"o"}
dx_err = {3009:-4, 3010:-2, 3011:0, 3012:2, 3013:4}

lw = 1.5

yfac = 0.1

years = [1992, 2050, 2070]

ylim = (0, 130)

fig = pl.figure(figsize=(8, 9))
ax00 = fig.add_subplot(421)
ax01 = fig.add_subplot(422)
ax10 = fig.add_subplot(423)
ax11 = fig.add_subplot(424)
ax20 = fig.add_subplot(425)
ax21 = fig.add_subplot(426)
ax30 = fig.add_subplot(427)
ax31 = fig.add_subplot(428)

axes0 = [ax00, ax10, ax20, ax30]
axes1 = [ax01, ax11, ax21, ax31]

for a_p, ax0, ax1 in zip(a_ps.keys(), axes0, axes1):
    for reg_code in regions.keys():
        ax0.errorbar(x=np.array(years) + dx_err[reg_code], y=r45[reg_code][a_p], yerr=r45_std[reg_code][a_p],
                     c=colors[reg_code], elinewidth=1, linewidth=0, capsize=2.5)
        ax0.plot(np.array(years), r45[reg_code][a_p],
                 marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])

        ax1.errorbar(x=np.array(years) + dx_err[reg_code], y=r85[reg_code][a_p], yerr=r85_std[reg_code][a_p],
                     c=colors[reg_code], elinewidth=1, linewidth=0, capsize=2.5)
        ax1.plot(np.array(years), r85[reg_code][a_p],
                 marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    # end for reg_code

    ax0.set_ylim(ylim)
    ax1.set_ylim(ylim)
    y_min, y_max = ax0.get_ylim()
    dylim = y_max - y_min
    ax0.text(1990, y_max-dylim*yfac, f"{ax_labels[a_p][0]} " + a_ps[a_p], fontdict={"fontsize":13})
    ax1.text(1990, y_max-dylim*yfac, f"{ax_labels[a_p][1]} " + a_ps[a_p], fontdict={"fontsize":13})

    ax0.set_ylabel("ADF in days")
# end for a_p, ax0, ax1

ax00.set_title("Historical & RCP4.5")
ax01.set_title("Historical & RCP8.5")

p10, = ax00.plot([], [], c=colors[3009], linestyle=stys[3009], label=regions_pl[3009], linewidth=lw, marker=marks[3009])
p11, = ax00.plot([], [], c=colors[3010], linestyle=stys[3010], label=regions_pl[3010], linewidth=lw, marker=marks[3010])
p12, = ax00.plot([], [], c=colors[3011], linestyle=stys[3011], label=regions_pl[3011], linewidth=lw, marker=marks[3011])
p13, = ax00.plot([], [], c=colors[3012], linestyle=stys[3012], label=regions_pl[3012], linewidth=lw, marker=marks[3012])
p14, = ax00.plot([], [], c=colors[3013], linestyle=stys[3013], label=regions_pl[3013], linewidth=lw, marker=marks[3013])
l00 = ax10.legend(handles=[p10, p11, p14], ncols=1, loc="upper right")  # loc=(0.19, 0.64))
l01 = ax11.legend(handles=[p13, p12], ncols=1, loc="upper right")  # loc=(0.515, 0.825))
# ax01.add_artist(l00)


ax01.set_yticklabels([])
ax11.set_yticklabels([])
ax21.set_yticklabels([])
ax31.set_yticklabels([])

ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax10.set_xticklabels([])
ax11.set_xticklabels([])
ax20.set_xticklabels([])
ax21.set_xticklabels([])

fig.suptitle(mod)
fig.subplots_adjust(wspace=0.05, hspace=0.1, top=0.95)

pl.savefig(pl_path + f"ADF_Proj_EURO-CORDEX_{mod}_{a_p}_{sea}Season.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% stop execution
sys.exit()


#%% create a mask for the upper triangle, including the diagonal
d_adf = np.zeros((5, 5))
mask = np.triu(np.ones_like(d_adf, dtype=bool))

# also mask those cells which are uninteresting
gray_mask = np.zeros((5, 5))
gray_mask[mask] = 1
gray_mask[4, 1] = 2
gray_mask[3, 2] = 2
gray_mask = gray_mask.flatten()[gray_mask.flatten() != 1]


#%% set up a heat map with the changes
k_dict = {"hist":["hist", "hist"], "rcp45_MC":["rcp45", "MC"], "rcp45_LC":["rcp45", "LC"],
          "rcp85_MC":["rcp85", "MC"], "rcp85_LC":["rcp85", "LC"]}

years = {"hist":[1985, 2000], "MC":[2040, 2060], "LC":[2080, 2099]}

d_adf_dict = {}
p_adf_dict = {}
for reg_code in regions.keys():

    d_adf_dict[reg_code] = {}
    p_adf_dict[reg_code] = {}

    for a_p in a_ps.keys():

        print(f"\n{regions_pl[reg_code]} {a_ps[a_p]}\n")

        d_adf = np.zeros((5, 5))
        p_adf = np.zeros((5, 5))

        for i, ik in enumerate(["hist", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):
            for j, jk in enumerate(["hist", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):

                i_scen = k_dict[ik][0]
                i_per = k_dict[ik][1]
                j_scen = k_dict[jk][0]
                j_per = k_dict[jk][1]

                ista = years[i_per][0]
                iend = years[i_per][1]
                jsta = years[j_per][0]
                jend = years[j_per][1]

                d_adf[i, j] = adf_mean[reg_code][i_scen][a_p]["full"][i_per] - \
                                                                        adf_mean[reg_code][j_scen][a_p]["full"][j_per]

                p_adf[i, j] = monte_carlo_p2(adf[reg_code][scen][a_p][sea][1].loc[ista:iend].DL,
                                             adf[reg_code][scen][a_p][sea][1].loc[jsta:jend].DL,
                                             num_permutations=100000)

            # end for j, jk
        # end for i, ik

        d_adf_dict[reg_code][a_p] = np.ma.masked_array(d_adf, mask=mask)
        p_adf_dict[reg_code][a_p] = np.ma.masked_array(p_adf, mask=mask)

    # end for a_p
# end for reg_code


#%% plot the heat map for all regions
# a_p = "wet"

ticks = ["historical", "45-MC", "45-LC", "85-MC", "85-LC"]

for a_p in ["wind_slab", "pwl_slab", "wet", "y"]:
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
    heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax00, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None,
             annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax00.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax00.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    reg_code = 3010
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax01, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None,
             annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax01.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax01.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    reg_code = 3011
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax10, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None,
             annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax10.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax10.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))


    reg_code = 3012
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax11, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None,
             annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax11.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax11.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    reg_code = 3013
    title = f"{regions_pl[reg_code]}"
    heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax20, title=title,
             title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None,
             annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax20.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax20.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    ax21.text(-0.1, 0.6, f"model: {mod}", fontdict={'weight':'bold', "fontsize":15})
    ax21.text(-0.1, 0.5, f"avalanche problem: {a_ps[a_p]}", fontdict={'weight':'bold', "fontsize":15})
    ax21.text(-0.1, 0.4, "Read: [value in cell] = [vertical] $-$ [horizontal]", fontdict={"fontsize":13})
    ax21.text(-0.1, 0.3, "bold values indicate $p$ < 0.05", fontdict={"fontsize":13})
    ax21.axis('off')

    pl.savefig(pl_path + f"/Heat_Maps/ADF_EURO-CORDEX_{mod}_{a_p}_{sea}Season.pdf", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()
# end for a_p