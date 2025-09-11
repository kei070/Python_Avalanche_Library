#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the averages of the ADFs for the different regions and APs.

NOTE: You first need to generate the ADF data with the script Gen_ADF_Model_With_SNOWPACK.py
"""

#%% imports
import sys
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from joblib import load, dump
from ava_functions.MonteCarlo_P_test_With_Plot import monte_carlo_p, monte_carlo_p2
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Plot_Functions import heat_map


#%% parameters
sea = "full"  # full, winter, spring


#%% set paths
out_path = f"{path_par}/IMPETUS/NorCP/ML_Predictions/"
pl_path = f"{path_par}/IMPETUS/NorCP/Plots/"


#%% load the data
adf_dict = {}
try:
    for reg_code in regions.keys():
        adf_dict[reg_code] = load(out_path + f"ADF_EC-Earth_{reg_code}_{regions[reg_code]}.joblib")
    # end for reg_code
except:
    print("\nData loading failed. Have you generated the ADF data with Test_Model_With_SNOWPACK_NorCP.py?\n")
# end try except


#%% years and colours and APs
years = [1995, 2011, 2050, 2090]
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}
a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}


#%% plot with RCP4.5
yfac = 0.1
tick_size = 12
title_size = 14
label_size = 13

y_max = 80  # None

years = {'historical':1995, 'evaluation':2011, 'rcp45_MC':2050, 'rcp45_LC':2090}
scens = ['historical', 'rcp45_MC', 'rcp45_LC']

fig = pl.figure(figsize=(8, 7))
ax00 = fig.add_subplot(311)
ax01 = fig.add_subplot(312)
ax02 = fig.add_subplot(313)

a_p = "wind_slab"
for reg_code in regions.keys():
    ax00.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days", fontsize=label_size)
# ax00.set_title(a_ps[a_p])
p00, = ax00.plot([], [], c=colors[3009], label=regions_pl[3009], linewidth=0.5, marker="o")
p01, = ax00.plot([], [], c=colors[3010], label=regions_pl[3010], linewidth=0.5, marker="o")
p02, = ax00.plot([], [], c=colors[3011], label=regions_pl[3011], linewidth=0.5, marker="o")
p03, = ax00.plot([], [], c=colors[3012], label=regions_pl[3012], linewidth=0.5, marker="o")
p04, = ax00.plot([], [], c=colors[3013], label=regions_pl[3013], linewidth=0.5, marker="o")
l00 = ax00.legend(handles=[p00, p01, p04], ncols=3)
l01 = ax00.legend(handles=[p03, p02], ncols=3, loc=(0.613, 0.64))
ax00.add_artist(l00)
ax00.set_ylim(0, y_max)
y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
ax00.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "pwl_slab"
for reg_code in regions.keys():
    ax01.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax01.set_xticklabels([])
ax01.set_ylabel("ADF in days", fontsize=label_size)
# ax01.set_title(a_ps[a_p])
ax01.set_ylim(0, y_max)
y_min, y_max = ax01.get_ylim()
dylim = y_max - y_min
ax01.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "wet"
for reg_code in regions.keys():
    ax02.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax02.set_ylabel("ADF in days", fontsize=label_size)
ax02.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax02.set_ylim(0, y_max)
y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)

fig.suptitle("EC-Earth historical & RCP4.5", fontsize=title_size)
fig.subplots_adjust(top=0.94, hspace=0.1)

pl.savefig(pl_path + f"TimeSeries_ADF_NorCP_EC-Earth_RCP45_{sea}Season.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot with RCP8.5
yfac = 0.1
tick_size = 12
title_size = 14
label_size = 13

y_max = 80  # None

years = {'historical':1995, 'evaluation':2011, 'rcp85_MC':2050, 'rcp85_LC':2090}
scens = ['historical', 'rcp85_MC', 'rcp85_LC']

fig = pl.figure(figsize=(8, 7))
ax00 = fig.add_subplot(311)
ax01 = fig.add_subplot(312)
ax02 = fig.add_subplot(313)

a_p = "wind_slab"
for reg_code in regions.keys():
    ax00.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days", fontsize=label_size)
# ax00.set_title(a_ps[a_p])
p00, = ax00.plot([], [], c=colors[3009], label=regions_pl[3009], linewidth=0.5, marker="o")
p01, = ax00.plot([], [], c=colors[3010], label=regions_pl[3010], linewidth=0.5, marker="o")
p02, = ax00.plot([], [], c=colors[3011], label=regions_pl[3011], linewidth=0.5, marker="o")
p03, = ax00.plot([], [], c=colors[3012], label=regions_pl[3012], linewidth=0.5, marker="o")
p04, = ax00.plot([], [], c=colors[3013], label=regions_pl[3013], linewidth=0.5, marker="o")
l00 = ax00.legend(handles=[p00, p01, p04], ncols=3)
l01 = ax00.legend(handles=[p03, p02], ncols=3, loc=(0.613, 0.64))
ax00.add_artist(l00)
ax00.set_ylim(0, y_max)
y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
ax00.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})


a_p = "pwl_slab"
for reg_code in regions.keys():
    ax01.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax01.set_xticklabels([])
ax01.set_ylabel("ADF in days", fontsize=label_size)
# ax01.set_title(a_ps[a_p])
ax01.set_ylim(0, y_max)
y_min, y_max = ax01.get_ylim()
dylim = y_max - y_min
ax01.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "wet"
for reg_code in regions.keys():
    ax02.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax02.set_ylabel("ADF in days", fontsize=label_size)
ax02.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax02.set_ylim(0, y_max)
y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)

fig.suptitle("EC-Earth historical & RCP8.5", fontsize=title_size)
fig.subplots_adjust(top=0.94, hspace=0.1)

pl.savefig(pl_path + f"TimeSeries_ADF_NorCP_EC-Earth_RCP85_{sea}Season.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot with RCP4.5 -- INCLUDING GENERAL PROBLEM
yfac = 0.1
tick_size = 12
title_size = 14
label_size = 13

y_max = 95  # None

years = {'historical':1995, 'evaluation':2011, 'rcp45_MC':2050, 'rcp45_LC':2090}
scens = ['historical', 'rcp45_MC', 'rcp45_LC']

fig = pl.figure(figsize=(8, 9))
ax00 = fig.add_subplot(411)
ax01 = fig.add_subplot(412)
ax02 = fig.add_subplot(413)
ax03 = fig.add_subplot(414)

a_p = "wind_slab"
for reg_code in regions.keys():
    ax00.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days", fontsize=label_size)
# ax00.set_title(a_ps[a_p])
p00, = ax00.plot([], [], c=colors[3009], label=regions_pl[3009], linewidth=0.5, marker="o")
p01, = ax00.plot([], [], c=colors[3010], label=regions_pl[3010], linewidth=0.5, marker="o")
p02, = ax00.plot([], [], c=colors[3011], label=regions_pl[3011], linewidth=0.5, marker="o")
p03, = ax00.plot([], [], c=colors[3012], label=regions_pl[3012], linewidth=0.5, marker="o")
p04, = ax00.plot([], [], c=colors[3013], label=regions_pl[3013], linewidth=0.5, marker="o")
l00 = ax00.legend(handles=[p00, p01, p04], ncols=3)
l01 = ax00.legend(handles=[p03, p02], ncols=3, loc=(0.613, 0.64))
ax00.add_artist(l00)
ax00.set_ylim(0, y_max)
y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
ax00.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "pwl_slab"
for reg_code in regions.keys():
    ax01.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax01.set_xticklabels([])
ax01.set_ylabel("ADF in days", fontsize=label_size)
# ax01.set_title(a_ps[a_p])
ax01.set_ylim(0, y_max)
y_min, y_max = ax01.get_ylim()
dylim = y_max - y_min
ax01.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "wet"
for reg_code in regions.keys():
    ax02.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax02.set_xticklabels([])
ax02.set_ylabel("ADF in days", fontsize=label_size)
# ax02.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax02.set_ylim(0, y_max)
y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "y"
for reg_code in regions.keys():
    ax03.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax03.set_ylabel("ADF in days", fontsize=label_size)
ax03.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax03.set_ylim(0, y_max)
y_min, y_max = ax03.get_ylim()
dylim = y_max - y_min
ax03.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)
ax03.tick_params(axis='both', which='major', labelsize=tick_size)

fig.suptitle("EC-Earth historical & RCP4.5", fontsize=title_size)
fig.subplots_adjust(top=0.95, hspace=0.1)

pl.savefig(pl_path + f"TimeSeries_ADF_NorCP_EC-Earth_RCP45_{sea}Season.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot with RCP8.5 -- INCLUDING GENERAL PROBLEM
yfac = 0.1
tick_size = 12
title_size = 14
label_size = 13

y_max = 95  # None

years = {'historical':1995, 'evaluation':2011, 'rcp85_MC':2050, 'rcp85_LC':2090}
scens = ['historical', 'rcp85_MC', 'rcp85_LC']

fig = pl.figure(figsize=(8, 9))
ax00 = fig.add_subplot(411)
ax01 = fig.add_subplot(412)
ax02 = fig.add_subplot(413)
ax03 = fig.add_subplot(414)

a_p = "wind_slab"
for reg_code in regions.keys():
    ax00.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days", fontsize=label_size)
# ax00.set_title(a_ps[a_p])
p00, = ax00.plot([], [], c=colors[3009], label=regions_pl[3009], linewidth=0.5, marker="o")
p01, = ax00.plot([], [], c=colors[3010], label=regions_pl[3010], linewidth=0.5, marker="o")
p02, = ax00.plot([], [], c=colors[3011], label=regions_pl[3011], linewidth=0.5, marker="o")
p03, = ax00.plot([], [], c=colors[3012], label=regions_pl[3012], linewidth=0.5, marker="o")
p04, = ax00.plot([], [], c=colors[3013], label=regions_pl[3013], linewidth=0.5, marker="o")
l00 = ax00.legend(handles=[p00, p01, p04], ncols=3)
l01 = ax00.legend(handles=[p03, p02], ncols=3, loc=(0.613, 0.64))
ax00.add_artist(l00)
ax00.set_ylim(0, y_max)
y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
ax00.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})


a_p = "pwl_slab"
for reg_code in regions.keys():
    ax01.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax01.set_xticklabels([])
ax01.set_ylabel("ADF in days", fontsize=label_size)
# ax01.set_title(a_ps[a_p])
ax01.set_ylim(0, y_max)
y_min, y_max = ax01.get_ylim()
dylim = y_max - y_min
ax01.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "wet"
for reg_code in regions.keys():
    ax02.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax02.set_xticklabels([])
ax02.set_ylabel("ADF in days", fontsize=label_size)
ax02.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax02.set_ylim(0, y_max)
y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "y"
for reg_code in regions.keys():
    ax03.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker="o", linewidth=0.5, c=colors[reg_code])
# end for reg_code
ax03.set_ylabel("ADF in days", fontsize=label_size)
ax03.set_xlabel("Year", fontsize=label_size)
# ax03.set_title(a_ps[a_p])
ax03.set_ylim(0, y_max)
y_min, y_max = ax03.get_ylim()
dylim = y_max - y_min
ax03.text(2000, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})


ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)
ax03.tick_params(axis='both', which='major', labelsize=tick_size)

fig.suptitle("EC-Earth historical & RCP8.5", fontsize=title_size)
fig.subplots_adjust(top=0.95, hspace=0.1)

pl.savefig(pl_path + f"TimeSeries_ADF_NorCP_EC-Earth_RCP85_{sea}Season.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot both scenarios
colors = {3009:"black", 3010:"gray", 3011:"red", 3012:"orange", 3013:"blue"}
stys = {3009:"-", 3010:"-", 3011:"--", 3012:"--", 3013:"-"}
marks = {3009:"o", 3010:"o", 3011:"o", 3012:"o", 3013:"o"}
dx_err = {3009:-4, 3010:-2, 3011:0, 3012:2, 3013:4}

lw = 1.25

yfac = 0.1
tick_size = 12
title_size = 14
label_size = 13

y_max = 95  # None

fig = pl.figure(figsize=(8, 9))
ax00 = fig.add_subplot(421)
ax01 = fig.add_subplot(423)
ax02 = fig.add_subplot(425)
ax03 = fig.add_subplot(427)

ax10 = fig.add_subplot(422)
ax11 = fig.add_subplot(424)
ax12 = fig.add_subplot(426)
ax13 = fig.add_subplot(428)

years = {'historical':1995, 'evaluation':2011, 'rcp45_MC':2050, 'rcp45_LC':2090}
scens = ['historical', 'rcp45_MC', 'rcp45_LC']

a_p = "wind_slab"
for reg_code in regions.keys():
    # ax00.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax00.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax00.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days", fontsize=label_size)
# ax00.set_title(a_ps[a_p])
p00, = ax00.plot([], [], c=colors[3009], linestyle=stys[3009], label=regions_pl[3009], linewidth=lw)
p01, = ax00.plot([], [], c=colors[3010], linestyle=stys[3010], label=regions_pl[3010], linewidth=lw)
p02, = ax00.plot([], [], c=colors[3011], linestyle=stys[3011], label=regions_pl[3011], linewidth=lw)
p03, = ax00.plot([], [], c=colors[3012], linestyle=stys[3012], label=regions_pl[3012], linewidth=lw)
p04, = ax00.plot([], [], c=colors[3013], linestyle=stys[3013], label=regions_pl[3013], linewidth=lw)
# l00 = ax00.legend(handles=[p00, p01, p04], ncols=3)
# l01 = ax00.legend(handles=[p03, p02], ncols=3, loc=(0.613, 0.64))
# ax00.add_artist(l00)
ax00.set_ylim(0, y_max)
y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
ax00.text(2000, y_max-dylim*yfac, "(a) " + a_ps[a_p], fontdict={"fontsize":13})


a_p = "pwl_slab"
for reg_code in regions.keys():
    # ax01.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax01.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax01.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax01.set_xticklabels([])
ax01.set_ylabel("ADF in days", fontsize=label_size)
# ax01.set_title(a_ps[a_p])
ax01.set_ylim(0, y_max)
y_min, y_max = ax01.get_ylim()
dylim = y_max - y_min
ax01.text(2000, y_max-dylim*yfac, "(c) " + a_ps[a_p], fontdict={"fontsize":13})

a_p = "wet"
for reg_code in regions.keys():
    ax02.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax02.set_xticklabels([])
ax02.set_ylabel("ADF in days", fontsize=label_size)
ax02.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax02.set_ylim(0, y_max)
y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(2000, y_max-dylim*yfac, "(e) " + a_ps[a_p], fontdict={"fontsize":13})

a_p = "y"
for reg_code in regions.keys():
    # ax03.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax03.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax03.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax03.set_ylabel("ADF in days", fontsize=label_size)
ax03.set_xlabel("Year", fontsize=label_size)
# ax03.set_title(a_ps[a_p])
ax03.set_ylim(0, y_max)
y_min, y_max = ax03.get_ylim()
dylim = y_max - y_min
ax03.text(2000, y_max-dylim*yfac, "(g) " + a_ps[a_p], fontdict={"fontsize":13})


ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)
ax03.tick_params(axis='both', which='major', labelsize=tick_size)


years = {'historical':1995, 'evaluation':2011, 'rcp85_MC':2050, 'rcp85_LC':2090}
scens = ['historical', 'rcp85_MC', 'rcp85_LC']

a_p = "wind_slab"
for reg_code in regions.keys():
    # ax10.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax10.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax10.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax10.set_xticklabels([])
ax10.set_yticklabels([])
# ax10.set_ylabel("ADF in days", fontsize=label_size)
# ax00.set_title(a_ps[a_p])
ax10.set_ylim(0, y_max)
y_min, y_max = ax10.get_ylim()
dylim = y_max - y_min
ax10.text(2000, y_max-dylim*yfac, "(b) " + a_ps[a_p], fontdict={"fontsize":13})


a_p = "pwl_slab"
for reg_code in regions.keys():
    ax11.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax11.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax11.set_xticklabels([])
ax11.set_yticklabels([])
# ax11.set_ylabel("ADF in days", fontsize=label_size)
# ax01.set_title(a_ps[a_p])
ax11.set_ylim(0, y_max)
y_min, y_max = ax11.get_ylim()
dylim = y_max - y_min
ax11.text(2000, y_max-dylim*yfac, "(d) " + a_ps[a_p], fontdict={"fontsize":13})

a_p = "wet"
for reg_code in regions.keys():
    # ax12.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax12.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax12.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax12.set_xticklabels([])
ax12.set_yticklabels([])
# ax12.set_ylabel("ADF in days", fontsize=label_size)
ax12.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax12.set_ylim(0, y_max)
y_min, y_max = ax12.get_ylim()
dylim = y_max - y_min
ax12.text(2000, y_max-dylim*yfac, "(f) " + a_ps[a_p], fontdict={"fontsize":13})

a_p = "y"
for reg_code in regions.keys():
    # ax13.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax13.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax13.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax13.set_yticklabels([])
# ax13.set_ylabel("ADF in days", fontsize=label_size)
ax13.set_xlabel("Year", fontsize=label_size)
# ax03.set_title(a_ps[a_p])
ax13.set_ylim(0, y_max)
y_min, y_max = ax13.get_ylim()
dylim = y_max - y_min
ax13.text(2000, y_max-dylim*yfac, "(h) " + a_ps[a_p], fontdict={"fontsize":13})


p10, = ax00.plot([], [], c=colors[3009], linestyle=stys[3009], label=regions_pl[3009], linewidth=lw, marker=marks[3009])
p11, = ax00.plot([], [], c=colors[3010], linestyle=stys[3010], label=regions_pl[3010], linewidth=lw, marker=marks[3010])
p12, = ax00.plot([], [], c=colors[3011], linestyle=stys[3011], label=regions_pl[3011], linewidth=lw, marker=marks[3011])
p13, = ax00.plot([], [], c=colors[3012], linestyle=stys[3012], label=regions_pl[3012], linewidth=lw, marker=marks[3012])
p14, = ax00.plot([], [], c=colors[3013], linestyle=stys[3013], label=regions_pl[3013], linewidth=lw, marker=marks[3013])
l00 = ax10.legend(handles=[p00, p01, p04], ncols=1, loc="upper right")  # , loc=(0.31, 0.825))
l01 = ax00.legend(handles=[p03, p02], ncols=1, loc="upper right")  # , loc=(0.585, 0.64))
# ax02.add_artist(l00)


ax10.tick_params(axis='both', which='major', labelsize=tick_size)
ax11.tick_params(axis='both', which='major', labelsize=tick_size)
ax12.tick_params(axis='both', which='major', labelsize=tick_size)
ax13.tick_params(axis='both', which='major', labelsize=tick_size)

ax00.set_title("Historical & RCP4.5")
ax10.set_title("Historical & RCP8.5")

fig.suptitle("EC-Earth", fontsize=title_size)
fig.subplots_adjust(top=0.95, hspace=0.1, wspace=0.025)

pl.savefig(pl_path + f"TimeSeries_ADF_NorCP_EC-Earth_{sea}Season.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()

##### stop here to reduce execution time #####
# sys.exit()
#### ----------------------------------- #####


#%% test the significance of the change using a Monte-Carlo test
"""
a_p = "wet"
p_hist_45_mc = monte_carlo_p2(adf_dict[reg_code]["adf"]["historical"][a_p][sea][1].DL,
                              adf_dict[reg_code]["adf"]["rcp45_MC"][a_p][sea][1].DL, num_permutations=100000)
p_45_mc_lc = monte_carlo_p2(adf_dict[reg_code]["adf"]["rcp45_MC"][a_p][sea][1].DL,
                            adf_dict[reg_code]["adf"]["rcp45_LC"][a_p][sea][1].DL, num_permutations=100000)
p_hist_45_lc = monte_carlo_p2(adf_dict[reg_code]["adf"]["historical"][a_p][sea][1].DL,
                              adf_dict[reg_code]["adf"]["rcp45_LC"][a_p][sea][1].DL, num_permutations=100000)
p_hist_85_mc = monte_carlo_p2(adf_dict[reg_code]["adf"]["historical"][a_p][sea][1].DL,
                              adf_dict[reg_code]["adf"]["rcp85_MC"][a_p][sea][1].DL, num_permutations=100000)
p_85_mc_lc = monte_carlo_p2(adf_dict[reg_code]["adf"]["rcp85_MC"][a_p][sea][1].DL,
                            adf_dict[reg_code]["adf"]["rcp85_LC"][a_p][sea][1].DL, num_permutations=100000)
p_hist_85_lc = monte_carlo_p2(adf_dict[reg_code]["adf"]["historical"][a_p][sea][1].DL,
                              adf_dict[reg_code]["adf"]["rcp85_LC"][a_p][sea][1].DL, num_permutations=100000)

"""
#% print a table
# print(f"""
#       p values
#       {a_p} ADF change {regions_pl[reg_code]}

#       RCP4.5
#       hist-MC {p_hist_45_mc:.4f}
#       MC-LC   {p_45_mc_lc:.4f}
#       hist-LC {p_hist_45_lc:.4f}

#       RCP8.5
#       hist-MC {p_hist_85_mc:.4f}
#       MC-LC   {p_85_mc_lc:.4f}
#       hist-LC {p_hist_85_lc:.4f}

#       """)


#%% create a mask for the upper triangle, including the diagonal
d_adf = np.zeros((5, 5))
mask = np.triu(np.ones_like(d_adf, dtype=bool))

# also mask those cells which are uninteresting
gray_mask = np.zeros((5, 5))
gray_mask[mask] = 1
gray_mask[4, 1] = 2
gray_mask[3, 2] = 2
gray_mask = gray_mask.flatten()[gray_mask.flatten() != 1]


#%% test if the late-century wind slab ADF is significantly different from zero
a_p = "wind_slab"
sea = "full"
diff_zero_ws = {}
for reg_code in regions.keys():
    diff_zero_ws[reg_code] = monte_carlo_p(adf_dict[reg_code]["adf"]["rcp85_LC"][a_p][sea][1].DL,
                                           np.array([0]), num_permutations=10000)
# end for reg_code

a_p = "pwl_slab"
sea = "full"
diff_zero_pwl = {}
for reg_code in regions.keys():
    diff_zero_pwl[reg_code] = monte_carlo_p(adf_dict[reg_code]["adf"]["rcp85_LC"][a_p][sea][1].DL,
                                            np.array([0]), num_permutations=10000)
# end for reg_code


print("p values wind slab:")
print(diff_zero_ws)

print("p values PWL slab:")
print(diff_zero_pwl)


#%% set up a heat map with the changes
d_adf_dict = {}
p_adf_dict = {}
for reg_code in regions.keys():

    d_adf_dict[reg_code] = {}
    p_adf_dict[reg_code] = {}

    for a_p in a_ps.keys():

        print(f"\n{regions_pl[reg_code]} {a_ps[a_p]}\n")

        d_adf = np.zeros((5, 5))
        p_adf = np.zeros((5, 5))

        for i, ik in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):
            for j, jk in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):

                d_adf[i, j] = adf_dict[reg_code]["adf_mean"][ik][a_p][sea] - \
                                                                         adf_dict[reg_code]["adf_mean"][jk][a_p][sea]

                p_adf[i, j] = monte_carlo_p2(adf_dict[reg_code]["adf"][ik][a_p][sea][1].DL,
                                             adf_dict[reg_code]["adf"][jk][a_p][sea][1].DL, num_permutations=100000)

            # end for j, jk
        # end for i, ik

        d_adf_dict[reg_code][a_p] = np.ma.masked_array(d_adf, mask=mask)
        p_adf_dict[reg_code][a_p] = np.ma.masked_array(p_adf, mask=mask)

    # end for a_p
# end for reg_code


#%% plot a heat map
reg_code = 3009
a_p = "pwl_slab"

ticks = ["historical", "45-MC", "45-LC", "85-MC", "85-LC"]

title = f"{regions_pl[reg_code]} EC-Earth {a_ps[a_p]}"

fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(111)

heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax00, title=title, mask=mask,
         ticks=ticks, vmin=-20, vmax=20, rects=None, annot_size=12, gray_text_mask=gray_mask,
         gray_mask_color="lightgray")

pl.gca().add_patch(pl.Rectangle((1, 4), 1, 1, color='gray', alpha=0.5))
pl.gca().add_patch(pl.Rectangle((2, 3), 1, 1, color='gray', alpha=0.5))

ax00.text(1.1, 1.5, "Read: [value in cell] = [vertical] $-$ [horizontal]")
ax00.text(1.1, 1.775, "bold values indicate $p$ < 0.05")

pl.show()
pl.close()


#%% plot the heat map for all regions
# a_p = "wet"
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
    hm = heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax20, title=title,
                  title_size=title_size, mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None,
                  annot_size=annot_size, gray_text_mask=gray_mask, gray_mask_color=gray_mask_color)
    ax20.add_patch(pl.Rectangle((1, 4), 1, 1, color='white', alpha=1))
    ax20.add_patch(pl.Rectangle((2, 3), 1, 1, color='white', alpha=1))

    pl_col = ax21.imshow([[np.nan, np.nan], [np.nan, np.nan]], cmap="coolwarm", vmin=-20, vmax=20)

    colbar = fig.colorbar(pl_col, ax=ax21, location="top", shrink=0.95, extend="both",
                          anchor=(0.0, -0.95), aspect=15)
    colbar.ax.tick_params(labelsize=13)
    colbar.set_label("Difference", size=13)

    ax21.text(-0.1, 0.675, "model: EC-Earth", fontdict={'weight':'bold', "fontsize":15})
    ax21.text(-0.1, 0.45, f"AP: {a_ps[a_p]}", fontdict={'weight':'bold', "fontsize":15})

    # ax21.text(-0.1, 0.4, "Read: [value in cell] = [vertical] $-$ [horizontal]", fontdict={"fontsize":13})
    # ax21.text(-0.1, 0.3, "bold values indicate $p$ < 0.05", fontdict={"fontsize":13})
    ax21.axis('off')

    pl.savefig(pl_path + f"ADF_NorCP_EC-Earth_{a_p}_{sea}Season.pdf", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()
# end for a_p


#%% Monte-Carlo investigation
"""
ik = "historical"
jk = "rcp85_LC"

test_mc = monte_carlo_p2(adf_dict[reg_code]["adf"][ik][a_p][sea][1].DL,
                                 adf_dict[reg_code]["adf"][jk][a_p][sea][1].DL, num_permutations=100000)

print(test_mc)
"""