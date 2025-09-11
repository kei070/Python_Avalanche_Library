#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the averages of the ADFs for the different regions and APs.
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
for reg_code in regions.keys():
    adf_dict[reg_code] = load(out_path + f"ADF_GFDL-CM3_{reg_code}_{regions[reg_code]}.joblib")
# end for reg_code


#%% years and colours and APs
years = [1995, 2011, 2050, 2090]
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}
a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}



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

fig.suptitle("GFDL-CM3 historical & RCP8.5", fontsize=title_size)
fig.subplots_adjust(top=0.94, hspace=0.1)

pl.savefig(pl_path + f"TimeSeries_ADF_NorCP_GFDL-CM3_RCP85_{sea}Season.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot with RCP8.5  -- INCLUDING GENERAL PROBLEM
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

y_max = 95  # None

years = {'historical':1995, 'evaluation':2011, 'rcp85_MC':2050, 'rcp85_LC':2090}
scens = ['historical', 'rcp85_MC', 'rcp85_LC']

fig = pl.figure(figsize=(4, 9))
ax00 = fig.add_subplot(411)
ax01 = fig.add_subplot(412)
ax02 = fig.add_subplot(413)
ax03 = fig.add_subplot(414)

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
l00 = ax00.legend(handles=[p00, p01, p04], ncols=1)
l01 = ax02.legend(handles=[p03, p02], ncols=1, loc="upper right")  # (0.613, 0.64))
# ax00.add_artist(l00)
ax00.set_ylim(0, y_max)
y_min, y_max = ax00.get_ylim()
dylim = y_max - y_min
ax00.text(2005, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})


a_p = "pwl_slab"
for reg_code in regions.keys():
    # ax01.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
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
ax01.text(2005, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "wet"
for reg_code in regions.keys():
    # ax02.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
    ax02.errorbar(x=np.array([years[k] for k in scens]) + dx_err[reg_code],
                  y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
                  yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                  linewidth=0, c=colors[reg_code], elinewidth=1)
    ax02.plot([years[k] for k in scens], [adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
              marker=marks[reg_code], linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
# end for reg_code
ax02.set_xticklabels([])
ax02.set_ylabel("ADF in days", fontsize=label_size)
ax02.set_xlabel("Year", fontsize=label_size)
# ax02.set_title(a_ps[a_p])
ax02.set_ylim(0, y_max)
y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax02.text(2005, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

a_p = "y"
for reg_code in regions.keys():
    # ax03.errorbar(x=[years[k] for k in scens], y=[adf_dict[reg_code]["adf_mean"][k][a_p][sea] for k in scens],
    #              yerr=[adf_dict[reg_code]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
    #              marker="o", linewidth=lw, c=colors[reg_code], linestyle=stys[reg_code])
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
y_min, y_max = ax02.get_ylim()
dylim = y_max - y_min
ax03.text(2005, y_max-dylim*yfac, a_ps[a_p], fontdict={"fontsize":13})

ax00.tick_params(axis='both', which='major', labelsize=tick_size)
ax01.tick_params(axis='both', which='major', labelsize=tick_size)
ax02.tick_params(axis='both', which='major', labelsize=tick_size)
ax03.tick_params(axis='both', which='major', labelsize=tick_size)

fig.suptitle("GFDL-CM3 historical & RCP8.5", fontsize=title_size)
fig.subplots_adjust(top=0.95, hspace=0.1)

pl.savefig(pl_path + f"TimeSeries_ADF_NorCP_GFDL-CM3_RCP85_{sea}Season.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


##### stop here to reduce execution time #####
sys.exit()
#### ----------------------------------- #####


#%% test the significance of the change using a Monte-Carlo test
reg_code = 3013
a_p = "wet"
p_hist_85_mc = monte_carlo_p2(adf_dict[reg_code]["adf"]["historical"][a_p][sea][1].DL,
                              adf_dict[reg_code]["adf"]["rcp85_MC"][a_p][sea][1].DL, num_permutations=100000)
p_85_mc_lc = monte_carlo_p2(adf_dict[reg_code]["adf"]["rcp85_MC"][a_p][sea][1].DL,
                            adf_dict[reg_code]["adf"]["rcp85_LC"][a_p][sea][1].DL, num_permutations=100000)
p_hist_85_lc = monte_carlo_p2(adf_dict[reg_code]["adf"]["historical"][a_p][sea][1].DL,
                              adf_dict[reg_code]["adf"]["rcp85_LC"][a_p][sea][1].DL, num_permutations=100000)


#% print a table
print(f"""
      p values
      {a_p} ADF change {regions_pl[reg_code]}

      RCP8.5
      hist-MC {p_hist_85_mc:.4f}
      MC-LC   {p_85_mc_lc:.4f}
      hist-LC {p_hist_85_lc:.4f}

      """)


#%% create a mask for the upper triangle, including the diagonal
d_adf = np.zeros((3, 3))
mask = np.triu(np.ones_like(d_adf, dtype=bool))


#%% set up a heat map with the changes
d_adf_dict = {}
p_adf_dict = {}
for reg_code in regions.keys():

    d_adf_dict[reg_code] = {}
    p_adf_dict[reg_code] = {}

    for a_p in a_ps.keys():

        print(f"\n{regions_pl[reg_code]} {a_ps[a_p]}\n")

        d_adf = np.zeros((3, 3))
        p_adf = np.zeros((3, 3))

        for i, ik in enumerate(["historical", "rcp85_MC", "rcp85_LC"]):
            for j, jk in enumerate(["historical", "rcp85_MC", "rcp85_LC"]):

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

ticks = ["historical", "85-MC", "85-LC"]

title = f"{regions_pl[reg_code]} GFDL-CM3 {a_ps[a_p]}"

fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(111)

heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax00, title=title, mask=mask,
         ticks=ticks, vmin=-20, vmax=20, rects=None, annot_size=12)

ax00.text(1.1, 1.5, "Read: [value in cell] = [vertical] $-$ [horizontal]")
ax00.text(1.1, 1.775, "bold values indicate $p$ < 0.05")

pl.show()
pl.close()


#%% plot the heat map for all regions
annot_size = 16
tick_size = 14
title_size = 17
a_p = "wet"
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
heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax00, title=title, title_size=title_size,
         mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None, annot_size=annot_size)

reg_code = 3010
title = f"{regions_pl[reg_code]}"
heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax01, title=title, title_size=title_size,
         mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None, annot_size=annot_size)

reg_code = 3011
title = f"{regions_pl[reg_code]}"
heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax10, title=title, title_size=title_size,
         mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None, annot_size=annot_size)

reg_code = 3012
title = f"{regions_pl[reg_code]}"
heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax11, title=title, title_size=title_size,
         mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None, annot_size=annot_size)

reg_code = 3013
title = f"{regions_pl[reg_code]}"
heat_map(diffs=d_adf_dict[reg_code][a_p], p_vals=p_adf_dict[reg_code][a_p], ax=ax20, title=title, title_size=title_size,
         mask=mask, ticks=ticks, tick_size=tick_size, vmin=-20, vmax=20, rects=None, annot_size=annot_size)

ax21.text(-0.1, 0.6, "model: GFDL-CM3", fontdict={'weight':'bold', "fontsize":15})
ax21.text(-0.1, 0.5, f"avalanche problem: {a_ps[a_p]}", fontdict={'weight':'bold', "fontsize":15})
ax21.text(-0.1, 0.4, "Read: [value in cell] = [vertical] $-$ [horizontal]", fontdict={"fontsize":13})
ax21.text(-0.1, 0.3, "bold values indicate $p$ < 0.05", fontdict={"fontsize":13})
ax21.axis('off')

pl.savefig(pl_path + f"ADF_NorCP_GFDL-CM3_{a_p}_{sea}Season.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% Monte-Carlo investigation
"""
ik = "historical"
jk = "rcp85_LC"

test_mc = monte_carlo_p2(adf_dict[reg_code]["adf"][ik][a_p][sea][1].DL,
                                 adf_dict[reg_code]["adf"][jk][a_p][sea][1].DL, num_permutations=100000)

print(test_mc)
"""