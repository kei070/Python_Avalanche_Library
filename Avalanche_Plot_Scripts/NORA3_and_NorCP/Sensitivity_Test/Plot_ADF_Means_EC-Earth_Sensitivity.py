#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENSITIVITY ANALYSIS FOR RF MODEL UNCERTAINTY

Plot the averages of the ADFs for the different regions and APs.

NOTE: You first need to generate the ADF data with the script Gen_ADF_Model_With_SNOWPACK.py
"""

#%% imports
import sys
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from scipy import stats
from joblib import load, dump
from ava_functions.MonteCarlo_P_test_With_Plot import monte_carlo_p, monte_carlo_p2
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Plot_Functions import heat_map


#%% parameters
sea = "full"  # full, winter, spring

# set the test years
test_yrs = ["21_23", "18_22", "19_24", "20_22"]


#%% set paths
data_path = f"{path_par}/IMPETUS/NorCP/ML_Predictions/"
pl_path = f"{path_par}/IMPETUS/NorCP/Plots/ADF/Sensitivity/EC-Earth/"


#%% load the data
adf_dict = {}
try:
    for reg_code in regions.keys():
        adf_dict[reg_code] = {}
        for test_yr in test_yrs:
            adf_dict[reg_code][test_yr] = load(data_path +
                                               f"ADF_EC-Earth_{reg_code}_{regions[reg_code]}_wo_{test_yr}.joblib")
        # end for test_yr
    # end for reg_code
except:
    print("\nData loading failed. Have you generated the ADF data with Test_Model_With_SNOWPACK_NorCP.py?\n")
# end try except


#%% years and colours and APs
years = [1995, 2011, 2050, 2090]
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}
a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}
panel = {3009:"(a)", 3010:"(b)", 3011:"(c)", 3012:"(d)", 3013:"(e)"}


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
d_adf_dict = {}
p_adf_dict = {}
for reg_code in regions.keys():

    d_adf_dict[reg_code] = {}
    p_adf_dict[reg_code] = {}

    for a_p in a_ps.keys():

        d_adf_dict[reg_code][a_p] = {}
        p_adf_dict[reg_code][a_p] = {}

        for test_yr in test_yrs:

            print(f"\n{regions_pl[reg_code]} {a_ps[a_p]} {test_yr}\n")

            d_adf = np.zeros((5, 5))
            p_adf = np.zeros((5, 5))

            for i, ik in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):
                for j, jk in enumerate(["historical", "rcp45_MC", "rcp45_LC", "rcp85_MC", "rcp85_LC"]):

                    d_adf[i, j] = adf_dict[reg_code][test_yr]["adf_mean"][ik][a_p][sea] - \
                                                                   adf_dict[reg_code][test_yr]["adf_mean"][jk][a_p][sea]

                    p_adf[i, j] = monte_carlo_p2(adf_dict[reg_code][test_yr]["adf"][ik][a_p][sea][1].DL,
                                                 adf_dict[reg_code][test_yr]["adf"][jk][a_p][sea][1].DL,
                                                 num_permutations=100000)

                # end for j, jk
            # end for i, ik

            d_adf_dict[reg_code][a_p][test_yr] = np.ma.masked_array(d_adf, mask=mask)
            p_adf_dict[reg_code][a_p][test_yr] = np.ma.masked_array(p_adf, mask=mask)
        # end for test_yr
    # end for a_p
# end for reg_code


#%% plot RCP4.5
years = {'historical':1995, 'evaluation':2011, 'rcp45_MC':2050, 'rcp45_LC':2090}
scens = ['historical', 'rcp45_MC', 'rcp45_LC']

test_yr_cs = {"21_23":"black", "18_22":"red", "19_24":"blue", "20_22":"orange"}

fig = pl.figure(figsize=(8, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)
axes = [ax00, ax01, ax10, ax11, ax20]

ax21 = fig.add_subplot(326)
ax21.axis("off")

a_p = "y"

for reg_code, ax in zip(regions.keys(), axes):
    for test_yr in test_yrs:
        ax.errorbar(x=[years[k] for k in scens],
                    y=[adf_dict[reg_code][test_yr]["adf_mean"][k][a_p][sea] for k in scens],
                    yerr=[adf_dict[reg_code][test_yr]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                    marker="o", linewidth=0.5, c=test_yr_cs[test_yr])
    # end for test_yr
    ax.set_title(f"{panel[reg_code]} {regions_pl[reg_code]}")
# end for reg_code, ax

for test_yr in test_yrs[:2]:
    ax10.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_cs[test_yr])
# end for test_yr, test_yr_c
ax10.legend()

for test_yr in test_yrs[2:]:
    ax11.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_cs[test_yr])
# end for test_yr, test_yr_c
ax11.legend()

ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax10.set_xticklabels([])

ax01.set_yticklabels([])
ax11.set_yticklabels([])

ax00.set_ylabel("ADF in d")
ax10.set_ylabel("ADF in d")
ax20.set_ylabel("ADF in d")

ax20.set_xlabel("Year")
ax11.set_xlabel("Year")

i = 0.5
y_col = 0.62
ax21.text(0.2, 0.81, "Change LC minus historical")
for reg_code in regions_pl.keys():
    ax21.text(0.05, i, regions_pl[reg_code])

    ax21.text(0.35, y_col, "21\n23", fontdict={"weight":"normal"})
    # [2, 0] is from historical to RCP4.5-LC; NOTE: We must take the abs() because for negative changes the p value is
    #                                         here negative!
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['21_23'][2, 0]) < 0.05 else "normal"
    ax21.text(0.35, i, f"{d_adf_dict[reg_code][a_p]['21_23'][2, 0]:.0f}", fontdict={"weight":font_weight})

    ax21.text(0.5, y_col, "18\n22")
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['18_22'][2, 0]) < 0.05 else "normal"
    ax21.text(0.5, i, f"{d_adf_dict[reg_code][a_p]['18_22'][2, 0]:.0f}", fontdict={"weight":font_weight})

    ax21.text(0.65, y_col, "19\n24")
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['19_24'][2, 0]) < 0.05 else "normal"
    ax21.text(0.65, i, f"{d_adf_dict[reg_code][a_p]['19_24'][2, 0]:.0f}", fontdict={"weight":font_weight})

    ax21.text(0.8, y_col, "20\n22")
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['20_22'][2, 0]) < 0.05 else "normal"
    ax21.text(0.8, i, f"{d_adf_dict[reg_code][a_p]['20_22'][2, 0]:.0f}", fontdict={"weight":font_weight})

    i -= 0.12
# end for reg

fig.suptitle(a_ps[a_p] + " " + sea + " RCP4.5")

fig.subplots_adjust(wspace=0.05, top=0.95)

pl.savefig(pl_path + f"/ADF_EC-Earth_RCP45_{a_p}.pdf", bbox_inches="tight", dpi=150)

pl.show()
pl.close()


#%% plot RCP8.5
years = {'historical':1995, 'evaluation':2011, 'rcp85_MC':2050, 'rcp85_LC':2090}
scens = ['historical', 'rcp85_MC', 'rcp85_LC']

test_yr_cs = {"21_23":"black", "18_22":"red", "19_24":"blue", "20_22":"orange"}

fig = pl.figure(figsize=(8, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)
axes = [ax00, ax01, ax10, ax11, ax20]

ax21 = fig.add_subplot(326)
ax21.axis("off")

a_p = "wet"

for reg_code, ax in zip(regions.keys(), axes):
    for test_yr in test_yrs:
        ax.errorbar(x=[years[k] for k in scens],
                    y=[adf_dict[reg_code][test_yr]["adf_mean"][k][a_p][sea] for k in scens],
                    yerr=[adf_dict[reg_code][test_yr]["adf_std"][k][a_p][sea] for k in scens], capsize=2.5,
                    marker="o", linewidth=0.5, c=test_yr_cs[test_yr])
    # end for test_yr
    ax.set_title(f"{panel[reg_code]} {regions_pl[reg_code]}")
# end for reg_code, ax

for test_yr in test_yrs[:2]:
    ax10.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_cs[test_yr])
# end for test_yr, test_yr_c
ax10.legend()

for test_yr in test_yrs[2:]:
    ax11.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_cs[test_yr])
# end for test_yr, test_yr_c
ax11.legend()

ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax10.set_xticklabels([])

ax01.set_yticklabels([])
ax11.set_yticklabels([])

ax00.set_ylabel("ADF in d")
ax10.set_ylabel("ADF in d")
ax20.set_ylabel("ADF in d")

ax20.set_xlabel("Year")
ax11.set_xlabel("Year")

i = 0.5
y_col = 0.62
ax21.text(0.2, 0.81, "Change LC minus historical")
for reg_code in regions_pl.keys():
    ax21.text(0.05, i, regions_pl[reg_code])

    ax21.text(0.35, y_col, "21\n23", fontdict={"weight":"normal"})
    # [4, 0] is from historical to RCP8.5-LC; NOTE: We must take the abs() because for negative changes the p value is
    #                                         here negative!
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['21_23'][4, 0]) < 0.05 else "normal"
    ax21.text(0.35, i, f"{d_adf_dict[reg_code][a_p]['21_23'][4, 0]:.0f}", fontdict={"weight":font_weight})

    ax21.text(0.5, y_col, "18\n22")
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['18_22'][4, 0]) < 0.05 else "normal"
    ax21.text(0.5, i, f"{d_adf_dict[reg_code][a_p]['18_22'][4, 0]:.0f}", fontdict={"weight":font_weight})

    ax21.text(0.65, y_col, "19\n24")
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['19_24'][4, 0]) < 0.05 else "normal"
    ax21.text(0.65, i, f"{d_adf_dict[reg_code][a_p]['19_24'][4, 0]:.0f}", fontdict={"weight":font_weight})

    ax21.text(0.8, y_col, "20\n22")
    font_weight = "bold" if abs(p_adf_dict[reg_code][a_p]['20_22'][4, 0]) < 0.05 else "normal"
    ax21.text(0.8, i, f"{d_adf_dict[reg_code][a_p]['20_22'][4, 0]:.0f}", fontdict={"weight":font_weight})

    i -= 0.12
# end for reg

fig.suptitle(a_ps[a_p] + " " + sea + " RCP8.5")

fig.subplots_adjust(wspace=0.05, top=0.95)

pl.savefig(pl_path + f"/ADF_EC-Earth_RCP85_{a_p}.pdf", bbox_inches="tight", dpi=150)

pl.show()
pl.close()
