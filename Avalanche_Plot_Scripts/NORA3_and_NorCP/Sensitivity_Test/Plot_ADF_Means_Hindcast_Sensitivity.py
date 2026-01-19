#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENSITIVITY ANALYSIS FOR RF MODEL UNCERTAINTY

Plot the averages of the ADFs for the different regions and APs.

First generate the ADF data with Gen_ADF_Model_With_SNOWPACK_fullNORA3.py !!!
"""

#%% imports
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from joblib import load, dump
from scipy.stats import linregress as lr
from ava_functions.MonteCarlo_P_test_With_Plot import monte_carlo_p, monte_carlo_p2
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Data_Loading import load_ci
from ava_functions.LeadLag_Corr import lead_lag_corr


#%% some parameters
sta_yr = 1970
end_yr = 2024

a_p_plot = "wet"

# set the test years
test_yrs = ["21_23", "18_22", "19_24", "20_22"]

# window for the rolling means
r_win = 7


#%% years and colours and APs and panel indicators
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}
a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}
panel = {3009:"(a)", 3010:"(b)", 3011:"(c)", 3012:"(d)", 3013:"(e)"}


#%% set paths
out_path = f"{path_par}/IMPETUS/NORA3/ML_Predictions/"
pl_path = f"{path_par}/IMPETUS/NORA3/Plots/ADF/Sensitivity/"


#%% load the data
adf_dict = {}
try:
    for reg_code in regions.keys():
        adf_dict[reg_code] = {}
        for test_yr in test_yrs:
            adf_dict[reg_code][test_yr] = load(out_path +
                                               f"ADF_{reg_code}_{regions[reg_code]}_NORA3_wo_{test_yr}.joblib")
        # end for test_yr
    # end for reg_code
except:
    print("\nData loading failed. Have you generated the ADF data with Test_Model_With_SNOWPACK_fullNORA3.py?\n")
# end try except


#%% calculate the sum of high and low avalanche danger per season
years = np.arange(sta_yr+1, end_yr+1, 1)


#%% load NAO -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
data_path = f"{path_par}/IMPETUS/NAO/"

naos = load_ci(fname="NAO_Index.txt", data_path=data_path, years=years)
naos_e5 = load_ci(fname="NAOi_ERA5.txt", data_path=data_path, years=years)
naos = load_ci(fname="NAO_Index_Hurrell.txt", data_path=data_path, years=years)


#%% load AO ------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
data_path = f"{path_par}/IMPETUS/AO/"

aos = load_ci(fname="AO_Index.txt", data_path=data_path, years=years)


#%% perform a running mean over AA, AOi, NAOi
aos_ro = {k:aos[k].rolling(window=r_win, center=True).mean() for k in aos.keys()}
naos_ro = {k:naos[k].rolling(window=r_win, center=True).mean() for k in naos.keys()}


#%% calculate the correlations
ao_ro_ll_corr_d = {}
ao_ro_ll_p_d = {}
ao_ll_corr_d = {}
ao_ll_p_d = {}

# a_p = "wind_slab"
for pred_reg in regions.keys():

    ao_ro_ll_corr_d[pred_reg] = {}
    ao_ro_ll_p_d[pred_reg] = {}
    ao_ll_corr_d[pred_reg] = {}
    ao_ll_p_d[pred_reg] = {}

    for a_p in a_ps.keys():

        ao_ro_ll_corr_d[pred_reg][a_p] = {}
        ao_ro_ll_p_d[pred_reg][a_p] = {}
        ao_ll_corr_d[pred_reg][a_p] = {}
        ao_ll_p_d[pred_reg][a_p] = {}

        for test_yr in test_yrs:

            ava_ro = adf_dict[pred_reg][test_yr]["adf_roll"][a_p]
            ava = adf_dict[pred_reg][test_yr]["adf"][a_p]

            ao_ro_ll_corr = {k:pd.Series(lead_lag_corr(aos_ro[k], ava_ro[k], max_lag=6)[0]) for k in ava_ro.keys()}
            ao_ro_ll_p = {k:pd.Series(lead_lag_corr(aos_ro[k], ava_ro[k], max_lag=6)[1]) for k in ava_ro.keys()}
            ao_ll_corr = {k:pd.Series(lead_lag_corr(aos[k], ava[k][1], max_lag=6)[0]) for k in ava.keys()}
            ao_ll_p = {k:pd.Series(lead_lag_corr(aos[k], ava[k][1], max_lag=6)[1]) for k in ava.keys()}

            ao_ro_ll_corr_d[pred_reg][a_p][test_yr] = ao_ro_ll_corr
            ao_ro_ll_p_d[pred_reg][a_p][test_yr] = ao_ro_ll_p
            ao_ll_corr_d[pred_reg][a_p][test_yr] = ao_ll_corr
            ao_ll_p_d[pred_reg][a_p][test_yr] = ao_ll_p

        # end for test_yr
    # end for a_p
# end for pred_reg


#%% calculate linear trends
sta_y = None
end_y = None

lr_s = {}
lr_r = {}
lr_p = {}
lr_y = {}
lr_e = {}
lr_ye = {}

for sea in ["full", "winter", "spring"]:
    lr_s[sea] = {}
    lr_r[sea] = {}
    lr_p[sea] = {}
    lr_y[sea] = {}
    lr_e[sea] = {}
    lr_ye[sea] = {}
    for reg_code in regions.keys():
        lr_s[sea][reg_code] = {}
        lr_r[sea][reg_code] = {}
        lr_p[sea][reg_code] = {}
        lr_y[sea][reg_code] = {}
        lr_e[sea][reg_code] = {}
        lr_ye[sea][reg_code] = {}
        for a_p in a_ps.keys():
            lr_s[sea][reg_code][a_p] = {}
            lr_r[sea][reg_code][a_p] = {}
            lr_p[sea][reg_code][a_p] = {}
            lr_y[sea][reg_code][a_p] = {}
            lr_e[sea][reg_code][a_p] = {}
            lr_ye[sea][reg_code][a_p] = {}
            for test_yr in test_yrs:
                l_mod = lr(adf_dict[reg_code][test_yr]["adf"][a_p][sea][1].index[sta_y:end_y],
                                               adf_dict[reg_code][test_yr]["adf"][a_p][sea][1].DL[sta_y:end_y])

                slo, y_int, r_v, p_v, err = l_mod

                lr_s[sea][reg_code][a_p][test_yr] = slo
                lr_r[sea][reg_code][a_p][test_yr] = r_v
                lr_p[sea][reg_code][a_p][test_yr] = p_v
                lr_y[sea][reg_code][a_p][test_yr] = y_int
                lr_e[sea][reg_code][a_p][test_yr] = err
                lr_ye[sea][reg_code][a_p][test_yr] = l_mod.intercept_stderr
        # end for a_p
    # end for reg_code
# end for sea


#%% plot: AO correlation
a_p_plot = "wind_slab"
sea = "winter"

test_yr_cs = ["black", "red", "blue", "orange"]

if a_p_plot == "wind_slab":
    y_lim = (0, 50)
elif a_p_plot == "wet":
    y_lim = (0, 11)
elif a_p_plot == "pwl_slab":
    y_lim = (0, 70)
elif a_p_plot == "y":
    y_lim = (0, 60)
# end if elif

fig = pl.figure(figsize=(8, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)
axes = [ax00, ax01, ax10, ax11, ax20]

ax21 = fig.add_subplot(326)
ax21.axis("off")

for pred_reg, ax in zip(regions.keys(), axes):
    for test_yr, test_yr_c in zip(test_yrs, test_yr_cs):
        ax.plot(adf_dict[pred_reg][test_yr]["adf_roll"][a_p_plot][sea], c=test_yr_c)
    # end for test_yr, test_yr_c
    ax.set_title(f"{panel[pred_reg]} {regions_pl[pred_reg]}")
    ax.set_ylim(y_lim)
# end for pred_reg, ax

for test_yr, test_yr_c in zip(test_yrs[:2], test_yr_cs[:2]):
    ax10.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_c)
# end for test_yr, test_yr_c
ax10.legend()

for test_yr, test_yr_c in zip(test_yrs[2:], test_yr_cs[2:]):
    ax11.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_c)
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

font_weight = "normal"

i = 0.5
y_col = 0.62
ax21.text(0.3, 0.81, "ADF$-$AO Correlation")
for reg_code in regions_pl.keys():
    ax21.text(0.05, i, regions_pl[reg_code])

    ax21.text(0.35, y_col, "21\n23", fontdict={"weight":"normal"})
    # font_weight = "bold" if ao_ro_ll_p_d[reg_code][a_p_plot]['21_23'][sea][0] < 0.05 else "normal"
    ax21.text(0.35, i, f"{ao_ro_ll_corr_d[reg_code][a_p_plot]['21_23'][sea][0]:.2f}", fontdict={"weight":font_weight})

    ax21.text(0.5, y_col, "18\n22")
    # font_weight = "bold" if ao_ro_ll_p_d[reg_code][a_p_plot]['18_22'][sea][0] < 0.05 else "normal"
    ax21.text(0.5, i, f"{ao_ro_ll_corr_d[reg_code][a_p_plot]['18_22'][sea][0]:.2f}", fontdict={"weight":font_weight})

    ax21.text(0.65, y_col, "19\n24")
    # font_weight = "bold" if ao_ro_ll_p_d[reg_code][a_p_plot]['19_24'][sea][0] < 0.05 else "normal"
    ax21.text(0.65, i, f"{ao_ro_ll_corr_d[reg_code][a_p_plot]['19_24'][sea][0]:.2f}", fontdict={"weight":font_weight})

    ax21.text(0.8, y_col, "20\n22")
    # font_weight = "bold" if ao_ro_ll_p_d[reg_code][a_p_plot]['20_22'][sea][0] < 0.05 else "normal"
    ax21.text(0.8, i, f"{ao_ro_ll_corr_d[reg_code][a_p_plot]['20_22'][sea][0]:.2f}", fontdict={"weight":font_weight})

    i -= 0.12
# end for reg

fig.suptitle(a_ps[a_p_plot] + " " + sea)

fig.subplots_adjust(wspace=0.05, top=0.95)

pl.savefig(pl_path + f"/AO_Corr/ADF_Hindcast_AO_Corr_{a_p_plot}_{sea}.pdf", bbox_inches="tight", dpi=150)

pl.show()
pl.close()


#%% plot: linear trends
a_p_plot = "y"
# sea = "full"

for sea in ["winter", "full", "spring"]:
    test_yr_cs = ["black", "red", "blue", "orange"]

    if a_p == "wind_slab":
        y_lim = (0, 70)
    elif a_p == "wet":
        y_lim = (0, 40)
    elif a_p == "pwl_slab":
        y_lim = (0, 80)
    elif a_p == "y":
        y_lim = (0, 125)
    # end if elif

    fig = pl.figure(figsize=(8, 8))
    ax00 = fig.add_subplot(321)
    ax01 = fig.add_subplot(322)
    ax10 = fig.add_subplot(323)
    ax11 = fig.add_subplot(324)
    ax20 = fig.add_subplot(325)
    axes = [ax00, ax01, ax10, ax11, ax20]

    ax21 = fig.add_subplot(326)
    ax21.axis("off")

    for pred_reg, ax in zip(regions.keys(), axes):
        for test_yr, test_yr_c in zip(test_yrs, test_yr_cs):
            ax.plot(adf_dict[pred_reg][test_yr]["adf"][a_p_plot][sea][1], c=test_yr_c, linewidth=0.75)
        # end for test_yr, test_yr_c
        ax.set_title(f"{panel[pred_reg]} {regions_pl[pred_reg]}")
        ax.set_ylim(y_lim)
    # end for pred_reg, ax

    for test_yr, test_yr_c in zip(test_yrs[:2], test_yr_cs[:2]):
        ax10.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_c)
    # end for test_yr, test_yr_c
    ax10.legend()

    for test_yr, test_yr_c in zip(test_yrs[2:], test_yr_cs[2:]):
        ax11.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_c)
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
    ax21.text(0.3, 0.81, "ADF Linear Trends")
    for reg_code in regions_pl.keys():
        ax21.text(0.05, i, regions_pl[reg_code])

        ax21.text(0.35, y_col, "21\n23", fontdict={"weight":"normal"})
        font_weight = "bold" if lr_p[sea][reg_code][a_p_plot]['21_23'] < 0.05 else "normal"
        ax21.text(0.35, i, f"{lr_s[sea][reg_code][a_p_plot]['21_23']*10:.1f}", fontdict={"weight":font_weight})

        ax21.text(0.5, y_col, "18\n22")
        font_weight = "bold" if lr_p[sea][reg_code][a_p_plot]['18_22'] < 0.05 else "normal"
        ax21.text(0.5, i, f"{lr_s[sea][reg_code][a_p_plot]['18_22']*10:.1f}", fontdict={"weight":font_weight})

        ax21.text(0.65, y_col, "19\n24")
        font_weight = "bold" if lr_p[sea][reg_code][a_p_plot]['19_24'] < 0.05 else "normal"
        ax21.text(0.65, i, f"{lr_s[sea][reg_code][a_p_plot]['19_24']*10:.1f}", fontdict={"weight":font_weight})

        ax21.text(0.8, y_col, "20\n22")
        font_weight = "bold" if lr_p[sea][reg_code][a_p_plot]['20_22'] < 0.05 else "normal"
        ax21.text(0.8, i, f"{lr_s[sea][reg_code][a_p_plot]['20_22']*10:.1f}", fontdict={"weight":font_weight})

        i -= 0.12
    # end for reg

    fig.suptitle(a_ps[a_p_plot] + " " + sea)

    fig.subplots_adjust(wspace=0.05, top=0.95)

    pl.savefig(pl_path + f"/Linear_Trend/ADF_Hindcast_LinearTrends_{a_p_plot}_{sea}.pdf", bbox_inches="tight", dpi=150)

    pl.show()
    pl.close()

# end for sea


#%% plot: AO correlation --> additional for annual values
a_p_plot = "wind_slab"
sea = "winter"

test_yr_cs = ["black", "red", "blue", "orange"]

if a_p_plot == "wind_slab":
    y_lim = (0, 60)
elif a_p_plot == "wet":
    y_lim = (0, 15)
elif a_p_plot == "pwl_slab":
    y_lim = (0, 70)
elif a_p_plot == "y":
    y_lim = (0, 80)
# end if elif

fig = pl.figure(figsize=(8, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)
axes = [ax00, ax01, ax10, ax11, ax20]

ax21 = fig.add_subplot(326)
ax21.axis("off")

for pred_reg, ax in zip(regions.keys(), axes):
    for test_yr, test_yr_c in zip(test_yrs, test_yr_cs):
        ax.plot(adf_dict[pred_reg][test_yr]["adf"][a_p_plot][sea][1], c=test_yr_c)
    # end for test_yr, test_yr_c
    ax.set_title(f"{panel[pred_reg]} {regions_pl[pred_reg]}")
    ax.set_ylim(y_lim)
# end for pred_reg, ax

for test_yr, test_yr_c in zip(test_yrs[:2], test_yr_cs[:2]):
    ax10.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_c)
# end for test_yr, test_yr_c
ax10.legend()

for test_yr, test_yr_c in zip(test_yrs[2:], test_yr_cs[2:]):
    ax11.plot([], [], label=test_yr.replace("_", "&"), color=test_yr_c)
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

font_weight = "normal"

i = 0.5
y_col = 0.62
ax21.text(0.3, 0.81, "ADF$-$AO Correlation")
for reg_code in regions_pl.keys():
    ax21.text(0.05, i, regions_pl[reg_code])

    ax21.text(0.35, y_col, "21\n23", fontdict={"weight":"normal"})
    font_weight = "bold" if ao_ll_p_d[reg_code][a_p_plot]['21_23'][sea][0] < 0.05 else "normal"
    ax21.text(0.35, i, f"{ao_ll_corr_d[reg_code][a_p_plot]['21_23'][sea][0]:.2f}", fontdict={"weight":font_weight})

    ax21.text(0.5, y_col, "18\n22")
    font_weight = "bold" if ao_ll_p_d[reg_code][a_p_plot]['18_22'][sea][0] < 0.05 else "normal"
    ax21.text(0.5, i, f"{ao_ll_corr_d[reg_code][a_p_plot]['18_22'][sea][0]:.2f}", fontdict={"weight":font_weight})

    ax21.text(0.65, y_col, "19\n24")
    font_weight = "bold" if ao_ll_p_d[reg_code][a_p_plot]['19_24'][sea][0] < 0.05 else "normal"
    ax21.text(0.65, i, f"{ao_ll_corr_d[reg_code][a_p_plot]['19_24'][sea][0]:.2f}", fontdict={"weight":font_weight})

    ax21.text(0.8, y_col, "20\n22")
    font_weight = "bold" if ao_ll_p_d[reg_code][a_p_plot]['20_22'][sea][0] < 0.05 else "normal"
    ax21.text(0.8, i, f"{ao_ll_corr_d[reg_code][a_p_plot]['20_22'][sea][0]:.2f}", fontdict={"weight":font_weight})

    i -= 0.12
# end for reg

fig.suptitle(a_ps[a_p_plot] + " " + sea)

fig.subplots_adjust(wspace=0.05, top=0.95)

# pl.savefig(pl_path + f"/AO_Corr/ADF_Hindcast_AO_Corr_{a_p_plot}_{sea}.pdf", bbox_inches="tight", dpi=150)

pl.show()
pl.close()