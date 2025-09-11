#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

# window for the rolling means
r_win = 7


#%% years and colours and APs
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}
a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}


#%% set paths
out_path = f"{path_par}/IMPETUS/NORA3/ML_Predictions/"
pl_path = f"{path_par}/IMPETUS/NORA3/Plots/ADF/"


#%% load the data
adf_dict = {}
try:
    for reg_code in regions.keys():
        adf_dict[reg_code] = load(out_path + f"ADF_{reg_code}_{regions[reg_code]}_NORA3.joblib")
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

a_p = "wind_slab"
for pred_reg in regions.keys():

    ao_ro_ll_corr_d[pred_reg] = {}
    ao_ro_ll_p_d[pred_reg] = {}
    ao_ll_corr_d[pred_reg] = {}
    ao_ll_p_d[pred_reg] = {}

    for a_p in a_ps.keys():

        ava_ro = adf_dict[pred_reg]["adf_roll"][a_p]
        ava = adf_dict[pred_reg]["adf"][a_p]

        ao_ro_ll_corr = {k:pd.Series(lead_lag_corr(aos_ro[k], ava_ro[k], max_lag=6)[0]) for k in ava_ro.keys()}
        ao_ro_ll_p = {k:pd.Series(lead_lag_corr(aos_ro[k], ava_ro[k], max_lag=6)[1]) for k in ava_ro.keys()}
        ao_ll_corr = {k:pd.Series(lead_lag_corr(aos[k], ava[k][1], max_lag=6)[0]) for k in ava.keys()}
        ao_ll_p = {k:pd.Series(lead_lag_corr(aos[k], ava[k][1], max_lag=6)[1]) for k in ava.keys()}

        ao_ro_ll_corr_d[pred_reg][a_p] = ao_ro_ll_corr
        ao_ro_ll_p_d[pred_reg][a_p] = ao_ro_ll_p
        ao_ll_corr_d[pred_reg][a_p] = ao_ll_corr
        ao_ll_p_d[pred_reg][a_p] = ao_ll_p

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

            l_mod = lr(adf_dict[reg_code]["adf"][a_p][sea][1].index[sta_y:end_y],
                                           adf_dict[reg_code]["adf"][a_p][sea][1].DL[sta_y:end_y])

            slo, y_int, r_v, p_v, err = l_mod

            lr_s[sea][reg_code][a_p] = slo
            lr_r[sea][reg_code][a_p] = r_v
            lr_p[sea][reg_code][a_p] = p_v
            lr_y[sea][reg_code][a_p] = y_int
            lr_e[sea][reg_code][a_p] = err
            lr_ye[sea][reg_code][a_p] = l_mod.intercept_stderr
        # end for a_p
    # end for reg_code
# end for sea



#%% plot
reg_code = 3013
sea = "winter"
a_p = "wind_slab"

adf = adf_dict[reg_code]["adf"]
adf_roll = adf_dict[reg_code]["adf_roll"]

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


fig = pl.figure(figsize=(8, 7))
ax00 = fig.add_subplot(311)
ax01 = fig.add_subplot(312)
ax02 = fig.add_subplot(313)

a_p = "wind_slab"
ax00.plot(adf[a_p][sea][1], c="black", linewidth=0.5)
ax00.plot(adf_roll[a_p][sea], c="black", linewidth=1.5)
ax00.axhline(y=0, c="black", linewidth=0.5)
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days")
ax00.text(1972, y_max00-d_y00, a_p, horizontalalignment="center")

a_p = "pwl_slab"
ax01.plot(adf[a_p][sea][1], c="black", linewidth=0.5)
ax01.plot(adf_roll[a_p][sea], c="black", linewidth=1.5)
ax01.axhline(y=0, c="black", linewidth=0.5)
ax01.set_ylabel("ADF in days")
ax01.set_xticklabels([])
ax01.text(1972, y_max01-d_y01, a_p, horizontalalignment="center")

a_p = "wet"
ax02.plot(adf[a_p][sea][1], c="black", linewidth=0.5)
ax02.plot(adf_roll[a_p][sea], c="black", linewidth=1.5)
ax02.axhline(y=0, c="black", linewidth=0.5)
ax02.set_xlabel("Year")
ax02.set_ylabel("ADF in days")
ax02.text(1972, y_max02-d_y02, a_p, horizontalalignment="center")

fig.suptitle(regions_pl[reg_code] + f" ({reg_code}) $-$ {sea}")
fig.subplots_adjust(top=0.94, hspace=0.1)

pl.show()
pl.close()


#%% plot for all regions and problems
colors = {3009:"orange", 3010:"purple", 3011:"blue", 3012:"red", 3013:"black"}
reg_code = 3013
a_p = "wind_slab"

adf = adf_dict[reg_code]["adf"]
adf_roll = adf_dict[reg_code]["adf_roll"]

sea = "winter"
y_max00 = np.nanmax([adf_dict[k]["adf_roll"][a_p][sea] for k in regions.keys()])
y_min00 = 0 #  np.nanmin([adf_roll[a_p][sea]])
d_y00 = (y_max00 - y_min00) * 0.075
a_p = "pwl_slab"
y_max01 = np.nanmax([adf_dict[k]["adf_roll"][a_p][sea] for k in regions.keys()])
y_min01 = 0 #  np.nanmin([adf_roll[a_p][sea]])
d_y01 = (y_max01 - y_min01) * 0.075
a_p = "wet"
y_max02 = np.nanmax([adf_dict[k]["adf_roll"][a_p][sea] for k in regions.keys()])
y_min02 = 0 #  np.nanmin([adf_roll[a_p][sea]])
d_y02 = (y_max02 - y_min02) * 0.075


fig = pl.figure(figsize=(16, 7))
ax00 = fig.add_subplot(331)
ax10 = fig.add_subplot(334)
ax20 = fig.add_subplot(337)

ax01 = fig.add_subplot(332)
ax11 = fig.add_subplot(335)
ax21 = fig.add_subplot(338)

ax02 = fig.add_subplot(333)
ax12 = fig.add_subplot(336)
ax22 = fig.add_subplot(339)

sea = "full"
a_p = "wind_slab"
lwd = 1
ax00.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd, label=regions_pl[3009])
ax00.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd, label=regions_pl[3010])
ax00.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd, label=regions_pl[3011])
ax00.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd)
ax00.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd)
ax00.legend(ncol=3)
ax00.axhline(y=0, c="black", linewidth=0.5)
ax00.set_xticklabels([])
ax00.set_ylabel("ADF in days")
# ax00.text(2010, y_max00-d_y00, a_p, horizontalalignment="center")

a_p = "pwl_slab"
ax10.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd)
ax10.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd)
ax10.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd)
ax10.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd)
ax10.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd)
ax10.axhline(y=0, c="black", linewidth=0.5)
ax10.set_ylabel("ADF in days")
ax10.set_xticklabels([])
# ax10.text(2010, y_max01-d_y01, a_p, horizontalalignment="center")

a_p = "wet"
ax20.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd)
ax20.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd)
ax20.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd)
ax20.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd, label=regions_pl[3012])
ax20.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd, label=regions_pl[3013])
ax20.legend(ncol=2)
ax20.axhline(y=0, c="black", linewidth=0.5)
ax20.set_xlabel("Year")
ax20.set_ylabel("ADF in days")
# ax20.text(2010, y_max02-d_y02, a_p, horizontalalignment="center")

ax00.set_title(f"{sea.capitalize()}")

text_x = 2013
sea = "winter"
a_p = "wind_slab"
lwd = 1
ax01.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd, label=regions_pl[3009])
ax01.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd, label=regions_pl[3010])
ax01.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd, label=regions_pl[3011])
ax01.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd)
ax01.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd)
# ax01.legend(ncol=3)
ax01.axhline(y=0, c="black", linewidth=0.5)
ax01.set_xticklabels([])
ax01.set_ylabel("ADF in days")
ax01.text(text_x, y_max00-d_y00, a_ps[a_p], horizontalalignment="center")

a_p = "pwl_slab"
ax11.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd)
ax11.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd)
ax11.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd)
ax11.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd)
ax11.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd)
ax11.axhline(y=0, c="black", linewidth=0.5)
ax11.set_ylabel("ADF in days")
ax11.set_xticklabels([])
ax11.text(text_x, y_max01-d_y01, a_ps[a_p], horizontalalignment="center")

a_p = "wet"
ax21.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd)
ax21.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd)
ax21.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd)
ax21.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd, label=regions_pl[3012])
ax21.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd, label=regions_pl[3013])
# ax21.legend(ncol=2)
ax21.axhline(y=0, c="black", linewidth=0.5)
ax21.set_xlabel("Year")
ax21.set_ylabel("ADF in days")
ax21.text(text_x, y_max02-d_y02, a_ps[a_p], horizontalalignment="center")

ax01.set_title(f"{sea.capitalize()}")


sea = "spring"
a_p = "wind_slab"
lwd = 1
ax02.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd, label=regions_pl[3009])
ax02.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd, label=regions_pl[3010])
ax02.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd, label=regions_pl[3011])
ax02.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd)
ax02.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd)
# ax02.legend(ncol=3)
ax02.axhline(y=0, c="black", linewidth=0.5)
ax02.set_xticklabels([])
ax02.set_ylabel("ADF in days")
# ax02.text(2010, y_max00-d_y00, a_p, horizontalalignment="center")

a_p = "pwl_slab"
ax12.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd)
ax12.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd)
ax12.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd)
ax12.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd)
ax12.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd)
ax12.axhline(y=0, c="black", linewidth=0.5)
ax12.set_ylabel("ADF in days")
ax12.set_xticklabels([])
# ax12.text(2010, y_max01-d_y01, a_p, horizontalalignment="center")

a_p = "wet"
ax22.plot(adf_dict[3009]["adf_roll"][a_p][sea], c=colors[3009], linewidth=lwd)
ax22.plot(adf_dict[3010]["adf_roll"][a_p][sea], c=colors[3010], linewidth=lwd)
ax22.plot(adf_dict[3011]["adf_roll"][a_p][sea], c=colors[3011], linewidth=lwd)
ax22.plot(adf_dict[3012]["adf_roll"][a_p][sea], c=colors[3012], linewidth=lwd, label=regions_pl[3012])
ax22.plot(adf_dict[3013]["adf_roll"][a_p][sea], c=colors[3013], linewidth=lwd, label=regions_pl[3013])
# ax22.legend(ncol=2)
ax22.axhline(y=0, c="black", linewidth=0.5)
ax22.set_xlabel("Year")
ax22.set_ylabel("ADF in days")
# ax22.text(2010, y_max02-d_y02, a_p, horizontalalignment="center")

ax02.set_title(f"{sea.capitalize()}")

fig.subplots_adjust(top=0.94, hspace=0.1, wspace=0.15)

pl.savefig(pl_path + "ADF_NORA3_1970_2024.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot with AO, NAO
sea = "spring"
a_p = "wet"
reg_code = 3010

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)
ax00_1 = ax00.twinx()

ax00.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax00_1.plot(aos_ro[sea], c="black")
ax00.plot([], c="red", label="ADF")
ax00.plot([], c="black", label="AO")
ax00.legend(loc="upper left")

ax00.set_ylabel("ADF in days")
ax00_1.set_ylabel("AO index")
ax00.set_xlabel("Year")

ax00.set_title(f"{regions_pl[reg_code]} $-$ {a_ps[a_p]}")

pl.show()
pl.close()


#%% plot all regions in one plot - rolling means
sea = "full"
a_p = "wet"

fig = pl.figure(figsize=(16, 7))
ax00 = fig.add_subplot(321)
ax00_1 = ax00.twinx()
ax01 = fig.add_subplot(322)
ax01_1 = ax01.twinx()
ax10 = fig.add_subplot(323)
ax10_1 = ax10.twinx()
ax11 = fig.add_subplot(324)
ax11_1 = ax11.twinx()
ax20 = fig.add_subplot(325)
ax20_1 = ax20.twinx()
ax21 = fig.add_subplot(326)

reg_code = 3009
ax00.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax00_1.plot(aos_ro[sea], c="black")
ax00.plot([], c="red", label="ADF")
ax00.plot([], c="black", label="AO")
ax00.legend(loc="upper left")

ax00.set_ylabel("ADF in days")
ax00_1.set_ylabel("AO index")
ax00.set_xticklabels([])
ax00.set_title(f"{regions_pl[reg_code]}")

reg_code = 3010
ax01.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax01_1.plot(aos_ro[sea], c="black")

ax01.set_ylabel("ADF in days")
ax01_1.set_ylabel("AO index")
ax01.set_xticklabels([])
ax01.set_title(f"{regions_pl[reg_code]}")

reg_code = 3011
ax10.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax10_1.plot(aos_ro[sea], c="black")
ax10.plot([], c="red", label="ADF")
ax10.plot([], c="black", label="AO")

ax10.set_ylabel("ADF in days")
ax10_1.set_ylabel("AO index")
ax10.set_xticklabels([])
ax10.set_title(f"{regions_pl[reg_code]}")

reg_code = 3012
ax11.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax11_1.plot(aos_ro[sea], c="black")

ax11.set_ylabel("ADF in days")
ax11_1.set_ylabel("AO index")
ax11.set_title(f"{regions_pl[reg_code]}")
ax11.set_xlabel("Year")

reg_code = 3013
ax20.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax20_1.plot(aos_ro[sea], c="black")
ax20.plot([], c="red", label="ADF")
ax20.plot([], c="black", label="AO")

ax20.set_ylabel("ADF in days")
ax20_1.set_ylabel("AO index")
ax20.set_title(f"{regions_pl[reg_code]}")
ax20.set_xlabel("Year")

ax21.plot()
x_txt0 = -0.045
dy0 = -0.005
ax21.text(x=x_txt0, y=0.02, s="Annual correlation")
ax21.text(x=x_txt0, y=0.01+dy0, s=f"{regions_pl[3009]}: R={ao_ll_corr_d[3009][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3009][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=0.0+dy0, s=f"{regions_pl[3010]}:        R={ao_ll_corr_d[3010][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3010][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=-0.01+dy0, s=f"{regions_pl[3011]}:        R={ao_ll_corr_d[3011][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3011][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=-0.02+dy0, s=f"{regions_pl[3012]}:    R={ao_ll_corr_d[3012][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3012][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=-0.03+dy0, s=f"{regions_pl[3013]}: R={ao_ll_corr_d[3013][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3013][a_p][sea][0]:.4f}")

x_txt1 = 0
dy1 = -0.005
ax21.text(x=x_txt1, y=0.02, s="7-year correlation")
ax21.text(x=x_txt1, y=0.01+dy1, s=f"{regions_pl[3009]}: R={ao_ro_ll_corr_d[3009][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3009][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=0.0+dy1, s=f"{regions_pl[3010]}:        R={ao_ro_ll_corr_d[3010][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3010][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=-0.01+dy1, s=f"{regions_pl[3011]}:        R={ao_ro_ll_corr_d[3011][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3011][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=-0.02+dy1, s=f"{regions_pl[3012]}:    R={ao_ro_ll_corr_d[3012][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3012][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=-0.03+dy1, s=f"{regions_pl[3013]}: R={ao_ro_ll_corr_d[3013][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3013][a_p][sea][0]:.4f}")

ax21.set_axis_off()

fig.suptitle(a_ps[a_p])
fig.subplots_adjust(top=0.93)

pl.savefig(pl_path + f"/Rolling/NORA3_Corr_With_ADF_AO_{r_win}Rolling_{sea}_{a_p}.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#% plot all regions in one plot - annual means
# sea = "spring"
# a_p = "wind_slab"

fig = pl.figure(figsize=(16, 7))
ax00 = fig.add_subplot(321)
ax00_1 = ax00.twinx()
ax01 = fig.add_subplot(322)
ax01_1 = ax01.twinx()
ax10 = fig.add_subplot(323)
ax10_1 = ax10.twinx()
ax11 = fig.add_subplot(324)
ax11_1 = ax11.twinx()
ax20 = fig.add_subplot(325)
ax20_1 = ax20.twinx()
ax21 = fig.add_subplot(326)

reg_code = 3009
ax00.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax00_1.plot(aos[sea], c="black")
ax00.plot([], c="red", label="ADF")
ax00.plot([], c="black", label="AO")
ax00.legend(loc="upper left")

ax00.set_ylabel("ADF in days")
ax00_1.set_ylabel("AO index")
ax00.set_xticklabels([])
ax00.set_title(f"{regions_pl[reg_code]}")

reg_code = 3010
ax01.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax01_1.plot(aos[sea], c="black")

ax01.set_ylabel("ADF in days")
ax01_1.set_ylabel("AO index")
ax01.set_xticklabels([])
ax01.set_title(f"{regions_pl[reg_code]}")

reg_code = 3011
ax10.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax10_1.plot(aos[sea], c="black")
ax10.plot([], c="red", label="ADF")
ax10.plot([], c="black", label="AO")

ax10.set_ylabel("ADF in days")
ax10_1.set_ylabel("AO index")
ax10.set_xticklabels([])
ax10.set_title(f"{regions_pl[reg_code]}")

reg_code = 3012
ax11.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax11_1.plot(aos[sea], c="black")

ax11.set_ylabel("ADF in days")
ax11_1.set_ylabel("AO index")
ax11.set_title(f"{regions_pl[reg_code]}")
ax11.set_xlabel("Year")

reg_code = 3013
ax20.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax20_1.plot(aos[sea], c="black")
ax20.plot([], c="red", label="ADF")
ax20.plot([], c="black", label="AO")

ax20.set_ylabel("ADF in days")
ax20_1.set_ylabel("AO index")
ax20.set_title(f"{regions_pl[reg_code]}")
ax20.set_xlabel("Year")

ax21.plot()
x_txt0 = -0.045
dy0 = -0.005
ax21.text(x=x_txt0, y=0.02, s="Annual correlation")
ax21.text(x=x_txt0, y=0.01+dy0, s=f"{regions_pl[3009]}: R={ao_ll_corr_d[3009][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3009][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=0.0+dy0, s=f"{regions_pl[3010]}:        R={ao_ll_corr_d[3010][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3010][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=-0.01+dy0, s=f"{regions_pl[3011]}:        R={ao_ll_corr_d[3011][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3011][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=-0.02+dy0, s=f"{regions_pl[3012]}:    R={ao_ll_corr_d[3012][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3012][a_p][sea][0]:.4f}")
ax21.text(x=x_txt0, y=-0.03+dy0, s=f"{regions_pl[3013]}: R={ao_ll_corr_d[3013][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ll_p_d[3013][a_p][sea][0]:.4f}")

x_txt1 = 0
dy1 = -0.005
ax21.text(x=x_txt1, y=0.02, s="7-year correlation")
ax21.text(x=x_txt1, y=0.01+dy1, s=f"{regions_pl[3009]}: R={ao_ro_ll_corr_d[3009][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3009][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=0.0+dy1, s=f"{regions_pl[3010]}:        R={ao_ro_ll_corr_d[3010][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3010][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=-0.01+dy1, s=f"{regions_pl[3011]}:        R={ao_ro_ll_corr_d[3011][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3011][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=-0.02+dy1, s=f"{regions_pl[3012]}:    R={ao_ro_ll_corr_d[3012][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3012][a_p][sea][0]:.4f}")
ax21.text(x=x_txt1, y=-0.03+dy1, s=f"{regions_pl[3013]}: R={ao_ro_ll_corr_d[3013][a_p][sea][0]:.2f} $p$=" +
          f"{ao_ro_ll_p_d[3013][a_p][sea][0]:.4f}")

ax21.set_axis_off()

fig.suptitle(a_ps[a_p])
fig.subplots_adjust(top=0.93)

pl.savefig(pl_path + f"/Annual/NORA3_Corr_With_ADF_AO_Annual_{sea}_{a_p}.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% paper plot: one region -- all problems -- both annual and 7-year
title_s = 14
label_s = 14
tickl_s = 12.5

fig = pl.figure(figsize=(16, 9))
ax00 = fig.add_subplot(421)
ax00_1 = ax00.twinx()
ax01 = fig.add_subplot(422)
ax01_1 = ax01.twinx()
ax10 = fig.add_subplot(423)
ax10_1 = ax10.twinx()
ax11 = fig.add_subplot(424)
ax11_1 = ax11.twinx()
ax20 = fig.add_subplot(425)
ax20_1 = ax20.twinx()
ax21 = fig.add_subplot(426)
ax21_1 = ax21.twinx()
ax30 = fig.add_subplot(427)
ax30_1 = ax30.twinx()
ax31 = fig.add_subplot(428)
ax31_1 = ax31.twinx()

reg_code = 3010
sea = "winter"

a_p = "wind_slab"
x = adf_dict[reg_code]["adf"][a_p][sea][1].index[sta_y:end_y]
y = lr_y[sea][reg_code][a_p] + (x * lr_s[sea][reg_code][a_p])
y1 = lr_y[sea][reg_code][a_p] + lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] - lr_e[sea][reg_code][a_p]))
y2 = lr_y[sea][reg_code][a_p] - lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] + lr_e[sea][reg_code][a_p]))
ax00.plot(x, y, c="black")
ax00.plot(x, y1, c="gray")
ax00.plot(x, y2, c="gray")
ax00.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax00_1.plot(aos[sea], c="black")
ax00.plot([], c="red", label="ADF")
ax00.plot([], c="black", label="AO")
# ax00.legend(loc="upper left")

ax00.set_ylabel("ADF in days", size=label_s)
ax00_1.set_ylabel("AO index", size=label_s)
ax00.set_xticklabels([])
ax00.set_title(f"(a) {a_ps[a_p]}", size=title_s)

a_p = "pwl_slab"
x = adf_dict[reg_code]["adf"][a_p][sea][1].index[sta_y:end_y]
y = lr_y[sea][reg_code][a_p] + (x * lr_s[sea][reg_code][a_p])
y1 = lr_y[sea][reg_code][a_p] + lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] - lr_e[sea][reg_code][a_p]))
y2 = lr_y[sea][reg_code][a_p] - lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] + lr_e[sea][reg_code][a_p]))
ax10.plot(x, y, c="black")
ax10.plot(x, y1, c="gray")
ax10.plot(x, y2, c="gray")
ax10.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax10_1.plot(aos[sea], c="black")

ax10.set_ylabel("ADF in days", size=label_s)
ax10_1.set_ylabel("AO index", size=label_s)
ax10.set_xticklabels([])
ax10.set_title(f"(c) {a_ps[a_p]}", size=title_s)

a_p = "wet"
x = adf_dict[reg_code]["adf"][a_p][sea][1].index[sta_y:end_y]
y = lr_y[sea][reg_code][a_p] + (x * lr_s[sea][reg_code][a_p])
y1 = lr_y[sea][reg_code][a_p] + lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] - lr_e[sea][reg_code][a_p]))
y2 = lr_y[sea][reg_code][a_p] - lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] + lr_e[sea][reg_code][a_p]))
ax20.plot(x, y, c="black")
ax20.plot(x, y1, c="gray")
ax20.plot(x, y2, c="gray")
ax20.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax20_1.plot(aos[sea], c="black")

ax20.set_ylabel("ADF in days", size=label_s)
ax20_1.set_ylabel("AO index", size=label_s)
ax20.set_xticklabels([])
ax20.set_title(f"(e) {a_ps[a_p]}", size=title_s)

a_p = "y"
x = adf_dict[reg_code]["adf"][a_p][sea][1].index[sta_y:end_y]
y = lr_y[sea][reg_code][a_p] + (x * lr_s[sea][reg_code][a_p])
y1 = lr_y[sea][reg_code][a_p] + lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] - lr_e[sea][reg_code][a_p]))
y2 = lr_y[sea][reg_code][a_p] - lr_ye[sea][reg_code][a_p] + (x * (lr_s[sea][reg_code][a_p] + lr_e[sea][reg_code][a_p]))
ax30.plot(x, y, c="black", linestyle="-")
ax30.plot(x, y1, c="gray")
ax30.plot(x, y2, c="gray")
ax30.plot(adf_dict[reg_code]["adf"][a_p][sea][1].DL, c="red")
ax30_1.plot(aos[sea], c="black")

ax30.set_ylabel("ADF in days", size=label_s)
ax30_1.set_ylabel("AO index", size=label_s)
# ax30.set_xticklabels([])
ax30.set_title(f"(g) {a_ps[a_p]}", size=title_s)


a_p = "wind_slab"
ax01.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax01_1.plot(aos_ro[sea], c="black")
ax01.plot([], c="red", label="ADF")
ax01.plot([], c="black", label="AO")
ax01.legend(loc="upper left", fontsize=14)

ax01.set_ylabel("ADF in days", size=label_s)
ax01_1.set_ylabel("AO index", size=label_s)
ax01.set_xticklabels([])
ax01.set_title(f"(b) {a_ps[a_p]}", size=title_s)

a_p = "pwl_slab"
ax11.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax11_1.plot(aos_ro[sea], c="black")

ax11.set_ylabel("ADF in days", size=label_s)
ax11_1.set_ylabel("AO index", size=label_s)
ax11.set_xticklabels([])
ax11.set_title(f"(d) {a_ps[a_p]}", size=title_s)

a_p = "wet"
ax21.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax21_1.plot(aos_ro[sea], c="black")

ax21.set_ylabel("ADF in days", size=label_s)
ax21_1.set_ylabel("AO index", size=label_s)
ax21.set_xticklabels([])
ax21.set_title(f"(f) {a_ps[a_p]}", size=title_s)

a_p = "y"
ax31.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="red")
ax31_1.plot(aos_ro[sea], c="black")

ax31.set_ylabel("ADF in days", size=label_s)
ax31_1.set_ylabel("AO index", size=label_s)
# ax31.set_xticklabels([])
ax31.set_title(f"(h) {a_ps[a_p]}", size=title_s)

ax00.tick_params(axis='y', labelsize=tickl_s)
ax00_1.tick_params(axis='y', labelsize=tickl_s)
ax01.tick_params(axis='y', labelsize=tickl_s)
ax01_1.tick_params(axis='y', labelsize=tickl_s)

ax10.tick_params(axis='y', labelsize=tickl_s)
ax10_1.tick_params(axis='y', labelsize=tickl_s)
ax11.tick_params(axis='y', labelsize=tickl_s)
ax11_1.tick_params(axis='y', labelsize=tickl_s)

ax20.tick_params(axis='y', labelsize=tickl_s)
ax20_1.tick_params(axis='y', labelsize=tickl_s)
ax21.tick_params(axis='y', labelsize=tickl_s)
ax21_1.tick_params(axis='y', labelsize=tickl_s)

ax30.tick_params(axis='y', labelsize=tickl_s)
ax30_1.tick_params(axis='y', labelsize=tickl_s)
ax31.tick_params(axis='y', labelsize=tickl_s)
ax31_1.tick_params(axis='y', labelsize=tickl_s)

ax30.tick_params(axis='x', labelsize=tickl_s)
ax31.tick_params(axis='x', labelsize=tickl_s)

fig.suptitle(regions_pl[reg_code], size=15)
fig.subplots_adjust(top=0.93, wspace=0.25)

pl.savefig(pl_path + f"/NORA3_Corr_With_ADF_AO_Annual_{sea}_{reg_code}.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot the linear correlations
reg_markers = {3009:"o", 3010:"s", 3011:"v", 3012:"^", 3013:"d"}

fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(111)

for reg_code in regions.keys():

    # for the legend
    ax00.scatter([], [], marker=reg_markers[reg_code], edgecolor="black", facecolor="none", label=regions_pl[reg_code])

    sea, col = "full", "black"
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if lr_p[sea][reg_code][a_p] < 0.05 else 1
        ax00.scatter(i, lr_r[sea][reg_code][a_p], marker=reg_markers[reg_code], edgecolor=col, facecolor="none",
                     linewidth=lw)
    # end for i, a_p

    sea, col = "winter", "gray"
    dx = -0.1
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if lr_p[sea][reg_code][a_p] < 0.05 else 1
        ax00.scatter(i+dx, lr_r[sea][reg_code][a_p], marker=reg_markers[reg_code], edgecolor=col, facecolor="none",
                     linewidth=lw)
    # end for i, a_p

    sea, col = "spring", "red"
    dx = 0.1
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if lr_p[sea][reg_code][a_p] < 0.05 else 1
        ax00.scatter(i+dx, lr_r[sea][reg_code][a_p], marker=reg_markers[reg_code], edgecolor=col, facecolor="none",
                     linewidth=lw)
    # end for i, a_p

# for reg_code

ax00.legend(ncol=2)

ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylabel("Trend coefficient")
ax00.set_xticks([0, 1, 2, 3])
ax00.set_xticklabels(["Wind slab", "PWL slab", "Wet", "General"])
ax00.set_title("Linear ADF trends NORA3 hindcast 1970-2024")

pl.show()
pl.close()


#%% plot the linear trend slopes -- NOW IN PAPER II
reg_markers = {3009:"o", 3010:"s", 3011:"v", 3012:"^", 3013:"d"}

fig = pl.figure(figsize=(5, 3))
ax00 = fig.add_subplot(111)

for reg_code in regions.keys():

    # for the legend
    ax00.scatter([], [], marker=reg_markers[reg_code], edgecolor="black", facecolor="none", label=regions_pl[reg_code])

    sea, col = "full", "black"
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if lr_p[sea][reg_code][a_p] < 0.05 else 1
        ax00.scatter(i, lr_s[sea][reg_code][a_p]*10, marker=reg_markers[reg_code], edgecolor=col, facecolor="none",
                     linewidth=lw)
    # end for i, a_p

    sea, col = "winter", "gray"
    dx = -0.1
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if lr_p[sea][reg_code][a_p] < 0.05 else 1
        ax00.scatter(i+dx, lr_s[sea][reg_code][a_p]*10, marker=reg_markers[reg_code], edgecolor=col, facecolor="none",
                     linewidth=lw)
    # end for i, a_p

    sea, col = "spring", "red"
    dx = 0.1
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if lr_p[sea][reg_code][a_p] < 0.05 else 1
        ax00.scatter(i+dx, lr_s[sea][reg_code][a_p]*10, marker=reg_markers[reg_code], edgecolor=col, facecolor="none",
                     linewidth=lw)
    # end for i, a_p

# for reg_code

for i in np.arange(3):
    ax00.axvline(x=i+0.5, c="gray", linewidth=0.5, linestyle=":")
# end for i

l00 = ax00.legend(ncol=2, fontsize=11, loc=(0.2, 0.01),
                  handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)

# for the second legend
p00 = ax00.scatter([], [], marker="s", c="black", label="full")
p01 = ax00.scatter([], [], marker="s", c="gray", label="winter")
p02 = ax00.scatter([], [], marker="s", c="red", label="spring")
l01 = ax00.legend(handles=[p00, p01], loc=(0.5, 0.86), ncol=2, fontsize=11,
                  handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)
ax00.legend(handles=[p02], loc=(0.05, 0.86), ncol=1, fontsize=11,
            handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0)
ax00.add_artist(l00)
ax00.add_artist(l01)

ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylabel("Linear trend in days dec$^{-1}$", fontsize=12)
ax00.set_xticks([0, 1, 2, 3])
ax00.set_xticklabels(["Wind slab", "PWL slab", "Wet", "General"])
ax00.xaxis.set_tick_params(labelsize=12)
ax00.yaxis.set_tick_params(labelsize=12)
ax00.set_title("Linear ADF trends NORA3 hindcast 1970-2024", fontsize=12.5)
pl.savefig(pl_path + "/NORA3_ADF_LinearTrend.pdf", bbox_inches="tight", dpi=200)
pl.show()
pl.close()


#%% plot the AO-ADF correlations
y_lim = (-0.82, 0.82)

fig = pl.figure(figsize=(9, 3))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

for reg_code in regions.keys():

    sea, col = "full", "black"
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if ao_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax00.scatter(i, ao_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)

        lw = 2.5 if ao_ro_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax01.scatter(i, ao_ro_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
    # end for i, a_p

    sea, col = "winter", "gray"
    dx = -0.1
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if ao_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax00.scatter(i+dx, ao_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
        lw = 2.5 if ao_ro_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax01.scatter(i+dx, ao_ro_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
    # end for i, a_p

    sea, col = "spring", "red"
    dx = 0.1
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if ao_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax00.scatter(i+dx, ao_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
        lw = 2.5 if ao_ro_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax01.scatter(i+dx, ao_ro_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
    # end for i, a_p

# end for reg_code

# for the legend
for reg_code in [3009, 3010, 3011]:
    ax00.scatter([], [], marker=reg_markers[reg_code], edgecolor="black", facecolor="none", label=regions_pl[reg_code])
# end for
l00 = ax00.legend(ncol=1, fontsize=11, loc="lower right")

# for the legend
for reg_code in [3012, 3013]:
    ax01.scatter([], [], marker=reg_markers[reg_code], edgecolor="black", facecolor="none", label=regions_pl[reg_code])
# end for
ax01.legend(ncol=1, fontsize=11, loc="lower right")


# for the second legend
p00 = ax00.scatter([], [], marker="s", c="black", label="full")
p01 = ax00.scatter([], [], marker="s", c="gray", label="winter")
p02 = ax00.scatter([], [], marker="s", c="red", label="spring")
l01 = ax00.legend(handles=[p00, p01, p02], loc="lower left", ncol=1, fontsize=11)
#ax00.legend(handles=[p02], loc=(0.05, 0.875), ncol=1, fontsize=11)
ax00.add_artist(l00)
#ax00.add_artist(l01)

ax00.axhline(y=0, c="black", linewidth=0.5)
ax01.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylabel("Correlation R")
ax00.set_title("(a) Annual")
ax01.set_title("(b) 7-year rolling")

ax00.set_ylim(y_lim)
ax01.set_ylim(y_lim)

ax01.set_yticklabels([])

ax00.set_xticks([0, 1, 2, 3])
ax00.set_xticklabels(["Wind slab", "PWL slab", "Wet", "General"])
ax01.set_xticks([0, 1, 2, 3])
ax01.set_xticklabels(["Wind slab", "PWL slab", "Wet", "General"])

fig.suptitle("ADF and AO correlation")
fig.subplots_adjust(wspace=0.05)

# pl.savefig(pl_path + "/NORA3_ADF_AO Corr.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% some extra plotting for the general ADF
reg_code = 3010
a_p = "y"
lw1 = 1.25
lw2 = 0.5

fig = pl.figure(figsize=(7, 3))
ax00 = fig.add_subplot(111)

sea = "spring"
ax00.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="black", linewidth=lw1, label="spring", linestyle=":")
ax00.plot(adf_dict[reg_code]["adf"][a_p][sea][1], c="black", linewidth=lw2, linestyle=":")

sea = "winter"
ax00.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="black", linewidth=lw1, label="winter", linestyle="--")
ax00.plot(adf_dict[reg_code]["adf"][a_p][sea][1], c="black", linewidth=lw2, linestyle="--")

sea = "full"
ax00.plot(adf_dict[reg_code]["adf_roll"][a_p][sea], c="black", linewidth=lw1, label="full", linestyle="-")
ax00.plot(adf_dict[reg_code]["adf"][a_p][sea][1], c="black", linewidth=lw2, linestyle="-")

ax00.legend(loc="upper left")

ax00.set_xlabel("Year")
ax00.set_ylabel("ADF in days")
ax00.set_title(f"ADF in {regions_pl[reg_code]}")

pl.savefig(pl_path + f"ADF_{regions[reg_code]}_AllSea.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot the AO-ADF correlations, second attempt -- NOW IN PAPER II
y_lim = (-0.82, 0.82)

fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

for reg_code in regions.keys():

    sea, col = "full", "black"
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if ao_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax00.scatter(i, ao_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)

        lw = 2.5 if ao_ro_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax01.scatter(i, ao_ro_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
    # end for i, a_p

    sea, col = "winter", "gray"
    dx = -0.2
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if ao_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax00.scatter(i+dx, ao_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
        lw = 2.5 if ao_ro_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax01.scatter(i+dx, ao_ro_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
    # end for i, a_p

    sea, col = "spring", "red"
    dx = 0.2
    for i, a_p in enumerate(["wind_slab", "pwl_slab", "wet", "y"]):
        lw = 2.5 if ao_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax00.scatter(i+dx, ao_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
        lw = 2.5 if ao_ro_ll_p_d[reg_code][a_p][sea][0] < 0.05 else 1
        ax01.scatter(i+dx, ao_ro_ll_corr_d[reg_code][a_p][sea][0], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
    # end for i, a_p

# end for reg_code

for i in np.arange(3):
    ax00.axvline(x=i+0.5, c="gray", linewidth=0.5, linestyle=":")
    ax01.axvline(x=i+0.5, c="gray", linewidth=0.5, linestyle=":")
# end for i

# for the legend
for reg_code in [3009, 3012, 3013]:
    ax00.scatter([], [], marker=reg_markers[reg_code], edgecolor="black", facecolor="none", label=regions_pl[reg_code])
# end for
l00 = ax00.legend(ncol=1, fontsize=10.5, loc="lower right",
                  handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)

reg_code = 3010
p001 = ax00.scatter([], [], marker=reg_markers[reg_code], edgecolor="black", facecolor="none",
                    label=regions_pl[reg_code])
reg_code = 3011
p002 = ax00.scatter([], [], marker=reg_markers[reg_code], edgecolor="black", facecolor="none",
                    label=regions_pl[reg_code])
ax00.legend(handles=[p001, p002], loc="lower left", fontsize=10.5,  # loc=(0.625, 0.26),
            handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)
ax00.add_artist(l00)

# for the second legend
p00 = ax00.scatter([], [], marker="s", c="black", label="full")
p01 = ax00.scatter([], [], marker="s", c="gray", label="winter")
p02 = ax00.scatter([], [], marker="s", c="red", label="spring")
l01 = ax01.legend(handles=[p00, p01, p02], loc="lower right", ncol=1, fontsize=10.5,
                  handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)

ax00.axhline(y=0, c="black", linewidth=0.5)
ax01.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylabel("Correlation R")
ax00.set_title("(a) Annual")
ax01.set_title("(b) 7-year rolling")

ax00.set_ylim(y_lim)
ax01.set_ylim(y_lim)

ax01.set_yticklabels([])

ax00.set_xticks([0, 1, 2, 3])
ax00.set_xticklabels(["Wind\nslab", "PWL\nslab", "Wet", "General"])
ax01.set_xticks([0, 1, 2, 3])
ax01.set_xticklabels(["Wind\nslab", "PWL\nslab", "Wet", "General"])

fig.suptitle("ADF and AO correlation")
fig.subplots_adjust(wspace=0.05)

pl.savefig(pl_path + "/NORA3_ADF_AO_Corr.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()



#%% print the table of linear trends
sea = "full"
a1, a2, a3, a4 = "wind_slab", "pwl_slab", "wet", "y"
r1, r2, r3, r4, r5 = 3009, 3010, 3011, 3012, 3013
print(f"""
                           {sea} season Pearson R

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_r[sea][r1][a1]:5.2f}        {lr_r[sea][r1][a2]:5.2f}      {lr_r[sea][r1][a3]:5.2f}   {lr_r[sea][r1][a4]:5.2f}

      Lyngen       {lr_r[sea][r2][a1]:5.2f}        {lr_r[sea][r2][a2]:5.2f}      {lr_r[sea][r2][a3]:5.2f}   {lr_r[sea][r2][a4]:5.2f}

      Tromsoe      {lr_r[sea][r3][a1]:5.2f}        {lr_r[sea][r3][a2]:5.2f}      {lr_r[sea][r3][a3]:5.2f}   {lr_r[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_r[sea][r4][a1]:5.2f}        {lr_r[sea][r4][a2]:5.2f}      {lr_r[sea][r4][a3]:5.2f}   {lr_r[sea][r4][a4]:5.2f}

      Indre Troms  {lr_r[sea][r5][a1]:5.2f}        {lr_r[sea][r5][a2]:5.2f}      {lr_r[sea][r5][a3]:5.2f}   {lr_r[sea][r5][a4]:5.2f}
      """)

print(f"""
                           {sea} season p value

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_p[sea][r1][a1]:5.2f}        {lr_p[sea][r1][a2]:5.2f}      {lr_p[sea][r1][a3]:5.2f}   {lr_p[sea][r1][a4]:5.2f}

      Lyngen       {lr_p[sea][r2][a1]:5.2f}        {lr_p[sea][r2][a2]:5.2f}      {lr_p[sea][r2][a3]:5.2f}   {lr_p[sea][r2][a4]:5.2f}

      Tromsoe      {lr_p[sea][r3][a1]:5.2f}        {lr_p[sea][r3][a2]:5.2f}      {lr_p[sea][r3][a3]:5.2f}   {lr_p[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_p[sea][r4][a1]:5.2f}        {lr_p[sea][r4][a2]:5.2f}      {lr_p[sea][r4][a3]:5.2f}   {lr_p[sea][r4][a4]:5.2f}

      Indre Troms  {lr_p[sea][r5][a1]:5.2f}        {lr_p[sea][r5][a2]:5.2f}      {lr_p[sea][r5][a3]:5.2f}   {lr_p[sea][r5][a4]:5.2f}
      """)

sea = "winter"
print(f"""
                           {sea} season Pearson R

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_r[sea][r1][a1]:5.2f}        {lr_r[sea][r1][a2]:5.2f}      {lr_r[sea][r1][a3]:5.2f}   {lr_r[sea][r1][a4]:5.2f}

      Lyngen       {lr_r[sea][r2][a1]:5.2f}        {lr_r[sea][r2][a2]:5.2f}      {lr_r[sea][r2][a3]:5.2f}   {lr_r[sea][r2][a4]:5.2f}

      Tromsoe      {lr_r[sea][r3][a1]:5.2f}        {lr_r[sea][r3][a2]:5.2f}      {lr_r[sea][r3][a3]:5.2f}   {lr_r[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_r[sea][r4][a1]:5.2f}        {lr_r[sea][r4][a2]:5.2f}      {lr_r[sea][r4][a3]:5.2f}   {lr_r[sea][r4][a4]:5.2f}

      Indre Troms  {lr_r[sea][r5][a1]:5.2f}        {lr_r[sea][r5][a2]:5.2f}      {lr_r[sea][r5][a3]:5.2f}   {lr_r[sea][r5][a4]:5.2f}
      """)

print(f"""
                           {sea} season p value

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_p[sea][r1][a1]:5.2f}        {lr_p[sea][r1][a2]:5.2f}      {lr_p[sea][r1][a3]:5.2f}   {lr_p[sea][r1][a4]:5.2f}

      Lyngen       {lr_p[sea][r2][a1]:5.2f}        {lr_p[sea][r2][a2]:5.2f}      {lr_p[sea][r2][a3]:5.2f}   {lr_p[sea][r2][a4]:5.2f}

      Tromsoe      {lr_p[sea][r3][a1]:5.2f}        {lr_p[sea][r3][a2]:5.2f}      {lr_p[sea][r3][a3]:5.2f}   {lr_p[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_p[sea][r4][a1]:5.2f}        {lr_p[sea][r4][a2]:5.2f}      {lr_p[sea][r4][a3]:5.2f}   {lr_p[sea][r4][a4]:5.2f}

      Indre Troms  {lr_p[sea][r5][a1]:5.2f}        {lr_p[sea][r5][a2]:5.2f}      {lr_p[sea][r5][a3]:5.2f}   {lr_p[sea][r5][a4]:5.2f}
      """)

sea = "spring"
print(f"""
                           {sea} season Pearson R

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_r[sea][r1][a1]:5.2f}        {lr_r[sea][r1][a2]:5.2f}      {lr_r[sea][r1][a3]:5.2f}   {lr_r[sea][r1][a4]:5.2f}

      Lyngen       {lr_r[sea][r2][a1]:5.2f}        {lr_r[sea][r2][a2]:5.2f}      {lr_r[sea][r2][a3]:5.2f}   {lr_r[sea][r2][a4]:5.2f}

      Tromsoe      {lr_r[sea][r3][a1]:5.2f}        {lr_r[sea][r3][a2]:5.2f}      {lr_r[sea][r3][a3]:5.2f}   {lr_r[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_r[sea][r4][a1]:5.2f}        {lr_r[sea][r4][a2]:5.2f}      {lr_r[sea][r4][a3]:5.2f}   {lr_r[sea][r4][a4]:5.2f}

      Indre Troms  {lr_r[sea][r5][a1]:5.2f}        {lr_r[sea][r5][a2]:5.2f}      {lr_r[sea][r5][a3]:5.2f}   {lr_r[sea][r5][a4]:5.2f}
      """)

print(f"""
                           {sea} season p value

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_p[sea][r1][a1]:5.2f}        {lr_p[sea][r1][a2]:5.2f}      {lr_p[sea][r1][a3]:5.2f}   {lr_p[sea][r1][a4]:5.2f}

      Lyngen       {lr_p[sea][r2][a1]:5.2f}        {lr_p[sea][r2][a2]:5.2f}      {lr_p[sea][r2][a3]:5.2f}   {lr_p[sea][r2][a4]:5.2f}

      Tromsoe      {lr_p[sea][r3][a1]:5.2f}        {lr_p[sea][r3][a2]:5.2f}      {lr_p[sea][r3][a3]:5.2f}   {lr_p[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_p[sea][r4][a1]:5.2f}        {lr_p[sea][r4][a2]:5.2f}      {lr_p[sea][r4][a3]:5.2f}   {lr_p[sea][r4][a4]:5.2f}

      Indre Troms  {lr_p[sea][r5][a1]:5.2f}        {lr_p[sea][r5][a2]:5.2f}      {lr_p[sea][r5][a3]:5.2f}   {lr_p[sea][r5][a4]:5.2f}
      """)


#%% print the slopes
sea = "full"
print(f"""
                           {sea} season slopes

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_s[sea][r1][a1]:5.2f}        {lr_s[sea][r1][a2]:5.2f}      {lr_s[sea][r1][a3]:5.2f}   {lr_s[sea][r1][a4]:5.2f}

      Lyngen       {lr_s[sea][r2][a1]:5.2f}        {lr_s[sea][r2][a2]:5.2f}      {lr_s[sea][r2][a3]:5.2f}   {lr_s[sea][r2][a4]:5.2f}

      Tromsoe      {lr_s[sea][r3][a1]:5.2f}        {lr_s[sea][r3][a2]:5.2f}      {lr_s[sea][r3][a3]:5.2f}   {lr_s[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_s[sea][r4][a1]:5.2f}        {lr_s[sea][r4][a2]:5.2f}      {lr_s[sea][r4][a3]:5.2f}   {lr_s[sea][r4][a4]:5.2f}

      Indre Troms  {lr_s[sea][r5][a1]:5.2f}        {lr_s[sea][r5][a2]:5.2f}      {lr_s[sea][r5][a3]:5.2f}   {lr_s[sea][r5][a4]:5.2f}
      """)

sea = "winter"
print(f"""
                           {sea} season slopes

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_s[sea][r1][a1]:5.2f}        {lr_s[sea][r1][a2]:5.2f}      {lr_s[sea][r1][a3]:5.2f}   {lr_s[sea][r1][a4]:5.2f}

      Lyngen       {lr_s[sea][r2][a1]:5.2f}        {lr_s[sea][r2][a2]:5.2f}      {lr_s[sea][r2][a3]:5.2f}   {lr_s[sea][r2][a4]:5.2f}

      Tromsoe      {lr_s[sea][r3][a1]:5.2f}        {lr_s[sea][r3][a2]:5.2f}      {lr_s[sea][r3][a3]:5.2f}   {lr_s[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_s[sea][r4][a1]:5.2f}        {lr_s[sea][r4][a2]:5.2f}      {lr_s[sea][r4][a3]:5.2f}   {lr_s[sea][r4][a4]:5.2f}

      Indre Troms  {lr_s[sea][r5][a1]:5.2f}        {lr_s[sea][r5][a2]:5.2f}      {lr_s[sea][r5][a3]:5.2f}   {lr_s[sea][r5][a4]:5.2f}
      """)

sea = "spring"
print(f"""
                           {sea} season slopes

                   wind slab    PWL slab    wet    general

      Nord-Troms   {lr_s[sea][r1][a1]:5.2f}        {lr_s[sea][r1][a2]:5.2f}      {lr_s[sea][r1][a3]:5.2f}   {lr_s[sea][r1][a4]:5.2f}

      Lyngen       {lr_s[sea][r2][a1]:5.2f}        {lr_s[sea][r2][a2]:5.2f}      {lr_s[sea][r2][a3]:5.2f}   {lr_s[sea][r2][a4]:5.2f}

      Tromsoe      {lr_s[sea][r3][a1]:5.2f}        {lr_s[sea][r3][a2]:5.2f}      {lr_s[sea][r3][a3]:5.2f}   {lr_s[sea][r3][a4]:5.2f}

      Soer-Troms   {lr_s[sea][r4][a1]:5.2f}        {lr_s[sea][r4][a2]:5.2f}      {lr_s[sea][r4][a3]:5.2f}   {lr_s[sea][r4][a4]:5.2f}

      Indre Troms  {lr_s[sea][r5][a1]:5.2f}        {lr_s[sea][r5][a2]:5.2f}      {lr_s[sea][r5][a3]:5.2f}   {lr_s[sea][r5][a4]:5.2f}
      """)


#%%
"""
x = adf_dict[reg_code]["adf"][a_p][sea][1].index[sta_y:end_y]
slop = lr_s[sea][reg_code][a_p]
y_i = lr_y[sea][reg_code][a_p]
y_err = lr_ye[sea][reg_code][a_p]
s_err = lr_e[sea][reg_code][a_p]

y = y_i + (x * slop)
y1 = y_i + y_err + (x * (slop - s_err))
y2 = y_i - y_err + (x * (slop + s_err))

pl.plot(x, y)
pl.plot(x, y1)
pl.plot(x, y2)

"""




