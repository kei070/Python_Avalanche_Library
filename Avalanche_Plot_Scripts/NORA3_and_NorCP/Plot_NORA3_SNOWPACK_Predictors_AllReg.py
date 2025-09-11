#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the NORA3 predictive features.
"""

#%% imports
import numpy as np
import pylab as pl
from scipy.stats import linregress as lr

from ava_functions.Data_Loading import load_agg_feats_no3, load_snowpack_stab
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Assign_Winter_Year import assign_winter_year
from ava_functions.Data_Loading import load_ci
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Features import feats_all


#%% set paths
pl_path = f"{path_par}/IMPETUS/NORA3/Plots/Features/"


#%% calculate the sum of high and low avalanche danger per season
sta_yr = 1970
end_yr = 2024
years = np.arange(sta_yr+1, end_yr+1, 1)


#%% load AO ------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
data_path = f"{path_par}/IMPETUS/AO/"

aos = load_ci(fname="AO_Index.txt", data_path=data_path, years=years)


#%% perform a running mean over AA, AOi, NAOi
r_win = 7
aos_ro = {k:aos[k].rolling(window=r_win, center=True).mean() for k in aos.keys()}


#%% load the data
# feats_no3 = {}
# sno_stab = {}
feats = {}
for reg_code in regions.keys():
    feats_no3 = load_agg_feats_no3(reg_codes=reg_code)
    sno_stab = load_snowpack_stab(reg_codes=reg_code, slope_angle=0, slope_azi=0)

    feats[reg_code] = feats_no3.merge(sno_stab, how="inner", left_index=True, right_index=True)
# end for reg_code


#%% split the data up into full, winter, and spring
feats_full = {}
feats_wint = {}
feats_spri = {}
for reg_code in regions.keys():
    full_inds = feats[reg_code].index.month == 0
    winter_inds = feats[reg_code].index.month == 0
    spring_inds = feats[reg_code].index.month == 0

    for mon in [12, 1, 2, 3, 4, 5]:
        full_inds = full_inds | (feats[reg_code].index.month == mon)
    # end for mon
    for mon in [12, 1, 2]:
        winter_inds = winter_inds | (feats[reg_code].index.month == mon)
    # end for mon
    for mon in [3, 4, 5]:
        spring_inds = spring_inds | (feats[reg_code].index.month == mon)
    # end for mon

    feats_full[reg_code] = feats[reg_code][full_inds]
    feats_wint[reg_code] = feats[reg_code][winter_inds]
    feats_spri[reg_code] = feats[reg_code][spring_inds]

# end for reg_code


#%% add winter year for aggreation
for reg_code in regions.keys():
    feats_full[reg_code]["winter_year"] = assign_winter_year(feats_full[reg_code].index, start_month=10)
    feats_wint[reg_code]["winter_year"] = assign_winter_year(feats_wint[reg_code].index, start_month=10)
    feats_spri[reg_code]["winter_year"] = assign_winter_year(feats_spri[reg_code].index, start_month=10)
# end for reg_code


#%% caclulate annual means
feats_full_an = {}
feats_wint_an = {}
feats_spri_an = {}
for reg_code in regions.keys():
    feats_full_an[reg_code] = feats_full[reg_code].groupby(feats_full[reg_code]["winter_year"]).mean()
    feats_wint_an[reg_code] = feats_wint[reg_code].groupby(feats_wint[reg_code]["winter_year"]).mean()
    feats_spri_an[reg_code] = feats_spri[reg_code].groupby(feats_spri[reg_code]["winter_year"]).mean()
# end for reg_code


#%% calculate 7-year rolling means
feats_full_ro = {}
feats_wint_ro = {}
feats_spri_ro = {}
for reg_code in regions.keys():
    feats_full_ro[reg_code] = feats_full_an[reg_code].rolling(window=r_win, center=True).mean()
    feats_wint_ro[reg_code] = feats_wint_an[reg_code].rolling(window=r_win, center=True).mean()
    feats_spri_ro[reg_code] = feats_spri_an[reg_code].rolling(window=r_win, center=True).mean()
# end for reg_code


#%% calculate the correlations
corr_an = {}
corr_ro = {}
p_an = {}
p_ro = {}
for reg_code in regions.keys():
    corr_an[reg_code] = {"full":{}, "winter":{}, "spring":{}}
    corr_ro[reg_code] = {"full":{}, "winter":{}, "spring":{}}
    p_an[reg_code] = {"full":{}, "winter":{}, "spring":{}}
    p_ro[reg_code] = {"full":{}, "winter":{}, "spring":{}}

    for fe in feats_full_an[reg_code].columns:
        l_mod_full_an = lr(aos["full"], feats_full_an[reg_code][fe])
        l_mod_wint_an = lr(aos["winter"], feats_wint_an[reg_code][fe])
        l_mod_spri_an = lr(aos["spring"], feats_spri_an[reg_code][fe])
        l_mod_full_ro = lr(aos_ro["full"][3:-3], feats_full_ro[reg_code][fe][3:-3])
        l_mod_wint_ro = lr(aos_ro["winter"][3:-3], feats_wint_ro[reg_code][fe][3:-3])
        l_mod_spri_ro = lr(aos_ro["spring"][3:-3], feats_spri_ro[reg_code][fe][3:-3])

        corr_an[reg_code]["full"][fe] = l_mod_full_an.rvalue
        corr_ro[reg_code]["full"][fe] = l_mod_full_ro.rvalue
        corr_an[reg_code]["winter"][fe] = l_mod_wint_an.rvalue
        corr_ro[reg_code]["winter"][fe] = l_mod_wint_ro.rvalue
        corr_an[reg_code]["spring"][fe] = l_mod_spri_an.rvalue
        corr_ro[reg_code]["spring"][fe] = l_mod_spri_ro.rvalue

        p_an[reg_code]["full"][fe] = l_mod_full_an.pvalue
        p_ro[reg_code]["full"][fe] = l_mod_full_ro.pvalue
        p_an[reg_code]["winter"][fe] = l_mod_wint_an.pvalue
        p_ro[reg_code]["winter"][fe] = l_mod_wint_ro.pvalue
        p_an[reg_code]["spring"][fe] = l_mod_spri_an.pvalue
        p_ro[reg_code]["spring"][fe] = l_mod_spri_ro.pvalue
# end for reg_code


#%% plot the rolling means
var = "t_max_emax"
reg_code = 3009

lw1 = 0.75
lw2 = 1.

fig = pl.figure(figure=(6, 10))
ax00 = fig.add_subplot(311)
ax10 = fig.add_subplot(312)
ax20 = fig.add_subplot(313)

ax00_1 = ax00.twinx()
ax10_1 = ax10.twinx()
ax20_1 = ax20.twinx()

# ax00.plot(feats_full_an[reg_code][var], linewidth=lw1)
ax00.plot(feats_full_ro[reg_code][var], label=var, linewidth=lw2)
ax00.plot([], label="AO", linewidth=lw2, c="black")
ax00_1.plot(aos_ro["full"], c="black")
ax00.set_title(f"R={corr_ro[reg_code]['full'][var]:.2f}, p={p_ro[reg_code]['full'][var]:.3f}")

# ax10.plot(feats_wint_an[reg_code][var], linewidth=lw1)
ax10.plot(feats_wint_ro[reg_code][var], label="winter", linewidth=lw2)
ax10_1.plot(aos_ro["winter"], c="black")
ax10.set_title(f"R={corr_ro[reg_code]['winter'][var]:.2f}, p={p_ro[reg_code]['winter'][var]:.3f}")

# ax20.plot(feats_spri_an[reg_code][var], linewidth=lw1)
ax20.plot(feats_spri_ro[reg_code][var], label="spring", linewidth=lw2)
ax20_1.plot(aos_ro["spring"], c="black")
ax20.set_title(f"R={corr_ro[reg_code]['spring'][var]:.2f}, p={p_ro[reg_code]['spring'][var]:.3f}")

ax00.legend()

ax00.set_xticklabels([])
ax10.set_xticklabels([])

ax00.set_ylabel(var)
ax10.set_ylabel(var)
ax20.set_ylabel(var)

ax00_1.set_ylabel("AO index")
ax10_1.set_ylabel("AO index")
ax20_1.set_ylabel("AO index")

fig.subplots_adjust(hspace=0.35)

pl.show()
pl.close()


#%% plot the annual means
var = "snow_depth_d1_emin"
reg_code = 3013

lw1 = 0.75
lw2 = 1.

fig = pl.figure(figure=(6, 10))
ax00 = fig.add_subplot(311)
ax10 = fig.add_subplot(312)
ax20 = fig.add_subplot(313)

ax00_1 = ax00.twinx()
ax10_1 = ax10.twinx()
ax20_1 = ax20.twinx()

# ax00.plot(feats_full_an[reg_code][var], linewidth=lw1)
ax00.plot(feats_full_an[reg_code][var], label=var, linewidth=lw2)
ax00.plot([], label="AO", linewidth=lw2, c="black")
ax00_1.plot(aos["full"], c="black")
ax00.set_title(f"Full $-$ R={corr_an[reg_code]['full'][var]:.2f}, p={p_an[reg_code]['full'][var]:.3f}")

# ax10.plot(feats_wint_an[reg_code][var], linewidth=lw1)
ax10.plot(feats_wint_an[reg_code][var], label="winter", linewidth=lw2)
ax10_1.plot(aos["winter"], c="black")
ax10.set_title(f"Winter $-$ R={corr_an[reg_code]['winter'][var]:.2f}, p={p_an[reg_code]['winter'][var]:.3f}")

# ax20.plot(feats_spri_an[reg_code][var], linewidth=lw1)
ax20.plot(feats_spri_an[reg_code][var], label="spring", linewidth=lw2)
ax20_1.plot(aos["spring"], c="black")
ax20.set_title(f"Spring $-$ R={corr_an[reg_code]['spring'][var]:.2f}, p={p_an[reg_code]['spring'][var]:.3f}")

# ax00.legend()

ax00.set_xticklabels([])
ax10.set_xticklabels([])

ax00.set_ylabel(var)
ax10.set_ylabel(var)
ax20.set_ylabel(var)

ax00_1.set_ylabel("AO index")
ax10_1.set_ylabel("AO index")
ax20_1.set_ylabel("AO index")

fig.subplots_adjust(hspace=0.35)

pl.show()
pl.close()


#%% plot some predictors
var = "t_max_emax"
reg_code = 3010


lw1 = 1.25
lw2 = 0.5

fig = pl.figure(figsize=(7, 3))
ax00 = fig.add_subplot(111)
ax00_1 = ax00.twinx()

# ax00.plot(feats_full_an[reg_code][var], linewidth=lw2, c="black")
# ax00.plot(feats_full_ro[reg_code][var], label="full", linewidth=lw1, c="black")
# ax00.plot(feats_wint_an[reg_code][var], linewidth=lw2, c="blue")
# ax00.plot(feats_wint_ro[reg_code][var], label="winter", linewidth=lw1, c="blue")

# ax00.plot(feats_spri_an[reg_code][var], linewidth=lw2, c="red")
ax00.plot(feats_spri_ro[reg_code][var], label="spring", linewidth=lw1, c="red")

ax00_1.plot(aos_ro["spring"], c="black")
# ax00_1.plot(aos["spring"])

ax00.set_xlabel("Year")
ax00_1.set_ylabel("AO index")
ax00.set_ylabel(var + " in K")
ax00.set_title(f"Spring {var} in {regions_pl[reg_code]}")

pl.savefig(pl_path + f"Tmax_{regions[reg_code]}_Spring.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot some correlation coefficients -- NOW IN PAPER II
var_l = ["s7_emin", "s3_emax", "t_max_emax", "lwc_max_d1", "lwc_max", "RTA_2_d3", "wmax3_emax", "r1_emax"]

feats_pl = np.array([feats_all[k] for k in var_l])

reg_markers = {3009:"o", 3010:"s", 3011:"v", 3012:"^", 3013:"d"}

y_lim = (-0.95, 0.95)

fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

for reg_code in regions.keys():
    for i, var in enumerate(var_l):

        sea, col = "winter", "gray"
        dx = -0.2
        lw = 2.5 if p_ro[reg_code][sea][var] < 0.05 else 1
        ax01.scatter(i+dx, corr_ro[reg_code][sea][var], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
        lw = 2.5 if p_an[reg_code][sea][var] < 0.05 else 1
        ax00.scatter(i+dx, corr_an[reg_code][sea][var], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)


        sea, col = "full", "black"
        dx = 0
        lw = 2.5 if p_ro[reg_code][sea][var] < 0.05 else 1
        ax01.scatter(i+dx, corr_ro[reg_code][sea][var], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
        lw = 2.5 if p_an[reg_code][sea][var] < 0.05 else 1
        ax00.scatter(i+dx, corr_an[reg_code][sea][var], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)

        sea, col = "spring", "red"
        dx = 0.2
        lw = 2.5 if p_ro[reg_code][sea][var] < 0.05 else 1
        ax01.scatter(i+dx, corr_ro[reg_code][sea][var], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
        lw = 2.5 if p_an[reg_code][sea][var] < 0.05 else 1
        ax00.scatter(i+dx, corr_an[reg_code][sea][var], marker=reg_markers[reg_code], edgecolor=col,
                     facecolor="none", linewidth=lw)
    # end for i, var
# end for reg_code

# for the legend
p_l = []
for reg_code in regions.keys():
    p00 = ax00.scatter([], [], edgecolor="black", facecolor="none", marker=reg_markers[reg_code],
                       label=regions_pl[reg_code])
    p_l.append(p00)
# end for reg_code
ax00.legend(handles=p_l,
            handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)

p010 = ax01.scatter([], [], c="gray", marker="s", label="winter")
p011 = ax01.scatter([], [], c="black", marker="s", label="full")
p012 = ax01.scatter([], [], c="red", marker="s", label="spring")
ax01.legend(handles=[p010, p011, p012],
            handletextpad=0.2, labelspacing=0.3, borderpad=0.5, handlelength=1.0, columnspacing=0.5)

for i in np.arange(len(var_l)-1):
    ax00.axvline(x=i+0.5, c="gray", linewidth=0.5, linestyle=":")
    ax01.axvline(x=i+0.5, c="gray", linewidth=0.5, linestyle=":")
# end for i

ax00.axhline(y=0, c="black", linewidth=0.5)
ax01.axhline(y=0, c="black", linewidth=0.5)

ax00.set_xticks(np.arange(len(var_l)))
ax01.set_xticks(np.arange(len(var_l)))

# ax00.tick_params(axis='x', which='major', length=10, pad=0.2)
# ax01.tick_params(axis='x', which='major', length=10, pad=0.2)

ax00.set_xticklabels(feats_pl, rotation=90)
ax01.set_xticklabels(feats_pl, rotation=90)

ax01.set_yticklabels([])

ax00.set_ylabel("Correlation R")

ax00.set_ylim(y_lim)
ax01.set_ylim(y_lim)

ax00.set_title("(a) Annual")
ax01.set_title("(b) 7-year rolling")

fig.suptitle("Feature correlation with AO index")
fig.subplots_adjust(wspace=0.02, top=0.87)

pl.savefig(pl_path + "/NORA3_Feats_Corr_With_AO.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()
