#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the avalanche problem frequencies.
Also calculate the average danger level per avalanche problem.
"""


#%% imports
import os
import sys
import pandas as pd
import pylab as pl
import numpy as np

from ava_functions.Lists_and_Dictionaries.Paths import obs_path, path_par
from ava_functions.DatetimeSimple import date_dt
from ava_functions.Assign_Winter_Year import assign_winter_year
from ava_functions.Discrete_Hist import disc_hist, disc_hist_data

path_ehd = path_par


#%% set up a dictionary for the avalanche problem names
ava_p_pl = {"Avalanche":"General", "glide_slab":"Glide slab", "new_loose":"New loose", "new_slab":"New slab",
            "pwl_slab":"PWL slab", "wet_loose":"Wet loose", "wet_slab":"Wet slab", "wet":"Wet", "wind_slab":"Wind slab"}


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
# f_name = "Avalanche_Danger_List.csv"

levels = [1, 2, 3, 4, 5]
ndlev = {}
avg_dl = {}
std_dl = {}
sea_avd = {}
for ap in ava_p_pl.keys():
    f_name = ap + "_Danger_List.csv"

    ar_df = pd.read_csv(f_path + f_name)

    # convert the date column to datetime format
    ar_df["date"] = pd.to_datetime(ar_df["date"])
    ar_df.set_index("date", inplace=True, drop=False)

    # extract the different northern regions
    ar_it = ar_df.loc[ar_df.region == 3013, :]  # Indre Troms (it)
    ar_st = ar_df.loc[ar_df.region == 3012, :]  # Sør-Troms (st)
    ar_tr = ar_df.loc[ar_df.region == 3011, :]  # Tromsoe (tr)
    ar_ly = ar_df.loc[ar_df.region == 3010, :]  # Lyngen (ly)
    ar_nt = ar_df.loc[ar_df.region == 3009, :]  # Nord-Troms (nt)
    # ar_fi = ar_df.loc[ar_df.region == 3008, :]  # Finnmarksvidda (fi)

    # assign a winter year
    ar_it["win_year"] = assign_winter_year(ar_it.index, 8)
    ar_st["win_year"] = assign_winter_year(ar_st.index, 8)
    ar_tr["win_year"] = assign_winter_year(ar_tr.index, 8)
    ar_ly["win_year"] = assign_winter_year(ar_ly.index, 8)
    ar_nt["win_year"] = assign_winter_year(ar_nt.index, 8)

    # generate AvDs and non-AvDs
    ar_it["AvD"] = np.zeros(len(ar_it))
    ar_it["AvD"][ar_it["danger_level"] > 2] = 1
    ar_st["AvD"] = np.zeros(len(ar_st))
    ar_st["AvD"][ar_st["danger_level"] > 2] = 1
    ar_tr["AvD"] = np.zeros(len(ar_tr))
    ar_tr["AvD"][ar_tr["danger_level"] > 2] = 1
    ar_ly["AvD"] = np.zeros(len(ar_ly))
    ar_ly["AvD"][ar_ly["danger_level"] > 2] = 1
    ar_nt["AvD"] = np.zeros(len(ar_nt))
    ar_nt["AvD"][ar_nt["danger_level"] > 2] = 1

    win_years = np.unique(ar_it["win_year"])

    # concatenate the regions
    avds_allreg = pd.concat([ar_it, ar_st, ar_tr, ar_ly, ar_nt], axis=0)

    # count the AvDs and non-AvDs
    sea_avd[ap] = {}
    sea_avd[ap]["AvD"] = [np.sum(avds_allreg[avds_allreg["win_year"] == k]["AvD"] == 1) for k in win_years]
    sea_avd[ap]["non-AvD"] = [np.sum(avds_allreg[avds_allreg["win_year"] == k]["AvD"] == 0) for k in win_years]

    # get the numbers of danger levels per region
    ndlev_it = {dlev:np.sum(ar_it.danger_level == dlev) for dlev in levels}
    ndlev_ly = {dlev:np.sum(ar_ly.danger_level == dlev) for dlev in levels}
    ndlev_nt = {dlev:np.sum(ar_nt.danger_level == dlev) for dlev in levels}
    ndlev_st = {dlev:np.sum(ar_st.danger_level == dlev) for dlev in levels}
    ndlev_tr = {dlev:np.sum(ar_tr.danger_level == dlev) for dlev in levels}
    ndlev_tot = {dlev:ndlev_it[dlev]+ndlev_ly[dlev]+ndlev_nt[dlev]+ndlev_st[dlev]+ndlev_tr[dlev] for dlev in levels}
    ndlev_avg = {dlev:ndlev_tot[dlev]/5 for dlev in levels}

    ndlev[ap] = {"IT":ndlev_it, "LY":ndlev_ly, "NT":ndlev_nt, "ST":ndlev_st, "TR":ndlev_tr, "tot":ndlev_tot,
                 "avg":ndlev_avg}

    avg_dl[ap] = {"IT":np.mean(ar_it["danger_level"]), "LY":np.mean(ar_ly["danger_level"]),
                  "NT":np.mean(ar_nt["danger_level"]), "ST":np.mean(ar_st["danger_level"]),
                  "TR":np.mean(ar_tr["danger_level"])}
    avg_dl[ap]["tot"] = np.mean(np.array(list(avg_dl[ap].values())))

    std_dl[ap] = {"IT":np.std(ar_it["danger_level"]), "LY":np.std(ar_ly["danger_level"]),
                  "NT":np.std(ar_nt["danger_level"]), "ST":np.std(ar_st["danger_level"]),
                  "TR":np.std(ar_tr["danger_level"])}

    all_vals = np.concatenate([np.array(ar_it["danger_level"]), np.array(ar_ly["danger_level"]),
                               np.array(ar_nt["danger_level"]), np.array(ar_st["danger_level"]),
                               np.array(ar_tr["danger_level"])])

    std_dl[ap]["tot"] = np.nanstd(all_vals)

# end for ap


#%% loop over all regions and sum the two wet types
"""
ndlev["wet"] = {}
for reg in ndlev["wet_loose"].keys():
    wet = {lev:ndlev["wet_loose"][reg][lev] + ndlev["wet_slab"][reg][lev] for lev in levels}
    ndlev["wet"][reg] = wet
# end for reg
"""

#%% aggregate the ADLs 1&2 and 3-5
avds = {}

for ap in ndlev.keys():
    avds[ap] = {}
    for reg in ndlev[ap].keys():
        avds[ap][reg] = {}
        avds[ap][reg]["non"] = ndlev[ap][reg][1] + ndlev[ap][reg][2]
        avds[ap][reg]["AvD"] = ndlev[ap][reg][3] + ndlev[ap][reg][4] + ndlev[ap][reg][5]
    # end for reg
# end for ap


#%% plot
pl_path = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Avalanche_Paper_2/00_Figures/"

reg = "avg"
fcl = "none"

fig = pl.figure(figsize=(4, 3))
ax00 = fig.add_subplot()

ax00.scatter(levels, ndlev["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o", label="wind")
ax00.scatter(levels, ndlev["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax00.scatter(levels, ndlev["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax00.scatter(levels, ndlev["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^", label="wet (slab)")
ax00.scatter(levels, ndlev["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax00.scatter(levels, ndlev["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax00.scatter(levels, ndlev["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax00.scatter(levels, ndlev["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.legend()

ax00.set_xticks(levels)

ax00.set_xlabel("Danger level")
ax00.set_ylabel("Number of cases")
ax00.set_title("AP frequency in northern Norway")

pl.savefig(pl_path + "00_Fig01_AP_Frequency_Regional_Avg.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot the average danger level and its standard deviation per avalanche problem
tot_avg_dl = [avg_dl[k]["tot"] for k in avg_dl.keys()]
tot_std_dl = [std_dl[k]["tot"] for k in std_dl.keys()]

fig = pl.figure(figsize=(4, 3))
ax00 = fig.add_subplot()

ax00.errorbar(x=np.arange(len(tot_avg_dl)), y=tot_avg_dl, yerr=tot_std_dl, linewidth=1, marker="o")

ax00.set_xticks(np.arange(len(tot_avg_dl)))
ax00.set_xticklabels(ava_p_pl.values(), rotation=45)

ax00.set_ylabel("Danger level")

ax00.set_title("Average danger level per problem")

pl.savefig(pl_path + "01_Fig02_Avg_DL_per_AP.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% combine the two above figures
fig = pl.figure(figsize=(9.5, 3))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

ax00.scatter(levels, ndlev["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o", label="wind")
ax00.scatter(levels, ndlev["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax00.scatter(levels, ndlev["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax00.scatter(levels, ndlev["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^", label="wet (slab)")
ax00.scatter(levels, ndlev["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax00.scatter(levels, ndlev["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax00.scatter(levels, ndlev["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax00.scatter(levels, ndlev["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.legend()

ax00.set_xticks(levels)

ax00.set_xlabel("Danger level")
ax00.set_ylabel("Number of cases")
ax00.set_title("(a) AP frequency in northern Norway")


ax01.errorbar(x=np.arange(len(tot_avg_dl)), y=tot_avg_dl, yerr=tot_std_dl, linewidth=1, marker="o")

ax01.set_xticks(np.arange(len(tot_avg_dl)))
ax01.set_xticklabels(ava_p_pl.values(), rotation=45)

ax01.set_ylabel("Danger level")

ax01.set_title("(b) Average danger level per problem")

pl.savefig(pl_path + "02_Fig03_Avg_DL_per_AP_and_AP_Freq.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot the number of AP-levels per region
ylim = (-20, 500)

fig = pl.figure(figsize=(10, 6))
ax00 = fig.add_subplot(231)
ax01 = fig.add_subplot(232)
ax02 = fig.add_subplot(233)
ax10 = fig.add_subplot(234)
ax11 = fig.add_subplot(235)
ax12 = fig.add_subplot(236)

reg = "NT"
p000 = ax00.scatter(levels, ndlev["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
p001 = ax00.scatter(levels, ndlev["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
p002 = ax00.scatter(levels, ndlev["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
p003 = ax00.scatter(levels, ndlev["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
p004 = ax00.scatter(levels, ndlev["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
p005 = ax00.scatter(levels, ndlev["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
p006 = ax00.scatter(levels, ndlev["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
p007 = ax00.scatter(levels, ndlev["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_xticklabels([])
ax00.set_ylabel("Number of cases")
ax00.set_title("Nord-Troms")
ax00.set_ylim(ylim)


reg = "TR"
ax01.scatter(levels, ndlev["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax01.scatter(levels, ndlev["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax01.scatter(levels, ndlev["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax01.scatter(levels, ndlev["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax01.scatter(levels, ndlev["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax01.scatter(levels, ndlev["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax01.scatter(levels, ndlev["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax01.scatter(levels, ndlev["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax01.axhline(y=0, c="black", linewidth=0.5)

ax01.set_xticklabels([])
ax01.set_yticklabels([])
ax01.set_title("Tromsø")
ax01.set_ylim(ylim)


reg = "LY"
ax10.scatter(levels, ndlev["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax10.scatter(levels, ndlev["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax10.scatter(levels, ndlev["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax10.scatter(levels, ndlev["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax10.scatter(levels, ndlev["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax10.scatter(levels, ndlev["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax10.scatter(levels, ndlev["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax10.scatter(levels, ndlev["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax10.axhline(y=0, c="black", linewidth=0.5)

ax10.set_xticks(levels)
ax10.set_ylabel("Number of cases")
ax10.set_xlabel("Danger level")
ax10.set_title("Lyngen")
ax10.set_ylim(ylim)


reg = "ST"
ax11.scatter(levels, ndlev["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax11.scatter(levels, ndlev["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax11.scatter(levels, ndlev["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax11.scatter(levels, ndlev["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax11.scatter(levels, ndlev["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax11.scatter(levels, ndlev["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax11.scatter(levels, ndlev["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax11.scatter(levels, ndlev["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax11.axhline(y=0, c="black", linewidth=0.5)

ax11.set_xticks(levels)
ax11.set_yticklabels([])
ax11.set_xlabel("Danger level")
ax11.set_title("Sør-Troms")
ax11.set_ylim(ylim)


reg = "IT"
ax12.scatter(levels, ndlev["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax12.scatter(levels, ndlev["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax12.scatter(levels, ndlev["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax12.scatter(levels, ndlev["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax12.scatter(levels, ndlev["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax12.scatter(levels, ndlev["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax12.scatter(levels, ndlev["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax12.scatter(levels, ndlev["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax12.axhline(y=0, c="black", linewidth=0.5)

ax12.set_xticks(levels)
ax12.set_yticklabels([])
ax12.set_xlabel("Danger level")
ax12.set_title("Indre Troms")
ax12.set_ylim(ylim)


ax02.legend(handles=[p000, p001, p002, p003, p004, p005, p006, p007], loc="center")
ax02.set_xticks([])
ax02.set_yticks([])
ax02.spines['top'].set_visible(False)
ax02.spines['right'].set_visible(False)
ax02.spines['left'].set_visible(False)
ax02.spines['bottom'].set_visible(False)

fig.suptitle("AP frequency in northern Norway")
fig.subplots_adjust(wspace=0.05)

pl.show()
pl.close()


#%% plot the number of AP AvDs and non-AvDs per region
ylim = (-20, 500)
xlim = (-0.2, 1.2)

levels = [0, 1]

fig = pl.figure(figsize=(10, 6))
ax00 = fig.add_subplot(231)
ax01 = fig.add_subplot(232)
ax02 = fig.add_subplot(233)
ax10 = fig.add_subplot(234)
ax11 = fig.add_subplot(235)
ax12 = fig.add_subplot(236)

reg = "NT"
p000 = ax00.scatter(levels, avds["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
p001 = ax00.scatter(levels, avds["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
p002 = ax00.scatter(levels, avds["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
p003 = ax00.scatter(levels, avds["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
p004 = ax00.scatter(levels, avds["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
p005 = ax00.scatter(levels, avds["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
p006 = ax00.scatter(levels, avds["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
p007 = ax00.scatter(levels, avds["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_xticklabels([])
ax00.set_ylabel("Number of cases")
ax00.set_title("Nord-Troms")
ax00.set_ylim(ylim)
ax00.set_xlim(xlim)


reg = "TR"
ax01.scatter(levels, avds["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax01.scatter(levels, avds["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax01.scatter(levels, avds["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax01.scatter(levels, avds["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax01.scatter(levels, avds["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax01.scatter(levels, avds["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax01.scatter(levels, avds["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax01.scatter(levels, avds["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax01.axhline(y=0, c="black", linewidth=0.5)

ax01.set_xticklabels([])
ax01.set_yticklabels([])
ax01.set_title("Tromsø")
ax01.set_ylim(ylim)
ax01.set_xlim(xlim)


reg = "LY"
ax10.scatter(levels, avds["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax10.scatter(levels, avds["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax10.scatter(levels, avds["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax10.scatter(levels, avds["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax10.scatter(levels, avds["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax10.scatter(levels, avds["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax10.scatter(levels, avds["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax10.scatter(levels, avds["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax10.axhline(y=0, c="black", linewidth=0.5)

ax10.set_xticks(levels)
ax10.set_ylabel("Number of cases")
ax10.set_xlabel("Danger level")
ax10.set_title("Lyngen")
ax10.set_ylim(ylim)
ax10.set_xlim(xlim)


reg = "ST"
ax11.scatter(levels, avds["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax11.scatter(levels, avds["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax11.scatter(levels, avds["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax11.scatter(levels, avds["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax11.scatter(levels, avds["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax11.scatter(levels, avds["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax11.scatter(levels, avds["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax11.scatter(levels, avds["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax11.axhline(y=0, c="black", linewidth=0.5)

ax11.set_xticks(levels)
ax11.set_yticklabels([])
ax11.set_xlabel("Danger level")
ax11.set_title("Sør-Troms")
ax11.set_ylim(ylim)
ax11.set_xlim(xlim)


reg = "IT"
ax12.scatter(levels, avds["wind_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="o",
                    label="wind")
ax12.scatter(levels, avds["pwl_slab"][reg].values(), edgecolor="black", facecolor=fcl, marker="s", label="PWL")
ax12.scatter(levels, avds["wet"][reg].values(), edgecolor="black", facecolor=fcl, marker="d", label="wet (sum)")
ax12.scatter(levels, avds["wet_slab"][reg].values(), edgecolor="gray", facecolor=fcl, marker="^",
                   label="wet (slab)")
ax12.scatter(levels, avds["wet_loose"][reg].values(), edgecolor="gray", facecolor=fcl, marker="v",
             label="wet (loose)")
ax12.scatter(levels, avds["glide_slab"][reg].values(), marker="x", c="gray", label="glide")
ax12.scatter(levels, avds["new_slab"][reg].values(), marker="+", c="gray", label="new (slab)")
ax12.scatter(levels, avds["new_loose"][reg].values(), marker="_", c="gray", label="new (loose)")

ax12.axhline(y=0, c="black", linewidth=0.5)

ax12.set_xticks(levels)
ax12.set_yticklabels([])
ax12.set_xlabel("Danger level")
ax12.set_title("Indre Troms")
ax12.set_ylim(ylim)
ax12.set_xlim(xlim)


ax02.legend(handles=[p000, p001, p002, p003, p004, p005, p006, p007], loc="center")
ax02.set_xticks([])
ax02.set_yticks([])
ax02.spines['top'].set_visible(False)
ax02.spines['right'].set_visible(False)
ax02.spines['left'].set_visible(False)
ax02.spines['bottom'].set_visible(False)

fig.suptitle("AP frequency in northern Norway")
fig.subplots_adjust(wspace=0.05)

pl.show()
pl.close()


#%% plot the total northern Norwegian AvDs/non-AvDs
ap_name = ["Glide\nslab", "New\nloose", "New\nslab", "PWL\nslab", "Wet\nloose", "Wet\nslab", "Wind\nslab"]
stand_cols = pl.rcParams['axes.prop_cycle'].by_key()['color']

try:
    del (avds["wet"])
    del (avds["Avalanche"])
except:
    print("Problems already removed...")
# end try except

fig = pl.figure(figsize=(5, 3))
ax00 = fig.add_subplot()

dx = 0
i = 0
for ap in avds.keys():
    ax00.bar(-0.35 + dx, list(avds[ap]["tot"].values())[0], width=0.7, facecolor="none", edgecolor=stand_cols[i])
    ax00.bar(0.35 + dx, list(avds[ap]["tot"].values())[1], width=0.7, facecolor=stand_cols[i], edgecolor="none")
    dx += 2
    i += 1
# for ap

ax00.bar(0, 0, width=0.5, facecolor="black", edgecolor="none", label="AvD")
ax00.bar(0, 0, width=0.5, facecolor="none", edgecolor="black", label="non-AvD")
ax00.legend()

ax00.set_ylabel("Number of occurrences")
ax00.set_xticks(np.arange(0, len(avds)*2, 2))
ax00.set_xticklabels(ap_name, rotation=0)

ax00.set_title("AvDs and non-AvDs in northern Norway")

pl.show()
pl.close()


#%% plot ADLs stacked
adl_cols = ["#6bf198", "#ffd046", "#ff9a24", "#ff3131", "#0d0d0e"]  # with colour pipette from the Varsom website
# https://imagecolorpicker.com/

fig = pl.figure(figsize=(5, 3))
ax00 = fig.add_subplot()

dx = 0
for ap in avds.keys():

    bottoms = [0] + list(np.cumsum((list(ndlev[ap]["tot"].values())[:-1])))

    ax00.bar(x=np.zeros(len(ndlev[ap]["tot"])) + dx, height=list(ndlev[ap]["tot"].values()),
             bottom=bottoms, color=adl_cols, width=0.5)
    dx += 1
# end if

ax00.bar(0, 0, color="#6bf198", label="1")
ax00.bar(0, 0, color="#ffd046", label="2")
ax00.bar(0, 0, color="#ff9a24", label="3")
ax00.bar(0, 0, color="#ff3131", label="4")
ax00.bar(0, 0, color="#0d0d0e", label="5")
ax00.legend(title="ADLs")

ax00.set_ylabel("Number of occurrences")
ax00.set_xticks(np.arange(0, len(avds), 1))
ax00.set_xticklabels(ap_name, rotation=0)
ax00.set_ylim(0, 5000)

ax00.set_title("ADLs in northern Norway 2017/18-2024/25")

pl_path = "/media/kei070/One_Touch/IMPETUS/Avalanche_Danger_Data/"
pl.savefig(pl_path + "ADL_Statistics_NNorge.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% both above figures together
fig = pl.figure(figsize=(9, 3))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

dx = 0
i = 0
for ap in avds.keys():
    ax00.bar(-0.35 + dx, list(avds[ap]["tot"].values())[0], width=0.7, facecolor="none", edgecolor=stand_cols[i])
    ax00.bar(0.35 + dx, list(avds[ap]["tot"].values())[1], width=0.7, facecolor=stand_cols[i], edgecolor="none")
    dx += 2
    i += 1
# for ap

ax00.bar(0, 0, width=0.5, facecolor="black", edgecolor="none", label="AvD")
ax00.bar(0, 0, width=0.5, facecolor="none", edgecolor="black", label="non-AvD")
ax00.legend()

ax00.set_ylabel("Number of occurrences")
ax00.set_xticks(np.arange(0, len(avds)*2, 2))
ax00.set_xticklabels(ap_name, rotation=0)

ax00.set_title("AvDs and non-AvDs")


dx = 0
for ap in avds.keys():

    bottoms = [0] + list(np.cumsum((list(ndlev[ap]["tot"].values())[:-1])))

    ax01.bar(x=np.zeros(len(ndlev[ap]["tot"])) + dx, height=list(ndlev[ap]["tot"].values()),
             bottom=bottoms, color=adl_cols, width=0.5)
    dx += 1
# end if

ax01.bar(0, 0, color="#6bf198", label="1")
ax01.bar(0, 0, color="#ffd046", label="2")
ax01.bar(0, 0, color="#ff9a24", label="3")
ax01.bar(0, 0, color="#ff3131", label="4")
ax01.bar(0, 0, color="#0d0d0e", label="5")
ax01.legend(title="ADLs")

ax01.set_ylabel("Number of occurrences")
ax01.set_xticks(np.arange(0, len(avds), 1))
ax01.set_xticklabels(ap_name, rotation=0)
ax01.set_ylim(0, 4150)

ax01.set_title("ADLs")

fig.suptitle("Northern Norway 2017/18-2024/25")
fig.subplots_adjust(top=0.835, wspace=0.25)

pl_path = "/media/kei070/One_Touch/IMPETUS/Avalanche_Danger_Data/"
pl.savefig(pl_path + "ADL_AvD_Statistics_NNorge.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot the AP-AvDs per season
width = 0.2
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.bar(x=win_years-0.3, height=sea_avd["Avalanche"]["AvD"], width=width, color="red", label="General")
ax00.bar(x=win_years-0.3, height=sea_avd["Avalanche"]["non-AvD"], bottom=sea_avd["Avalanche"]["AvD"], width=width,
         color="black")

ax00.bar(x=win_years-0.1, height=sea_avd["wind_slab"]["AvD"], width=width, color="grey", label="Wind slab")
ax00.bar(x=win_years-0.1, height=sea_avd["wind_slab"]["non-AvD"], bottom=sea_avd["wind_slab"]["AvD"], width=width,
         color="black")

ax00.bar(x=win_years+0.1, height=sea_avd["pwl_slab"]["AvD"], width=width, color="orange", label="PWL slab")
ax00.bar(x=win_years+0.1, height=sea_avd["pwl_slab"]["non-AvD"], bottom=sea_avd["pwl_slab"]["AvD"], width=width,
         color="black")

ax00.bar(x=win_years+0.3, height=sea_avd["wet"]["AvD"], width=width, color="blue", label="Wet")
ax00.bar(x=win_years+0.3, height=sea_avd["wet"]["non-AvD"], bottom=sea_avd["wet"]["AvD"], width=width,
         color="black")

ax00.legend(ncol=2)

ax00.set_ylabel("Number of instances")
ax00.set_xlabel("Winter season ending in")
ax00.set_title("AvDs per winter season")

pl.show()
pl.close()


