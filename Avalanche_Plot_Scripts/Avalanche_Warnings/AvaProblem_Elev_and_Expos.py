#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Look into the avalanche problem elevations and exposures.
"""

#%% imports
import numpy as np
import pandas as pd
import pylab as pl
from datetime import datetime

from ava_functions.Lists_and_Dictionaries.Paths import obs_path
from ava_functions.Discrete_Hist import disc_hist_data


#%% set the avalanche problem
a_p = "wind_slab"


#%% AP dictionary
a_ps = {"wind_slab":"Wind slab", "new_loose":"New loose", "new_slab":"New_slab",
        "pwl_slab":"PWL slab", "wet_loose":"Wet loose", "wet_slab":"Wet slab", "glide_slab":"Glide slab"}


#%% region dictionary
regs_pl = {3009:"Nord-Troms", 3010:"Lyngen", 3011:"Tromsø", 3012:"Sør-Troms", 3013:"Indre Troms"}


#%% set parameters
sta_date = "01-12-2016"
end_date = "01-06-2024"


#%% get start and end date in datatime format
sta_date_dt = datetime.strptime(sta_date, "%d-%m-%Y")
end_date_dt = datetime.strptime(end_date, "%d-%m-%Y")


#%% paths
path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"


#%% load a file
df = pd.read_csv(path + f"{a_p}_Danger_List.csv")


#%% extract the northern region
reg_dfs = {}
for reg_code in regs_pl.keys():
    df_reg = df[df["region"] == reg_code]
    df_reg["date"] = pd.to_datetime(df_reg["date"])
    df_reg.set_index("date", inplace=True)

    reg_dfs[reg_code] = df_reg
# end for reg_code


#%% loop over all exposition columns and turn NaN to False
# --> one exposition column with values from 1 to 8
expos_dict = {"N":1, "NE":2, "E":3, "SE":4, "S":5, "SW":6, "W":7, "NW":8}
expos = {}
for reg_code in regs_pl.keys():
    expos[reg_code] = np.zeros(len(reg_dfs[reg_code]))

    for k in expos_dict:
        reg_dfs[reg_code][k][reg_dfs[reg_code][k].isna()] = False

        # convert the expositions so as to be able to plot them
        expos[reg_code][reg_dfs[reg_code][k].astype(bool)] = expos_dict[k]
    # end for k
# end for reg_code


#%% plot the elevations
reg1, reg2, reg3, reg4, reg5 = 3009, 3010, 3011, 3012, 3013
ylim = (-5,  1550)


fig = pl.figure(figsize=(10, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)

ax00.plot(reg_dfs[reg1]["elev_min"][((reg_dfs[reg1].index > sta_date_dt) & (reg_dfs[reg1].index < end_date_dt))])
ax00.plot(reg_dfs[reg1]["elev_max"][((reg_dfs[reg1].index > sta_date_dt) & (reg_dfs[reg1].index < end_date_dt))])

ax00.set_ylim(ylim)
ax00.set_xticklabels([])
ax00.set_ylabel("Elevation in m")
ax00.set_title(f"{regs_pl[reg1]}")


ax01.plot(reg_dfs[reg2]["elev_min"][((reg_dfs[reg2].index > sta_date_dt) & (reg_dfs[reg2].index < end_date_dt))])
ax01.plot(reg_dfs[reg2]["elev_max"][((reg_dfs[reg2].index > sta_date_dt) & (reg_dfs[reg2].index < end_date_dt))])

ax01.set_ylim(ylim)
ax01.set_xticklabels([])
ax01.set_yticklabels([])
ax01.set_title(f"{regs_pl[reg2]}")


ax10.plot(reg_dfs[reg3]["elev_min"][((reg_dfs[reg3].index > sta_date_dt) & (reg_dfs[reg3].index < end_date_dt))])
ax10.plot(reg_dfs[reg3]["elev_max"][((reg_dfs[reg3].index > sta_date_dt) & (reg_dfs[reg3].index < end_date_dt))])

ax10.set_ylim(ylim)
ax10.set_xticklabels([])
ax10.set_ylabel("Elevation in m")
ax10.set_title(f"{regs_pl[reg3]}")


ax11.plot(reg_dfs[reg4]["elev_min"][((reg_dfs[reg4].index > sta_date_dt) & (reg_dfs[reg4].index < end_date_dt))])
ax11.plot(reg_dfs[reg4]["elev_max"][((reg_dfs[reg4].index > sta_date_dt) & (reg_dfs[reg4].index < end_date_dt))])

ax11.set_ylim(ylim)
ax11.set_yticklabels([])
ax11.set_title(f"{regs_pl[reg4]}")


ax20.plot(reg_dfs[reg5]["elev_min"][((reg_dfs[reg5].index > sta_date_dt) & (reg_dfs[reg5].index < end_date_dt))])
ax20.plot(reg_dfs[reg5]["elev_max"][((reg_dfs[reg5].index > sta_date_dt) & (reg_dfs[reg5].index < end_date_dt))])

ax20.set_ylim(ylim)
ax20.set_ylabel("Elevation in m")
ax20.set_title(f"{regs_pl[reg5]}")

fig.suptitle(a_ps[a_p])
fig.subplots_adjust(wspace=0.05, top=0.93)

pl.show()
pl.close()


#%% minimum elevation histograms
reg1, reg2, reg3, reg4, reg5 = 3009, 3010, 3011, 3012, 3013
ind_r1 = ((reg_dfs[reg1].index > sta_date_dt) & (reg_dfs[reg1].index < end_date_dt))
ind_r2 = ((reg_dfs[reg2].index > sta_date_dt) & (reg_dfs[reg2].index < end_date_dt))
ind_r3 = ((reg_dfs[reg3].index > sta_date_dt) & (reg_dfs[reg3].index < end_date_dt))
ind_r4 = ((reg_dfs[reg4].index > sta_date_dt) & (reg_dfs[reg4].index < end_date_dt))
ind_r5 = ((reg_dfs[reg5].index > sta_date_dt) & (reg_dfs[reg5].index < end_date_dt))

col = "black"
marker = "o"
ylim = (0, 320)

fig = pl.figure(figsize=(10, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)

ax00.hist(reg_dfs[reg1]["elev_min"][ind_r1])
ax00.set_ylim(ylim)
# ax00.set_xticks([0] + list(expos_dict.values()))
# ax00.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax00.set_title(f"{regs_pl[reg1]}")

ax01.hist(reg_dfs[reg2]["elev_min"][ind_r2])
ax01.set_ylim(ylim)
ax01.set_yticklabels([])
# ax01.set_xticks([0] + list(expos_dict.values()))
# ax01.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax01.set_title(f"{regs_pl[reg2]}")

ax10.hist(reg_dfs[reg3]["elev_min"][ind_r3])
ax10.set_ylim(ylim)
# ax10.set_xticks([0] + list(expos_dict.values()))
# ax10.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax10.set_title(f"{regs_pl[reg3]}")

ax11.hist(reg_dfs[reg4]["elev_min"][ind_r4])
ax11.set_ylim(ylim)
ax11.set_yticklabels([])
# ax11.set_xticks([0] + list(expos_dict.values()))
# ax11.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax11.set_xlabel("Min. elevation in m")
ax11.set_title(f"{regs_pl[reg4]}")

ax20.hist(reg_dfs[reg5]["elev_min"][ind_r5])
ax20.set_ylim(ylim)
# ax20.set_xticks([0] + list(expos_dict.values()))
# ax20.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax20.set_xlabel("Min. elevation in m")
ax20.set_title(f"{regs_pl[reg5]}")

fig.suptitle(a_ps[a_p])
fig.subplots_adjust(wspace=0.05, hspace=0.3, top=0.93)

pl.show()
pl.close()


#%% plot the expositions
reg1, reg2, reg3, reg4, reg5 = 3009, 3010, 3011, 3012, 3013
ind_r1 = ((reg_dfs[reg1].index > sta_date_dt) & (reg_dfs[reg1].index < end_date_dt))
ind_r2 = ((reg_dfs[reg2].index > sta_date_dt) & (reg_dfs[reg2].index < end_date_dt))
ind_r3 = ((reg_dfs[reg3].index > sta_date_dt) & (reg_dfs[reg3].index < end_date_dt))
ind_r4 = ((reg_dfs[reg4].index > sta_date_dt) & (reg_dfs[reg4].index < end_date_dt))
ind_r5 = ((reg_dfs[reg5].index > sta_date_dt) & (reg_dfs[reg5].index < end_date_dt))

col = "black"
marker = "o"

fig = pl.figure(figsize=(10, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)

ax00.scatter(reg_dfs[reg1].index[ind_r1], expos[reg1][ind_r1], facecolor="none", edgecolor=col, marker=marker)
ax00.set_xticklabels([])
ax00.set_yticks([0] + list(expos_dict.values()))
ax00.set_yticklabels(["NaN"] + list(expos_dict.keys()))
ax00.set_title(f"{regs_pl[reg1]}")

ax01.scatter(reg_dfs[reg2].index[ind_r2], expos[reg2][ind_r2], facecolor="none", edgecolor=col, marker=marker)
ax01.set_xticklabels([])
ax01.set_yticklabels([])
ax01.set_title(f"{regs_pl[reg2]}")

ax10.scatter(reg_dfs[reg3].index[ind_r3], expos[reg3][ind_r3], facecolor="none", edgecolor=col, marker=marker)
ax10.set_xticklabels([])
ax10.set_yticks([0] + list(expos_dict.values()))
ax10.set_yticklabels(["NaN"] + list(expos_dict.keys()))
ax10.set_title(f"{regs_pl[reg3]}")

ax11.scatter(reg_dfs[reg4].index[ind_r4], expos[reg4][ind_r4], facecolor="none", edgecolor=col, marker=marker)
ax11.set_yticklabels([])
ax11.set_title(f"{regs_pl[reg4]}")

ax20.scatter(reg_dfs[reg5].index[ind_r5], expos[reg5][ind_r5], facecolor="none", edgecolor=col, marker=marker)
ax20.set_yticks([0] + list(expos_dict.values()))
ax20.set_yticklabels(["NaN"] + list(expos_dict.keys()))
ax20.set_title(f"{regs_pl[reg5]}")

fig.suptitle(a_ps[a_p])
fig.subplots_adjust(wspace=0.05, hspace=0.2, top=0.93)

pl.show()
pl.close()


#%% exposition histograms
col = "black"
marker = "o"
ylim = (0, 650)

fig = pl.figure(figsize=(10, 8))
ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)

ax00.hist(expos[reg1][ind_r1], bins=[-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2,
                                     6.8, 7.2, 7.8, 8.2])
ax00.set_ylim(ylim)
ax00.set_xticks([0] + list(expos_dict.values()))
ax00.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax00.set_title(f"{regs_pl[reg1]}")

ax01.hist(expos[reg2][ind_r2], bins=[-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2,
                                     6.8, 7.2, 7.8, 8.2])
ax01.set_ylim(ylim)
ax01.set_yticklabels([])
ax01.set_xticks([0] + list(expos_dict.values()))
ax01.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax01.set_title(f"{regs_pl[reg2]}")

ax10.hist(expos[reg3][ind_r3], bins=[-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2,
                                     6.8, 7.2, 7.8, 8.2])
ax10.set_ylim(ylim)
ax10.set_xticks([0] + list(expos_dict.values()))
ax10.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax10.set_title(f"{regs_pl[reg3]}")

ax11.hist(expos[reg4][ind_r4], bins=[-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2,
                                     6.8, 7.2, 7.8, 8.2])
ax11.set_ylim(ylim)
ax11.set_yticklabels([])
ax11.set_xticks([0] + list(expos_dict.values()))
ax11.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax11.set_title(f"{regs_pl[reg4]}")

ax20.hist(expos[reg5][ind_r5], bins=[-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2,
                                     6.8, 7.2, 7.8, 8.2])
ax20.set_ylim(ylim)
ax20.set_xticks([0] + list(expos_dict.values()))
ax20.set_xticklabels(["NaN"] + list(expos_dict.keys()))
ax20.set_title(f"{regs_pl[reg5]}")

fig.suptitle(a_ps[a_p])
fig.subplots_adjust(wspace=0.05, hspace=0.3, top=0.93)

pl.show()
pl.close()


#%% prepare the data for a pie-chart
regs = [reg1, reg2, reg3, reg4, reg5]
ind_rs = [ind_r1, ind_r2, ind_r3, ind_r4, ind_r5]
exp_freqs = disc_hist_data([expos[reg][ind_r] for reg, ind_r in zip(regs, ind_rs)], classes=np.arange(9))[1]


#%% generate the pie chart
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.pie(exp_freqs[0][1:], labels=expos_dict.keys())
ax00.set_title("Nord-Troms exposures " + a_ps[a_p].lower())

pl.show()
pl.close()


#%% stacked bar plot
colors = ["black", "gray", "red", "blue", "brown", "orange", "violet", "lightblue"]

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax00.bar(x=x, height=exp_freqs[x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs[x][i]
    # end for i
# end for x

for i in np.arange(1, 9, 1):
    ax00.bar(x=np.zeros(8), height=np.zeros(8), color=colors[i-1], label=list(expos_dict.keys())[i-1])
# end for i
ax00.legend()

ax00.set_xticks(np.arange(len(regs_pl)))
ax00.set_xticklabels(regs_pl.values())
ax00.set_title("Exposures " + a_ps[a_p].lower())

pl.show()
pl.close()




