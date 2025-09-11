"""
Analyse the avalanche risk list.
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


#%% set up a dictionary for the avalanche problem names
ava_p_pl = {"glide_slab":"Glide slab", "new_loose":"New loose", "new_slab":"New slab", "pwl_slab":"PWL slab",
            "wet_loose":"Wet loose", "wet_slab":"Wet slab", "wind_slab":"Wind slab", "Avalanche":"General"}


#%% set the avalanche problem
#   choices are glide_slab, new_loose, new_slab, pwl_slab, wet_loose, wet_slab, wind_slab
#   for the general danger level set to "Avalanche"
ava_p = "Avalanche"

path_ehd = path_par


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
# f_name = "Avalanche_Danger_List.csv"
f_name = ava_p + "_Danger_List.csv"

ar_df = pd.read_csv(f_path + f_name)


#%% convert the date column to datetime format
ar_df["date"] = pd.to_datetime(ar_df["date"])
ar_df.set_index("date", inplace=True, drop=False)


#%% extract the different northern regions
ar_it = ar_df.loc[ar_df.region == 3013, :]  # Indre Troms (it)
ar_st = ar_df.loc[ar_df.region == 3012, :]  # Sør-Troms (st)
ar_tr = ar_df.loc[ar_df.region == 3011, :]  # Tromsoe (tr)
ar_ly = ar_df.loc[ar_df.region == 3010, :]  # Lyngen (ly)
ar_nt = ar_df.loc[ar_df.region == 3009, :]  # Nord-Troms (nt)
ar_fi = ar_df.loc[ar_df.region == 3008, :]  # Finnmarksvidda (fi)


#%% assign a winter year
ar_it["win_year"] = assign_winter_year(ar_it.index, 8)
ar_st["win_year"] = assign_winter_year(ar_st.index, 8)
ar_tr["win_year"] = assign_winter_year(ar_tr.index, 8)
ar_ly["win_year"] = assign_winter_year(ar_ly.index, 8)
ar_nt["win_year"] = assign_winter_year(ar_nt.index, 8)


#%% concatenate the regions of interest
ar_nnor = pd.concat([ar_it, ar_st, ar_tr, ar_ly, ar_nt])


#%% extract the avalanche risk >= 4
ar_it_l4 = ar_it[ar_it.danger_level >= 4]
ar_st_l4 = ar_st[ar_st.danger_level >= 4]
ar_ly_l4 = ar_ly[ar_ly.danger_level >= 4]
ar_nt_l4 = ar_nt[ar_nt.danger_level >= 4]
ar_fi_l4 = ar_fi[ar_fi.danger_level >= 4]


#%% plot the "events"
pl_path = f"{obs_path}/IMPETUS/Plots/"

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 3))

axes.scatter(ar_it_l4.date, ar_it_l4.danger_level, marker="o", linewidth=0.5, facecolor="none", edgecolor="orange")
axes.scatter(ar_st_l4.date, ar_st_l4.danger_level+0.1, marker="o", linewidth=0.5, facecolor="none", edgecolor="skyblue")
axes.scatter(ar_ly_l4.date, ar_ly_l4.danger_level+0.2, marker="o", linewidth=0.5, facecolor="none", edgecolor="red")
axes.scatter(ar_nt_l4.date, ar_nt_l4.danger_level+0.3, marker="o", linewidth=0.5, facecolor="none", edgecolor="tan")
axes.scatter(ar_fi_l4.date, ar_fi_l4.danger_level+0.4, marker="o", linewidth=0.5, facecolor="none", edgecolor="blue")

axes.set_yticks([4, 4.1, 4.2, 4.3, 4.4])
axes.set_yticklabels([f"Indre Troms (#{len(ar_it_l4)})", f"Sør-Troms (#{len(ar_st_l4)})", f"Lyngen (#{len(ar_ly_l4)})",
                      f"Nord-Troms (#{len(ar_nt_l4)})", f"Finnmarksvidda (#{len(ar_fi_l4)})"])

axes.set_xlabel("Date")

axes.set_title("Avalanche Risk $\geq 4$ $-$ Troms and Finnmark\nWinter 2017/18-2022/23")

axes.set_ylim((3.95, 4.45))

# pl.savefig(pl_path + "Northern_Regions_Avalanche_Risk_2017_2023.png", bbox_inches="tight", dpi=300)

pl.show()
pl.close()


#%% Is there even a single level-5 day?
ar_df[ar_df.danger_level == 5]

# --> Yes, there is exactly one: 2018-04-21 in region 3018 = Helgeland


#%% look specifically into the data for individual avalanche seasons:
#     - When does the typical avalanche season start and end?
#     - Are there ever any days without a risk assessment between the first and last in the avalanche season?

# extract first and last day of avalanche season for ever season and year
years = np.unique(ar_it.date.dt.year)

# generate a dictionary to store start and end date per region per season
season_dic = {}

# extract the first and last day in the following for-loop
for reg_code in [3009, 3010, 3011, 3012, 3013]:
    season_dic[reg_code] = {"year":[], "start":[], "end":[]}

    ar = ar_df[ar_df.region == reg_code]

    for yr in years[:-1]:

        season_dic[reg_code]["year"].append(yr)

        # get the indices for fall of current year and spring of next year
        i_fall = ar.date[ar.date.dt.year == yr][ar.date[ar.date.dt.year == yr] > date_dt(yr, 9, 1)]
        i_spring = ar.date[ar.date.dt.year == yr+1][ar.date[ar.date.dt.year == yr+1] < date_dt(yr+1, 9, 1)]

        # extract the first value of the fall and the last of the spring
        s_start = np.array(i_fall)[0]
        s_end = np.array(i_spring)[-1]

        season_dic[reg_code]["start"].append(s_start)
        season_dic[reg_code]["end"].append(s_end)

    # end for yr
# end for reg_code


#%% plot the start and end dates per season per year
sta_m = "+"
end_m = "x"
col = "black"

nyrs = 9

fig = pl.figure()

ax00 = fig.add_subplot()

ax00.plot(season_dic[3009]["start"], np.zeros(nyrs), linewidth=0, marker=sta_m, c=col)
ax00.plot(season_dic[3009]["end"], np.zeros(nyrs), linewidth=0, marker=end_m, c=col)

ax00.plot(season_dic[3010]["start"], np.zeros(nyrs)+1, linewidth=0, marker=sta_m, c=col)
ax00.plot(season_dic[3010]["end"], np.zeros(nyrs)+1, linewidth=0, marker=end_m, c=col)

ax00.plot(season_dic[3011]["start"], np.zeros(nyrs)+2, linewidth=0, marker=sta_m, c=col)
ax00.plot(season_dic[3011]["end"], np.zeros(nyrs)+2, linewidth=0, marker=end_m, c=col)

ax00.plot(season_dic[3012]["start"], np.zeros(nyrs)+3, linewidth=0, marker=sta_m, c=col)
ax00.plot(season_dic[3012]["end"], np.zeros(nyrs)+3, linewidth=0, marker=end_m, c=col)

ax00.plot(season_dic[3013]["start"], np.zeros(nyrs)+4, linewidth=0, marker=sta_m, c=col)
ax00.plot(season_dic[3013]["end"], np.zeros(nyrs)+4, linewidth=0, marker=end_m, c=col)

for yr in season_dic[3009]["year"]:
    ax00.axvline(x=date_dt(yr+1, 6, 30), c="blue")
    ax00.axvline(x=date_dt(yr, 11, 1), c="red")
# end for yr

ax00.set_yticks(np.arange(len(season_dic)))
ax00.set_yticklabels(["Nord-Troms", "Lyngen", "Troms${\o}$", "S${\o}$r-Troms", "Indre Troms"])

ax00.set_title("Begin and end of avalanche warning")

pl.show()
pl.close()


#%% remove the year so as to just plot the date
start_md = {}
end_md = {}
for reg in np.arange(3009, 3013+1):
    start_md[reg] = []
    for start_ymd in season_dic[reg]["start"]:
        start_md[reg].append(pd.Timestamp(start_ymd).strftime("%m-%d"))
    # end for start_ymd
    end_md[reg] = []
    for end_ymd in season_dic[reg]["end"]:
        end_md[reg].append(pd.Timestamp(end_ymd).strftime("%m-%d"))
    # end for end_ymd
# end for reg


print(start_md)
print(end_md)


#%% get the numbers of danger levels per region
levels = [1, 2, 3, 4, 5]
ndlev_it = {dlev:np.sum(ar_it.danger_level == dlev) for dlev in levels}
ndlev_ly = {dlev:np.sum(ar_ly.danger_level == dlev) for dlev in levels}
ndlev_nt = {dlev:np.sum(ar_nt.danger_level == dlev) for dlev in levels}
ndlev_st = {dlev:np.sum(ar_st.danger_level == dlev) for dlev in levels}
ndlev_tr = {dlev:np.sum(ar_tr.danger_level == dlev) for dlev in levels}

ndlev_tot = {dlev:ndlev_it[dlev]+ndlev_ly[dlev]+ndlev_nt[dlev]+ndlev_st[dlev]+ndlev_tr[dlev] for dlev in levels}


#%% plot the avalanche danger per region and per "total" region
f_col = "none"
marker = "_"

fig = pl.figure(figsize=(6, 3))
ax00 = fig.add_subplot(111)

ax00.scatter(levels, ndlev_it.values(), label="Indre Troms", marker=marker, color="black")  #, facecolor=f_col)
ax00.scatter(levels, ndlev_tr.values(), label="Troms${\o}$", marker=marker, color="blue")  #, facecolor=f_col)
ax00.scatter(levels, ndlev_st.values(), label="S${\o}$r-Troms", marker=marker, color="red")  #, facecolor=f_col)
ax00.scatter(levels, ndlev_nt.values(), label="Nord-Troms", marker=marker, color="orange")  #, facecolor=f_col)
ax00.scatter(levels, ndlev_ly.values(), label="Lyngen", marker=marker, color="violet")  #, facecolor=f_col)
ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylim((-85, 650))

y_text = -70
for x_text, s_text in ndlev_tot.items():
    ax00.text(x_text, y_text, s_text, horizontalalignment="center")
# end for
ax00.text(0.45, y_text, "total")

ax00.set_xticks(levels)

ax00.legend()

ax00.set_xlabel("Danger level")
ax00.set_ylabel("Number of days")

pl_path = "/home/kei070/Documents/IMPETUS/Plots/"
# pl.savefig(pl_path + "DangerLevelSummary.png", bbox_inches="tight", dpi=100)
# pl.savefig(pl_path + "DangerLevelSummary.pdf", bbox_inches="tight", dpi=100)

pl.show()
pl.close()


#%% avalanche danger per year
disc_hist([ar_nnor[ar_nnor.win_year == yr]["danger_level"] for yr in np.unique(ar_nnor.win_year)], classes=[1, 2, 3, 4],
          width=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
          color=["black", "gray", "red", "blue", "orange", "violet", "brown", "green", "lime"],
          labels=np.unique(ar_nnor.win_year), ylabel="# days", xlabel="Danger level", linewidth=3)


#%% prepare the data for the per-year plot
data_uni_list, data_hist_list = disc_hist_data([ar_nnor[ar_nnor.win_year == yr]["danger_level"] for yr in
                                                np.unique(ar_nnor.win_year)], classes=None)


#%% combine the per-region and per-year plots --> Fig. 2 in the manuscript
y_lim = (-85, 850)
x_lim = (0.7, 5.3)
marker = "_"
markers = ["_", "_", "_", "_", "x", "_", "x", "_", "_"]
lwd = 3
lwds = [3, 3, 3, 3, 1, 3, 1, 3, 3]
win_year = np.unique(ar_nnor.win_year)
colors = ["black", "grey", "blue", "red", "orange", "violet", "grey", "green", "lime"]

fig = pl.figure(figsize=(8, 3))

ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

ax00.scatter(levels, ndlev_it.values(), label="Indre Troms", marker=marker, color="black", linewidth=lwd)
ax00.scatter(levels, ndlev_tr.values(), label="Tromsø", marker=marker, color="blue", linewidth=lwd)
ax00.scatter(levels, ndlev_st.values(), label=r"Sør-Troms", marker=marker, color="red", linewidth=lwd)
ax00.scatter(levels, ndlev_nt.values(), label="Nord-Troms", marker=marker, color="orange", linewidth=lwd)
ax00.scatter(levels, ndlev_ly.values(), label="Lyngen", marker=marker, color="violet", linewidth=lwd)
ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylim(y_lim)
ax00.set_xlim(x_lim)

y_text = -70
for x_text, s_text in ndlev_tot.items():
    ax00.text(x_text, y_text, s_text, horizontalalignment="center")
# end for
ax00.text(0.21, y_text, "total")

ax00.set_xticks(levels)

ax00.legend()

ax00.set_xlabel("ADL")
ax00.set_ylabel("Number of occurrences")
ax00.set_title("(a) ADL per region")


i = 0
for data_uni, data_hist in zip(data_uni_list, data_hist_list):
    ax01.scatter(np.array(data_uni), np.array(data_hist), marker=markers[i], linewidth=lwds[i], label=win_year[i],
                 c=colors[i])
    i += 1
# end for

ax01.scatter(5, 0, linewidth=lwd, marker=marker, c="black")

ax01.legend()
ax01.axhline(y=0, c="black", linewidth=0.5)

ax01.set_xticks(levels)
ax01.set_yticks([])
ax01.set_ylim(y_lim)
ax01.set_xlim(x_lim)

ax01.set_xlabel("ADL")
ax01.set_title("(b) ADL per year")

fig.suptitle(ava_p_pl[ava_p])
fig.subplots_adjust(wspace=0.05)

pl_path = f"{path_ehd}/IMPETUS/Avalanche_Danger_Data/Plots/"
pl.savefig(pl_path + f"{ava_p}_Danger_Summary_NorthernNorway_Suppl.pdf", bbox_inches="tight", dpi=150)
pl.savefig(pl_path + f"{ava_p}_Danger_Summary_NorthernNorway_Suppl.png", bbox_inches="tight", dpi=150)

pl.show()
pl.close()


#%% prepare the data for the train-test split bar plot
split = [2021, 2023]
data_te_list, data_te_hist_list = disc_hist_data([ar_nnor[ar_nnor.win_year.isin(split)]["danger_level"]],
                                                 classes=None)
split = [2017, 2018, 2019, 2020, 2022, 2024, 2025]
data_tr_list, data_tr_hist_list = disc_hist_data([ar_nnor[ar_nnor.win_year.isin(split)]["danger_level"]],
                                                 classes=None)


#%% aggregate the data into the training an test sets and generate a bar plot
y_lim = (-85, 850)
x_lim = (0.7, 5.3)
marker = "_"
lwd = 3
win_year = np.unique(ar_nnor.win_year)
colors = ["black", "grey", "blue", "red", "orange", "violet", "grey", "green", "lime"]

fig = pl.figure(figsize=(9, 3))

ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

ax00.scatter(levels, ndlev_it.values(), label="Indre Troms", marker=marker, color="black", linewidth=lwd)
ax00.scatter(levels, ndlev_tr.values(), label="Tromsø", marker=marker, color="blue", linewidth=lwd)
ax00.scatter(levels, ndlev_st.values(), label=r"Sør-Troms", marker=marker, color="red", linewidth=lwd)
ax00.scatter(levels, ndlev_nt.values(), label="Nord-Troms", marker=marker, color="orange", linewidth=lwd)
ax00.scatter(levels, ndlev_ly.values(), label="Lyngen", marker=marker, color="violet", linewidth=lwd)
ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylim(y_lim)
ax00.set_xlim(x_lim)

y_text = -70
for x_text, s_text in ndlev_tot.items():
    ax00.text(x_text, y_text, s_text, horizontalalignment="center")
# end for
ax00.text(0.21, y_text, "total")

ax00.set_xticks(levels)

ax00.legend()

ax00.set_xlabel("ADL")
ax00.set_ylabel("Number of occurrences")
ax00.set_title("(a) ADL per region")

for data_tr, data_hist in zip(data_tr_list, data_tr_hist_list):
    # ax01.scatter(np.array(data_tr), np.array(data_hist)/np.sum(np.array(data_hist)), marker=marker, linewidth=lwd,
    #              label="training", c="black")
    ax01.bar(np.array(data_tr), np.array(data_hist)/np.sum(np.array(data_hist)), width=0.3,
             label="training", facecolor="none", edgecolor="black")
# end for

for data_te, data_hist in zip(data_te_list, data_te_hist_list):
    # ax01.scatter(np.array(data_te), np.array(data_hist)/np.sum(np.array(data_hist)), marker=marker, linewidth=lwd,
    #              label="test", c="red")
    ax01.bar(np.array(data_te), np.array(data_hist)/np.sum(np.array(data_hist)), width=0.4,
             label="test", facecolor="none", edgecolor="red")
# end for

ax01.scatter(5, 0, linewidth=0.5, marker=marker, c="black")

ax01.legend()
ax01.axhline(y=0, c="black", linewidth=0.5)

ax01.set_xticks(levels)
# ax01.set_yticks([])
ax01.set_ylim((-0.08, 0.575))
ax01.set_xlim(x_lim)

y_text = -0.0375
for x_text, s_text in zip(data_te_list[0], data_te_hist_list[0]):
    ax01.text(x_text, y_text, s_text, horizontalalignment="center", color="red")
# end for
ax01.text(5, y_text, 0, horizontalalignment="center", color="red")
y_text = -0.075
for x_text, s_text in zip(data_tr_list[0], data_tr_hist_list[0]):
    ax01.text(x_text, y_text, s_text, horizontalalignment="center", color="black")
# end for
ax01.text(5, y_text, 0, horizontalalignment="center", color="black")

ax01.set_xlabel("ADL")
ax01.set_title("(b) ADL for training and test data")
ax01.set_ylabel("Fraction")

# fig.suptitle(ava_p_pl[ava_p])
fig.subplots_adjust(wspace=0.2)

pl_path = f"{path_ehd}/IMPETUS/Avalanche_Danger_Data/Plots/"
pl.savefig(pl_path + f"{ava_p}_Danger_Summary_NorthernNorway.pdf", bbox_inches="tight", dpi=150)
pl.savefig(pl_path + f"{ava_p}_Danger_Summary_NorthernNorway.png", bbox_inches="tight", dpi=150)


pl.show()
pl.close()


#%% aggregate the data into the training an test sets and generate a bar plot
y_lim = (-85, 850)
x_lim = (0.7, 5.3)
marker = "_"
lwd = 3
win_year = np.unique(ar_nnor.win_year)
colors = ["black", "grey", "blue", "red", "orange", "violet", "grey", "green", "lime"]

fig = pl.figure(figsize=(13, 3))

ax00 = fig.add_subplot(131)
ax01 = fig.add_subplot(132)
ax02 = fig.add_subplot(133)

ax00.scatter(levels, ndlev_it.values(), label="Indre Troms", marker=marker, color="black", linewidth=lwd)
ax00.scatter(levels, ndlev_tr.values(), label="Tromsø", marker=marker, color="blue", linewidth=lwd)
ax00.scatter(levels, ndlev_st.values(), label=r"Sør-Troms", marker=marker, color="red", linewidth=lwd)
ax00.scatter(levels, ndlev_nt.values(), label="Nord-Troms", marker=marker, color="orange", linewidth=lwd)
ax00.scatter(levels, ndlev_ly.values(), label="Lyngen", marker=marker, color="violet", linewidth=lwd)
ax00.axhline(y=0, c="black", linewidth=0.5)

ax00.set_ylim(y_lim)
ax00.set_xlim(x_lim)

y_text = -70
for x_text, s_text in ndlev_tot.items():
    ax00.text(x_text, y_text, s_text, horizontalalignment="center")
# end for
ax00.text(0.21, y_text, "total")

ax00.set_xticks(levels)

ax00.legend()

ax00.set_xlabel("ADL")
ax00.set_ylabel("Number of occurrences")
ax00.set_title("(a) ADL per region")



i = 0
for data_uni, data_hist in zip(data_uni_list, data_hist_list):
    ax01.scatter(np.array(data_uni), np.array(data_hist), marker=markers[i], linewidth=lwds[i], label=win_year[i],
                 c=colors[i])
    i += 1
# end for

ax01.scatter(5, 0, linewidth=lwd, marker=marker, c="black")

ax01.legend()
ax01.axhline(y=0, c="black", linewidth=0.5)

ax01.set_xticks(levels)
ax01.set_yticklabels([])
ax01.set_ylim(y_lim)
ax01.set_xlim(x_lim)

ax01.set_xlabel("ADL")
ax01.set_title("(b) ADL per year")



for data_tr, data_hist in zip(data_tr_list, data_tr_hist_list):
    # ax01.scatter(np.array(data_tr), np.array(data_hist)/np.sum(np.array(data_hist)), marker=marker, linewidth=lwd,
    #              label="training", c="black")
    ax02.bar(np.array(data_tr), np.array(data_hist)/np.sum(np.array(data_hist)), width=0.3,
             label="training", facecolor="none", edgecolor="black")
# end for

for data_te, data_hist in zip(data_te_list, data_te_hist_list):
    # ax01.scatter(np.array(data_te), np.array(data_hist)/np.sum(np.array(data_hist)), marker=marker, linewidth=lwd,
    #              label="test", c="red")
    ax02.bar(np.array(data_te), np.array(data_hist)/np.sum(np.array(data_hist)), width=0.4,
             label="test", facecolor="none", edgecolor="red")
# end for

ax02.scatter(5, 0, linewidth=0.5, marker=marker, c="black")

ax02.legend()
ax02.axhline(y=0, c="black", linewidth=0.5)

ax02.set_xticks(levels)
# ax02.set_yticks([])
ax02.set_ylim((-0.08, 0.575))
ax02.set_xlim(x_lim)

y_text = -0.0375
for x_text, s_text in zip(data_te_list[0], data_te_hist_list[0]):
    ax02.text(x_text, y_text, s_text, horizontalalignment="center", color="red")
# end for
ax02.text(5, y_text, 0, horizontalalignment="center", color="red")
y_text = -0.075
for x_text, s_text in zip(data_tr_list[0], data_tr_hist_list[0]):
    ax02.text(x_text, y_text, s_text, horizontalalignment="center", color="black")
# end for
ax02.text(5, y_text, 0, horizontalalignment="center", color="black")

ax02.set_xlabel("ADL")
ax02.set_title("(c) ADL for training and test data")
ax02.set_ylabel("Fraction")

# move ticks and label to the right
ax02.yaxis.set_ticks_position('right')
ax02.yaxis.set_label_position('right')

# fig.suptitle(ava_p_pl[ava_p])
fig.subplots_adjust(wspace=0.05)

pl_path = f"{path_ehd}/IMPETUS/Avalanche_Danger_Data/Plots/"
pl.savefig(pl_path + f"{ava_p}_Danger_Summary_NorthernNorway.pdf", bbox_inches="tight", dpi=150)
pl.savefig(pl_path + f"{ava_p}_Danger_Summary_NorthernNorway.png", bbox_inches="tight", dpi=150)


pl.show()
pl.close()




