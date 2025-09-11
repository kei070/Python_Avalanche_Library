"""
Analyse the avalanche list.
"""


#%% imports
import pandas as pd
import pylab as pl
import numpy as np
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
f_name = "Avalanche_Problems_Reduced.csv"

df = pd.read_csv(f_path + f_name, header=0)  # , low_memory=False)

# df1 = np.array(pd.read_csv(f_path + f_name, nrows=1, header=None))
# df2 = np.array(pd.read_csv(f_path + f_name, nrows=2, header=None), dtype=str)


#%% convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"])


#%% extract one year
# df = df[df.date.dt.year == 2018]


#%% regions dictionary
regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}
regions_pl = {3009:"Nord-Troms", 3010:"Lyngen", 3011:"Tromsø", 3012:"Sør-Troms", 3013:"Indre Troms"}


#%% extract the different northern regions
pr_it = df[df.region == 3013]  # Indre Troms (it)
pr_st = df[df.region == 3012]  # Sør-Troms (st)
pr_tr = df[df.region == 3011]  # Tromsoe (tr)
pr_ly = df[df.region == 3010]  # Lyngen (ly)
pr_nt = df[df.region == 3009]  # Nord-Troms (nt)
# pr_fi = df[df.region == 3008]  # Finnmarksvidda (fi)


#%% set the column
param = "size"
# param = "sensitivity"
# param = "distribution"


#%% extract the index of a specific column
col_name = "wind_slab_" + param


#%% plot the "events"
fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 5))

axes.scatter(pr_it.date, pr_it[col_name], label="Wind Slab", marker="o", facecolor="none", edgecolor="gray",
             linewidth=0.5)

pl.show()
pl.close()


#%% generate a histogram
fig = pl.figure(figsize=(8, 5))

ax00 = fig.add_subplot(111)

ax00.hist(pr_it[col_name])

ax00.set_xlabel(param)
ax00.set_ylabel("Number of days")
ax00.set_title("Indre Troms")

pl.show()
pl.close()


#%% get the unique names of the avalanche problems from the original file
f_orig = "Avalanche_Problems_List.csv"
ava_ps = np.unique(np.array(pd.read_csv(f_path + f_orig, nrows=1, index_col=0, header=None).dropna(axis=1)))


#%% for each avalanche problem count for how many days it exists
avap_count = {}
reg_codes = [3009, 3010, 3011, 3012, 3013]
for reg_c in reg_codes:
    avap_count[reg_c] = {}
    for ava_p in ava_ps:
        # set the column name
        col_name = ava_p + "_size"  # arbitrarily take size as parameter; this should be irrelevant (but check!)

        # extract the region
        df_reg = df[df.region == reg_c]

        # count the number of days with the given avalanche problem
        avap_count[reg_c][ava_p] = len(df_reg[col_name].dropna(axis=0))

        # print(col_name + " " + str(reg_c))
        # print(len(df_reg[col_name].dropna(axis=0)))
        # print("")
    # end for ava_p
#  end for reg_c


#%% plot
ylim = (0, 750)

fig = pl.figure(figsize=(10, 10))

ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)

regc = 3009
ax00.bar(avap_count[regc].keys(), avap_count[regc].values())
ax00.set_title(regions_pl[regc])
ax00.set_ylabel("Number of Days")
ax00.set_ylim(ylim)
ax00.set_xticklabels([])

regc = 3010
ax01.bar(avap_count[regc].keys(), avap_count[regc].values())
ax01.set_title(regions_pl[regc])
# ax01.set_ylabel("Number of Days")
ax01.set_ylim(ylim)
ax01.set_xticklabels([])
ax01.set_yticklabels([])

regc = 3011
ax10.bar(avap_count[regc].keys(), avap_count[regc].values())
ax10.set_title(regions_pl[regc])
ax10.set_ylabel("Number of Days")
ax10.set_ylim(ylim)
ax10.set_xticklabels([])

regc = 3012
ax11.bar(avap_count[regc].keys(), avap_count[regc].values())
ax11.set_title(regions_pl[regc])
# ax11.set_ylabel("Number of Days")
ax11.set_ylim(ylim)
ax11.set_xticklabels(avap_count[regc].keys(), rotation=30)
ax11.set_yticklabels([])

regc = 3013
ax20.bar(avap_count[regc].keys(), avap_count[regc].values())
ax20.set_title(regions_pl[regc])
ax20.set_ylabel("Number of Days")
ax20.set_ylim(ylim)
ax20.set_xticklabels(avap_count[regc].keys(), rotation=30)

fig.subplots_adjust(wspace=0.05)

pl.show()
pl.close()




