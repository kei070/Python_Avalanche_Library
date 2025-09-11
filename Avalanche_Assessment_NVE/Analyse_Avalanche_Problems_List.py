"""
Analyse the avalanche risk list.
"""


#%% imports
import pandas as pd
import pylab as pl
import numpy as np
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
f_name = "Avalanche_Problems_List.csv"

df = pd.read_csv(f_path + f_name, header=2)  # , low_memory=False)

df1 = np.array(pd.read_csv(f_path + f_name, nrows=1, header=None))
df2 = np.array(pd.read_csv(f_path + f_name, nrows=2, header=None), dtype=str)


#%% concatenate the column names
col_names = []
for a, b in zip(df2[0, :], df2[1, :]): col_names.append(a + "_" + b)
col_names = np.array(col_names)


#%% convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"])


#%% extract one year
# df = df[df.date.dt.year == 2018]


#%% extract the different northern regions
pr_it = df[df.region == 3013]  # Indre Troms (it)
pr_st = df[df.region == 3012]  # Sør-Troms (st)
pr_tr = df[df.region == 3011]  # Tromsoe (tr)
pr_ly = df[df.region == 3010]  # Lyngen (ly)
pr_nt = df[df.region == 3009]  # Nord-Troms (nt)
# pr_fi = df[df.region == 3008]  # Finnmarksvidda (fi)


#%% set the column
param = "size"


#%% extract the index of a specific column
col_name1 = "new_loose_" + param
col_i1 = np.argmax(col_names == col_name1)

col_name2 = "wet_loose_" + param
col_i2 = np.argmax(col_names == col_name2)

col_name3 = "new_slab_" + param
col_i3 = np.argmax(col_names == col_name3)

col_name4 = "wind_slab_" + param
col_i4 = np.argmax(col_names == col_name4)

col_name5 = "pwl_slab_" + param
col_i5 = np.argmax(col_names == col_name5)

col_name6 = "wet_slab_" + param
col_i6 = np.argmax(col_names == col_name6)

col_name7 = "glide_slab_" + param
col_i7 = np.argmax(col_names == col_name7)


#%% plot the "events"
fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 5))

axes.scatter(pr_it.date, pr_it.iloc[:, col_i1], label="New Loose", marker="o", facecolor="none", edgecolor="blue",
             linewidth=0.5)
axes.scatter(pr_it.date, pr_it.iloc[:, col_i2], label="Wet Loose", marker="o", facecolor="none", edgecolor="red",
             linewidth=0.5)
axes.scatter(pr_it.date, pr_it.iloc[:, col_i3], label="New Slab", marker="o", facecolor="none", edgecolor="orange",
             linewidth=0.5)
axes.scatter(pr_it.date, pr_it.iloc[:, col_i4], label="Wind Slab", marker="o", facecolor="none", edgecolor="gray",
             linewidth=0.5)
axes.scatter(pr_it.date, pr_it.iloc[:, col_i5], label="Pwl Slab", marker="o", facecolor="none", edgecolor="black",
             linewidth=0.5)
axes.scatter(pr_it.date, pr_it.iloc[:, col_i6], label="Wet Slab", marker="o", facecolor="none", edgecolor="green",
             linewidth=0.5)
axes.scatter(pr_it.date, pr_it.iloc[:, col_i7], label="Glide Slab", marker="o", facecolor="none", edgecolor="violet",
             linewidth=0.5)

pl.show()
pl.close()

