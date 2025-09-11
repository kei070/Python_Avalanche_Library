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


#%% set the column
col_names1 = ["new_loose_", "wet_loose_", "new_slab_", "wind_slab_", "pwl_slab_", "wet_slab_", "glide_slab_"]
col_names2 = ["size", "distribution", "priority", "sensitivity", "elevation_min", "elevation_max",
              "exposition_N", "exposition_NE", "exposition_E", "exposition_SE", "exposition_S", "exposition_SW",
              "exposition_W", "exposition_NW"]

df_new = {"date":df.date, "region":df["region"]}
for n1 in col_names1:
    for n2 in col_names2:
        col_name = n1 + n2
        df_new[col_name] = df.iloc[:, np.argmax(col_names == col_name)]
    # end for n2
# end for n1


#%% convert the dictionary to a dataframe
pd.DataFrame(df_new).to_csv(f_path + "Avalanche_Problems_Reduced.csv")

