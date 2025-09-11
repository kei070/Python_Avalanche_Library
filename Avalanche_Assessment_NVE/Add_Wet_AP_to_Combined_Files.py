#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add the wet avalanche problem (combined slab and loose) to the combined file.
"""


#%% imports
import numpy as np
import pandas as pd
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
f_name = "Avalanche_Problems_ADL.csv"

df = pd.read_csv(f_path + f_name, header=0)  # , low_memory=False)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)


#%% load the wet file
wet_name = "wet_Danger_List.csv"

wet_df = pd.read_csv(f_path + wet_name, header=0)  # , low_memory=False)
wet_df["date"] = pd.to_datetime(wet_df["date"])
wet_df.set_index("date", inplace=True)

wet_dl = wet_df["danger_level"]
wet_dl.rename("wet", inplace=True)


#%% combine the wet DL with the combined file
df_dict = {k:df[k] for k in df.keys()}
df_dict["wet"] = wet_dl

wet_nan = df["wet_slab"].isna() & df["wet_loose"].isna()
df_dict["wet"][wet_nan] = np.nan

df_comb = pd.DataFrame(df_dict)


#%% store the date
df_comb.to_csv(f_path + "Avalanche_Problems_ADL_Extended.csv")



