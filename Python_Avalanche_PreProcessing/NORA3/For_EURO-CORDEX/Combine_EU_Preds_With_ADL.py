#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine the NORA3-based EURO-CORDEX cordex predictors with the ADLs.
"""

#%% imports
import os
import sys
import pandas as pd
import argparse

from ava_functions.Lists_and_Dictionaries.Paths import path_par, obs_path


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Predictors with Danger Levels",
                    description="""Combine danger levels and predictors.""")

# ...and add the arguments
parser.add_argument('--reg_code', default=3009, type=int, help='Code of the region.')
parser.add_argument("--agg_type", default="Mean", type=str, choices=["mean", "median", "percentile", "perc"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=0, help="""The percentile used in the grid-cell aggregation.""")
args = parser.parse_args()


#%% get the parameters from the parser
reg_code = args.reg_code
agg_type = "percentile" if args.agg_type == "perc" else args.agg_type
perc = args.perc


#%% AP list
a_ps = ["wind_slab", "pwl_slab", "wet"]


#%% get the region
regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}
region = regions[reg_code]


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if
agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% set data path and read the avalanche predictor files
pred_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/EURO-CORDEX/{agg_type}/"
pred_name = f"{region}_EU_ElevAgg_Predictors_MultiCell{agg_type}.csv"

df = pd.read_csv(pred_path + pred_name)
df.date = pd.to_datetime(df.date)


#%% get all the danger-level "observations"
dl_path = f"/{obs_path}/IMPETUS/Avalanches_Danger_Files/"
dl_name = "Avalanche_Danger_List.csv"

dl_df = pd.read_csv(dl_path + dl_name)
dl_df.date = pd.to_datetime(dl_df.date)


#%% extract the region according to the region code
dl_df_reg = dl_df[dl_df.region == reg_code]


#%% perform an inner join with the predictors and the danger-level "observations" to merge the files
df_obs_data = df.merge(dl_df_reg, how="inner", on="date")
# df_obs_data.set_index("date", inplace=True)


#%% load the danger levels per avalanche problem
# ap_name = "Avalanche_Problems_ADL.csv"
ap_name = "Avalanche_Problems_ADL_Extended.csv"  # including the "wet" problem (loose & slab combined)

ap_df = pd.read_csv(dl_path + ap_name)
ap_df.date = pd.to_datetime(ap_df.date)


#%% extract the region according to the region code
ap_df_reg = ap_df[ap_df.region == reg_code].copy()
ap_df_reg.drop(columns="region", inplace=True)


#%% perform an left join with the predictors & general ADL on the left and the AP ADLs on the right
df_obs_data = df_obs_data.merge(ap_df_reg, how="left", on="date")
df_obs_data.set_index("date", inplace=True)

df_obs_data.drop("ADL", axis=1, inplace=True)


#%% store the predictors
out_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors/EURO-CORDEX/{agg_str}/"

os.makedirs(out_path, exist_ok=True)

df_obs_data.to_csv(out_path + region + f"_EU_Predictors_MultiCell{agg_str}_ElevAgg.csv")
