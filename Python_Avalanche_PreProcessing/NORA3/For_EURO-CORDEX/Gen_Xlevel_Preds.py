#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the NORA3-based EURO-CORDEX predictors with a custom number of danger levels.
"""


#%% imports
import os
import sys
import numpy as np
import pandas as pd
import argparse

from ava_functions.Lists_and_Dictionaries.Paths import path_par, obs_path


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Predictors with Danger Levels",
                    description="""Combine danger levels and predictors.""")

# ...and add the arguments
parser.add_argument('--reg_code', default=3009, type=int, help='Code of the region.')
parser.add_argument('--ndlev', default=2, type=int, help='Custom danger level scale.')
parser.add_argument("--agg_type", default="Mean", type=str, choices=["mean", "median", "percentile", "perc"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=0, help="""The percentile used in the grid-cell aggregation.""")
args = parser.parse_args()


#%% get the parameters from the parser
reg_code = args.reg_code
ndlev = args.ndlev
agg_type = "percentile" if args.agg_type == "perc" else args.agg_type
perc = args.perc


#%% AP list
a_ps = ["danger_level", "glide_slab", "new_slab", "new_loose", "wind_slab", "pwl_slab", "wet_loose", "wet_slab", "wet"]


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
pred_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors/EURO-CORDEX/{agg_type}/"
pred_name = f"{region}_EU_Predictors_MultiCell{agg_type}_ElevAgg.csv"

df = pd.read_csv(pred_path + pred_name, index_col=0, parse_dates=True)
# df.date = pd.to_datetime(df.date)


#%% convert the danger levels to the requested number of levels; i.e. ndlev=2 means levels 1&2 = 0, 3&4&5 = 1
dl_converted = {}

if ndlev == 2:
    for a_p in a_ps:
        dl_converted[a_p] = np.zeros(len(df))
        dl_converted[a_p][df[a_p] > 2] = 1
    # end for a_p
elif ndlev == 4:
    for a_p in a_ps:
        dl_converted[a_p] = np.zeros(len(df))
        dl_converted[a_p][df[a_p] == 2] = 1
        dl_converted[a_p][df[a_p] == 3] = 2
        dl_converted[a_p][df[a_p] > 3] = 3
    # end for a_p
# if elif

df_converted = pd.DataFrame({a_p:dl_converted[a_p] for a_p in a_ps}, index=df.index)


#%% "replace" the original danger levels with the custom ones
df_out = pd.concat([df.drop(a_ps, axis=1), df_converted], axis=1)


#%% store
out_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors_{ndlev}Level/EURO-CORDEX/{agg_type}/"
out_name = f"{region}_EU_Predictors_{ndlev}Level_MultiCell{agg_type}_ElevAgg.csv"

os.makedirs(out_path, exist_ok=True)

df_out.to_csv(out_path + out_name)