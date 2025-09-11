#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the predictors based on NORA3 for the EURO-CORDEX variables.
"""


#%% imports
import os
import sys
import numpy as np
import pandas as pd
import argparse

import ava_functions.Calc_Preds as cprf
from ava_functions.Lists_and_Dictionaries.Paths import path_par, path_par3
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Lists_and_Dictionaries.Avalanche_Climate import ava_clim_code
from ava_functions.Lists_and_Dictionaries.Variable_Name_NORA3_NorCP import varn_nora3


#%% parser
parser = argparse.ArgumentParser(
                    description="""Calculate the predictors for NORA3 reanalysis.""")
parser.add_argument('--reg_code', default=3010, type=int, help='Code of the region.')
parser.add_argument("--agg_type", default="mean", type=str, choices=["mean", "median", "percentile", "perc"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=0, help="""The percentile used in the grid-cell aggregation.""")
parser.add_argument('--low', default=0, type=int, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', default=300, type=int, help='The upper threshold for the altitude band to extract.')

args = parser.parse_args()


#%% get the arguments from the parser
reg_code = args.reg_code
agg_type = "percentile" if args.agg_type == "perc" else args.agg_type
perc = args.perc
h_low = args.low  # 400  # args.low
h_hi = args.high  # 900  # args.high


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% set the data path
data_path = f"/{path_par3}/IMPETUS/NORA3/Avalanche_Region_Data/Between{h_low}_and_{h_hi}m/"


#%% execute the functions
unclean = False

# temperature predictors
temp_pred = cprf.calc_temp_pred(data_path=data_path, reg_code=reg_code, h_low=h_low, h_hi=h_hi,
                                agg_type=agg_type, perc=perc, varn_dic=varn_nora3, unclean=unclean)

# precipitation predictors
prec_pred = cprf.calc_prec_pred_n3_eu(data_path=data_path, reg_code=reg_code, h_low=h_low, h_hi=h_hi,
                                      agg_type=agg_type, perc=perc, varn_dic=varn_nora3, unclean=unclean)


#%% concatenate the predictors
print("\nConcatenating and storing predictors...\n")
df = pd.concat([temp_pred.drop("date", axis=1), prec_pred.drop("date", axis=1)], axis=1)


#%% generate a column with the "avalanche climate index" taken from the Avalanche_Climate script
df["ava_clim"] = np.repeat(a=ava_clim_code[reg_code], repeats=len(df))


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% store the data as csv
region = regions[reg_code]
out_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/EURO-CORDEX/{agg_str}/" + \
                                                                                          f"Between{h_low}_and_{h_hi}m/"
os.makedirs(out_path, exist_ok=True)
df.to_csv(out_path + f"{region}_EU_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}m.csv")


#%% make NaN test
# import pylab as pl
# pl.plot(df_obs_data.isna().sum())