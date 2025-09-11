#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate the annual hourly mean nc files to annual files.
"""

#%% imports
import os
import sys
import glob
import argparse
import xarray as xr

from ava_functions.Concat_NORA3_Day import concat_nora3_ann
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set the year via the command line
parser = argparse.ArgumentParser(description="Concatenate the monthly files to an annual file.")

parser.add_argument("--yr", type=int, default=1985, help="The year over which the files will be concatenated.")
parser.add_argument("--delete", action="store_true",
                    help="If --delete is added the original monthly files will be deleted.")

args = parser.parse_args()


#%% run the function
data_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/"
out_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
out_name = f"NORA3_NorthNorway_Sub_{args.yr}.nc"

concat_nora3_ann(yr=args.yr, data_path=data_path, out_path=out_path, out_name=out_name, delete=args.delete)