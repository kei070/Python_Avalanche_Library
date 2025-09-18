#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the concatination of extracted NORA3 gridcells.
"""

#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par


#%%% set the paths to the scripts
py_path = f"{py_path_par}/Python_Avalanche_Library/Python_Avalanche_PreProcessing/NORA3/"


#%% set start and end year
sta_yr = 1970
end_yr = 2024


#%% execute the script
for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):
    print(f"\nBetween {h_low} and {h_hi} m\n")

    subprocess.call(["python", py_path + "Concat_Extracted_NORA3_GridCell_Data_NC.py", "--sta_yr1", str(sta_yr),
                     "--sta_yr2", str(sta_yr), "--end_yr1", str(end_yr), "--end_yr2", str(end_yr),
                     "--low", str(h_low), "--high", str(h_hi)])

# end for h_low, h_hi
