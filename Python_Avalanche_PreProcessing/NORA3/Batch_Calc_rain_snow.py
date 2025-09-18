#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch calculate the rain and snow for NORA3.
"""


#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%%% set the paths to the scripts
py_path = f"{py_path_par}/Python_Avalanche_Library/Python_Avalanche_PreProcessing/NORA3/"


#%% execute the script
for reg_code in regions.keys():
    for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):
        print(f"\nBetween {h_low} and {h_hi} m\n")

        subprocess.call(["python", py_path + "Calc_snow_rain.py", "--reg_code", str(reg_code),
                         "--low", str(h_low), "--high", str(h_hi)])

    # end for h_low, h_hi
# end reg_code
