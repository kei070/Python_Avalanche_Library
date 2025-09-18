#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the script Calc_and_Store_NORA3_Predictors.py
"""

#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par


#%% set the paths to the scripts
py_path = f"/{py_path_par}/Python_Avalanche_Library/Python_Avalanche_PreProcessing/NORA3/For_EURO-CORDEX/"


#%% set height thresholds
agg_type = "mean"
perc = 0


#%% send the scripts

# calculate the predictors only for those days with available risk assessment
for reg_code in [3009, 3010, 3011, 3012, 3013]:
    print(reg_code)
    for h_lo, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):

        if ((reg_code in [3011, 3012]) & (h_lo == 900)):
            continue
        # end if

        subprocess.call(["python", py_path + "Calc_and_Store_NORA3_Predictors_EU.py", "--reg_code", str(reg_code),
                         "--low", str(h_lo), "--high", str(h_hi), "--agg_type", agg_type, "--perc", str(perc)])
    # end for h_low, h_hi
# end for reg_code