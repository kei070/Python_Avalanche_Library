#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the script Calc_and_Store_NORA3_Predictors.py
"""

#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par


#%% set the paths to the scripts
py_path = f"/{py_path_par}/NORA3/"


#%% set height thresholds
agg_type = "mean"
perc = 0
a_p = "all"
ndlev = "2"


#%% send the scripts

# calculate the predictors only for those days with available risk assessment
for reg_code in [3009, 3010, 3011, 3012, 3013]:
    print(reg_code)
    for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):
        subprocess.call(["python", py_path + "Gen_Store_XLevel_Balanced_Predictors.py", "--reg_code", str(reg_code),
                         "--a_p", a_p, "--ndlev", ndlev,
                         "--h_low", str(h_low), "--h_hi", str(h_hi), "--agg_type", agg_type, "--perc", str(perc)])
    # end for h_low, h_hi
# end for reg_code