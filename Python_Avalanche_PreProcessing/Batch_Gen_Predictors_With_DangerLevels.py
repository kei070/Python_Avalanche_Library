#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the script Calc_and_Store_NORA3_Predictors.py
"""

#%% imports
import subprocess

from Lists_and_Dictionaries.Paths import py_path_par


#%% set the paths to the scripts
py_path = f"/{py_path_par}/"


#%% set height thresholds
h_low = 600
h_hi = 1100
agg_type = "mean"
perc = 90


#%% send the scripts

# calculate the predictors only for those days with available risk assessment
for reg_code in [3009, 3010, 3011, 3012, 3013]:
    print(reg_code)
    subprocess.call(["python", py_path + "Gen_Preds_With_DangerLevels.py",
                     str(reg_code), str(h_low), str(h_hi), agg_type, str(perc)])
# end for reg_code