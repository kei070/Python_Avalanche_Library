#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete binary predictor calculation.
"""

#%% imports
import subprocess

from Lists_and_Dictionaries.Paths import py_path_par


#%% set the paths to the scripts
py_path = f"/{py_path_par}/"


#%% set height thresholds
agg_type = "mean"
perc = 90
a_p = "all"


#%% send the scripts

# calculate the binary predictors for 5 different permutations
for ndlev in [2]:

    for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):

        for reg_code in [0, 3009, 3010, 3011, 3012, 3013]:
            subprocess.call(["python", py_path + "Gen_Store_XLevel_Balanced_Predictors.py", "--reg_code", str(reg_code),
                             "--h_low", str(h_low), "--h_hi", str(h_hi), "--ndlev", str(ndlev), "--agg_type", agg_type,
                             "--perc", str(perc), "--a_p", a_p, "--feats", "nora3_clean"])
        # end for reg_code
    # end for h_low, h_hi
# end for ndlev

