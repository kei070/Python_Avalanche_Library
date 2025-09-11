#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the script Aggregate_EURO-CORDEX_Predictors_Across_ElevBands.py.
"""


#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_par_path
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set the paths to the scripts
py_path = f"/{py_par_path}/EURO-CORDEX_Downscalings/Predictor_Calculations/"


#%% set the scenarios
mods = ["IPSL_RCA", "CNRM_RCA"]
scens = ["hist", "rcp45", "rcp85"]


#%% batch execution
for mod in mods:
    for scen in scens:
        for reg_code in regions.keys():

            print(f"\n{reg_code}...\n")

            subprocess.call(["python", py_path + "Aggregate_EURO-CORDEX_Predictors_Across_ElevBands.py", mod,
                             scen, str(reg_code)])

        # for reg_code
    # end for scen
# end for mod