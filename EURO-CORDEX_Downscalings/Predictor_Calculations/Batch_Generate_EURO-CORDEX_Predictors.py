#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the script Generate_EURO-CORDEX_Predictors.py
"""

#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_par_path
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set the paths to the scripts
py_path = f"/{py_par_path}/EURO-CORDEX_Downscalings/Predictor_Calculations/"


#%% parameters to loop over
h_lo_l = [0, 300, 600, 900]
h_hi_l = [300, 600, 900, 1200]

mods = ["IPSL_RCA", "CNRM_RCA"]
scens = ["hist", "rcp45", "rcp85"]


#%% batch execution
for mod in mods:
    for scen in scens:
        for reg_code in regions.keys():

            print(f"\n{reg_code}...\n")

            for h_lo, h_hi in zip(h_lo_l, h_hi_l):

                if ((reg_code in [3011, 3012]) & (h_lo == 900)):
                    continue
                # end if

                subprocess.call(["python", py_path + "Generate_EURO-CORDEX_Predictors.py",
                                 str(reg_code), str(h_lo), str(h_hi), scen, mod])

            # end for h_lo, h_hi
        # end for reg_code
    # end for scen
# end for mod
