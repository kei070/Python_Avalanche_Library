#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the script Extract_EURO-CORDEX_GridCells.py
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


#%% batch execution
for mod in ["IPSL_RCA", "CNRM_RCA"]:
    for scen in ["hist", "rcp45", "rcp85"]:

        print(f"\n{scen}...\n")

        for h_lo, h_hi in zip(h_lo_l, h_hi_l):

            subprocess.call(["python", py_path + "Extract_EURO-CORDEX_GridCells.py", "--model", mod, "--scen",
                             scen, "--low", str(h_lo), "--high", str(h_hi)])

        # for h_lo, h_hi
    # end for scen
# end for mod