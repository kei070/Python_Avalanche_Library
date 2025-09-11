#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch execute the Aggregate_SNOWPACK_Stab_Inds_Across_ElevBands.py script.
"""

#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% set paths and name
py_path = f"{obs_path}/Python_Avalanche_Library/Snowpack/NorCP/"
py_script = "Aggregate_SNOWPACK_Stab_Inds_Across_ElevBands.py"


#%% set parameters
reg_codes = [3009, 3010, 3011, 3012, 3013]
models = ["EC-Earth"]  # ["ERAINT", "GFDL-CM3", "EC-Earth"]
scens = {"EC-Earth":["historical", "rcp45", "rcp85"], "GFDL-CM3":["historical", "rcp85"], "ERAINT":["evaluation"]}
periods = {"evaluation":[""], "historical":[""], "rcp45":["_MC", "_LC"], "rcp85":["_MC", "_LC"]}



#%% loop over the parameters
for reg_code in reg_codes:
    for model in models:
        for scen in scens[model]:
            for period in periods[scen]:

                print(f"{reg_code}  {model}  {scen}  {period}")

                subprocess.call(["python", py_path + py_script, str(reg_code), model, scen, period])

            # end for period
        # end for scen
    # end for model
# end for reg_code