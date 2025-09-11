#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch execute the NorCP predictor aggregation across elevation bands.
"""

#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set the paths to the scripts
py_path = f"/{py_path_par}/NorCP/"


#%% set height thresholds
models = ["ERAINT", "GFDL-CM3", "EC-Earth"]
scens = {"EC-Earth":["historical", "rcp45", "rcp85"], "GFDL-CM3":["historical", "rcp85"], "ERAINT":["evaluation"]}
periods = {"evaluation":[""], "historical":[""], "rcp45":["_MC", "_LC"], "rcp85":["_MC", "_LC"]}


#%% execute
for model in models:
    print(f"\nmodel: {model}\n")

    for scen in scens[model]:
        print(f"\nscenario: {scen}\n")

        for period in periods[scen]:
            print(f"\nperiod: {period}\n")
            for reg_code in regions.keys():
                print(reg_code)
                subprocess.call(["python", py_path + "Aggregate_NorCP_Predictors_Across_ElevBands.py",
                                  model, scen, period, str(reg_code)])
            # end for reg_code
        # end for period
    # end for scen
# end for model
