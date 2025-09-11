#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the script Calc_and_Store_NORA3_Predictors.py
"""

#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par


#%% set the paths to the scripts
py_path = f"/{py_path_par}/NorCP/"


#%% set height thresholds
models = ["GFDL-CM3"]
scens = {"EC-Earth":["historical", "rcp45", "rcp85"], "GFDL-CM3":["historical", "rcp85"], "ERAINT":["evaluation"]}
periods = {"historical":[""], "rcp45":["MC", "LC"], "rcp85":["MC", "LC"], "evaluation":[""]}
h_low = 0
h_hi = 300
agg_type = "mean"
perc = 0


#%% send the scripts
for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):
    print(f"\n{h_low}-{h_hi}m\n")

    for model in models:
        print(f"\nmodel: {model}\n")

        # if model == "GFDL-CM3":
        #    continue
        # end if

        for scen in scens[model]:
            print(f"\nscenario: {scen}\n")

            # if scen != "historical":
            #     continue
            # end if¨

            for period in periods[scen]:
                print(f"\nperiod: {period}\n")
                for reg_code in [3009, 3010, 3011, 3012, 3013]:
                    print(reg_code)
                    subprocess.call(["python", py_path + "Calc_and_Store_NorCP_Predictors.py",
                                     "--reg_code", str(reg_code), "--agg_type", agg_type, "--perc", str(perc),
                                     "--model", model, "--scen", scen, "--period", period,
                                     "--low", str(h_low), "--high", str(h_hi)])
                # end for reg_code
            # end for period
        # end for scen
    # end for model
# end for h_low, h_hi
