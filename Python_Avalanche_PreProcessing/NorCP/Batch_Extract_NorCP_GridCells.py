#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch script for extraction of NorCP gridcells.
"""

#%% imports
import subprocess
import numpy as np

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par


#%%% set the paths to the scripts
py_path = f"{py_path_par}/NorCP/"


#%% execute the script

for model in ["GFDL-CM3"]:

    for scen in ["historical"]:

        for period in ["", "MC", "LC"]:
        # for period in ["LC"]:

            if ((scen in ["historical", "evaluation"]) & (period != "")):
                continue
            elif ((scen not in ["historical", "evaluation"]) & (period == "")):
                continue
            # end if elif

            for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):
                print(f"\nBetween {h_low} and {h_hi} m\n")
                subprocess.call(["python", py_path + "Extract_NorCP_GridCells_Between_NC.py", "--model", model,
                                 "--scen", scen, "--period", period, "--low", str(h_low), "--high", str(h_hi)])
            # end for h_low, h_hi
        # end for period
    # end for scen
# end for model