#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-execute the preparation of NorCP data for SNOWPACK.
"""


#%% imports
import subprocess
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% set paths and name
py_path = f"{obs_path}/Python_Avalanche_Library/Snowpack/NorCP/"
py_script = "Prep_NorCP_Data_for_SNOWPACK.py"


#%% set parameter lists
reg_codes = [3009, 3010, 3011, 3012, 3013]
h_lows = [0, 300, 600, 900]
h_his = [300, 600, 900, 1200]

models = ["GFDL-CM3"]
scens = ["historical", "rcp85"]  # ["rcp45", "rcp85"]
periods = ["", "MC", "LC"]  # []


#%% loop over the parameters
for model in models:

    print(model)

    for scen in scens:

        print(scen)

        for period in periods:

            if ((scen == "historical") & (period in ["MC", "LC"])):
                continue
            # end if

            print(period)

            for reg_code in reg_codes:

                for h_low, h_hi in zip(h_lows, h_his):

                    print(f"\n{reg_code} between {h_low} and {h_hi} m...\n")

                    subprocess.call(["python", py_path + py_script, "--reg_code", str(reg_code), "--low", str(h_low),
                                     "--high", str(h_hi), "--model", model, "--scen", scen, "--period", period])

                # end for h_low, h_hi
# end for reg_code

