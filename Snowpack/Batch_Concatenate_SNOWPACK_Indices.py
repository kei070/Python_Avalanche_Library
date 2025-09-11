#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-execute the preparation of SNOWPACK-derived indices.
"""


#%% imports
import subprocess
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% set paths and name
py_path = f"{obs_path}/Python_Avalanche_Library/Snowpack/"
py_script = "Concatenate_SNOWPACK_Indices.py"


#%% set the parameters
reg_codes = [3009, 3010, 3011, 3012, 3013]
h_lows = [0, 300, 600, 900]
h_his = [300, 600, 900, 1200]

slope_angle = 0
slope_azi = 0

source = "NORA3"  # args.source




#%% loop over the parameters
for reg_code in reg_codes:
    for h_low, h_hi in zip(h_lows, h_his):

        if reg_code in [3011, 3012]:
            if h_low == 900:
                continue
            # end if
        # end if

        print(f"\n{reg_code} between {h_low} and {h_hi} m...\n")

        subprocess.call(["python", py_path + py_script, str(reg_code), str(h_low), str(h_hi), str(slope_angle),
                         str(slope_azi)])

    # end for h_low, h_hi
# end for reg_code

