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
py_script = "Prep_Indices_from_SNOWPACK_Output.py"


#%% set the parameters
reg_codes = [3012]  # [3009, 3010, 3011, 3012, 3013]
h_lows = [300]  # [0, 300, 600, 900]
h_his = [600]  # [300, 600, 900, 1200]
slope_angle = 0
slope_azi = 0

models = ["GFDL-CM3"]
scens = ["historical"]  # ["historical", "rcp45", "rcp85"]

periods = ["", "MC", "LC"]  # THIS CAN BE LEFT AS IS (if queries are in place to ensure correct combinations)

source = "NORA3"

per_years = {"":zip([1985], [2005]), "MC":zip([2040], [2060]), "LC":zip([2080], [2100])}


#%% loop over the parameters
# if source is NorCP
if source == "NorCP":
    for model in models:
        print(model)

        for scen in scens:

            print(scen)

            for period in periods:

                if ((scen == "historical") & (period != "")):
                    continue
                if ((scen in ["rcp45", "rcp85"]) & (period == "")):
                    continue
                # end if

                print(period)
                for sta_yr, end_yr in per_years[period]:
                    for reg_code in reg_codes:
                        for h_low, h_hi in zip(h_lows, h_his):

                            print(f"\n{reg_code} between {h_low} and {h_hi} m...\n")

                            subprocess.call(["python", py_path + py_script, "--reg_code", str(reg_code),
                                             "--low", str(h_low), "--source", source, "--model", model, "--scen", scen,
                                             "--period", period, "--slope_angle", str(slope_angle),
                                             "--slope_azi", str(slope_azi), "--cut_30june",
                                             "--high", str(h_hi), "--sta_yr", str(sta_yr), "--end_yr", str(end_yr)])

                        # end for h_low, h_hi
                    # end for reg_code
                # end for sta_yr, end_yr

            # end for period
        # end for scen
    # end for model
# end if


# if source is NORA3
n3_sta_y = [1970, 1985, 2000, 2015]
n3_end_y = [1985, 2000, 2015, 2024]

n_iter = len(n3_end_y) * len(reg_codes) * len(h_lows)
i_iter = 0
print(f"\nGoing through {n_iter} iterations...\n")
if source == "NORA3":
    for sta_yr, end_yr in zip(n3_sta_y, n3_end_y):
        for reg_code in reg_codes:
            for h_low, h_hi in zip(h_lows, h_his):

                print(f"\n{reg_code} between {h_low} and {h_hi} m...\n")

                call = ["python", py_path + py_script, "--reg_code", str(reg_code), "--low", str(h_low),
                                 "--source", source, "--retry_wo_yr", "--slope_angle", str(slope_angle),
                                 "--slope_azi", str(slope_azi),
                                 "--high", str(h_hi), "--sta_yr", str(sta_yr), "--end_yr", str(end_yr)]
                if end_yr == 2024:
                    call = ["python", py_path + py_script, "--reg_code", str(reg_code), "--low", str(h_low),
                                     "--source", source, "--retry_wo_yr", "--slope_angle", str(slope_angle),
                                     "--slope_azi", str(slope_azi), "--cut_30june",
                                     "--high", str(h_hi), "--sta_yr", str(sta_yr), "--end_yr", str(end_yr)]
                # end if
                subprocess.call(call)

                i_iter += 1
                print(f"\n{i_iter / n_iter * 100:.1f}% ({i_iter}/{n_iter} iterations) completed\n")
            # end for h_low, h_hi
        # end for reg_code
    # end for sta_yr, end_yr
# end if

