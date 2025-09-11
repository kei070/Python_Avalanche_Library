#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call the regrid NorCP to NORA3 script.
"""


#%% imports
import subprocess
from ava_functions.Lists_and_Dictionaries.Paths import path_scripts as py_path


#%% set paths
py_name = "Regrid_NorCP_to_NORA3_Grid.py"


#%% parameter sets
params_list = [["GFDL-CM3", "historical", "", "rsds"]]

"""
params_list = [["EC-EARTH", "rcp85", "LC", "rsds"]] #,
["EC-EARTH", "historical", "", "rsds"],
               ["EC-EARTH", "rcp45", "MC", "rlds"],
               ["EC-EARTH", "rcp45", "MC", "rsds"],
               ["EC-EARTH", "rcp45", "LC", "rlds"],
               ["EC-EARTH", "rcp45", "LC", "rsds"],
               ["EC-EARTH", "rcp85", "MC", "rlds"],
               ["EC-EARTH", "rcp85", "MC", "rsds"],
               ["EC-EARTH", "rcp85", "LC", "rlds"],
               ["EC-EARTH", "rcp85", "LC", "rsds"]]"""

"""
params_list = [["GFDL-CM3", "rcp85", "MC", "pr"],
               ["GFDL-CM3", "rcp85", "MC", "tas"],
               ["GFDL-CM3", "rcp85", "MC", "pr"],
               ["GFDL-CM3", "historical", "", "pr"],
               ["GFDL-CM3", "historical", "", "tas"],
               ["GFDL-CM3", "historical", "", "uas"],
               ["GFDL-CM3", "historical", "", "vas"]]
params_list = [["ERAINT", "evaluation", "", "hurs"],
               ["ERAINT", "evaluation", "", "tas"],
               ["ERAINT", "evaluation", "", "vas"],
               ["ERAINT", "evaluation", "", "uas"],
               ["ERAINT", "evaluation", "", "rlns"],
               ["ERAINT", "evaluation", "", "rsns"]]
"""

#%% set up the subprocess call
call_p1 = ["python", py_path + py_name]


#%% call the script with the required set of parameters
for params in params_list:
    print("")
    print(params)
    print("")
    call = call_p1 + params
    subprocess.call(call)
# end for params