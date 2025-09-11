#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch execute the Gen_ADF_Model_With_SNOWPACK_NorCP.py
"""

#%% imports
import subprocess
from ava_functions.Lists_and_Dictionaries.Paths import obs_path
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set paths and name
py_path = f"{obs_path}/Python_Avalanche_Libraries/Avalanche_Plot_Scripts/NORA3_and_NorCP/With_SNOWPACK/"
py_script = "Gen_ADF_Model_With_SNOWPACK_fullNORA3.py"


#%% execute
for reg_code in regions.keys():
    subprocess.call(["python", py_path + py_script, str(reg_code)])
# end for reg_code
