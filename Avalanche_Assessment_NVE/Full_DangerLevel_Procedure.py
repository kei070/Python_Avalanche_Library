#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform the full procedure to download and pre-process the avalanche danger levels (including the avalanche problems)
so that they can be used the machine-learning methodology.

Note that we need the conda environment n3_down to execute the download scripts.
"""


#%% imports
import subprocess
from ava_functions.Lists_and_Dictionaries.Paths import py_path_par


#%% parameters
perform_download = False  # set to true if the download is desired


#%% set paths
py_path = f"{py_path_par}/Python_Avalanche_Library/Avalanche_Assessment_NVE/"


#%% perform the procedure
if perform_download:
    print("Downloading danger levels and problems...")
    subprocess.call(["python", py_path + "Download_Avalanche_Danger_Events.py"])
    print("General avalanche danger levels downloaded. Downloading the problems...")
    subprocess.call(["python", py_path + "Download_Avalanche_Problem_Events.py"])
# end if

print("Generating danger level and problem datasets...")

subprocess.call(["python", py_path + "Adjust_Store_Avalanche_Problems_List.py"])

subprocess.call(["python", py_path + "Convert_Sensitivity_to_Readable.py"])

subprocess.call(["python", py_path + "Gen_ADL_per_AP_File.py"])

subprocess.call(["python", py_path + "Combine_Wet_Loose_and_Slab.py"])

subprocess.call(["python", py_path + "Add_Wet_AP_to_Combined_Files.py"])
