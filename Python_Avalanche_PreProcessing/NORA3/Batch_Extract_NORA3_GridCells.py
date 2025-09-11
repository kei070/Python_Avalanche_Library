"""
Batch script for extraction of NORA3 gridcells. Now also includes the extract seNorge grid cells script.
"""


#%% imports
import subprocess
import numpy as np

from ava_functions.Lists_and_Dictionaries.Paths import py_path_par


#%%% set the paths to the scripts
py_path = f"{py_path_par}/NORA3/"


#%% execute the script
for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):
    print(f"\nBetween {h_low} and {h_hi} m\n")
    for syr, eyr in zip(np.arange(1970, 2024+1, 1), np.arange(1970, 2024+1, 1)):
        print(f"\nYears {syr}-{eyr}\n")

        subprocess.call(["python", py_path + "Extract_NORA3_GridCells_Between_NC.py", "--start_year", str(syr),
                         "--end_year", str(eyr), "--low", str(h_low), "--high", str(h_hi)])
    # end for syr, eyr
# end for h_low, h_hi
