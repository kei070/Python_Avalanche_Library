"""
Batch extract the EURO-CORDEX data.
"""


#%% imports
import subprocess

from ava_functions.Lists_and_Dictionaries.Paths import py_par_path
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set the paths to the scripts
py_path = f"/{py_par_path}/EURO-CORDEX_Downscalings/Preprocessing/"


#%% set the scenarios
scens = ["hist"]


#%% set the models
models = ["CNRM_RCA", "IPSL_RCA"]


#%% set the variables
variables = ["TM", "TN", "TX", "RR"]


#%% batch execution
for mod in models:
    for scen in scens:
        for var in variables:

            print(f"\n{mod} -- {scen} -- {var}...\n")

            subprocess.call(["python", py_path + "Extract_EURO-CORDEX_NorthNorwaySub.py", mod, scen, var])

        # for reg_code
    # end for scen
# end for mod