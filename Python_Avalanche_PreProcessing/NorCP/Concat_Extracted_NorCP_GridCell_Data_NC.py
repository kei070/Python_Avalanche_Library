#%% imports
import sys
import glob
import numpy as np
import xarray as xr
import argparse
from netCDF4 import Dataset

from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Variable_Name_NORA3_NorCP import var1h, var3h


#%% parser
parser = argparse.ArgumentParser(
                    description="""Concatenates that annual files of the extracted grid cells from NorCP for the
                    predictors.""")
parser.add_argument('--model', default="GFDL-CM3", choices=["EC-Earth", "GFDL-CM3", "ERAINT"], type=str,
                    help='Model used in the downscaling.')
parser.add_argument('--scen', default="rcp85", choices=["historical", "evaluation", "rcp45", "rcp85"], type=str,
                    help='Scenario used in the downscaling.')
parser.add_argument('--period', default="", choices=["", "MC", "LC"], type=str,
                    help='Period used in the downscaling.')
parser.add_argument('--low', default=0, type=int, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', default=300, type=int, help='The upper threshold for the altitude band to extract.')
parser.add_argument("--reg_codes", nargs="*", default=[3009, 3010, 3011, 3012, 3013],
                    help="""The numerical codes of the regions.""")

args = parser.parse_args()


#%% get the arguments from the parser
model = args.model  # "EC-Earth"
scen = args.scen  # "historical"
period = args.period  # ""  # either "", "MC", or "LC"
h_low = args.low  # 400  # args.low
h_hi = args.high  # 900  # args.high
reg_codes = args.reg_codes


#%% adjust the period string
if period in ["MC", "LC"]:
    period = f"_{period}"
# end if


#%% set data path
data_path = (f"{path_par}/IMPETUS/NorCP/Avalanche_Region_Data/Between{h_low}_and_{h_hi}m" +
             f"/{model.upper()}_{scen}{period}/")


#%% loop over the regions
for reg_code in reg_codes:

    # set region name according to region code
    if reg_code == 3009:
        region = "NordTroms"
    elif reg_code == 3010:
        region = "Lyngen"
    elif reg_code == 3011:
        region = "Tromsoe"
    elif reg_code == 3012:
        region = "SoerTroms"
    elif reg_code == 3013:
        region = "IndreTroms"
    # end if elif

    # loop over the time periods to load the necessary file names
    fn_list = sorted(glob.glob(data_path + f"/Annual_Files/{region}/*.nc"))

    if len(fn_list) == 0:
        print(f"No files available for {reg_code}. Continuing...")
        continue
    # end if
    print()
    print(f"Number of year intervals for {region}: {np.shape(fn_list)}\n")

    # get start and end year
    sta_yr = fn_list[0].split("_")[-1][:-3]
    end_yr = fn_list[-1].split("_")[-1][:-3]

    # print(sta_yr, end_yr)
    datasets = [xr.open_dataset(f) for f in fn_list]

    vars_1h = []
    vars_3h = []
    for var_1h in var1h:
        vars_1h.append(xr.concat([ds[var_1h] for ds in datasets], dim='time'))
    for var_3h in var3h:
        vars_3h.append(xr.concat([ds[var_3h] for ds in datasets], dim='time3h'))
    # end for
    vars_xh = vars_1h + vars_3h
    varn_xh = var1h + var3h

    # Combine into one dataset
    combined_ds = xr.Dataset({varn_xh_k:vars_xh_k for varn_xh_k, vars_xh_k in zip(varn_xh, vars_xh)})
    # combined = {varn_xh_k:vars_xh_k for varn_xh_k, vars_xh_k in zip(varn_xh, vars_xh)}

    # open the nc-files
    # nc = xr.open_mfdataset(fn_list, combine="by_coords", data_vars="minimal")
    combined_ds.to_netcdf(data_path + f"NorCP_{region}_Between{h_low}_and_{h_hi}_{sta_yr}-{end_yr}.nc")
# end for reg_code


#%% test
# nc = Dataset(data_path + f"NorCP_{region}_Between{h_low}_and_{h_hi}_{sta_yr}-{end_yr}.nc")
# nc1 = xr.open_dataset(data_path + f"NorCP_Tromsoe_Between{h_low}_and_{h_hi}_{sta_yr}-{end_yr}.nc")
# nc2 = xr.open_dataset(data_path + f"NorCP_NordTroms_{sta_yr}-{end_yr}.nc", decode_times=False)
# print(nc1)
# print(nc2)



