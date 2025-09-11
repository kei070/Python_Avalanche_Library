#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate the NORA3 daily files to monthly files. Note that the values are not changed; the remain hourly.
"""

# imports
import os
import glob
import threading
import xarray as xr

def concat_nora3_ann(yr, data_path, out_path, out_name, delete=False):
    """
    Concatenate the NORA3 monthly files to annual files. Note that the values are not changed; the remain hourly.

    Parameter:
        yr     Integer. The year for which the files are concatenated.
        delete Boolean. If True (NOT default) the daily files are will be deleted after the concatenation.
    """

    print(f"Thread ID: {threading.get_ident()}, year: {yr}\n")

    #% load the file list
    fn_list = glob.glob(data_path + f"*_{yr}*.nc")


    # print(fn_list)


    #% load the files via xarrays multi-file function
    try:
        print("\ndata_vars='minimal'\n")
        ncs = xr.open_mfdataset(fn_list, data_vars="minimal")
    except:
        print("...didn't work. Using default instead.\n")
        ncs = xr.open_mfdataset(fn_list)
    # end try except
    # print(ncs)

    # ncs = xr.open_mfdataset(fn_list)

    #% store the data
    print(f"\n{len(fn_list)} Files loaded. Concatenating...\n")
    # print(fn_list)
    ncs.to_netcdf(out_path + out_name)

    #% remove the monthly files
    if delete:
        for f in fn_list:
            try:
                os.remove(f)
                # print(f"The file {f} has been deleted successfully.")
            except FileNotFoundError:
                print(f"The file {f} does not exist.")
            except PermissionError:
                print(f"Permission denied: You do not have the necessary permissions to delete the file {f}.")
            except OSError as e:
                print(f"Error: {e.strerror}")
            # end try except'
        # end for f
        print("\nMonthly files deleted.\n")
    # end if

# end def


