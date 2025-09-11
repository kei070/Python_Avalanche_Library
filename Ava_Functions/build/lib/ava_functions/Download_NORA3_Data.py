#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the functions provided in the metocean-api to download NORA3 subsets. The "novelty" here is that the function will
allow to generate >>spatial<< subsets and not only temporal subsets.
"""

#%% imports --> import my adjusted ts module
import os
import sys
import numpy as np
from netCDF4 import Dataset
from nco import Nco
from metocean_api import ts_KUE as ts


#%% function
def download_nora3(x1, y1, x2, y2, product='NORA3_atm_sub', start_date='1985-01-01', end_date='1985-01-31',
                   var_l=["air_temperature_2m"],
                   outpath="/media/kei070/One Touch/IMPETUS/NORA3/",
                   outname_add="NORA3_NorthNorway_sub_",
                   gen_dir=True,
                   nora3_aux="/media/kei070/One Touch/IMPETUS/NORA3/arome3km_1hr_198602.nc"):

    """
    Function for downloading NORA3 data subsets employing some of the metocean-api functions. The novelty of the
    function is that in addition to temporal also >>spatial<< subsets can be created. That is, here we download gridded
    data instead of time-series data for individual gridcells.

    Parameters:
        x1:          Integer. x index of the south-west corner of the spatial subset. Units: degrees east.
        y1:          Integer. y index of the south-west corner of the spatial subset. Units: degrees north.
        x2:          Integer. x index of the north-east corner of the spatial subset. Units: degrees east.
        y2:          Integer. y index of the north-east corner of the spatial subset. Units: degrees north.
        product:     String. The NORA3 product name. Defaults to NORA3_atm_sub.
        start_date:  String. Start data of temporal subset. Format: YYYY-MM-DD. Defaults to 1985-01-01.
        end_date:    String. End data of temporal subset. Format: YYYY-MM-DD. Defaults to 1985-01-31.
        var_l:       List of strings. A list of the names of variables to be downloaded. Defaults to
                                      ["air_temperature_2m"].
        outpath:     String. Path to the directory where the downloaded data will be stored.
        outname_add: String. This will be added in front of the date in the output file name. Defaults to
                             NORA3_NorthNorway_sub.nc.
        gen_dir:     Boolean. If True (default), the directory specified in the parameter outpath will be created.
        nora3_aux:   String. The path and filename of an already downloaded NORA3 netcdf file that will be used in this
                             function as grid information.
    """

    # generate the output director if requested
    if gen_dir:
        os.makedirs(outpath, exist_ok=True)
    # ednd if

    # generate the date list and get the URL info of the file
    date_list = ts.get_date_list(product=product, start_date=start_date, end_date=end_date)

    x_coor_str, y_coor_str, infile = ts.get_url_info(product, date_list[0])

    # load a NORA3 file to retrieve the grid --> this is needed to retrieve the rlon and rlat coordinates that are then
    # used in the NCO command to extract the correct spatial subset
    nora3_aux_nc = Dataset(nora3_aux)

    # get the x and y coordinates on the NORA3 grid
    if np.any(np.array([x1, x2, y1, y2]) < 0):
        # construct the NCO command
        opt = ['-O -v ' + ",".join(var_l) + ",longitude,latitude,x,y"]

        print("\nThe whole NORA3 grid will be downloaded.\n")
    else:
        x_coor1 = nora3_aux_nc.variables["x"][x1]
        x_coor2 = nora3_aux_nc.variables["x"][x2]
        y_coor1 = nora3_aux_nc.variables["y"][y1]
        y_coor2 = nora3_aux_nc.variables["y"][y2]

        # inform the user if x2 < x1 and y2 < y1
        if x_coor2 < x_coor1:
            print("\nNote that x2 < x1\n")
        if y_coor2 < y_coor1:
            print("\nNote that y2 < y1\n")
        # end if

        # construct the NCO command
        opt = ['-O -v ' + ",".join(var_l) + ",longitude,latitude" +
                                                                  f' -d x,{x_coor1},{x_coor2} -d y,{y_coor1},{y_coor2}']

    # end if else
    # get the nearest coordinates on the NORA3 grid
    # x_coor, y_coor, lon_near, lat_near = ts.get_near_coord(infile=infile, lon=lon1, lat=lat1, product=product)
    # x_coor2, y_coor2, lon_near2, lat_near2 = ts.get_near_coord(infile=infile, lon=lon2, lat=lat2, product=product)

    # print the NCO command
    print("\nThe NCO command is:")
    print(opt[0])
    print("\n")

    # ask the user if - after seeing the NCO command - the execution should be stopped
    proceed = input("\nProceed? (Y/no)\n")
    if proceed == "":
        proceed = "yes"
    if (proceed.lower() == "N") | (proceed.lower() == "no"):
        sys.exit("Stopping execution.")
    # end if

    # execute the NCO commend and subsequently the download
    for i in range(len(date_list)):
        outname = outname_add + date_list[i].strftime('%Y%m') + ".nc"
        x_coor_str, y_coor_str, infile = ts.get_url_info(product, date_list[i])
        nco = Nco()
        nco.ncks(input=infile, output=outpath+outname, options=opt)
    # end for i
# end def
