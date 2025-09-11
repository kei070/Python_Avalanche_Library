#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for extracting subset.
"""

# imports
import numpy as np
import netCDF4


# definition
def extract_subset(arr_large, arr_small=None, x=None, y=None, x_f=None, y_f=None, x_vn="x", y_vn="y", topo_vn="ZS"):

    """
    Function tailored to extract a subset of the NORA3 grid based on a subset already in existence.

    Parameters:
        arr_large   Either a numpy array or a netCDF4 dataset containing the full NORA3-grid dataset from which the
                    subset will be extracted.
        arr_small   Submit the netCDF4 dataset of the NORA3 subset-grid dataset if you do not want to give all the grid
                    information in the other parameters. Defaults to None.
        x           1d-array containing the x-coordinates of the NORA3 subset. Defaults to None.
        y           1d-array containing the y-coordinates of the NORA3 subset. Defaults to None.
        x_f         1d-array containing the x-coordinates of the full NORA3-grid dataset. Defaults to None.
        y_f         1d-array containing the y-coordinates of the full NORA3-grid dataset. Defaults to None.
        x_vn        Name of the x variable (string) in the netCDF4 dataset. Only used if either arr_large or arr_small
                    are netCDF4 datasets. Defaults to "x".
        y_vn        Name of the y variable (string) in the netCDF4 dataset. Only used if either arr_large or arr_small
                    are netCDF4 datasets. Defaults to "y".
        topo_vn     Name of the topography variable in the netCDF4 dataset. Only used if arr_large is a netCDF4 dataset.
                    Defaults to "ZS".
    """

    if type(arr_small) == netCDF4._netCDF4.Dataset:
        x = arr_small[x_vn][:].squeeze()
        y = arr_small[y_vn][:].squeeze()
    # end if else

    if type(arr_large) == netCDF4._netCDF4.Dataset:
        x_f = arr_large[x_vn][:].squeeze()
        y_f = arr_large[y_vn][:].squeeze()
        large_sz = arr_large[topo_vn][:].squeeze()
    else:
        large_sz = arr_large
    # end if else

    # get the indices corresponding to the subset
    x_inds = np.where(np.isin(x_f, x))[0]
    y_inds = np.where(np.isin(y_f, y))[0]

    # get the DEM subset
    sz_sub = large_sz[y_inds, :][:, x_inds]

    # return
    return sz_sub, y_inds, x_inds

# end def
