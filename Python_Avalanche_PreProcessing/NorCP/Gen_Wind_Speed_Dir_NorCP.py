#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute the function to generate the rain and snow distinction for the NorCP data.
"""

#%% imports
# from Gridded_Predictors.Calc_Wind_Gridded_NorCP import calc_wspeed_dir_ncp
from ava_functions.Calc_Wind_Gridded_NorCP import calc_wspeed_dir_ncp


#%% call the function for the required parameter settings
params_list = [["EC-Earth", "historical", ""],
               ["EC-Earth", "rcp45", "MC"],
               ["EC-Earth", "rcp45", "LC"],
               ["EC-Earth", "rcp85", "MC"],
               ["EC-Earth", "rcp85", "LC"],
               ["GFDL-CM3", "historical", ""],
               ["GFDL-CM3", "rcp85", "MC"],
               ["GFDL-CM3", "rcp85", "LC"]]

params_list = [["ERAINT", "evaluation", ""]]


#%% loop over the parameter sets and execute the function
for params in params_list:
    calc_wspeed_dir_ncp(params)
# end for params