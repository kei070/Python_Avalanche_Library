#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the NORA3 gridded rain and snow.
"""

#%% import
# from Gridded_Predictors.Calc_Rain_Snow_Gridded import calc_rain_snow
from ava_functions.Calc_Rain_Snow_Gridded import calc_rain_snow


#%% execute
calc_rain_snow(sta_yr=1970, end_yr=2016)
# LAST YEAR NOT INCLUDED!