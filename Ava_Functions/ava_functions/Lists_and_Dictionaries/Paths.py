#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parent directory path for avalanche analysis --> change when porting to another system.
For the scripts to work properly (without further changes), the directory structure must be as follows:

    The avalanche features/predictors must be stored under
        h_low = 400  # change depending on requested height threshold
        h_hi = 900  # change depending on requested height threshold
        agg_type = "Mean"
        perc = "0"
        agg_str = ...
        feat_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors/{agg_str}/Between_{h_low}_and_{h_hi}m/"

    Note that these are the features together with the original 5-level target variable. The features with reduced
    danger level number are in
        ndlev = 2
        feat_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors_{ndlev}Level/{agg_str}/Between_{h_low}_and_{h_hi}m/"

    The naming convention of the feature/predictor files is
        region = "IndreTroms"  # "NordTroms", ...
        f_name = f"{region}_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}.csv"

    in case of the 5-level target and
        reg_code = 3013  # 3009, ...
        f_name_all = f"Features_{ndlev}Level_All_{agg_str}_Between400_900m_{reg_code}_{region}.csv"

    After the predictive features are aggregated across elevation bands, the will be stored in
        f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors_{ndlev}Level/{agg_str}/"

    with file names such as Features_2Level_All_Mean_ElevAgg_3009_NordTroms.csv

    The machine-learning models are stored in the directory
        f"{path_par}/IMPETUS/NORA3/Stored_Models/Mean/Elev_Agg/"

    The avalanche danger data from NVE are expected to be stored in
        f"{obs_path}/IMPETUS/Avalanche_Danger_Data/"

    The NORA3 annual files after concatination with
        Concat_Monthly_NORA3_Files_To_Annual.py
    are expected to be stored in
        f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"

    The Python parent path (py_path_par) should point to the directory in which Python_Avalanche_Library is located.

    To be able to execute SNOWPACK from the Python scripts in the Snowpack directory in this Package the SNOWPACK .sif
    container should be located in:
        f"{path_par}/IMPETUS/{source}/Snowpack/"
    where source is either "NORA3" or "NorCP".

    The SNOWPACK-derived indices are stored in
        f"{path_par}/IMPETUS/NORA3/Snowpack/Timeseries/Daily/Flat/"
"""


# external hard drive --- you can of course set all to the same directory
path_par = ""    # NORA3 & NorCP data (mostly predictive features); the Skred data observations (Fig. 1 in the paper)
path_par3 = ""   # EURO-CORDEX data; some raw NORA3 and NorCP data

# path to danger level "observations"
obs_path = ""

# python scripts path
py_path_par = ""
