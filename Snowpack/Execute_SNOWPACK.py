#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for executing SNOWPACK. Set the important parameters here. The script will produce the .ini file necessary to
run SNOWPACK.
"""


#%% imports
import os
import sys
import glob
import subprocess
import numpy as np
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set parameters
reg_codes = [3012]  # [3009, 3010, 3011, 3012, 3013]
h_lows = [300]  # , 300, 600]  # , 900]
h_his = [600]  # , 600, 900]  # , 1200]
slope_angle = 0
slope_azi = 0  # the aspect

ta_int = "nearest"  # "linear"

source = "NORA3"

# NORA3:
sta_yrs = [1970, 1985, 2000, 2015]
sta_mons = [10, 9, 9, 9]
sta_day = 1
end_yrs = [1985, 2000, 2015, 2024]
end_mons = [8, 8, 8, 6]
end_days = [31, 31, 31, 30]

"""
# NorCP Lyngen 900-1200
sta_yrs = [1985, 1992, 1999]
sta_mons = [10, 9, 9]
sta_day = 1
end_yrs = [1992, 1999, 2005]
end_mons = [8, 8, 6]
end_days = [31, 31, 30]

sta_yrs = [1999]
sta_mons = [9]
sta_day = 1
end_yrs = [2005]
end_mons = [12]
end_days = [31]

sta_yrs = [2040]
sta_mons = [10]
sta_day = 1
end_yrs = [2060]
end_mons = [12]
end_days = [31]
"""

model = "EC-Earth"
scen = "rcp45"
period = "_MC"  # as of now either "", "_MC", or "_LC"


#%% generate the strings based on slope and aspect
slope_path = "Flat"
if slope_angle > 0:
    aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

    slope_path = f"{slope_angle}" + aspect
# end if

if slope_angle == 0:
    aspect = ""
# end if


#%% set the temperature interpolation
if ta_int == "nearest":
    ta_nearest1 = "TA::resample = nearest"
    ta_nearest2 = "TA::nearest::extrapolate = true"
else:
    ta_nearest1 = ""
    ta_nearest2 = ""
# end if else


#%% loop over the different time periods
# for sta_yr, end_yr, sta_mon, end_mon, end_day in zip(sta_yrs, end_yrs, sta_mons, end_mons, end_days):
for reg_code in reg_codes:
    for h_low, h_hi in zip(h_lows, h_his):

        # for 3011 and 3012 there no grid cells > 900 m -- NORA3
        if ((h_low >= 900) & (reg_code in [3011, 3012])):
            continue
        # end if

        # for 3012 the SNOWPACK calculation do not work for the 300-600m elevation band (don't know why) -- NORA3
        # if ((h_low == 300) & (reg_code == 3012)):
        #     continue
        # end if

        for i in np.arange(len(sta_yrs)):
            sta_yr = sta_yrs[i]
            end_yr = end_yrs[i]
            sta_mon = sta_mons[i]
            end_mon = end_mons[i]
            end_day = end_days[i]


            #% depending on the period an scenario set the end year
            if end_yr == 0:
                if source == "NorCP":
                    if period[-2:] == "MC":
                        end_yr = 2060
                    elif period[-2:] == "LC":
                        end_yr = 2100
                    else:
                        if scen == "historical":
                            end_yr = 2005
                        elif scen == "evaluation":
                            end_yr = 2018
                        # end if elif
                    # end if elif else
                elif source == "NORA3":
                    end_yr = 2024
                # end if
            # end if

            if source == "NORA3":
                out_dir = ""
            elif source == "NorCP":
                out_dir = f"{model}_{scen}{period}/"
            # end if elif


            #% set paths
            ini_path = f"{path_par}/IMPETUS/{source}/Snowpack/"
            snowpack_path = f"{path_par}/IMPETUS/{source}/Snowpack/"
            out_path = f"{path_par}/IMPETUS/{source}/Snowpack/Output/{slope_path}/" + \
                                                                  f"Between{h_low}_{h_hi}m/{out_dir}/{sta_yr}_{end_yr}/"
            inp_path = f"/{path_par}/IMPETUS/{source}/Snowpack/Input/Elev_Mean_SMET/Between{h_low}_{h_hi}m/{out_dir}/"


            #% generate the output file
            os.makedirs(out_path, exist_ok=True)


            #% load the .sno file to adjust the start date and the file name
            print(f"\nUpdating the .sno file with start year {sta_yr}...\n")

            with open(inp_path + f"{regions[reg_code]}_{h_low}_{h_hi}m_{slope_path}.sno", 'r') as file:
                lines = file.readlines()
            # end with

            # loop over the lines and find the ProfileDate
            prof_ind = np.where(np.array([k[:11] == "ProfileDate" for k in lines]))[0][0]

            # change the profile date
            lines[prof_ind] = f"ProfileDate = {sta_yr}-{sta_mon:02}-{sta_day:02}T00:00:00\n"

            # loop over the lines and find the ProfileDate
            # id_ind = np.where(np.array([k[:10] == "station_id" for k in lines]))[0][0]
            # name_ind = np.where(np.array([k[:12] == "station_name" for k in lines]))[0][0]

            # lines[id_ind] = f"station_id = {regions[reg_code]}_{h_low}_{h_hi}_{sta_yr}_{end_yr}{slope}\n"
            # lines[name_ind] = f"station_name = {regions[reg_code]}_{h_low}_{h_hi}_{sta_yr}_{end_yr}{slope}\n"

            # write the .sno file
            with open(inp_path + f"{regions[reg_code]}_{h_low}_{h_hi}m_{slope_path}.sno", 'w') as file:
                file.writelines(lines)
            # end with


            #% set up the .ini script
            ini_str = f"""[General]
            BUFFER_SIZE = 370
            BUFF_BEFORE = 1.5
            BUFF_GRIDS = 10

            [Input]
            COORDSYS = CH1903
            TIME_ZONE = 1.00
            METEO = SMET
            METEOPATH = /{path_par}/IMPETUS/{source}/Snowpack/Input/Elev_Mean_SMET/Between{h_low}_{h_hi}m/{out_dir}/
            STATION1 = {regions[reg_code]}_{h_low}_{h_hi}m.smet
            SNOWPACK_SLOPES = false
            SNOW = SMET
            SNOWPATH = /{path_par}/IMPETUS/{source}/Snowpack/Input/Elev_Mean_SMET/Between{h_low}_{h_hi}m/{out_dir}/
            SNOWFILE1 = {regions[reg_code]}_{h_low}_{h_hi}m_{slope_path}.sno
            SLOPE_FROM_SNO = TRUE

            [InputEditing]
            ENABLE_TIMESERIES_EDITING = TRUE

            [Snowpack]
            CALCULATION_STEP_LENGTH = 15.000
            ROUGHNESS_LENGTH = 0.002
            HEIGHT_OF_METEO_VALUES = 5.0
            HEIGHT_OF_WIND_VALUE = 5.0
            ENFORCE_MEASURED_SNOW_HEIGHTS = FALSE
            SW_MODE = BOTH
            ATMOSPHERIC_STABILITY = RICHARDSON
            CANOPY = FALSE
            MEAS_TSS = FALSE
            CHANGE_BC = FALSE
            SNP_SOIL = FALSE

            [SnowpackAdvanced]
            FIXED_POSITIONS = 0.25 0.5 1.0 -0.25 -0.10
            NUMBER_SLOPES = 1

            [SnowpackSeaice]
            CHECK_INITIAL_CONDITIONS = FALSE

            [TechSnow]
            SNOW_GROOMING = FALSE

            [Filters]
            ENABLE_METEO_FILTERS = TRUE
            ENABLE_TIME_FILTERS = TRUE

            PSUM::filter1 = min
            PSUM::arg1::soft = true
            PSUM::arg1::min = 0.0

            TA::filter1 = min_max
            TA::arg1::min = 200
            TA::arg1::max = 320

            RH::filter1 = min_max
            RH::arg1::min = 0.01
            RH::arg1::max = 1.2
            RH::filter2 = min_max
            RH::arg2::soft = true
            RH::arg2::min = 0.05
            RH::arg2::max = 1.0

            ISWR::filter1 = min_max
            ISWR::arg1::min = -10
            ISWR::arg1::max = 1500
            ISWR::filter2 = min_max
            ISWR::arg2::soft = true
            ISWR::arg2::min = 0
            ISWR::arg2::max = 1500

            RSWR::filter1 = min_max
            RSWR::arg1::min = -10
            RSWR::arg1::max = 1500
            RSWR::filter2 = min_max
            RSWR::arg2::soft = true
            RSWR::arg2::min = 0
            RSWR::arg2::max = 1500

            ILWR::filter1 = min_max
            ILWR::arg1::min = 188
            ILWR::arg1::max = 600
            ILWR::filter2 = min_max
            ILWR::arg2::soft = true
            ILWR::arg2::min = 200
            ILWR::arg2::max = 400

            TSS::filter1	= min_max
            TSS::arg1::min = 200
            TSS::arg1::max = 320

            TSG::filter1 = min_max
            TSG::arg1::min = 200
            TSG::arg1::max = 320

            HS::filter1 = min
            HS::arg1::soft = true
            HS::arg1::min = 0.0
            HS::filter2 = rate
            HS::arg2::max = 5.55e-5 ;0.20 m/h

            VW::filter1 = min_max
            VW::arg1::min = -2
            VW::arg1::max = 70
            VW::filter2 = min_max
            VW::arg2::soft = true
            VW::arg2::min = 0.0
            VW::arg2::max = 50.0

            [Interpolations1D]
            ENABLE_RESAMPLING = TRUE
            WINDOW_SIZE = 43200

            PSUM::resample = accumulate ;cf interractions with CALCULATION_STEP_LENGTH
            PSUM::accumulate::period = 900

            HS::resample = linear
            HS::linear::window_size	=	43200

            VW::resample = nearest
            VW::nearest::extrapolate = true

            DW::resample = nearest
            DW::nearest::extrapolate = true

            TSS::resample = linear
            TSS::linear::extrapolate = true
            TSS::linear::window_size	=	43200

            {ta_nearest1}
            {ta_nearest2}

            [Output]
            COORDSYS = CH1903
            TIME_ZONE = 1.00
            METEOPATH = /{out_path}/
            WRITE_PROCESSED_METEO = FALSE
            EXPERIMENT = {slope_path}
            SNOW_WRITE = FALSE
            PROF_WRITE = TRUE
            PROF_FORMAT = PRO
            AGGREGATE_PRO = FALSE
            AGGREGATE_PRF = FALSE
            PROF_START = 0
            PROF_DAYS_BETWEEN = 4.1666e-2
            PROF_ID_OR_MK = ID
            PROF_AGE_OR_DATE = AGE
            HARDNESS_IN_NEWTON = FALSE
            CLASSIFY_PROFILE = FALSE
            TS_WRITE = TRUE
            TS_FORMAT = SMET
            ACDD_WRITE = FALSE
            TS_START = 0
            TS_DAYS_BETWEEN = 4.1666e-2
            AVGSUM_TIME_SERIES = TRUE
            CUMSUM_MASS = FALSE
            PRECIP_RATES = TRUE
            OUT_CANOPY = FALSE
            OUT_HAZ = FALSE
            OUT_SOILEB = FALSE
            OUT_HEAT = TRUE
            OUT_T = TRUE
            OUT_LW = TRUE
            OUT_SW = TRUE
            OUT_MASS = TRUE
            OUT_METEO = TRUE
            OUT_STAB = TRUE"""


            """
            TA::resample = linear
            TA::linear::window_size	=	43200

            TA::resample = nearest
            TA::nearest::extrapolate = true

            TSS::resample = nearest
            TSS::nearest::extrapolate = true
            """

            #% write the ini-file
            ini_fn = f"io_{source}_{reg_code}_Mean_Between{h_low}_{h_hi}m.ini"

            try:
                with open(ini_path + ini_fn, "w") as fsno:
                    fsno.write(ini_str)

                    print(f"\nNew .ini file produced:\n{ini_fn}\n\n")
                # end with

                #% execute the SNOWPACK model
                os.chdir(snowpack_path)
                subprocess.call(["./snowpack.sif", "-c", ini_fn, "-e", f"{end_yr}-{end_mon:02}-{end_day:02}T23:00:00"])
                # subprocess.call(["./snowpack.sif", "-c", ini_fn, "-e", "2042-12-31T23:00:00"])
            except:
                print(f"{regions[reg_code]} {h_low} {h_hi}m not executed.")
                continue
            # end try except

            #% remove the backup files
            del_files = glob.glob(out_path + "*.sno*") + glob.glob(out_path + "*.haz*")

            # loop over the files and delete them
            for file_path in del_files:
                try:
                    os.remove(file_path)
                    # print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
                # end try except
            # end for file_path

        # end for ... in zip
