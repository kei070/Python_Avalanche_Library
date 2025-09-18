#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess SNOWPACK output to prepare the index time series to be used in a statistical prediction model.

See Perez-Guillen et al. (2022), Data-driven automated predictions..., p. 2036 (section 3.2.2)  --> PG22

To get more info about the SNOWPACK output variables look into

from snowpacktools.snowpro import pro_helper
pro_helper.get_pro_code_dict()
# snp_names, snp_nums = pro_helper.get_pro_code_dict()
# print(snp_names)


"0500": "date",
"0501": "height",                           # height [> 0: top, < 0: bottom of elem.] (cm)
"0502": "density",                          # element density (kg m-3)
"0503": "temperature",                      # element temperature (degC)
"0504": "element ID (1)",
"0505": "element deposition date (ISO)",    # or "element age (days)" --> see ini key PROF_AGE_OR_DATE
"0506": "lwc",                              # liquid water content by volume (%)
"0508": "dendricity (1)",
"0509": "sphericity (1)",
"0510": "coordination number (1)",
"0511": "bond size (mm)",
"0512": "grain size (mm)",
"0513": "grain type (Swiss Code F1F2F3)",
"0514": "grain type, grain size (mm), and density (kg m-3) of SH at surface", # 0514,3,660,0.9,100 vs. 0514,3,-999,-999.0,-999.0
"0515": "ice volume fraction (%)",
"0516": "air volume fraction (%)",
"0517": "stress in (kPa)",
"0518": "viscosity (GPa s)",
"0519": "soil volume fraction (%)",
"0520": "temperature gradient (K m-1)",
"0521": "thermal conductivity (W K-1 m-1)",
"0522": "absorbed shortwave radiation (W m-2)",
"0523": "viscous deformation rate",         # (1.e-6 s-1)
"0531": "deformation rate stability index Sdef",
"0532": "Sn38",                             # natural stability index Sn38
"0533": "Sk38",                             # stability index Sk38
"0534": "hand hardness",                    # either (N) or index steps (1)",
"0535": "optical equivalent grain size (mm)",
"0540": "bulk salinity (g/kg)",
"0541": "brine salinity (g/kg)",
"0601": "snow shear strength (kPa)",
"0602": "grain size difference (mm)",
"0603": "hardness difference (1)",
"0604": "RTA",                              # "Structural stability" index or with SNP-HACKING "Relative threshold sum approach (RTA)"
"0605": "inverse texture index ITI (Mg m-4)",
"0606": "critical cut length (m)"}
"""

#%% import
import os
import sys
import numpy as np
import pandas as pd
import pylab as pl
import argparse
from snowpacktools.snowpro import snowpro
from ava_functions.DatetimeSimple import date_dt
from ava_functions.Snowpack_Helpers import tsa
from ava_functions.Progressbar import print_progress_bar
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Prep SNOWPACK indices",
                    description="""Prepares the SNOWPACK output for use in statistical models.""")

# ...and add the arguments
parser.add_argument("--reg_code", default=3012, type=int, help="""The region code.""")
parser.add_argument("--low", default=300, type=int, help="""The lower elevation band threshold.""")
parser.add_argument("--high", default=600, type=int, help="""The upper elevation band threshold.""")
parser.add_argument("--source", default="NORA3", choices=["NORA3", "NorCP"], type=str, help="The dataset source.")
parser.add_argument("--model", default="EC-Earth", type=str, help="The NorCP model.")
parser.add_argument("--scen", default="rcp85", type=str, help="The NorCP scenario.")
parser.add_argument("--period", default="MC", type=str, help="The NorCP period.")
parser.add_argument("--sta_yr", default=1970, type=int, help="Start year if required.")
parser.add_argument("--end_yr", default=1985, type=int, help="End year if required.")
parser.add_argument("--slope_angle", default=0, type=int, help="Slope angle.")
parser.add_argument("--slope_azi", default=0, type=int, help="Slope azimuth angle -- i.e., the aspect.")
parser.add_argument("--retry_wo_yr", action="store_true", help="""Retry data loading without given years if loading
                    fails. This is mainly to make the application of batch script simpler.""")
parser.add_argument("--cut_30june", action="store_true", help="""Cut the data after 30 June. This should be done for the
                                                                 last year of the time series. That is, if the resulting
                                                                 files are supposed to be concatenated with
                                                                 Concatenate_SNOWPACK_Indices.py do NOT set this.""")
parser.add_argument("--plot", action="store_true", help="If set plots are generated.")

args = parser.parse_args()


#%% get the values from the parser
reg_code = args.reg_code
h_low = args.low
h_hi = args.high
sta_yr = args.sta_yr
end_yr = args.end_yr
source = args.source
slope_angle = args.slope_angle
slope_azi = args.slope_azi
retry_wo_yr = args.retry_wo_yr
retry_wo_yr = True
cut_30june = args.cut_30june


#%% fill the model, scen, and period variables with empty strings if source is NORA3
if source == "NORA3":
    model = ""
    scen = ""
    period = ""
    per_str = ""
elif source == "NorCP":
    model = args.model
    scen = "_" + args.scen
    period = args.period

    # set up the period string
    per_str = f"_{period}" if len(period) > 0 else ""
# end if elif


#%% generate the strings based on slope and aspect
"""
slope_str = "Flat"
if slope_angle > 0:
    slope_str = "Slope"
# end if
aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

if slope_angle == 0:
    aspect = ""
# end if
"""

#%% generate the strings based on slope and aspect
slope_path = "Flat"
if slope_angle > 0:
    aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

    slope_path = f"{slope_angle}" + aspect
# end if

if slope_angle == 0:
    aspect = ""
# end if


#%% add the year-string to the path if necessary
yr_str = ""
if ((sta_yr > 0) & (end_yr > 0)):
    yr_str = f"{sta_yr}_{end_yr}/"
# end if


#%% set path to some snow profile
spack_path = f"{path_par}/IMPETUS/{source}/Snowpack/Output/{slope_path}/" + \
                                                               f"Between{h_low}_{h_hi}m/{model}{scen}{per_str}/{yr_str}"

fn = f"{regions[reg_code]}_{h_low}_{h_hi}m_{slope_path}.pro"
path_to_pro = spack_path + fn

out_path = f"{path_par}/IMPETUS/{source}/Snowpack/Timeseries/Hourly/{model}{scen}{per_str}/{slope_path}/" + \
                                                                                      f"Between{h_low}_{h_hi}m/{yr_str}"
out_path2 = f"{path_par}/IMPETUS/{source}/Snowpack/Timeseries/Daily/{model}{scen}{per_str}/{slope_path}/" + \
                                                                                      f"Between{h_low}_{h_hi}m/{yr_str}"
os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path2, exist_ok=True)


#%% attempt the loading of a profile
if retry_wo_yr:
    try:
        profs, meta_dict = snowpro.read_pro(path_to_pro, res='1h', keep_soil=False, consider_surface_hoar=True)
    except:
        try:
            # stop if the iteration is not the first set of years (1970-1985) to prevent redundant calculations
            if yr_str != "1970_1985/":
                sys.exit("Calculation would likely be redundant. Stopping.")
            # end if
            print("\nSNOWPACK data loading failed. Retrying without the given year range...\n")
            spack_path = f"{path_par}/IMPETUS/{source}/Snowpack/Output/{slope_path}/" + \
                                                                       f"Between{h_low}_{h_hi}m/{model}{scen}{per_str}/"
            out_path = f"{path_par}/IMPETUS/{source}/Snowpack/Timeseries/Hourly/{model}{scen}{per_str}/" + \
                                                                                 f"{slope_path}/Between{h_low}_{h_hi}m/"
            out_path2 = f"{path_par}/IMPETUS/{source}/Snowpack/Timeseries/Daily/{model}{scen}{per_str}/" + \
                                                                                 f"{slope_path}/Between{h_low}_{h_hi}m/"
            os.makedirs(out_path, exist_ok=True)
            os.makedirs(out_path2, exist_ok=True)
            path_to_pro = spack_path + fn
            profs, meta_dict = snowpro.read_pro(path_to_pro, res='1h', keep_soil=False, consider_surface_hoar=True)
        except:
            sys.exit("Data not available for this region and elevation band. Stopping.")
        # end try except
    # end try except
else:
    profs, meta_dict = snowpro.read_pro(path_to_pro, res='1h', keep_soil=False, consider_surface_hoar=True)
# end if else


#%% get a timestamp
tstamps = list(profs.keys())


#%% apply the TSA to all profiles
l = len(tstamps)
count = 0
print("\nApplying the TSA to all profiles...\n")
print_progress_bar(count, l, suffix="")
t_i = 0
for t_i in np.arange(len(tstamps)):

    suffix = f"{tstamps[t_i]}"

    try:
        profs[tstamps[t_i]]["TSA"] = tsa(profs[tstamps[t_i]])
        count +=1
        print_progress_bar(count, l, suffix="  | " + suffix + "                            ")
    except:
        count +=1
        print_progress_bar(count, l, suffix="  | " + suffix + " TSA failed (likely no data)")
        # continue
    # end try except
# end for t_i


#%% identify the PWLs for every time step and extract other parameters such as snowdepth
pwl_100s = []
pwl_100_is = []
pwl_2s = []
pwl_2_is = []
sno_hei = []
l = len(tstamps)
count = 0
print("\nSearching for PWLs and extracting variables...\n")
# DO NOT USE THE PROGRESSBAR --> CAUSES SPYDER TO CRASH
# print_progress_bar(count, l, suffix="")
for tstamp in tstamps:

    suffix = f"{tstamp}"
    # print_progress_bar(count, l, suffix="  | " + suffix + "  |            ")


    # load the profile
    temp_prof = profs[tstamp]

    if len(temp_prof["height"]) > 0:  # proceed only if profiles exist

        # extract the height
        sno_hei.append(temp_prof["height"][-1])

        if len(temp_prof["TSA"]["pwl_heights"]) > 0:  # proceed only if PWLs exist
            # get the top PWL height
            pwl_0h = temp_prof["TSA"]["pwl_heights"][0]

            # if the top layer is within the first 100cm search for the next PWL
            if pwl_0h < 100:
                pwl_100 = pwl_0h
                pwl_100_i = np.where(temp_prof["TSA"]["tsa"])[0][0]

                pwl_100s.append(pwl_100)
                pwl_100_is.append(pwl_100_i)

                if len(temp_prof["TSA"]["pwl_heights"]) > 1:
                    pwl_2 = temp_prof["TSA"]["pwl_heights"][1]
                    pwl_2_i = np.where(temp_prof["TSA"]["tsa"])[0][1]

                    pwl_2s.append(pwl_2)
                    pwl_2_is.append(pwl_2_i)
                else:  # no second PWL exists
                    pwl_2s.append(np.nan)
                    pwl_2_is.append(np.nan)
                # end if else
            # if the top layer is not with the first 100cm
            else:
                pwl_100 = np.nan
                pwl_100_i = np.nan

                pwl_2 = pwl_0h
                pwl_2_i = np.where(temp_prof["TSA"]["tsa"])[0][0]

                pwl_100s.append(pwl_100)
                pwl_100_is.append(pwl_100_i)
                pwl_2s.append(pwl_2)
                pwl_2_is.append(pwl_2_i)
            # end if else
        else:  # no PWL exists
            pwl_100s.append(np.nan)
            pwl_100_is.append(np.nan)
            pwl_2s.append(np.nan)
            pwl_2_is.append(np.nan)
            # print_progress_bar(count, l, suffix="  | " + suffix + "  |  no PWL    ")
        # end if PWL
    else:  # no profile exists
        # print_progress_bar(count, l, suffix="  | " + suffix + "  |  no profile")
        pwl_100s.append(np.nan)
        pwl_100_is.append(np.nan)
        pwl_2s.append(np.nan)
        pwl_2_is.append(np.nan)
        sno_hei.append(np.nan)
    # end if profile

    count += 1
    # print_progress_bar(count, l, suffix="  | " + suffix)
# end for i_prof


#%% add the index to a Pandas dataframe
depth_df = pd.DataFrame({"snow_depth":sno_hei}, index=tstamps)

# again, fill the NaNs with the highest value (apparently 3)
depth_df.fillna(value=0, inplace=True)


#%% calculate the 1-day sum, 3-day sum, and 7-day mean of the snow depth
depth_df_day = depth_df.resample('D').mean()
depth_df_3d = depth_df_day.rolling(window=3).mean()
depth_df_7d = depth_df_day.rolling(window=7).mean()

depth_df_3d.rename(columns={"snow_depth":"snow_depth3"}, inplace=True)
depth_df_7d.rename(columns={"snow_depth":"snow_depth7"}, inplace=True)

depth_df_day.set_index(depth_df_day.index.date, inplace=True)
depth_df_3d.set_index(depth_df_3d.index.date, inplace=True)
depth_df_7d.set_index(depth_df_7d.index.date, inplace=True)


#%% combine the PWL arrays to a dataframe
pwl_df = pd.DataFrame({"pwl_100":pwl_100s, "pwl_100_i":pwl_100_is, "pwl":pwl_2s, "pwl_i":pwl_2_is},
                      index=tstamps)
# --> not the _i refers to this index of the layer in the profile, but:
#     the unstable layer is then probably the one above the interface and we thus must add 1 to the index!


#%% extract time series of the indices from the profile at the PWLs
s_i_ns = ["Sk38", "Sn38", "RTA"]
stab = {}
for s_i_n in s_i_ns:
    s_i_100s = []
    s_i_2s = []
    for tstamp in tstamps:

        # load the profile
        temp_prof = profs[tstamp]

        if len(temp_prof["height"]) == 0:
            s_i_100s.append(np.nan)
            s_i_2s.append(np.nan)

            continue
        # end if

        # extract the PWL indices --> add 1 to make sure the layer above the interface is extracted
        pwl_100_i = pwl_df.loc[tstamp]["pwl_100_i"] + 1
        pwl_2_i = pwl_df.loc[tstamp]["pwl_i"] + 1

        if np.isnan(pwl_100_i):
            # extract a stability index
            s_i_100 = np.nan
        else:
            s_i_100 = temp_prof[s_i_n][int(pwl_100_i)]
        if np.isnan(pwl_2_i):
            # extract a stability index
            s_i_2 = np.nan
        else:
            s_i_2 = temp_prof[s_i_n][int(pwl_2_i)]
        # end if else

        s_i_100s.append(s_i_100)
        s_i_2s.append(s_i_2)

    # end for tstamp

    stab[s_i_n + "_100"] = s_i_100s
    stab[s_i_n + "_2"] = s_i_2s

# end for s_i_n

stab_df = pd.DataFrame(stab, index=tstamps)


#%% fill the NaNs with the highest value (apparently 6)
for col in stab_df.columns:
    stab_df[col].fillna(value=stab_df.max()[col], inplace=True)
# end for col


#%% extract the minimum critical cut length
print("\nExtracting the minimum CCL from all profiles...\n")
ccl_l = []
for tstamp in tstamps:

    # load the profile
    temp_prof = profs[tstamp]

    if len(temp_prof["height"]) == 0:
        ccl_l.append(np.nan)
        continue
    # end if

    ccls = temp_prof["critical cut length (m)"]

    # NaNs are stored as -999 --> they would be taken as minimum so first convert them to NaN
    ccls[ccls <= -999] = np.nan

    # take the NaN-min
    ccl = np.nanmin(ccls)

    ccl_l.append(ccl)

# end for tstamp


#%% add the index to a Pandas dataframe
ccl_df = pd.DataFrame({"ccl":ccl_l}, index=tstamps)

# again, fill the NaNs with the highest value (apparently 3)
ccl_df.fillna(value=ccl_df.max(), inplace=True)


#%% combine the dataframes
snowpack_df = stab_df.merge(ccl_df, how="inner", left_index=True, right_index=True)
# snowpack_df = snowpack_df.merge(depth_df, how="inner", left_index=True, right_index=True)


#%% calculate the LWC_index from Mitterer et al. (2013) and also used in Hendrick et al. (2023)
#   (https://doi.org/10.1017/jog.2023.24)
print("\nCalculating some parameters from Hendrick et al. (2023; H23) for all profiles...\n")
t_i = 0
lwc_i_l = []
lwc_s_l = []
lwc_s_top_l = []
lwc_max_l = []
for t_i in np.arange(len(tstamps)):

    suffix = f"{tstamps[t_i]}"

    try:
        # NOTE: LWC_index is in effect the mean LWC!
        lwc_i_l.append(np.average(profs[tstamps[t_i]]["lwc"], weights=profs[tstamps[t_i]]["thickness"]) / 3)
        lwc_s_l.append(np.sum(profs[tstamps[t_i]]["lwc"]))
        lwc_max_l.append(np.max(profs[tstamps[t_i]]["lwc"]))

        # calculate the depth "into" the snowpack (i.e., relative to the snow surface) to then calculate the sum of the
        # LWC within the top 15 cm (see H23, sum_up)
        depth_into_snow = profs[tstamps[t_i]]["height"][-1] - profs[tstamps[t_i]]["bottom"]
        lwc_s_top_l.append(np.sum(profs[tstamps[t_i]]["lwc"][depth_into_snow < 15]))
    except:
        lwc_i_l.append(0)
        lwc_s_l.append(0)
        lwc_max_l.append(0)
        lwc_s_top_l.append(0)
    # end try except
# end for t_i


#%% Add some of the other parameters that are among the most important features in Hendrick et al. (2023)
t_i = 0
t_top_l = []
for t_i in np.arange(len(tstamps)):

    suffix = f"{tstamps[t_i]}"

    try:
        t_top_l.append(profs[tstamps[t_i]]["temperature"][-1])
    except:
        t_top_l.append(0)
    # end try except
# end for t_i


#%% add the index to a Pandas dataframe
h23_df = pd.DataFrame({"lwc_i":lwc_i_l, "lwc_sum":lwc_s_l, "lwc_max":lwc_max_l, "lwc_s_top":lwc_s_top_l,
                       "t_top":t_top_l},
                      index=tstamps)

# again, fill the NaNs with the highest value (apparently 3)
h23_df.fillna(value=0, inplace=True)


#%% combine the dataframes
snowpack_df = stab_df.merge(h23_df, how="inner", left_index=True, right_index=True)


#%% name the index
snowpack_df.index.rename("date", inplace=True)


#%% test plots
if args.plot:
    pl.plot(snowpack_df["ccl"])
    pl.show()
    pl.close()
# end if


#%% store the index dataframe as .csv
snowpack_df.to_csv(out_path + f"{regions[reg_code]}_SNOWPACK_Stability_TimeseriesHourly_" +
                                                                             f"Between{h_low}_{h_hi}m_{slope_path}.csv")


#%% extract the data at 12:00 pm (like in PG22)
snowpack_df12 = snowpack_df[snowpack_df.index.hour == 12]
snowpack_df15 = snowpack_df[snowpack_df.index.hour == 15]  # for the LWC params take 15:00 as in Hendrick et al. (2023)
snowpack_df12.set_index(snowpack_df12.index.date, inplace=True)
snowpack_df15.set_index(snowpack_df15.index.date, inplace=True)
for lwc_p in ["lwc_i", "lwc_sum", "lwc_max", "lwc_s_top"]:
    snowpack_df12.loc[:, lwc_p] = snowpack_df15[lwc_p]  # replace the value at 12:00 with the value at 15:00
# end for lwc_p

# calculate the LWC differences over 1 to 3 days
snowpack_df12_diff1 = snowpack_df12.diff(periods=1)
snowpack_df12_diff2 = snowpack_df12.diff(periods=2)
snowpack_df12_diff3 = snowpack_df12.diff(periods=3)

for k in snowpack_df12.keys():
    snowpack_df12_diff1.rename(columns={k:k + "_d1"}, inplace=True)
    snowpack_df12_diff2.rename(columns={k:k + "_d2"}, inplace=True)
    snowpack_df12_diff3.rename(columns={k:k + "_d3"}, inplace=True)
# end for k

# calculate the snow depth difference over 1 to 3 days
depth_df_day_diff1 = depth_df_day.diff(periods=1)
depth_df_day_diff2 = depth_df_day.diff(periods=1)
depth_df_day_diff3 = depth_df_day.diff(periods=1)

depth_df_day_diff1.fillna(0)
depth_df_day_diff2.fillna(0)
depth_df_day_diff3.fillna(0)

depth_df_day_diff1.rename(columns={"snow_depth":"snow_depth_d1"}, inplace=True)
depth_df_day_diff2.rename(columns={"snow_depth":"snow_depth_d2"}, inplace=True)
depth_df_day_diff3.rename(columns={"snow_depth":"snow_depth_d3"}, inplace=True)

snowpack_day_df = pd.concat([snowpack_df12, depth_df_day, depth_df_3d, depth_df_7d,
                             snowpack_df12_diff1, snowpack_df12_diff2, snowpack_df12_diff3,
                             depth_df_day_diff1, depth_df_day_diff2, depth_df_day_diff3],
                            axis=1)


#%% make sure that the last year is only stored until 30 June
if cut_30june:
    snowpack_day_df.index = pd.to_datetime(snowpack_day_df.index)
    snowpack_day_df = snowpack_day_df[snowpack_day_df.index < date_dt(end_yr, 7, 1)]
# end if


#%% test plots
if args.plot:
    pl.plot(snowpack_day_df["Sk38_100"])
    pl.show()
    pl.close()
# end if


#%% test plots
if args.plot:
    pl.plot(snowpack_day_df["snow_depth"])
    pl.show()
    pl.close()
# end if


#%% store the index dataframe as .csv
snowpack_day_df.to_csv(out_path2 + f"{regions[reg_code]}_SNOWPACK_Stability_TimeseriesDaily_" +
                                                                             f"Between{h_low}_{h_hi}m_{slope_path}.csv")