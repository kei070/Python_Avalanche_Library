#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for running SNOWPACK; partly based on functions provided via AWESOME.
"""

# imports
import os
import numpy as np
import pandas as pd
from datetime import datetime
from awsmet import SMETParser
from snowpacktools.snowpro import pro_helper, pro_plotter
from snowpacktools.snowpro.snowpro import _parse_date_like


# estimate the number of lines in a .txt-like file
def estimate_lines(data_path, chunk_size=1024*1024):
    line_count = 0
    with open(data_path, "rb") as f:
        while chunk := f.read(chunk_size):
            line_count += chunk.count(b'\n')
    # end with
    return line_count
# end def


# count the number of lines in a .txt-like file
def count_lines(data_path):
    with open(data_path, "r") as f:
        return sum(1 for _ in f)
    # end with
# end def


#% load only requested lines of a text file (or a .pro file for SNOWPACK output)
def load_lines(data_path, start, end):
    """
    Start and end inclusive!
    """
    read_lines = []

    with open(data_path, "r") as f:

        for i, line in enumerate(f):
            if i > end:
                break
            elif i >= start:
                read_lines.append(line.strip())
            else:
                continue
            # end if
        # end for i, line
    # end with
    return read_lines
# end def


#% specifically for the .pro SNOWPACK output: find the last line with a starting day
def find_last_line(lines):
    """
    The last line must be the line before a line including data and time 0:00:00 or an empty line (eof).
    """

    for i, l in enumerate(lines[::-1]):
        if ((l[:4] == "0500") & (l[-8:] == "00:00:00")):
            # print(l, i)
            break
    # end for i

    # if the found line is the FIRST line in the file, this means that we have reached the end of the file and the last
    # line in lines should be the last line of the file anyway --> take the last line of the file as the last line
    if l == lines[0]:
        print("\nReached end of file (presumably: l = line[0]).\n")
        return len(lines)
    else:
        return len(lines)-i-1
    # end if else
# end def


#% load the header of a .pro file; also returns the line number of the last line of the header
def load_pro_header(data_path):
    read_lines = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if line[:6] == "[DATA]":
                read_lines.append(line.strip())
                break
            else:
                read_lines.append(line.strip())
            # end if else
        # end for
    return read_lines, i
# end def


#% load lines with header; also cuts the lines at the last line found with the last line function and returns that line
def load_lines_whead(data_path, start, end):

    # load the header
    head = load_pro_header(data_path)

    # load the data
    read_lines = load_lines(data_path, start, end)

    # find the last line BEFORE the header is added
    last_line = find_last_line(read_lines)

    # check that the header is loaded
    if start > head[1]:
        return head[0] + read_lines[:last_line], last_line
    else:
        return read_lines[:last_line], last_line
    # end if else
# end def


# an adjusted read_pro function taken from snowpacktools to include my above reading functions
def read_pro_lines(path, start, end, res='1h', keep_soil=False, consider_surface_hoar=True):
    """Reads a .PRO file and returns a dictionary with timestamps as keys and values being another dictionary with
    profile parameters as keys and data as value representing the evolving state of the snowpack.

    Arguments:
        path (str):             String pointing to the location of the .PRO file to be read
        res (str):              temporal resolution
        keep_soil (bool):       Decide if soil layers are kept
        consider_surface_hoar (bool):   Decide if surface hoar should be added as another layer
    Returns:
        profs (dict):           Dictionary with timestamps as keys and values being another dictionary with profile
                                parameters
        meta_dict (dict):       Dictionary with metadata of snow profile
        header (list):          List containing the elements of the header in the .pro file.
    """

    w, hours = pro_helper.set_resolution(res)

    PRO_CODE_DICT, VAR_CODES = pro_helper.get_pro_code_dict()
    VAR_CODES_PROF = {}

    # print(PRO_CODE_DICT)
    # print(VAR_CODES)

    """Open the PRO file and generate dict of variables with list of lines for each variable"""
    ########################################
    # --> HERE THE BIG ADJUSTMENT HAPPENS! #
    ########################################
    file_content, last_line = load_lines_whead(path, start=start, end=end)
    # --> note that this makes sure that the last line will correspond to the end of a day! That means your end line
    #     will not correspond perfectly to the actually used last line!!
    ########################################
    ########################################
    ########################################

    """Dictionary with timestamps as keys and values being another dictionary with profile parameters as keys and data as value"""
    profs = {}
    meta_dict = {}
    header = []

    """ OLD --- CHANGED --- OLD --- CHANGED --- OLD
    Open the PRO file and generate dict of variables with list of lines for each variable
    with open(path, "r") as f:
        file_content = f.readlines()
    """

    section = '[STATION_PARAMETERS]'
    for line in file_content:
        line = line.rstrip('\n')

        if section=='[DATA]':
            if line[:4] == '0500':
                # """Check that all variable lists are same length"""
                # if not_first_timestamp:
                #     for varcode in VAR_CODES:
                #         if PRO_CODE_DICT[varcode] not in profs[ts].keys():
                #             # profs[ts][PRO_CODE_DICT[varcode]] = '-999'
                #             profs[ts][PRO_CODE_DICT[varcode]] = -999

                """Check if timestamp is of interest"""
                if int(line[-8:-6]) in hours:
                    timestamp_of_interest = True
                    ts = _parse_date_like(line.strip().split(',')[1])
                    # ts = datetime.strptime( line.strip().split(',')[1], '%d.%m.%Y %H:%M:%S' )
                    profs[ts] = {}
                    not_first_timestamp = True
                else:
                    timestamp_of_interest = False

            elif line[:4] in VAR_CODES and timestamp_of_interest:
                if line[:4] == "0513": # 'grain type (Swiss Code F1F2F3)':
                    profs[ts][PRO_CODE_DICT[line[:4]]] = np.array(line.strip().split(',')[2:-1],dtype=float)
                elif line[:4] == "0501":
                    height = np.array(line.strip().split(',')[2:],dtype=float)
                    if len(height) == 1 and height.item() == 0: profs[ts][PRO_CODE_DICT[line[:4]]] = np.array([])
                    else: profs[ts][PRO_CODE_DICT[line[:4]]] = height
                elif line[:4] == "0505":
                    try:
                        profs[ts][PRO_CODE_DICT[line[:4]]] = np.array(line.strip().split(',')[2:],dtype='datetime64[D]')
                        # profs[ts][PRO_CODE_DICT[line[:4]]] = np.array(line.strip().split(',')[2:],dtype='datetime64[s]')
                    except:
                        profs[ts][PRO_CODE_DICT[line[:4]]] = np.array(line.strip().split(',')[2:],dtype=float)
                elif line[:4] == "0530":
                    pass
                else:
                    try:
                        profs[ts][PRO_CODE_DICT[line[:4]]] = np.array(line.strip().split(',')[2:],dtype=float)
                    except:
                        # print(f"Waring, Key {line[:4]} not available for Prof: {path}")
                        pass

        elif section=='[HEADER]':
            if line == '[DATA]':
                # """Drop variables that are not present in header of .PRO file"""
                # keys_to_drop = []
                # for varcode in VAR_CODES:
                #     if varcode not in VAR_CODES_PROF:
                #         print('[i]  Variable ', varcode, ' is not found in the header of this .PRO file. It is dropped.')
                #         keys_to_drop.append(varcode)

                # for key in keys_to_drop:
                #     VAR_CODES.pop(key)

                section = '[DATA]'
                timestamp_of_interest = False
                not_first_timestamp   = False
                continue
            else:
                header.append(line)
                VAR_CODES_PROF[line.split(",")[0]] = line.split(",")[-1]
                VAR_CODES = VAR_CODES_PROF.keys()
        else: # - section=='[STATION_PARAMETERS]' - #
            if line == '[HEADER]':
                section = '[HEADER]'
                continue
            else:
                line_arr = line.split('= ')
                if len(line_arr) == 2:
                    meta_dict[line_arr[0]] = line_arr[1]


    """Check again that all variable lists are same length... (for no snow at end of season)"""
    # for varcode in VAR_CODES:
    #     if PRO_CODE_DICT[varcode] not in profs[ts].keys():
    #         profs[ts][PRO_CODE_DICT[varcode]] = -999

    """Check existence of soil layers (exist for negative height values)"""
    prof_dates    = list(profs)
    first_date    = prof_dates[0]
    soil_detected = False

    # if 'height' in profs[first_date].keys():
    nheight = len(profs[first_date]['height'])
    soil_vars     = []
    i_ground_surf = 0
    if nheight > 0:
        if profs[first_date]['height'][0] < 0:
            print('[i]  Soil layers detected')
            soil_detected = True
            i_ground_surf = list(profs[first_date]['height']).index(0.0)
            print('[i]  i_ground_surf: ', i_ground_surf)
            for varname in profs[first_date].keys():
                n = len(profs[first_date][varname])
                if n+1==nheight:
                    soil_vars.append(varname)
            print('[i]  Variables with soil layers: ', soil_vars)

    if keep_soil==False and soil_detected:
        for ts in profs.keys():
            profs[ts]['height'] = profs[ts]['height'][i_ground_surf+1:]
            for var in soil_vars:
                profs[ts][var] = profs[ts][var][i_ground_surf:]

    """Calculate thickness for each layer in current snow profile"""
    for ts in profs.keys():
        """Consider surface hoar at surface"""
        if consider_surface_hoar:
            if 'grain type, grain size (mm), and density (kg m-3) of SH at surface' in profs[ts].keys():
                surf_hoar = profs[ts]['grain type, grain size (mm), and density (kg m-3) of SH at surface']
                if not np.isscalar(surf_hoar):
                    if surf_hoar[0]!=-999:
                        for var in profs[ts].keys():
                            if var == 'grain type, grain size (mm), and density (kg m-3) of SH at surface':
                                continue
                            elif var == 'height':
                                profs[ts][var] = np.append(profs[ts][var], profs[ts][var][-1] + surf_hoar[1]/10) # or np.insert()
                            elif var == 'density':
                                profs[ts][var] = np.append(profs[ts][var], surf_hoar[2])
                            elif var == 'grain size (mm)':
                                profs[ts][var] = np.append(profs[ts][var], surf_hoar[1])
                            elif var == 'grain type (Swiss Code F1F2F3)':
                                profs[ts][var] = np.append(profs[ts][var], surf_hoar[0])
                            elif var == "element deposition date (ISO)":
                                try:
                                    profs[ts][var] = np.append(profs[ts][var], np.datetime64('NaT'))
                                except:
                                    profs[ts][var] = np.append(profs[ts][var], np.nan)
                            else:
                                profs[ts][var] = np.append(profs[ts][var], np.nan)

        """Calculate thickness and bottom"""
        # if 'height' in profs[ts].keys():
        if len(profs[ts]['height']) > 0:
            # profs[ts]['height']    = profs[ts]['height'] / 100 # transform to m (not used anymore)
            profs[ts]['thickness'] = profs[ts]['height'].copy() # Catches first layer (thickness=height)
            i = np.arange(1,len(profs[ts]['height']))
            profs[ts]['thickness'][i] = profs[ts]['height'][i] - profs[ts]['height'][i-1]
            profs[ts]['bottom'] = profs[ts]['height'].copy()
            profs[ts]['bottom'][0] = 0
            profs[ts]['bottom'][i] = profs[ts]['height'][i-1]

        """Transform SLF graintype code into ICSSG standard abbreviation"""
        if 'grain type (Swiss Code F1F2F3)' in profs[ts].keys():
            if not np.isscalar(profs[ts]['grain type (Swiss Code F1F2F3)']):
                profs[ts]['graintype'] = pro_helper.slf_graintypes_to_ICSSG(profs[ts]['grain type (Swiss Code F1F2F3)'])

        """NANs"""
        # for ts in prof['data'].keys():
        #     for var in prof['data'][ts].keys():
        #         data = prof['data'][ts][var]
        #         try: prof['data'][ts][var] = np.where((data==-999),np.nan,data)
        #         except: pass

        """Turn order around that highest layer is the 'first'"""
        # df = df[::-1]
        # df = df.reset_index(drop=True)

    return profs, meta_dict, header, last_line
# end def


# generate .sno file
def gen_sno(params:dict, prof_date:str, sno_dir:str, source="NORA3"):
    """
    Generate a .sno file for running SNOWPACK.

    """

    out_params = ["timestamp", "Layer_Thick", "T", "Vol_Frac_I", "Vol_Frac_W", "Vol_Frac_V", "Vol_Frac_S",
                               "Rho_S", "Conduc_S", "HeatCapac_S", "rg", "rb", "dd", "sp", "mk", "mass_hoar", "ne",
                               "CDot", "metamo"]

    sno_path = os.path.join(sno_dir, params["filename"] + ".sno")

    header = {}

    header['station_id']              = params["filename"]
    header['station_name']            = params['name']
    header['latitude']                = str(params['lat'])
    header['longitude']               = str(params['lon'])
    header['altitude']                = str(params['elev'])
    header["nodata"]                  = str(-999)
    header['easting']                 = str(-999)
    header['northing']                = str(-999)
    header['tz']                      = str(1)
    header['source']                  = source
    header['ProfileDate']             = prof_date
    header["HS_Last"]                 = str(0.0000)
    header["SlopeAngle"]              = params["SlopeAngle"]
    header["SlopeAzi"]                = params["SlopeAzi"]
    header["nSoilLayerData"]          = str(0)
    header["nSnowLayerData"]          = str(0)
    header["SoilAlbedo"]              = str(0.09)
    header["BareSoil_z0"]             = str(0.200)
    header["CanopyHeight"]            = str(0.00)
    header["CanopyLeafAreaIndex"]     = str(0.00)
    header["CanopyDirectThroughfall"] = str(1.00)
    header["WindScalingFactor"]       = str(1.00)
    header["ErosionLevel"]            = str(0)
    header["TimeCountDeltaHS"]        = str(0.000000)

    header['fields']                  = " ".join(out_params)

    head_str = SMETParser.put_header(header)

    # write the file
    with open(sno_path, "w") as fsno:
        fsno.write(head_str + "\n")
        fsno.write("[DATA]\n")
    # end with

# end def


# TSA
def tsa(prof):

    """
    Performs the threshold sum approach (TSA) as described in Monti et al. (2012) with the threshold values given in
    Monti et al. (2014).
    Parameters:
        prof   A One SNOWPACK profile (i.e., one timestep) containing hand hardness, grain size, height, graintype. This
               should work with a typical SNOWPACK run and the output loaded with the snowpacktools function read_pro.

    Output:
        A dictionary containing the elements "tsa" (= array with 0s and 1s where a potential PWL exists in the profile)
        and "pwl_heights" (the heights of the potential PWLs). Note that the height corresponds to the height of the
        interface
    """

    # calculate grain-size and hand-hardness difference --> dhh[0] = hh[1] - hh[0]
    prof["dhh"] = prof["hand hardness"][1:] - prof["hand hardness"][:-1]
    prof["dgs"] = (prof["grain size (mm)"][1:] - prof["grain size (mm)"][:-1]) / prof["grain size (mm)"][1:]

    # get the interface depth
    prof["ld"] = -(prof["height"][:-1] - prof["height"][-1])

    # set up a dictionary for (non-)persistent grain types and use to separate the grain types
    gt_persistent = {'PP':1, 'DF':0, 'RG':0, 'FC':1, 'DH':1, 'SH':1, 'MF':0, 'IF':0, 'FCxr':1, "MFcr":0}
    prof["gt_persistent"] = np.array([gt_persistent[k[0]] for k in prof["graintype"]])

    # loop over all the interfaces and apply the TSA
    tsai = []
    for i in np.arange(len(prof["dhh"])):

        # consider the interface
        interf_count = 0
        if prof["dhh"][i] >= 1.7:  # hand hardness difference >= 1.7
            interf_count += 1
        if prof["dgs"][i] >= 0.4:  # grain size difference >= 40 %
            interf_count += 1
        if prof["ld"][i] <= 100:  # layer depth <= 100 cm
            interf_count += 1
        # end if

        # consider the lower layer
        lay1_count = 0
        if prof["hand hardness"][i] <= 1.3:
            lay1_count += 1
        if prof["grain size (mm)"][i] >= 0.6:
            lay1_count += 1
        if prof["gt_persistent"][i] > 0:
            lay1_count += 1
        # end if

        # consider the upper layer
        lay2_count = 0
        if prof["hand hardness"][i+1] <= 1.3:
            lay2_count += 1
        if prof["grain size (mm)"][i+1] >= 0.6:
            lay2_count += 1
        if prof["gt_persistent"][i+1] > 0:
            lay2_count += 1
        # end if

        # add up the counts to obtain the TSA
        tsai.append(interf_count + np.max([lay1_count, lay2_count]))

    # end for i
    tsai = np.array(tsai)

    # identify the potential PWL
    tsa = np.zeros(len(prof["ld"]))
    tsa[tsai > 4] = 1
    tsa = tsa.astype(int)

    # get the heights of the PWLs
    pwl_heights = prof["height"][:-1][tsa.astype(bool)]

    # return
    return {"tsa":tsa, "pwl_heights":pwl_heights}

# end def


# adjusted get_smet_df function from snowpacktools to load the forcing data
def get_smet_df(path, forcing=False):
    """Generates dataframe for further processing out of .smet file."""

    var = pro_helper.get_var_smet(path)

    skip = 0
    with open(path, "r", encoding="utf-8") as f:
        for skiprow, line in enumerate(f):
            if line.strip() == "[DATA]":
                break
    df_smet = pd.read_csv(path, sep=" ", skiprows=skiprow+1, skipinitialspace=True, names =var)
    df_smet = df_smet.replace(-999.0, np.nan)

    if forcing:
        variables_of_intrest = ["timestamp", "DW", "NET_SW", "PSUM", "RAIN", "RH", "SNOW", "TA", "TSG", "TSS", "VW"]
        df_smet = df_smet.loc[:, df_smet.columns.intersection(variables_of_intrest)]

        # convert the timestep to datetime format
        df_smet["timestamp"] = pd.to_datetime(df_smet["timestamp"])
        df_smet.set_index("timestamp", inplace=True)

        return df_smet
    else:
        # Reduce dataframe to variables of interest
        variables_of_intrest = ['timestamp', 'TSS_mod', 'TSS_meas', 'T_bottom', 'TSG', 'VW', 'DW', 'wind_trans24',
                                'VW_drift', 'MS_Wind', 'HS_mod', 'HS_meas', 'MS_Snow','hoar_size', 'HN72_24',
                                'HN24', 'MS_Rain', 'SWE', 'MS_Water']
        df_smet = df_smet.loc[:, df_smet.columns.intersection(variables_of_intrest)]

        # convert the timestep to datetime format
        df_smet["timestamp"] = pd.to_datetime(df_smet["timestamp"])
        df_smet.set_index("timestamp", inplace=True)

        # KUE CHANGE START (added try except clause)
        try:
            df_smet['HS_mod']  = df_smet['HS_mod']/100
            df_smet['HS_meas'] = df_smet['HS_meas']/100
            df_smet['HN72_24'] = df_smet['HN72_24']/100
            df_smet['HN24']    = df_smet['HN24']/100
            return df_smet
        except:
            print("\nHN72 and HN24 parameters not available. If you want them, set OUT_HAZ = TRUE in the .ini file\n")
            return df_smet
        # end try except

# end def