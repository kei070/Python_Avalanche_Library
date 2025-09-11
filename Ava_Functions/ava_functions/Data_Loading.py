#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to load data in specific formats.
"""

# imports
import sys
import copy
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import proprietary functions
from .Balance_Data import balance_data
from .Helpers_Load_Data import extract_reg, extract_sea
from .Discrete_Hist import disc_hist

# import predefined parameters from script
from .Lists_and_Dictionaries.Features import features as all_features
from .Lists_and_Dictionaries.Features import nora3_clean
from .Lists_and_Dictionaries.Avalanche_Problems import ava_ps
from .Lists_and_Dictionaries.Region_Codes import regions
from .Lists_and_Dictionaries.Paths import path_par
from .DatetimeSimple import date_dt


# function for loading the aggregated features for NORA3 WITH ADLs
def load_agg_feats_adl(path=path_par, reg_codes=3009, features=slice(None), ndlev=4, agg_type="mean", perc=90):
    """
    Brief simple function for loading the predictive features for avalanche danger forecasting including the ADLs.
    """

    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        perc = 0
    # end if

    agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"

    if type(reg_codes) == int:
        reg_codes = [reg_codes]
    # end if

    # loop over the region codes
    feats_out = []
    for reg_code in reg_codes:

        # set file name
        fn = f"/IMPETUS/NORA3/Avalanche_Predictors_{ndlev}Level/{agg_str}/" + \
                                       f"Features_{ndlev}Level_All_{agg_str}_ElevAgg_{reg_code}_{regions[reg_code]}.csv"

        # load file
        feats_df = pd.read_csv(path+fn, index_col=0, parse_dates=True)[features]
        feats_df[feats_df.isna()] = 0

        feats_out.append(feats_df)

    # end for reg_code

    # concatenate the features
    feats_out = pd.concat(feats_out, axis=0)

    return feats_out

# end def


# function for loading the aggregated features for NORA3 WITHOUT
def load_agg_feats_no3(path=path_par, reg_codes=3009, features=slice(None), agg_type="mean", perc=90,
                       add_ava_clim=True):
    """
    Brief simple function for loading the predictive features for avalanche danger forecasting.
    """

    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        perc = 0
    # end if

    agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"

    if type(reg_codes) == int:
        reg_codes = [reg_codes]
    # end if

    # loop over the region codes
    feats_out = []
    for reg_code in reg_codes:

        # set file name
        fn = f"/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/{agg_str}/" + \
                                       f"{regions[reg_code]}_ElevAgg_Predictors_MultiCellMean.csv"

        # load file
        feats_df = pd.read_csv(path+fn, index_col=0, parse_dates=True)[features]
        feats_df[feats_df.isna()] = 0

        if "reg_code" not in feats_df.columns:
            feats_df["reg_code"] = np.repeat(reg_code, len(feats_df))
        # end if

        if add_ava_clim:
            if "ava_clim" not in feats_df.columns:
                ava_clim = 1
                if reg_code == 3010:
                    ava_clim = 2
                elif reg_code in [3009, 3013]:
                    ava_clim = 3
                # end if elif
                feats_df["ava_clim"] = np.repeat(ava_clim, len(feats_df))
            # end if
        # end if

        feats_out.append(feats_df)

    # end for reg_code

    # concatenate the features
    feats_out = pd.concat(feats_out, axis=0)

    return feats_out

# end def


# function for loading the aggregated features for NorCP
def load_agg_feats_ncp(path=path_par, reg_codes=3009, model="EC-Earth", scen="rcp45", period="MC",
                           features=slice(None), agg_type="mean", perc=90):
    """
    Brief simple function for loading the predictive features for avalanche danger forecasting.
    """

    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        perc = 0
    # end if

    agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"

    per_str = period
    if len(period) > 0:
        per_str = f"_{period}"
    # end if

    # set up the NorCP simulation string
    sim = f"{model}_{scen}{per_str}"

    if type(reg_codes) == int:
        reg_codes = [reg_codes]
    # end if

    # loop over the region codes
    feats_out = []
    for reg_code in reg_codes:

        # set file name
        fn = f"/IMPETUS/NorCP/Avalanche_Region_Predictors/{agg_str}/{sim}/" + \
                                                 f"{regions[reg_code]}_NorCP_{sim}_ElevAgg_Predictors_MultiCellMean.csv"

        # load file
        feats_df = pd.read_csv(path+fn, index_col=0, parse_dates=True)[features]
        feats_df[feats_df.isna()] = 0

        if "reg_code" not in feats_df.columns:
            feats_df["reg_code"] = np.repeat(reg_code, len(feats_df))
        # end if
        feats_out.append(feats_df)

    # end for reg_code

    # concatenate the features
    feats_out = pd.concat(feats_out, axis=0)

    return feats_out

# end def


# function for simple feature loading
def load_features(path, features=slice(None)):
    """
    Brief simple function for loading the predictive features for avalanche danger forecasting.
    """

    feats_df = pd.read_csv(path)
    feats_df["date"] = pd.to_datetime(feats_df["date"])
    feats_df.set_index("date", inplace=True)
    feats_df = feats_df[features]
    feats_df[feats_df.isna()] = 0

    return feats_df

# end def


# function for simple feature loading
def load_features2(path=path_par, ndlev=4, reg_codes=3009, h_low=400, h_hi=900,
                   features=slice(None), agg_type="mean", perc=90, nan_handling="drop"):
    """
    Brief simple function for loading the predictive features for avalanche danger forecasting.

    Parameters:
        path  String.

    """

    if type(features) == str:
        if features == "NORA3":
            features = list(nora3_clean.keys())
        # end if
    # end if

    # if reg_code is all set it to the list of all available regions
    if type(reg_codes) == str:
        reg_codes = list(regions.keys()) if reg_codes == "all" else sys.exit("reg_code not available. Aborting.")
    # end if

    # if the reg_code parameter is not a list convert it to one
    reg_codes = [reg_codes] if type(reg_codes) != list else reg_codes

    # handle the aggregation string
    # generate a name prefix/suffix depending on the gridcell aggregation
    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        perc = 0
    # end if
    agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"

    # loop over the regions
    out_data = []
    for reg_code in reg_codes:
        fn = f"/IMPETUS/NORA3/Avalanche_Predictors_{ndlev}Level/{agg_str}/Between{h_low}_and_{h_hi}m/"
        fn = fn + f"Features_{ndlev}Level_All_{agg_str}_Between{h_low}_{h_hi}m_{reg_code}_{regions[reg_code]}.csv"

        feats_df = pd.read_csv(path+fn, index_col=0, parse_dates=True)[features]
        feats_df[feats_df.isna()] = 0

        out_data.append(feats_df)
    # end for reg

    # convert to dataframe
    out_data = pd.concat(out_data, axis=0)

    return out_data

# end def


# function for loading the SNOWPACK-derived stability indices for NORA3
def load_snowpack_stab(path=path_par, reg_codes=3009, slope_angle="agg", slope_azi="agg", verbose=True):

    # generate the strings based on slope and aspect
    slope_n = "ElevAgg"
    slope_path = "Flat"
    if ((slope_angle == "agg") | (slope_azi == "agg")):
        slope_n = "ElevSlopeAgg"
        slope_path = ""
    elif slope_angle > 0:
        aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

        slope_path = f"{slope_angle}" + aspect
    # end if elif


    # if reg_code is all set it to the list of all available regions
    if type(reg_codes) == str:
        reg_codes = list(regions.keys()) if reg_codes == "all" else sys.exit("reg_code not available. Aborting.")
    # end if

    # if the reg_code parameter is not a list convert it to one
    reg_codes = [reg_codes] if type(reg_codes) != list else reg_codes

    if verbose:
        print("\nLoading data from:")
        print(f"/IMPETUS/NORA3/Snowpack/Timeseries/Daily/{slope_path}/" + "\n")
    # end if

    # loop over the regions
    out_data = []
    for reg_code in reg_codes:
        fn = f"/IMPETUS/NORA3/Snowpack/Timeseries/Daily/{slope_path}/"
        fn = fn + f"{regions[reg_code]}_SNOWPACK_Stability_TimeseriesDaily_{slope_n}.csv"

        feats_df = pd.read_csv(path+fn, index_col=0, parse_dates=True)
        feats_df[feats_df.isna()] = 0

        # add the region code as a column for disambiguation
        feats_df["reg_code"] = np.repeat(reg_code, repeats=len(feats_df))

        out_data.append(feats_df)
    # end for reg

    # convert to dataframe
    out_data = pd.concat(out_data, axis=0)

    return out_data

# end def


# function for loading the SNOWPACK-derived stability indices for NorCP
def load_snowpack_stab_ncp(path=path_par, model="EC-Earth", scen="rcp45", period="MC", reg_codes=3009):

    per_str = period
    if len(period) > 0:
        per_str = f"_{period}"
    # end if

    # set up the NorCP simulation string
    sim = f"{model}_{scen}{per_str}"

    # if reg_code is all set it to the list of all available regions
    if type(reg_codes) == str:
        reg_codes = list(regions.keys()) if reg_codes == "all" else sys.exit("reg_code not available. Aborting.")
    # end if

    # if the reg_code parameter is not a list convert it to one
    reg_codes = [reg_codes] if type(reg_codes) != list else reg_codes

    # loop over the regions
    out_data = []
    for reg_code in reg_codes:
        fn = f"/IMPETUS/NorCP/Snowpack/Timeseries/Daily/{sim}/Flat/"
        fn = fn + f"{regions[reg_code]}_{sim}_SNOWPACK_Stability_TimeseriesDaily_ElevAgg_Flat.csv"

        feats_df = pd.read_csv(path+fn, index_col=0, parse_dates=True)
        feats_df[feats_df.isna()] = 0

        # add the region code as a column for disambiguation
        feats_df["reg_code"] = np.repeat(reg_code, repeats=len(feats_df))

        out_data.append(feats_df)
    # end for reg

    # convert to dataframe
    out_data = pd.concat(out_data, axis=0)

    return out_data

# end def



# function for loading the predictive features with the danger level aggregated to x levels
# performs a train-test split
def load_xlevel_preds(data_path, sel_feats, reg_code, a_p="y", split=0.33, sea="Full", nlevels=2, nan_handling="drop",
                      balance_meth="SMOTE", sample_strat="auto", scale_x=False, verbose=True):

    """
    Function for loading the predictors (implictly performing a train-test split) for a given danger level number.

    Parameters:
        data_path
        sel_feats List of the features that will be loaded.
        reg_code  Region code (3009, 3010, ...)
        a_p            String. The avalanche problem for which the danger level is chosen. Defaults to "y",
                               implying the general danger level. The names of the available avalanche problems are
                               stored in Avalanche_Problems.py under Lists_and_Dictionaries.
        split     Possibility (1): 0. No train-test split is performed. Note that this changes the output. Note that
                                   this option is itntroduced for special purposes in can (so far) not be used as an
                                   option in scripts like RF_DT_FeatureImportance or Train_StatisticalModel.
                  Possibility (2): Float between 0 and 1 (excluding these boundaries) to perform the train-test split
                                   using the scikit-learn function train_test_split. The value given to split then
                                   represents the fraction of the data that is used as test data.
                  Possibility (3): Integer representing the year of the winter to be excluded. Note that the year is
                                   interpreted as the year in which the season ends. That is, for the winter of 2019/20
                                   one must submit 2020.
                  Possibility (4): Integer representing the region code of the region to be extracted as test data and
                                   removed from the training data. Note that this works slightly differently than the
                                   other two methods as the data must be balanced only AFTER the requested region has
                                   been extracted as test data.
                  Default: 0.33
        sea:          String. Full (default) for the full avalanche season, winter for Dec-Feb, and spring for Mar-May.
        nlevels       Integer. Number of danger levels to convert the raw data to. For 2, levels 1 & 2 are set to 0 and
                               3-5 are set to 1. For 3, level 1=0, 2=1, 3-5=2. For 4, level 1=0, 2=1, 3=2, 4&5=3.
        nan_handling  String. Controls how NaNs are handled. Two possibilities: 1) "drop" (default), i.e., dropping all
                              rows where the danger level is NaN and 2) "zero", i.e., converting the NaNs to 0s.
        balance_meth  String. The method used for balancing the data. Choices are the following:
                               -None data are not balanced
                               -undersample: uses the custom undersample function NOT FUNCTIONAL
                               -SMOTE: (default) uses the synthetic minority oversampling method from the
                                       imbalanced-learn library
                               -BSMOTE: BorderlineSMOTE
                               -SVMSMOTE: SMOTE using an SVM
                               -ADASYN: uses the adaptive synthetic sampling method from the imbalanced-learn library
                               -ros: uses the random oversampling method from the imbalanced-learn library
                               -rus: uses the random undersampling method from the imbalanced-learn library
                               -SMOTEENN: combination of over- and undersampling
                               -SMOTETomek: combination of over- and undersampling
                              There is considerable difference in procedure between the undersample case and the other
                              cases: In the undersample case the predictors have already been balanced (script
                              Gen_Store_XLevel_Balanced_Predictors.py). In the other cases the predictors are only
                              balanced here.
       sample_strat   Float, String, or dict: The strategy used in the class balancing algorithm.
                                              Float only for binary classification. String possibilities: auto, all,
                                              minority, not_majority, not_minority. dict: Keys indicate the targeted
                                              classes and the values to desired number of samples per class.
       scale_x        Logical. If true, the predictors will be scaled using the scikit-learn scaler. If false (default),
                              the predictors are used without being scaled.
       verbose        Logical. If True (default) print statements will made. If False they are suppressed.
    """

    """ --> probably remove this part
    if balance_meth == "undersample":

        #% load the data
        df = pd.read_csv(glob.glob(data_path + f"Features_{nlevels}Level_Balanced_*_{reg_code}*.csv")[0])
        all_df = pd.read_csv(glob.glob(data_path + f"Features_{nlevels}Level_All_*_{reg_code}*.csv")[0])


        #% convert the date to datetime
        df.date = pd.to_datetime(df.date)
        all_df.date = pd.to_datetime(all_df.date)
        df.set_index("date", inplace=True)
        all_df.set_index("date", inplace=True)

        # perform the scaling if requested
        df_x = df[sel_feats]
        all_df_x = all_df[sel_feats]
        if scale_x:
            scaler = StandardScaler()
            df_x_sc = scaler.fit_transform(df_x)
            df_x = pd.DataFrame({k:df_x_sc[:, i] for i, k in enumerate(df_x.columns)})
            df_x.set_index(df.index, inplace=True)

            all_df_x_sc = scaler.fit_transform(all_df_x)
            all_df_x = pd.DataFrame({k:all_df_x_sc[:, i] for i, k in enumerate(all_df_x.columns)})
            all_df_x.set_index(all_df.index, inplace=True)
        # end if

        df = pd.concat([df_x, df["y_balanced"]], axis=1)
        all_df = pd.concat([all_df_x, all_df["reg_code"], all_df["y"]], axis=1)

        #% extract the subseason (or not)
        if sea == "winter":
            df = df[((df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2))]
            all_df = all_df[((all_df.index.month == 12) | (all_df.index.month == 1) | (all_df.index.month == 2))]
        elif sea == "spring":
            df = df[((df.index.month == 3) | (df.index.month == 4) | (df.index.month == 5))]
            all_df = all_df[((all_df.index.month == 3) | (all_df.index.month == 4) | (all_df.index.month == 5))]
        # end if elif

        # if the split is not supposed to by OWO or ORO perform it with the standard function from scikit-learnd
        if split == 0:  # no train-test split

            if verbose:
                print("\nNo train-test split performed..\n")
            # end if

            #% extract the required features and prepare the data for the decision tree
            odata_x = df[sel_feats]
            odata_y = df["y_balanced"]

            #% make sure the data are balanced; note that they might not be because of the sub-season extraction
            # --> in case of the full season this should not have any effect
            bal_x, bal_y = balance_data(odata_x, odata_y, method="undersample")

            all_x = all_df[sel_feats]
            all_y = all_df["y"]

        elif (split > 0) & (split < 1):  # split = fraction of data to be extracted as test data

            if verbose:
                print(f"\n{split*100}% of data is randomly extracted as test data.\n")
            # end if

            #% extract the required features and prepare the data for the decision tree
            odata_x = df[sel_feats]
            odata_y = df["y_balanced"]

            #% make sure the data are balanced; note that they might not be because of the sub-season extraction
            # --> in case of the full season this should not have any effect
            odata_x, odata_y = balance_data(odata_x, odata_y, method="undersample")

            odata_x_all = all_df[sel_feats]
            odata_y_all = all_df["y"]

            train_x, test_x, train_y, test_y = train_test_split(odata_x, odata_y, test_size=split, shuffle=True,
                                                                stratify=odata_y)
            train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(odata_x_all, odata_y_all,
                                                                                test_size=split, shuffle=True,
                                                                                stratify=odata_y_all)
        elif split < 3000:  # split = year to be extracted as test data

            if verbose:
                print(f"\nYear {split} is extracted as test data.\n")
            # end if

            # extract the data for the requested avalanche season
            test_inds = (df.index > date_dt(split-1, 7, 1)) & (df.index < date_dt(split, 7, 1))
            test_all_inds = (all_df.index > date_dt(split-1, 7, 1)) & (all_df.index < date_dt(split, 7, 1))
            test_df = df[test_inds]
            test_all_df = all_df[test_all_inds]
            train_df = df[~test_inds]
            train_all_df = all_df[~test_all_inds]

            #% extract the required features and prepare the data for the decision tree
            train_x = train_df[sel_feats]
            train_y = train_df["y_balanced"]
            test_x = test_df[sel_feats]
            test_y = test_df["y_balanced"]

            #% make sure the data are balanced; note that they might not be because of the sub-season extraction
            # --> in case of the full season this should not have any effect
            train_x, train_y = balance_data(train_x, train_y, method="undersample")
            test_x, test_y = balance_data(test_x, test_y, method="undersample")

            train_x_all = train_all_df[sel_feats]
            train_y_all = train_all_df["y"]
            test_x_all = test_all_df[sel_feats]
            test_y_all = test_all_df["y"]

        else:  # split = region to be extracted as test data
            if verbose:
                print(f"\nRegion {split} is extracted as test data.\n")
            # end if

            # extract the data for the requested region
            # --> this only makes sense for the unbalanced data since the balanced data were randomly selected from the
            #     regions
            # --> this means we first extract the data from the unbalanced files and perform the train-test splint and
            #     balance the data only after
            test_all_inds = all_df["reg_code"] == split
            test_all_df = all_df[test_all_inds]
            train_all_df = all_df[~test_all_inds]

            # extract x and y data
            train_x_all = train_all_df[sel_feats]
            train_y_all = train_all_df["y"]
            test_x_all = test_all_df[sel_feats]
            test_y_all = test_all_df["y"]

            # generate the balanced data
            train_x, train_y = balance_data(train_x_all, train_y_all, method="undersample")
            test_x, test_y = balance_data(test_x_all, test_y_all, method="undersample")

        # end if elif else
    """

    # else:  # balance the data using scikit-learn's methods (ros, SMOTE, ADASYN, ...)

    # make sure that split is a list
    if type(split) != list:
        split = [split]
    # end if

    all_df = pd.read_csv(glob.glob(data_path + f"Features_{nlevels}Level_All_*_{reg_code}*.csv")[0])

    #% convert the date to datetime
    all_df.date = pd.to_datetime(all_df.date)
    all_df.set_index("date", inplace=True)

    # perform the scaling if requested
    all_df_x = all_df[sel_feats]
    if scale_x:
        scaler = StandardScaler()
        all_df_x_sc = scaler.fit_transform(all_df_x)
        all_df_x = pd.DataFrame({k:all_df_x_sc[:, i] for i, k in enumerate(all_df_x.columns)})
        all_df_x.set_index(all_df.index, inplace=True)
    # end if

    all_df = pd.concat([all_df_x, all_df["reg_code"], all_df[a_p]], axis=1)

    # rename the a_p column to "y" for later simpler use
    if a_p != "y":
        all_df.rename(columns={a_p:"y"}, inplace=True)
    # end if

    # HOW SHOULD NANs BE TREATED?
    # --> NaN essentially means the avalanche problem in question was not identified on that day
    if nan_handling == "drop":
    # 1) drop them
        all_df.dropna(axis=0, inplace=True)
    elif nan_handling == "zero":
    # 2) convert to zero
        all_df[all_df.isna()] = 0
    # end if

    #% extract the subseason (or not)
    if sea == "winter":
        all_df = all_df[((all_df.index.month == 12) | (all_df.index.month == 1) | (all_df.index.month == 2))]
    elif sea == "spring":
        all_df = all_df[((all_df.index.month == 3) | (all_df.index.month == 4) | (all_df.index.month == 5))]
    # end if elif

    # extract the data
    all_x = all_df[sel_feats]
    all_y = all_df["y"]

    # balance the data if requested
    if str(balance_meth) != "None":
        bal_x, bal_y = balance_data(all_x, all_y, method=balance_meth, sample_strat=sample_strat)
    # end if


    if split[0] == 0:  # no train-test split

        if verbose:
            print("\nNo train-test split performed..\n")
        # end if

    elif (split[0] > 0) & (split[0] < 1):  # split = fraction of data to be extracted as test data
        if verbose:
            print(f"\n{split[0]*100}% of data is randomly extracted as test data.\n")
        # end if

        # perform the split into training and test data
        if str(balance_meth) != "None":
            train_x, test_x, train_y, test_y = train_test_split(bal_x, bal_y, test_size=split[0], shuffle=True,
                                                                stratify=bal_y)
        # end if
        train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(all_x, all_y, test_size=split[0],
                                                                            shuffle=True, stratify=all_y)

    elif split[0] < 3000:  # split = year to be extracted as test data
        if verbose:
            print(f"\nYear(s) {split} is/are extracted as test data.\n")
        # end if

        if str(balance_meth) == "None":
            train_x_all, test_x_all, train_y_all, test_y_all = extract_sea(all_df, sel_feats, split, balance_meth)
        else:
            train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all =\
                                                                extract_sea(all_df, sel_feats, split, balance_meth)
        # end if else
    else:  # split = region to be extracted as test data
        if verbose:
            print(f"\nRegion(s) {split} is/are extracted as test data.\n")
        # end if

        if str(balance_meth) == "None":
            train_x_all, test_x_all, train_y_all, test_y_all = extract_reg(all_df, sel_feats, split, balance_meth)
        else:
            train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all =\
                                                                extract_reg(all_df, sel_feats, split, balance_meth)
        # end if else
    # end if elif else

    # return
    if split[0] != 0:  # train-test split performed
        if str(balance_meth) == "None":
            return train_x_all, test_x_all, train_y_all, test_y_all
        else:
            return train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all
        # end if else
    else:  # NO train-test split performed
        if str(balance_meth) == "None":
            return all_x, all_y
        else:
            return bal_x, bal_y, all_x, all_y
        # end if else
    # end if else

# end def


# function for loading the predictive features with specific danger-level setup, not performing a train-test split
def load_feats_xlevel(reg_codes, ndlev=2, exposure=None, sel_feats=None,
                      a_p="danger_level",
                      start_date="09-2017", end_date="09-2023",
                      h_low=400, h_hi=900,
                      agg_type="mean",
                      perc=90,
                      out_type="array",
                      plot_y=False,
                      data_path_par=f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors/",
                      pred_pl_path=""):

    """
    Parameters:
        reg_codes      List of integers. Region code(s) of the avalanche regions.
        ndlev          Number of danger levels. Possible levels are 2, 3, and 4.
        exposure       Either "west" or "east". Use only grid cells with western or eastern exposure, respectively.
                       Defaults to None, meaning all available grid cells are used. NOT USED
        sel_feats      List of features to be loaded.
        a_p            String or List. The avalanche problem for which the danger level is chosen. Defaults to
                               "danger_level", implying the general danger level. Use "all" to load all avalanche
                               problems. Or use a list of requested avalanche problems. Note that if more than one
                               avalanche problem is used no balancing is performed and the balanced data are returned
                               as NaN.
        agg_type       String. The type of the aggregation of the grid cells. Either mean, median, or percentile.
                               Defaults to mean.
        p              Integer. The percentile if the grid-cell aggregation type is percentile. Not used if the agg_type
                                is not percentile.
        out_type       String. Output type, either "array" or "dataframe" (selfexplanatory). Note that the output is a
                               dictionary in both cases, but either a dictionary of numpy arrays or pandas dataframes.
                               Note also that the keys are different for the two options. Defaults to "array".
        plot_y         Boolean. If True, a bar-plot of the y-values will be produced.
        data_path_par  Parent data path for the predictors/features.

    More information on the parameters:
        reg_code = 3009  # Nord-Troms
        reg_code = 3010  # Lyngen
        reg_code = 3011  # Tromsoe
        reg_code = 3012  # Soer-Troms
        reg_code = 3013  # Indre Troms
    """

    # if no features to select are provided use all features available
    if sel_feats == None:
        sel_feats = all_features
    elif sel_feats == "NORA3"    :
        sel_feats = list(nora3_clean.keys())
    # end if

    # if a_p == "all" load the list of avalanche problems, else make sure a_ps is a list
    if type(a_p) is list:
        a_ps = copy.deepcopy(a_p)

        # prepare the output
        a_p_out = ["y"] if a_p == ["danger_level"] else a_p
    elif a_p == "all":
        a_ps = copy.deepcopy(ava_ps)
        a_ps.insert(0, "danger_level")

        # prepare the output
        a_p_out = copy.deepcopy(ava_ps)
        a_p_out.insert(0, "y")
    else:
        a_ps = [a_p]

        # prepare the output
        a_p_out = ["y"] if a_p == "danger_level" else [a_p]
    # end if else

    # generate a name prefix/suffix depending on the gridcell aggregation
    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        perc = 0
    # end if

    agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"

    # use western/eastern exposure condition?
    if exposure == "west":
        expos = "w_expos"
        expos_add = "_WestExposed"
    elif exposure == "east":
        expos = "e_expos"
        expos_add = "_EastExposed"
    else:
        expos_add = ""
    # end if

    if type(reg_codes) == int:
        reg_codes = [reg_codes]
    # end if

    # loop over the region codes and load the data
    data_x_l = []
    data_y_l = []

    dates_l = []
    dates_l = []

    reg_code_l = []

    for reg_code in reg_codes:

        # set region name according to region code
        region = regions[reg_code]

        # NORA3 data path
        data_path = data_path_par

        # load the data
        data_fns = sorted(glob.glob(data_path + f"/{agg_str}/Between{h_low}_and_{h_hi}m" +
                                  f"/{region}_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}m{expos_add}.csv"))
        if len(data_fns) == 0:
            print(f"No predictors for {reg_code} between {h_low} and {h_hi} m. Continuing.")
            continue
        # end if

        # select and load a dataset
        data_i = 0
        data_fn = data_fns[data_i]

        data = pd.read_csv(data_fn)

        # convert date column to datetime
        data["date"] = pd.to_datetime(data["date"])

        # select the training data from training data set
        data_sub = data[sel_feats]

        # add the x and y data to list
        data_x_l.append(data_sub)
        data_y_l.append(np.array(data[a_ps]))

        # store the dates of the train and test data
        dates_l.append(data.date)

        # add the region coed
        reg_code_l.append(data["region"])

    # end for reg_code

    if len(data_x_l) == 0:
        sys.exit()
    # end if

    x_df = pd.concat(data_x_l)
    x_df["reg_code"] = pd.concat(reg_code_l, axis=0)
    data_y = np.concatenate(data_y_l)

    dates = pd.concat(dates_l)

    # convert the predictands to binary, 3-level, or 4-level
    # IMPORTANT: Like Python, we start at 0! --> the original danger-level list starts at 1
    y_bin = np.zeros(np.shape(data_y))

    if ndlev == 2:
        y_bin[data_y > 2] = 1
        # remainder is zero
    elif ndlev == 3:
        y_bin[data_y == 2] = 1
        y_bin[data_y > 2] = 2
        # remainder is zero
    elif ndlev == 4:
        y_bin[data_y == 2] = 1
        y_bin[data_y == 3] = 2
        y_bin[data_y > 3] = 3
        # remainder is zero
    # end if elif

    # make sure that instances of previous NaN remain NaN
    y_bin[np.isnan(data_y)] = np.nan

    # plot the number of the different values (0, 1, 2, 3) in the new classification
    if plot_y:
        disc_hist([y_bin])
    # end if

    # balancing can only be undertaken for one target varialbe, i.e., not for multiple avalanche problems at a time
    if len(a_ps) == 1:

        # squeeze to make the following operations possible
        # --> y_bin will be returned to its original shape at the end of this to ensure compatibility with the case
        #     without balancing
        # --> there are probably more elegant ways to do this
        y_bin = np.squeeze(y_bin)

        # first find out which level occurs the least often
        n_data = np.min([np.sum(y_bin == dlev) for dlev in np.arange(ndlev)])

        # to make sure the accuracy metrics are not just artifacts make sure there the data contain the same number of
        # danger levels
        y_bal = []
        dates_bal = []
        x_bal = []

        for dlev in np.arange(ndlev):

            # get the number of days with level dlev
            n_x = np.sum(y_bin == dlev)

            # select a random sample of length n_train as the training data -- NEW
            perm = np.random.choice(n_x, size=n_data, replace=False)

            # permute the predictands to get balanced predictands data
            y_bal.append(y_bin[y_bin == dlev][perm])

            # permutate the dates to get the dates of the balanced data
            dates_bal.append(dates.iloc[y_bin == dlev].iloc[perm])

            # extract the selected features
            x_bal.append(np.array(x_df[sel_feats])[y_bin == dlev, :][perm, :])

        # end for dlev
        y_bal = np.concatenate(y_bal)
        dates_bal = np.concatenate(dates_bal)
        x_bal = np.concatenate(x_bal)

        # return y_bin to its original shape to ensure compatibility with the code outside the if condition
        y_bin = y_bin[:, None]

    else:
        y_bal = [np.nan]
        dates_bal = [0]
        x_bal = {k:[np.nan] for k in sel_feats}
    # end if else

    x_all = np.array(x_df[sel_feats + ["reg_code"]])

    if out_type == "dataframe":
        output = {}

        # balanced data
        temp = pd.DataFrame(x_bal)
        temp.columns = sel_feats
        temp["date"] = dates_bal
        temp.set_index("date", inplace=True)
        temp["y_balanced"] = y_bal
        output["balanced"] = temp

        # all data
        temp = pd.DataFrame(x_all)
        temp.columns = sel_feats + ["reg_code"]
        temp.set_index(dates, inplace=True)
        for i, i_ap in enumerate(a_p_out):
            temp[i_ap] = y_bin[:, i]
        # end for i, i_ap
        output["all"] = temp

    elif out_type == "array":
        # add data to the function output

        # balanced data
        output = {"x_balanced":x_bal,
                  "x_all":x_all,
                  "dates_balanced":dates_bal,
                  "y_balanced":y_bal,
                  "y_all":y_bin,
                  "dates_all":dates}

        # all data
        output = {"x_all":x_all,
                  "y_all":y_bin,
                  "dates_all":dates}
    # end if elif

    # return
    return output

# end def


# function for loading seasonal predictors
def load_seasonal_preds(data_path, features, h_low=400, h_hi=900, agg_type="mean", perc=0):

    # generate a name prefix/suffix depending on the gridcell aggregation

    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        perc = 0
    # end if
    agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"

    # extract the data per region and calculate the seasonal means (full, winter, spring)
    regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}

    # set the aggregartion function
    agg_func = {k:np.mean for k in features}
    for k in [f"s{i}" for i in np.arange(8)]:
        agg_func[k] = np.sum
    for k in [f"r{i}" for i in np.arange(8)]:
        agg_func[k] = np.sum
    # end for k
    agg_func["total_prec_sum"] = np.sum

    reg_dic = {}
    for reg_code in regions.keys():
        fn = data_path + f"{regions[reg_code]}_Predictors_MultiCell{agg_str}_Between{h_low}_and_{h_hi}m.csv"
        df = pd.read_csv(fn)
        df["date"] = pd.to_datetime(df["date"])

        reg_dic[reg_code] = {k: {"full":[], "winter":[], "spring":[]} for k in features}

        years = np.unique(df.date.dt.year)[1:]
        for yr in years:
            full_inds = (df.date >= date_dt(yr-1, 12, 1)) & (df.date < date_dt(yr, 6, 1))
            win_inds = (df.date >= date_dt(yr-1, 12, 1)) & (df.date < date_dt(yr, 3, 1))
            spr_inds = (df.date >= date_dt(yr, 3, 1)) & (df.date < date_dt(yr, 6, 1))

            for var in reg_dic[reg_code].keys():
                reg_dic[reg_code][var]["full"].append(agg_func[var](df[var][full_inds]))
                reg_dic[reg_code][var]["winter"].append(agg_func[var](df[var][win_inds]))
                reg_dic[reg_code][var]["spring"].append(agg_func[var](df[var][spr_inds]))
            # end for var
        # end for yr

        # convert to Pandas DataFrame simplify later operations
        for var in reg_dic[reg_code].keys():
            reg_dic[reg_code][var] = pd.DataFrame(reg_dic[reg_code][var], index=years)
        # end for var

    # end fn

    return reg_dic
# end function


"""
Generalised function for loading a requested climate index. This should work for:
    -NAO (NOAA, Hurrell, and my proprietary ERA5-derived NAOi)
    -AO
    -AMO
    -SCA
"""
def load_ci(fname, data_path, sta_y=None, end_y=None, years=None, nan=None):

    """
    This function should work for the (1) NOAA, (2) Hurrell, (3) proprietary ERA5 NAO indices as well as the (4) NOAA
    AMO index, the (5) NOAA AO index, and the (6) NOAA SCA index.

    The NAO files are in:
        /media/kei070/One Touch/IMPETUS/NAO/
    and their names are:
        (1) NAO_Index.txt
        (2) NAO_Index_Hurrell.txt
        (3) NAOi_ERA5.txt
    The AMO index is in:
        /media/kei070/One Touch/IMPETUS/AMO/
    and its name is;
        (4) AMO_Index.txt
    The AO index is in:
        /media/kei070/One Touch/IMPETUS/AO/
    and its name is:
        (5) AO_Index.txt
    The SCA index is in:
        /media/kei070/One Touch/IMPETUS/SCA/
    and its name is:
        (6) SCA_Index.txt
    """

    # generate years array
    if years is None:
        years = np.arange(sta_y, end_y)
    # end if

    # load the index
    c_i = pd.read_csv(data_path + fname, index_col=0, engine="python", header=0, na_values=nan)

    # nao.set_index("year", inplace=True)

    # reduce the index to the time for which we have NORA3
    c_i = c_i[(c_i.index >= years[0]) & (c_i.index <= years[-1])]

    # calculate full-, winter- and spring-season mean as well as annual mean climate index
    try:  # depending on the column names
        c_is = pd.DataFrame({"full":c_i[["Dec", "Jan", "Feb", "Mar", "Apr", "May"]].mean(axis=1),
                             "winter":c_i[["Dec", "Jan", "Feb"]].mean(axis=1),
                             "spring":c_i[["Mar", "Apr", "May"]].mean(axis=1),
                             "full_yr":c_i.mean(axis=1)},
                            index=years)
    except:
        c_is = pd.DataFrame({"full":c_i[["12", "1", "2", "3", "4", "5"]].mean(axis=1),
                             "winter":c_i[["12", "1", "2"]].mean(axis=1),
                             "spring":c_i[["3", "4", "5"]].mean(axis=1),
                             "full_yr":c_i.mean(axis=1)},
                            index=years)
    # end try except

    return c_is

# end def

