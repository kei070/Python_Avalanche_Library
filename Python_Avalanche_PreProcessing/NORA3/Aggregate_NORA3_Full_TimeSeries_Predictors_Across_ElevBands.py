#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the NORA3 predictive features from several elevation bands (and aspects) by selecting the most extreme index
per region across the elevation bands and aspects.
This is similar to Aggregate_SNOWPACK_Stab_Inds_Across_ElevBands.py.
"""

#%% imports
import numpy as np
import pandas as pd
from ava_functions.Data_Loading import load_feats_xlevel, load_features
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set the region code
reg_code = 3009
agg_type = "mean"
perc = 0


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% set paths
data_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/{agg_str}/"


#%% loop over the eleveation bands and load the features
features = []
for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):

    fn_in = f"Between{h_low}_and_{h_hi}m/{regions[reg_code]}_Predictors_MultiCellMean_Between{h_low}_and_{h_hi}m.csv"

    try:

        features.append(load_features(data_path + fn_in))
    except:
        print(f"\nNo data for {reg_code} between {h_low} and {h_hi}. Continuing...\n")
        continue
    # end try except

# end for h_low, h_hi


#%% set lists of features over which maxima and minima should be calculated
max_feats = ['t_mean', 't_max', 't_range', 'ftc', 't3', 't7', 'tmax3', 'tmax7', 'dtemp1', 'dtemp2', 'dtemp3', 'dtempd1',
             'dtempd2', 'dtempd3', 'pdd', 'ws_mean', 'ws_max', 'ws_range', 'dws1', 'dws2', 'dws3', 'dwsd1', 'dwsd2',
             'dwsd3', 'wind_direction', 'dwdir', 'dwdir1', 'dwdir2', 'dwdir3', 'w3', 'wmax3', 'wmax7', 'total_prec_sum',
             's1', 'r1', 'r3', 'r7', 's3', 's7', 'wdrift', 'wdrift_3', 'wdrift3', 'wdrift3_3', 'RH', 'NLW', 'NSW',
             'RH3', 'RH7', 'NLW3', 'NLW7', 'NSW3', 'NSW7']
min_feats = ['t_min', 'ws_min', 's1', 'r1', 'r3', 'r7', 's3', 's7', 'RH', 'NLW', 'RH3', 'RH7', 'NLW3', 'NLW7']


#%% loop over the features and align them
columns_add = ["ava_clim", "reg_code"]
feature_names = features[0].columns
result_feat_dict = {}
for feat in feature_names:

    onef_list = [f[feat] for f in features]

    onef_df = pd.concat(onef_list, axis=1)

    if feat in max_feats:
        result_feat_dict[feat + "_emax"] = np.max(onef_df, axis=1)
    if feat in min_feats:
        result_feat_dict[feat + "_emin"] = np.min(onef_df, axis=1)
    if feat in columns_add:
        result_feat_dict[feat] = np.mean(onef_df, axis=1)
    else:
        continue
        # result_feat_dict[feat] = np.mean(onef_df, axis=1)
    # end if elif

# end for feat


#%% aggregate into a dataframe
result_feats = pd.DataFrame(result_feat_dict)


#%% store the features
result_feats.to_csv(data_path + f"{regions[reg_code]}_ElevAgg_Predictors_MultiCellMean.csv")
