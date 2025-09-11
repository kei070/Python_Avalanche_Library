#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the NORA3-based EURO-CORDEX predictors across the elevation bands.
"""


#%% imports
import sys
import numpy as np
import pandas as pd
from ava_functions.Data_Loading import load_features
from ava_functions.Lists_and_Dictionaries.Paths import path_par, path_par3
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set the region code
reg_code = 3013  # int(sys.argv[1])  # 3010
agg_type = "mean"
a_p = "all"


#%% set paths
data_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/EURO-CORDEX/Mean/"



#%% loop over the eleveation bands and load the features
features = []
for h_lo, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):

    fn_in = f"Between{h_lo}_and_{h_hi}m/{regions[reg_code]}_EU_Predictors_MultiCellMean_" + \
                                                                                        f"Between{h_lo}_and_{h_hi}m.csv"

    # load the data
    try:

        features.append(load_features(data_path + fn_in))
    except:
        print(f"\nNo data for {reg_code} between {h_lo} and {h_hi}. Continuing...\n")
        continue
    # end try except

# end for h_low, h_hi


#%% set lists of features over which maxima and minima should be calculated
max_feats = ['t_mean', 't_max', 't_range', 'ftc', 't3', 't7', 'tmax3', 'tmax7', 'dtemp1', 'dtemp2', 'dtemp3', 'dtempd1',
             'dtempd2', 'dtempd3', 'pdd', 'rr', 'rr3', 'rr7']
min_feats = ['t_min', 'ws_min', 'rr', 'rr3', 'rr7']


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
result_feats.to_csv(data_path + f"{regions[reg_code]}_EU_ElevAgg_Predictors_MultiCellMean.csv")