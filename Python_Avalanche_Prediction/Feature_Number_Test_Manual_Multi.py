#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the change of model skill with the number of features included in model training.
"""


#%% imports
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pylab as pl
from timeit import default_timer as timer
from joblib import load, Parallel, delayed
from sklearn.model_selection import PredefinedSplit, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support

from ava_functions.Model_Fidelity_Metrics import mod_metrics, dist_metrics
from ava_functions.Data_Loading import load_features2, load_snowpack_stab, load_agg_feats_adl
from ava_functions.Lists_and_Dictionaries.Features import se_norge_feats, nora3_clean
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Helpers_Load_Data import extract_sea
from ava_functions.StatMod import stat_mod
from ava_functions.Progressbar import print_progress_bar


#%% set up the parser
min_leaf = 10
min_split = 10

parser = argparse.ArgumentParser(
                    prog="FeatureNumberTest",
                    description="""Performs a feature number test.""",
                    epilog="For more information consult the documentation of the function feat_sel.")

# ...and add the arguments
parser.add_argument("--a_p", default="y", type=str, choices=["y", "glide_slab", "new_loose", "new_slab",
                                                                     "pwl_slab", "wet_loose", "wet_slab", "wind_slab",
                                                                     "wet"],
                    help="""The avalanche problem that is used. Standard is 'y', i.e., the general ADL.""")
parser.add_argument("--class_weight", type=str, help="""The internal class weights.
                    Must be submitted as a JSON-string that can be interpreted as a dictionary, e.g.:
                        '{0:1, 1:1}'. If empty, default values will be used.""")
parser.add_argument("--nan_handl", default="drop", type=str, choices=["drop", "zero"],
                    help="""How to handle NaNs in the danger level data.""")
parser.add_argument("--cv_type", type=str, default="seasonal", choices=["seasonal", "stratified"],
                    help="""The type of folds in the cross-validation during grid search.""")
parser.add_argument("--cv", type=int, default=3, help="""The number folds in the gridsearch cross-validation.""")
parser.add_argument("--cv_score", type=str, default="f1_macro", help="""The type of skill score used in the
                    cross-validation. For possible choices see
                    https://scikit-learn.org/stable/modules/model_evaluation.html""")
parser.add_argument("--cv_on_all", action="store_true", help="""If set, the k-fold cross-validation is performed on the
                    the whole dataset instead of only the training data.""")
parser.add_argument("--model_ty", default="RF", choices=["RF", "DT"],
                    help="""The type of statistical model that is used.""")
parser.add_argument("--ndlev", type=int, default=2, choices=[2, 3, 4],
                    help="""The number of danger levels to be predicted.""")
parser.add_argument("--split", nargs="*", default=[2021, 2023],
                    help="""Fraction of data to be used as test data.""")
parser.add_argument("--balancing", type=str, default="external", choices=["none", "internal", "external"],
                    help="""Controls the kind of balancing to be performed.""")
parser.add_argument("--balance_meth", type=str, default="SMOTE",
                    choices=["undersample", "rus", "ros", "SMOTE", "ADASYN"],
                    help="""The balancing method applied during the loading of the data. This is only relevant if
                    balancing is set to external.""")
parser.add_argument("--scale_x", action="store_true", help="Perform a predictor scaling.")
parser.add_argument("--sea", type=str, default="full", choices=["full", "winter", "spring"],
                    help="""The season for which the model is trained. The choices mean: full=Dec-May, winter=Dec-Feb,
                    spring=Mar-May.""")
parser.add_argument("--h_low", type=int, default=-1, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=-1, help="The upper threshold of the grid cell altitude.")
parser.add_argument("--reg_code", type=int, default=0, help="""The region code of the region for which the danger levels
                    will be predicted. Set 0 (default) to use all regions.""")
parser.add_argument("--agg_type", default="mean", type=str, choices=["mean", "median", "percentile"],
                    help="""The type of aggregation used for the grid cells.""")
parser.add_argument("--perc", type=int, default=90, help="""The percentile used in the grid-cell aggregation.""")

args = parser.parse_args()


#%% get the class weights dictionary
class_weight = {}
# convert the JSON string to a dictionary
if args.class_weight:
    try:
        class_weight = json.loads(args.class_weight)
    except json.JSONDecodeError:
        print("\nError while loading hyperp: Invalid JSON format\n")
    # end try except
else:
    print("\nNo internal class weighting performed.\n")
    class_weight = {k:1 for k in np.arange(args.ndlev)}
# end if else


#%% hyperparameters set and class weights

# set the class weights
cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight.values()])}"
print(f"\nUsing the following class weights: {class_weight}\n")

# two years as validation data
# splits = [[2017, 2018], [2019, 2020], [2022, 2024]] #  [2021, 2022], [2023, 2024]]
splits = [[2018], [2019], [2020], [2022], [2024]]

# leave-one-out validation
# splits = [[2017], [2018], [2019], [2020], [2022], [2024]] #  [2021, 2022], [2023, 2024]]

slope_angle = 0  # "agg"
slope_azi = 0  # "agg"
hypp_set = {'n_estimators':500,  # [100, 200, 300],
            'max_depth':50,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':min_leaf,
            'min_samples_split':min_split,
            'max_features':"log2",
            'bootstrap':True
            }

try:
    test_pers = [f"{k[0]}_{k[1]}" for k in splits]
except:
    test_pers = [f"{k[0]}" for k in splits]
# end try


#%% use some parameters to generate the n_best list
max_n_feat = 60
min_n_feat = 10
dn_feat = 10
add_ante = [15]
add_post = []

n_bests = add_ante + list(np.arange(min_n_feat, max_n_feat+1, dn_feat)) + add_post


#%% set with_snowpack to true for the time being
with_snowpack = True


#%% choose the model; choices are
#   DT  --> decision tree
#   RF  --> random forest
model_ty = args.model_ty

# the following should be superflous, but leave it in for the time being
if model_ty not in ("DT", "RF"):
    print("\nThe feature-importance is only available for DT and RF. Aborting.\n")
    sys.exit("model_ty not available.")
# end if


#%% set the train-test split
split = [float(i) for i in args.split]


#%% get the avalanche problem
a_p = args.a_p
nan_handl = args.nan_handl

a_p_str = a_p
if a_p == "y":
    a_p_str = "general"
# end if


#%% set parameters
ndlev = args.ndlev
cv = args.cv
cv_score = args.cv_score
cv_on_all = args.cv_on_all
h_low = args.h_low
h_hi = args.h_hi
sea = args.sea
balancing = args.balancing
balance_meth = args.balance_meth
scale_x = args.scale_x
agg_type = args.agg_type
perc = args.perc

if args.reg_code == 0:
    reg_code = "AllReg"  #  sys.argv[1]
    reg_codes = [3009, 3010, 3011, 3012, 3013]
else:
    reg_code = args.reg_code
    reg_codes = [reg_code]
# end if else


#%% set the drop list
if with_snowpack:
    drop = ["y", "wind_slab", "glide_slab", "new_loose", "new_slab", "pwl_slab", "wet_loose", "wet_slab", "wet",
            "reg_code_x", "reg_code_y"]
else:
    drop = ["y", "wind_slab", "glide_slab", "new_loose", "new_slab", "pwl_slab", "wet_loose", "wet_slab", "wet",
            "reg_code"]
# end if else


#%% APs for plotting
a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}


#%% score dictionary
score_pl = {"f1_macro":"F1-macro"}


#%% set up the elevation string
"""
elev_dir = "/ElevSlope_Agg/"
elev_n = ""
if ((h_low > -1) & (h_hi > -1)):
    elev_dir = f"/Between{h_low}_and_{h_hi}m/"
    elev_n = f"_Between{h_low}_and_{h_hi}m"
# end if
"""
#%% set up the elevation string
elev_dir = "/Elev_Agg/"
elev_n = ""
if ((slope_angle == "agg") | (slope_azi == "agg")):
    elev_dir = "/ElevSlope_Agg/"
if ((h_low > -1) & (h_hi > -1)):
    elev_dir = f"/Between{h_low}_and_{h_hi}m/"
    elev_n = f"_Between{h_low}_and_{h_hi}m"
# end if


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% paths and names
data_path = f"{path_par}/IMPETUS/NORA3/"  # data path
mod_path = f"{data_path}/Stored_Models/{agg_str}/{elev_dir}"  # directory to store the model in
pl_path = f"{data_path}/Plots/Feature_Testing/"


#%% remove the target variable from the drop list
drop.remove(a_p)


#%% prepare a suffix for the model name based on the data balancing
bal_suff = ""
if balancing == "internal":
    bal_suff = "_internal"
elif balancing == "external":
    bal_suff = f"_{balance_meth}"
# end if elif


#%% load the best-features list
best_path = f"{data_path}/Feature_Selection/{elev_dir}"
best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}{elev_n}_{sea}{bal_suff}_{cw_str}_{a_p_str}"
# test
try:
    sel_feats = np.array(pd.read_csv(best_path + best_name + "_PostSearch.csv", index_col=0).index)
    print("\nLoading post-search file...\n")
except:
    sel_feats = np.array(pd.read_csv(best_path + best_name + ".csv", index_col=0).index)
    print("\nPost-search file not available...\n")
# end try except


n_all_best = len(sel_feats)

sel_feats = np.append(sel_feats, a_p)


#%% load the NORA3-derived features
feats_n3 = load_agg_feats_adl(ndlev=ndlev, reg_codes=reg_codes, agg_type=agg_type)


#%% include SNOWPACK-derived stability indices if requested
if with_snowpack:
    #% load the SNOWPACK-derived stability indices
    sno_stab = load_snowpack_stab(reg_codes=reg_codes)


    #% merge the dataframes
    feats_df = []
    for reg_c in reg_codes:
        feats_df.append(feats_n3[feats_n3.reg_code == reg_c].merge(sno_stab[sno_stab.reg_code == reg_c],
                                                                   how="inner", left_index=True, right_index=True))
    # end for reg_c

    feats_df = pd.concat(feats_df, axis=0)
else:
    feats_df = feats_n3
# end if else

feats_df = feats_df[sel_feats]


#%% split training and test
data_dict = {}
for split in splits:
    try:
        test_per = "_".join([str(split[0]), str(split[1])])
    except:
        test_per = str(split[0])
    # end try
    # print(f"{test_per}...")
    data_dict[test_per] =\
                       extract_sea(all_df=feats_df, drop=drop, split=split, balance_meth=balance_meth,
                                   target_n=a_p, ord_vars=["ftc_emax", "ava_clim"],
                                   ord_lims={"ftc_emax":[0, 1], "ava_clim":[1, 3]})

# end for splits


#%% def
def calc_metrics(n_best, best_path, best_name, splits):

    best_feats = np.array(pd.read_csv(best_path + best_name, index_col=0).index)[:n_best]

    # set up the model
    model = stat_mod(model_ty=model_ty, ndlev=ndlev, class_weight=class_weight, hyperp=hypp_set, verbose=False)

    result = {}

    result["acc"] = {}
    result["metrics"] = {}
    result["score"] = {}
    result["cr_vals"] = {}
    result["dist_met"] = {}
    for split in splits:
        try:
            test_per = "_".join([str(split[0]), str(split[1])])
        except:
            test_per = str(split[0])
        # end try

        # train the model
        model.fit(data_dict[test_per][0][best_feats], data_dict[test_per][2])

        # perform prediction
        pred_test_all_d = model.predict(data_dict[test_per][5][best_feats])

        #% prepare the confusion matrix
        conf_test_all = confusion_matrix(data_dict[test_per][7], pred_test_all_d)

        result["acc"][test_per] = accuracy_score(data_dict[test_per][7], pred_test_all_d)
        result["cr_vals"][test_per] = precision_recall_fscore_support(data_dict[test_per][7], pred_test_all_d)

        if ndlev == 2:
            result["metrics"][test_per] = mod_metrics(conf_test_all)
            result["score"][test_per] = (2*result["metrics"][test_per]["POD"] + result["metrics"][test_per]["PON"] +
                                        (1-result["metrics"][test_per]["FAR"]) + result["metrics"][test_per]["RPC"]) / 5
            result["dist_met"][test_per] = dist_metrics(true_y=data_dict[test_per][7], pred_y=pred_test_all_d)
        # end if
    # end for split

    return result
# end def


#%% perform the manual cross-validation
best_path = f"{data_path}/Feature_Selection/{elev_dir}"
# load the best feats
print("\nUSING PostSearch BEST FEATURES LIST!!!\n")
best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}{elev_n}_" + \
                                                                    f"{sea}{bal_suff}_{cw_str}_{a_p_str}_PostSearch.csv"
# if n_best == n_all_best:
#     best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}{elev_n}_{sea}{bal_suff}_{a_p_str}.csv"
# end if

start_ti = timer()
metrics = Parallel(n_jobs=-1)(delayed(calc_metrics)(n_best, best_path, best_name, splits) for n_best in n_bests)
end_ti = timer()
elapsed_ti = end_ti - start_ti
print(f"Model training time: {elapsed_ti:.2f} seconds.")


#%%  lines plot  -- 4 panels
if len(splits) == 4:
    ylim = (0.1, 1.05)

    width = 0.1
    x1, x2 = np.array([-0.15, -0.05]), np.array([0.05, 0.15])

    a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

    fig = pl.figure(figsize=(6, 4.5))

    ax00 = fig.add_subplot(221)
    ax01 = fig.add_subplot(222)
    ax10 = fig.add_subplot(223)
    ax11 = fig.add_subplot(224)
    axes = [ax00, ax01, ax10, ax11]

    j = 0
    for test_per in test_pers:

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][0] for i in np.arange(len(n_bests))], color="black",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][1] for i in np.arange(len(n_bests))], color="black",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][0] for i in np.arange(len(n_bests))], color="red",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][1] for i in np.arange(len(n_bests))], color="red",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["acc"][test_per] for i in np.arange(len(n_bests))], color="blue", marker="s")
        axes[j].plot(n_bests, [metrics[i]["metrics"][test_per]["FAR"] for i in np.arange(len(n_bests))], color="gray",
                     marker="d")

        axes[j].set_title("-".join(test_pers[j].split("_")))

        j += 1
    # end for test_per

    ax00.plot([], [], marker="o", c="black", label="PR non-AvD")
    ax00.plot([], [], marker="x", c="black", label="PR AvD")
    ax10.plot([], [], marker="o", c="red", label="RC non-AvD")
    ax10.plot([], [], marker="x", c="red", label="RC AvD")
    ax11.plot([], [], marker="s", c="blue", label="ACC")

    ax00.legend(ncol=1)
    ax10.legend(ncol=1)
    ax11.legend(ncol=1)

    ax00.set_xticklabels([])
    ax01.set_xticklabels([])
    ax01.set_yticklabels([])
    ax11.set_yticklabels([])

    ax10.set_xticks(n_bests)
    ax10.set_xticklabels(n_bests)
    ax11.set_xticks(n_bests)
    ax11.set_xticklabels(n_bests)

    ax00.set_ylabel("Metric")
    ax10.set_ylabel("Metric")
    # ax10.set_xlabel("min_sample_leaf & _split")
    # ax11.set_xlabel("min_sample_leaf & _split")
    ax10.set_xlabel("Number of features")
    ax11.set_xlabel("Number of features")

    ax00.set_ylim(ylim)
    ax01.set_ylim(ylim)
    ax10.set_ylim(ylim)
    ax11.set_ylim(ylim)

    fig.suptitle(a_ps[a_p] + " problem")
    fig.subplots_adjust(wspace=0.1)

    pl.show()
    pl.close()

# end if


#%% lines plot -- 3 panels
if len(splits) == 3:

    fig = pl.figure(figsize=(8, 3))

    ax00 = fig.add_subplot(131)
    ax01 = fig.add_subplot(132)
    ax02 = fig.add_subplot(133)
    axes = [ax00, ax01, ax02]

    j = 0
    for test_per in test_pers:

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][0] for i in np.arange(len(n_bests))], color="black",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][1] for i in np.arange(len(n_bests))], color="black",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][0] for i in np.arange(len(n_bests))], color="red",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][1] for i in np.arange(len(n_bests))], color="red",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["acc"][test_per] for i in np.arange(len(n_bests))], color="blue", marker="s")
        axes[j].plot(n_bests, [metrics[i]["metrics"][test_per]["FAR"] for i in np.arange(len(n_bests))], color="gray",
                     marker="d")

        axes[j].set_title(" & ".join(test_pers[j].split("_")))

        j += 1
    # end for test_per

    ax00.plot([], [], marker="o", c="black", label="PR non-AvD")
    ax00.plot([], [], marker="x", c="black", label="PR AvD")
    ax01.plot([], [], marker="o", c="red", label="RC non-AvD")
    ax01.plot([], [], marker="x", c="red", label="RC AvD")
    ax02.plot([], [], marker="s", c="blue", label="ACC")
    ax02.plot([], [], marker="d", c="gray", label="FAR")

    ax00.legend(ncol=1)
    ax01.legend(ncol=1)
    ax02.legend(ncol=1)

    ax00.set_xlabel("Number of features")
    ax01.set_xlabel("Number of features")
    ax02.set_xlabel("Number of features")

    ax01.set_yticklabels([])
    ax02.set_yticklabels([])

    ax00.set_ylim(0, 1.05)
    ax01.set_ylim(0, 1.05)
    ax02.set_ylim(0, 1.05)
    fig.suptitle(a_ps[a_p] + " problem")
    fig.subplots_adjust(wspace=0.1, top=0.82)

    pl.show()
    pl.close()

# end if


#%% lines plot -- 6 panels
if len(splits) == 6:

    fig = pl.figure(figsize=(8, 6))

    ax00 = fig.add_subplot(331)
    ax01 = fig.add_subplot(332)
    ax02 = fig.add_subplot(333)
    ax10 = fig.add_subplot(334)
    ax11 = fig.add_subplot(335)
    ax12 = fig.add_subplot(336)

    axes = [ax00, ax01, ax02,
            ax10, ax11, ax12]

    j = 0
    for test_per in test_pers:

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][0] for i in np.arange(len(n_bests))], color="black",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][1] for i in np.arange(len(n_bests))], color="black",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][0] for i in np.arange(len(n_bests))], color="red",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][1] for i in np.arange(len(n_bests))], color="red",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["acc"][test_per] for i in np.arange(len(n_bests))], color="blue", marker="s")
        axes[j].plot(n_bests, [metrics[i]["metrics"][test_per]["FAR"] for i in np.arange(len(n_bests))], color="gray",
                     marker="d")

        axes[j].set_title(" & ".join(test_pers[j].split("_")))

        j += 1
    # end for test_per

    ax10.set_xlabel("Number of features")
    ax11.set_xlabel("Number of features")
    ax12.set_xlabel("Number of features")

    ax01.set_yticklabels([])
    ax02.set_yticklabels([])
    ax11.set_yticklabels([])
    ax12.set_yticklabels([])

    ax00.set_ylim(0, 1.05)
    ax01.set_ylim(0, 1.05)
    ax02.set_ylim(0, 1.05)
    ax10.set_ylim(0, 1.05)
    ax11.set_ylim(0, 1.05)
    ax12.set_ylim(0, 1.05)

    fig.suptitle(a_ps[a_p] + " problem")
    fig.subplots_adjust(wspace=0.1, top=0.82)

    pl.show()
    pl.close()
# end if


#%%  lines plot  -- 5 panels
if len(splits) == 5:
    ylim = (0.1, 1.05)

    width = 0.1
    x1, x2 = np.array([-0.15, -0.05]), np.array([0.05, 0.15])

    a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

    fig = pl.figure(figsize=(6, 6))

    ax00 = fig.add_subplot(321)
    ax01 = fig.add_subplot(322)
    ax10 = fig.add_subplot(323)
    ax11 = fig.add_subplot(324)
    ax20 = fig.add_subplot(325)
    ax21 = fig.add_subplot(326)
    axes = [ax00, ax01, ax10, ax11, ax20, ax21]

    j = 0
    for test_per in test_pers:

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][0] for i in np.arange(len(n_bests))], color="black",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][0][1] for i in np.arange(len(n_bests))], color="black",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][0] for i in np.arange(len(n_bests))], color="red",
                     marker="o")
        axes[j].plot(n_bests, [metrics[i]["cr_vals"][test_per][1][1] for i in np.arange(len(n_bests))], color="red",
                     marker="x")

        axes[j].plot(n_bests, [metrics[i]["acc"][test_per] for i in np.arange(len(n_bests))], color="blue", marker="s")
        axes[j].plot(n_bests, [metrics[i]["metrics"][test_per]["FAR"] for i in np.arange(len(n_bests))], color="gray",
                     marker="d")

        axes[j].set_title("-".join(test_pers[j].split("_")))

        j += 1
    # end for test_per

    ax21.plot([], [], marker="o", c="black", label="PR non-AvD")
    ax21.plot([], [], marker="x", c="black", label="PR AvD")
    ax21.plot([], [], marker="o", c="red", label="RC non-AvD")
    ax21.plot([], [], marker="x", c="red", label="RC AvD")
    ax21.plot([], [], marker="s", c="blue", label="ACC")
    ax21.plot([], [], marker="d", c="gray", label="FAR")

    ax21.axis("off")

    ax21.legend(ncol=1, loc=(0.2, -0.075))

    ax00.set_xticklabels([])
    ax01.set_xticklabels([])
    ax10.set_xticklabels([])

    ax01.set_yticklabels([])
    ax11.set_yticklabels([])

    ax20.set_xticks(n_bests)
    ax20.set_xticklabels(n_bests)
    ax11.set_xticks(n_bests)
    ax11.set_xticklabels(n_bests)

    ax00.set_ylabel("Metric")
    ax10.set_ylabel("Metric")
    ax20.set_ylabel("Metric")
    # ax10.set_xlabel("min_sample_leaf & _split")
    # ax11.set_xlabel("min_sample_leaf & _split")
    ax11.set_xlabel("Number of features")
    ax20.set_xlabel("Number of features")

    ax00.set_ylim(ylim)
    ax01.set_ylim(ylim)
    ax10.set_ylim(ylim)
    ax11.set_ylim(ylim)
    ax20.set_ylim(ylim)

    fig.suptitle(a_ps[a_p] + " problem")
    fig.subplots_adjust(wspace=0.1, hspace=0.25, top=0.92)

    pl.show()
    pl.close()

# end if


#%% calculate the means
wss = []
tss = []
hss = []
pss = []
acc = []
far = []
f1mac = []
avg_true0 = []
avg_true1 = []
avg_pred0 = []
avg_pred1 = []
reliab = []
discr1 = []
for test_per in test_pers:

    f1mac.append([np.mean(metrics[i]["cr_vals"][test_per][2][:]) for i in np.arange(len(n_bests))])
    far.append([metrics[i]["metrics"][test_per]["FAR"] for i in np.arange(len(n_bests))])

    acc.append([metrics[i]["acc"][test_per] for i in np.arange(len(n_bests))])
    wss.append([metrics[i]["score"][test_per] for i in np.arange(len(n_bests))])
    tss.append([metrics[i]["metrics"][test_per]["TSS"] for i in np.arange(len(n_bests))])
    hss.append([metrics[i]["metrics"][test_per]["HSS"] for i in np.arange(len(n_bests))])
    pss.append([metrics[i]["metrics"][test_per]["PSS"] for i in np.arange(len(n_bests))])

    avg_true0.append([metrics[i]["dist_met"][test_per]["avg_true"]["0"] for i in np.arange(len(n_bests))])
    avg_true1.append([metrics[i]["dist_met"][test_per]["avg_true"]["1"] for i in np.arange(len(n_bests))])
    avg_pred0.append([metrics[i]["dist_met"][test_per]["avg_pred"]["0"] for i in np.arange(len(n_bests))])
    avg_pred1.append([metrics[i]["dist_met"][test_per]["avg_pred"]["1"] for i in np.arange(len(n_bests))])

    reliab.append([metrics[i]["dist_met"][test_per]["reliability"] for i in np.arange(len(n_bests))])
    discr1.append([metrics[i]["dist_met"][test_per]["discrimination1"] for i in np.arange(len(n_bests))])
# end for test_per

far = np.array(far)
far_means = [np.mean(far[:, k]) for k in np.arange(len(n_bests))]
far_stds = [np.std(far[:, k]) for k in np.arange(len(n_bests))]

f1mac = np.array(f1mac)
f1mac_means = [np.mean(f1mac[:, k]) for k in np.arange(len(n_bests))]
f1mac_stds = [np.std(f1mac[:, k]) for k in np.arange(len(n_bests))]

acc = np.array(acc)
acc_means = [np.mean(acc[:, k]) for k in np.arange(len(n_bests))]
acc_stds = [np.std(acc[:, k]) for k in np.arange(len(n_bests))]

wss = np.array(wss)
wss_means = [np.mean(wss[:, k]) for k in np.arange(len(n_bests))]
wss_stds = [np.std(wss[:, k]) for k in np.arange(len(n_bests))]

tss = np.array(tss)
tss_means = [np.mean(tss[:, k]) for k in np.arange(len(n_bests))]
tss_stds = [np.std(tss[:, k]) for k in np.arange(len(n_bests))]

hss = np.array(hss)
hss_means = [np.mean(hss[:, k]) for k in np.arange(len(n_bests))]
hss_stds = [np.std(hss[:, k]) for k in np.arange(len(n_bests))]

pss = np.array(pss)
pss_means = [np.mean(pss[:, k]) for k in np.arange(len(n_bests))]
pss_stds = [np.std(pss[:, k]) for k in np.arange(len(n_bests))]

avg_true0 = np.array(avg_true0)
avg_true0_means = [np.mean(avg_true0[:, k]) for k in np.arange(len(n_bests))]
avg_true0_stds = [np.std(avg_true0[:, k]) for k in np.arange(len(n_bests))]

avg_true1 = np.array(avg_true1)
avg_true1_means = [np.mean(avg_true1[:, k]) for k in np.arange(len(n_bests))]
avg_true1_stds = [np.std(avg_true1[:, k]) for k in np.arange(len(n_bests))]

avg_pred0 = np.array(avg_pred0)
avg_pred0_means = [np.mean(avg_pred0[:, k]) for k in np.arange(len(n_bests))]
avg_pred0_stds = [np.std(avg_pred0[:, k]) for k in np.arange(len(n_bests))]

avg_pred1 = np.array(avg_pred1)
avg_pred1_means = [np.mean(avg_pred1[:, k]) for k in np.arange(len(n_bests))]
avg_pred1_stds = [np.std(avg_pred1[:, k]) for k in np.arange(len(n_bests))]

reliab = np.array(reliab)
reliab_means = [np.mean(reliab[:, k]) for k in np.arange(len(n_bests))]
reliab_stds = [np.std(reliab[:, k]) for k in np.arange(len(n_bests))]

discr1 = np.array(discr1)
discr1_means = [np.mean(discr1[:, k]) for k in np.arange(len(n_bests))]
discr1_stds = [np.std(discr1[:, k]) for k in np.arange(len(n_bests))]


#%% plot the means -- distribution-oriented measures
lw2 = 0.65

fig = pl.figure(figsize=(6, 3))
ax00 = fig.add_subplot(111)

ax00.scatter(n_bests, reliab_means, facecolor="none", edgecolor="blue", marker="o", s=80)
ax00.errorbar(n_bests, reliab_means, yerr=reliab_stds, color="blue", capsize=5)

ax00.scatter(n_bests, discr1_means, facecolor="none", edgecolor="red", marker="o", s=80)
ax00.errorbar(n_bests, discr1_means, yerr=reliab_stds, color="red", capsize=5)

ax00.scatter(n_bests, avg_true0_means, facecolor="none", edgecolor="gray", marker="o", s=80)
ax00.errorbar(n_bests, avg_true0_means, yerr=avg_true0_stds, color="gray", capsize=5, linewidth=lw2)

ax00.scatter(n_bests, avg_true1_means, facecolor="none", edgecolor="gray", marker="d", s=80)
ax00.errorbar(n_bests, avg_true1_means, yerr=avg_true1_stds, color="gray", capsize=5, linewidth=lw2)

ax00.scatter(n_bests, avg_pred0_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(n_bests, avg_pred0_means, yerr=avg_pred0_stds, color="black", capsize=5, linewidth=lw2)

ax00.scatter(n_bests, avg_pred1_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(n_bests, avg_pred1_means, yerr=avg_pred1_stds, color="black", capsize=5, linewidth=lw2)

# ax00.plot([], [], c="blue", label="PSS")
# ax00.legend(ncol=5)

ax00.set_ylabel("Score")
ax00.set_xlabel("Feature number")
ax00.set_title(f"Distribution-oriented measures {a_ps[a_p]} problem")

ax00.set_ylim(0, 1)

# pl_path = "/media/kei070/One_Touch/IMPETUS/NORA3/Plots/Model_Evaluation/With_SNOWPACK/WeightedScore_FeatNumTest/"
# pl.savefig(pl_path + f"DistributionOriented_{a_p}.png", dpi=200, bbox_inches="tight")
pl.show()
pl.close()


#%% plot the means -- scores
fig = pl.figure(figsize=(6, 3))
ax00 = fig.add_subplot(111)

ax00.scatter(n_bests, acc_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(n_bests, acc_means, yerr=acc_stds, color="black", capsize=5, linestyle="-")

ax00.scatter(n_bests, far_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(n_bests, far_means, yerr=far_stds, color="black", capsize=5, linestyle="--")

ax00.scatter(n_bests, f1mac_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(n_bests, f1mac_means, yerr=f1mac_stds, color="black", capsize=5, linestyle="-.")

ax00.scatter(n_bests, tss_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(n_bests, tss_means, yerr=tss_stds, color="black", capsize=5, linestyle=":")

"""
ax00.scatter(n_bests, hss_means, facecolor="none", edgecolor="orange", marker="o", s=80)
ax00.errorbar(n_bests, hss_means, yerr=hss_stds, color="orange", capsize=5)

ax00.scatter(n_bests, pss_means, facecolor="none", edgecolor="blue", marker="o", s=80)
ax00.errorbar(n_bests, pss_means, yerr=pss_stds, color="blue", capsize=5)

ax00.scatter(n_bests, wss_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(n_bests, wss_means, yerr=wss_stds, color="black", capsize=5)
"""

ax00.plot([], [], c="black", linestyle="-", label="PC")
ax00.plot([], [], c="black", linestyle="--", label="FAR")
ax00.plot([], [], c="black", linestyle="-.", label="F1-macro")
ax00.plot([], [], c="black", linestyle=":", label="TSS")

"""
ax00.plot([], [], c="black", label="WSS")
ax00.plot([], [], c="orange", label="HSS")
ax00.plot([], [], c="blue", label="PSS")
"""

ax00.legend(ncol=5)

ax00.set_ylabel("Score")
ax00.set_xlabel("Feature number")
ax00.set_title(f"Scores {a_ps[a_p]} problem")

ax00.set_ylim(0, 1)

pl_path = "/media/kei070/One_Touch/IMPETUS/NORA3/Plots/Model_Evaluation/With_SNOWPACK/WeightedScore_FeatNumTest/"
# pl.savefig(pl_path + f"WeightedScore_{a_p}_{p_add}.png", dpi=200, bbox_inches="tight")
pl.savefig(pl_path + f"PC_FAR_F1_TSS_{a_p}.png", dpi=200, bbox_inches="tight")
pl.show()
pl.close()

