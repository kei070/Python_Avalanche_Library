#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative feature search based on (1) feature importance and (2) feature cross-correlation.
The feature importance is derived from a random forest model and the feature cross-correlation is represented by the
Pearson R.
"""


#%% imports
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pylab as pl
from joblib import load

from ava_functions.Data_Loading import load_snowpack_stab, load_agg_feats_adl
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Feature_Selection import feat_sel


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="FeatureSelection",
                    description="""Performs the iterative feature selection for RF and DT.""",
                    epilog="For more information consult the documentation of the function feat_sel.")

# ...and add the arguments
parser.add_argument("--a_p", default="y", type=str, choices=["y", "glide_slab", "new_loose", "new_slab",
                                                                     "pwl_slab", "wet_loose", "wet_slab", "wet",
                                                                     "wind_slab"],
                    help="""The avalanche problem that is used. Standard is 'y', i.e., the general ADL.""")
parser.add_argument("--class_weight", type=str, help="""The internal class weights.
                    Must be submitted as a JSON-string that can be interpreted as a dictionary, e.g.:
                        '{0:1, 1:1}'. If empty, default values will be used.""")
parser.add_argument("--thres", type=float, default=0.9, help="""Value of cross-correlation R for dropping features""")
parser.add_argument("--r_coeff", default="R", type=str, choices=["R", "R**2", "R^2"],
                    help="""The type of cross-correlation R to be considered as threshold""")
parser.add_argument("--model_ty", default="RF", choices=["RF", "DT"],
                    help="""The type of statistical model that is used.""")
parser.add_argument("--ndlev", type=int, default=2, choices=[2, 3, 4],
                    help="""The number of danger levels to be predicted.""")
parser.add_argument("--balancing", type=str, default="external", choices=["none", "internal", "external"],
                    help="""Controls the kind of balancing to be performed.""")
parser.add_argument("--balance_meth", type=str, default="SMOTE",
                    choices=["undersample", "rus", "ros", "SMOTE", "SVMSMOTE", "ADASYN"],
                    help="""The balancing method applied during the loading of the data. This is only relevant if
                    balancing is set to external.""")
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


#%% set with_snowpack to true for the time being
with_snowpack = True


#%% choose the model; choices are
#   DT  --> decision tree
#   RF  --> random forest
model_ty = args.model_ty

if model_ty not in ["DT", "RF"]:
    print("\nThe feature-importance is only available for DT and RF. Aborting.\n")
    sys.exit("model_ty not available.")
# end if


#%% get the avalanche problem
a_p = args.a_p

a_p_str = a_p
if a_p == "y":
    a_p_str = "general"
# end if


#%% set the parameters
thres = args.thres
r_coeff = args.r_coeff
ndlev = args.ndlev
h_low = args.h_low
h_hi = args.h_hi
sea = args.sea
balancing = args.balancing
balance_meth = args.balance_meth
reg_code = args.reg_code
agg_type = args.agg_type
perc = args.perc
exposure = None
slope_angle = 0  # "agg"
slope_azi = 0  # "agg"


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
    class_weight = {k:1 for k in np.arange(ndlev)}
# end if else


#%% set the class weights string
cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight.values()])}"
print(f"\nUsing the following class weights: {class_weight}\n")


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


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


#%% paths
data_path = f"{path_par}/IMPETUS/NORA3/"  # data (=parent) path
mod_path = f"{data_path}/Stored_Models/{agg_str}/{elev_dir}/"  # directory to store the model in
best_feats_path = f"{data_path}/Feature_Selection/{elev_dir}/"  # output path


#%% handle region codes
if reg_code == 0:
    reg_code = "AllReg"
    reg_codes = [3009, 3010, 3011, 3012, 3013]
else:
    reg_codes = [reg_code]
# end if else


#%% prepare a suffix for the model name based on the data balancing
bal_suff = ""
if balancing == "internal":
    bal_suff = "_internal"
elif balancing == "external":
    bal_suff = f"_{balance_meth}"
# end if elif


#%% load the model
mod_name = f"{model_ty}_{ndlev}DL_{reg_code}_{agg_str}{elev_n}_{sea}{bal_suff}_{cw_str}_{a_p_str}"

print("\nLoading model:")
print(mod_name)
print(mod_path)
print()

try:  # try first the model with data
    model = load(f"{mod_path}/{mod_name}_wData.joblib")["model"]
except:  # and if that fails try the model without data

    model = load(f"{mod_path}/{mod_name}.joblib")
# end try except


#%% investigate feature importance
sel_feats = model.feature_names_in_
importances = model.feature_importances_
# print("\nFeature importances:\n", importances, "\n")

imp_sort = np.argsort(importances)[::-1]


#%% load the features
"""
feat_df = load_feats_xlevel(reg_codes=reg_codes, ndlev=ndlev, exposure=exposure, sel_feats=list(sel_feats),
                            a_p=a_p,
                            agg_type=agg_type, perc=perc,
                            out_type="dataframe",
                            h_low=h_low, h_hi=h_hi,
                            data_path_par=data_path + "Avalanche_Predictors/")["all"]

feat_df.drop(a_p, axis=1, inplace=True)
feat_df.drop("reg_code", axis=1, inplace=True)
"""

#%% load the NORA3-derived features
# feats_n3 = load_features2(ndlev=ndlev, reg_codes=reg_codes, h_low=h_low, h_hi=h_hi,
#                           agg_type="mean", perc=90, nan_handling="drop")
feats_n3 = load_agg_feats_adl(ndlev=ndlev, reg_codes=reg_codes, agg_type=agg_type)


#%% include SNOWPACK-derived stability indices if requested
if with_snowpack:
    #% load the SNOWPACK-derived stability indices
    sno_stab = load_snowpack_stab(reg_codes=reg_codes, slope_angle=slope_angle, slope_azi=slope_azi)


    #% merge the dataframes
    feat_df = []
    for reg_code in reg_codes:
        feat_df.append(feats_n3[feats_n3.reg_code == reg_code].merge(sno_stab[sno_stab.reg_code == reg_code],
                                                                     how="inner", left_index=True, right_index=True))
    # end for reg_code

    feat_df = pd.concat(feat_df, axis=0)
else:
    feat_df = feats_n3
# end if else


#%% reorder the features
sel_sub = sel_feats[imp_sort] # [20:30]
feat_sub = feat_df[sel_sub]


#%% calculate the cross-correlation
cm_pearson = feat_sub.corr()
cm_spearman = feat_sub.corr(method='spearman')
cm_kendall = feat_sub.corr(method='kendall')


#%% combine the Pearson and the Spearman matrices: Pearson is lower left, Spearman is upper right
cm_pear_spear = np.zeros(np.shape(cm_pearson))

pearson_ind = np.tril_indices_from(cm_pearson, k=-1)
spearman_ind = np.triu_indices_from(cm_spearman, k=1)

cm_pear_spear[pearson_ind] = np.array(cm_pearson)[pearson_ind]
cm_pear_spear[spearman_ind] = np.array(cm_spearman)[spearman_ind]

cm_pear_spear_df = pd.DataFrame(cm_pear_spear, index=np.array(sel_sub), columns=np.array(sel_sub))


#%% combine the Pearson and the Kendall matrices: Pearson is lower left, Kendall is upper right
cm_pear_ken = np.zeros(np.shape(cm_pearson))

pearson_ind = np.tril_indices_from(cm_pearson, k=-1)
kendall_ind = np.triu_indices_from(cm_kendall, k=1)

cm_pear_ken[pearson_ind] = np.array(cm_pearson)[pearson_ind]
cm_pear_ken[kendall_ind] = np.array(cm_kendall)[kendall_ind]

cm_pear_ken_df = pd.DataFrame(cm_pear_ken, index=np.array(sel_sub), columns=np.array(sel_sub))


#%% perform an iterative search over the features in terms of cross-correlation
best_feats = feat_sel(feats=sel_feats, cross_c=cm_pear_spear_df, importances=importances, thres=thres, r_coeff=r_coeff,
                      verbose=True)

print("\nBest features:\n")
print(best_feats)


#%% plot feature importances
fig = pl.figure(figsize=(10, 12))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

ax00.barh(sel_feats[imp_sort[::-1]], importances[imp_sort[::-1]])
ax00.set_xlabel("Importance")
ax00.set_title(f"All features (N={len(sel_feats)})")

ax00.tick_params(axis='x', labelrotation=0)

ax00.set_ylim((-1, len(sel_feats)))


ax01.barh(best_feats.index[::-1], best_feats.importance.iloc[::-1])
ax01.set_xlabel("Importance")
ax01.set_title(f"Best features (N={len(best_feats)})")

ax01.tick_params(axis='x', labelrotation=0)

ax01.set_ylim((-1, len(best_feats)))

fig.suptitle(f"Feature importance ndlev={ndlev}")
fig.subplots_adjust(top=0.93, wspace=0.4)

# os.makedirs(fi_path, exist_ok=True)
# pl.savefig(fi_path + f"{model_ty}_AllFeatImportance_{ndlev}Levels.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% store the best features to .csv
os.makedirs(best_feats_path, exist_ok=True)
best_feats.to_csv(best_feats_path + f"BestFeats_{mod_name}.csv")

print("\nBest features stored in:")
print("  " + best_feats_path)
print("as:")
print(f"  BestFeats_{mod_name}.csv\n")
