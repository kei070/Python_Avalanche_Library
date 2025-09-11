#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the statistical model including SNOWPACK output.
"""


#%% imports
import shap
import os
import sys
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from ava_functions.Model_Fidelity_Metrics import dist_metrics
from ava_functions.Data_Loading import load_features2, load_snowpack_stab, load_agg_feats_adl
from ava_functions.Helpers_Load_Data import extract_sea
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.StatMod import stat_mod
from ava_functions.ConfMat_Helper import conf_helper
from ava_functions.Lists_and_Dictionaries.Features import se_norge_feats, nora3_clean
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set parameters
n_best = 30  # set to 0 to use all best features -- this is required to generate a new best features list
feats = "best"
a_p = "wind_slab"

min_leaf = 15
min_split = 15

class_weight = {0:1, 1:1}
cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight.values()])}"
print(f"\nUsing the following class weights: {class_weight}\n")

slope_angle = 0 # "agg"
slope_azi = 0 # "agg"
gen_feats_list = True  # set to true if new best features list should be created (only if feats=best and n_best=0)
add_feats = [] # ["snow_depth3_emax"]
drop_feats = []  # ["r3_emax"]
with_snowpack = True
model_ty = "RF"
sea = "full"
balancing = "external"
balance_meth = "SMOTE"
ndlev = 2
split = [2021, 2023]
# reg_codes = [3010, 3011, 3012]
# reg_codes = [3009, 3013]
reg_code = 0
agg_type = "mean"
perc = 75
h_low = -1
h_hi = -1

"""
hypp_set = {'n_estimators':350,  # [100, 200, 300],
            'max_depth':70,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':2,
            'min_samples_split':12,
            'max_features':"log2",
            'bootstrap':True
            }
"""
hypp_set = {'n_estimators':500,  # [100, 200, 300],
            'max_depth':50,  # [2, 5, 8, 11, 14, 17, 20],
            'min_samples_leaf':min_leaf,
            'min_samples_split':min_split,
            'max_features':"log2",
            'bootstrap':True
            }

if with_snowpack:
    drop = ["y", "wind_slab", "glide_slab", "new_loose", "new_slab", "pwl_slab", "wet_loose", "wet_slab", "wet",
            "reg_code_x", "reg_code_y"]
else:
    drop = ["y", "wind_slab", "glide_slab", "new_loose", "new_slab", "pwl_slab", "wet_loose", "wet_slab", "wet",
            "reg_code"]
# end if else


#%% should a new best feats list be generated?
gen_new_best_feats = ((feats == "best") & (n_best == 0) & gen_feats_list)


#%% add a string to the file names depending on the inclusion of SNOWPACK-derived data
if with_snowpack:
    sp_str = ""
else:
    sp_str = "_wo_SNOWP"
# end if else


#%% generate the strings based on slope and aspect
"""
slope_path = "Flat"
if slope_angle > 0:
    aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

    slope_path = f"{slope_angle}" + aspect
# end if

if slope_angle == 0:
    aspect = ""
# end if
"""

#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% get the avalanche problem
a_p_str = a_p
if a_p == "y":
    a_p_str = "general"
# end if


#%% prepare a suffix for the model name based on the data balancing
bal_suff = ""
if balancing == "internal":
    bal_suff = "_internal"
elif balancing == "external":
    bal_suff = f"_{balance_meth}"
# end if elif


#%% handle region codes
if reg_code == 0:
    reg_code = "AllReg"
    reg_code_save = "AllReg"
    reg_codes = [3009, 3010, 3011, 3012, 3013]
else:
    reg_codes = [reg_code]
# end if else


#%% set up the elevation string
elev_dir = "/Elev_Agg/"
elev_n = ""
if ((slope_angle == "agg") | (slope_azi == "agg")):
    elev_dir = "/ElevSlope_Agg/"
if ((h_low > -1) & (h_hi > -1)):
    elev_dir = f"/Between{h_low}_and_{h_hi}m/"
    elev_n = f"_Between{h_low}_and_{h_hi}m"
# end if


#%% set paths and names
data_path = f"{path_par}/IMPETUS/NORA3/"  # data path
mod_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/{agg_str}/{elev_dir}/"
modev_path = f"{path_par}/IMPETUS/NORA3/Plots/Model_Evaluation/With_SNOWPACK/"
fi_path = f"{path_par}/IMPETUS/NORA3/Plots/Feature_Importance/With_SNOWPACK/"

print("\nLoading model from:")
print(mod_path + "\n")


#%% remove the target variable from the drop list
drop.remove(a_p)


#%% adjust the reg_codes
# reg_codes = list(regions.keys()) if reg_codes == "all" else reg_codes


#%% set the features
if feats == "all":
    sel_feats = slice(None)
elif feats == "best":
    try:
        post_search = "_PostSearch"
        post_search_print = " post-search"
        if n_best == 0:
            post_search = ""
            post_search_print = ""
        # end if
        best_path = f"{data_path}/Feature_Selection/{elev_dir}/"
        best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code}_{agg_str}" + \
                                                         f"{elev_n}_{sea}{bal_suff}_{cw_str}_{a_p_str}{post_search}.csv"
        if n_best == 0:
            n_best = None
        # end if
        sel_feats = np.array(pd.read_csv(best_path + best_name, index_col=0).index)[:n_best]
        n_best = len(sel_feats)

        print(f"\nLoading{post_search_print} best-feature list...\n")

        print(f"\nNumber of features: {len(sel_feats)}\n")

        sel_feats = np.append(sel_feats, a_p)

    except:
        print(f"\nNo{post_search_print} best-feature list found in\n")
        print(best_path + best_name)
        print("\nAborting\n")
        sys.exit()
    # end try except
# end if elif


#%% load the NORA3-derived features
# feats_n3 = load_features2(ndlev=ndlev, reg_codes=reg_codes, h_low=h_low, h_hi=h_hi,
#                           agg_type="mean", perc=90, nan_handling="drop")
feats_n3 = load_agg_feats_adl(ndlev=ndlev, reg_codes=reg_codes, agg_type=agg_type, perc=perc)


#%% include SNOWPACK-derived stability indices if requested
if with_snowpack:
    #% load the SNOWPACK-derived stability indices
    sno_stab = load_snowpack_stab(reg_codes=reg_codes, slope_angle=slope_angle, slope_azi=slope_azi)


    #% merge the dataframes
    feats_df = []
    for reg_code in reg_codes:
        feats_df.append(feats_n3[feats_n3.reg_code == reg_code].merge(sno_stab[sno_stab.reg_code == reg_code],
                                                                      how="inner", left_index=True, right_index=True))
    # end for reg_code

    feats_df = pd.concat(feats_df, axis=0)
else:
    feats_df = feats_n3
# end if else


#%% add and remove the manually requested features to the selected features
if feats != "all":
    sel_feats = np.concatenate([sel_feats, add_feats])
# end if

if len(drop_feats) > 0:
    for del_feat in drop_feats:
        sel_feats = np.delete(sel_feats, sel_feats == del_feat)
    # end for del_feat
# end if


#%% select the features
feats_df = feats_df[sel_feats]


#%% split training and test
train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all =\
                       extract_sea(all_df=feats_df, drop=drop, split=split, balance_meth=balance_meth,
                                   target_n=a_p, ord_vars=["ftc_emax", "ava_clim"],
                                   ord_lims={"ftc_emax":[0, 1], "ava_clim":[1, 3]})


#%% set up the model
hypp_grid = {'n_estimators': [500],  # [100, 200, 300],
             'max_depth': [50],  # [2, 5, 8, 11, 14, 17, 20],
             'min_samples_leaf': [2, 10, 40],
             'min_samples_split': [2, 10, 40],
             'max_features': ["sqrt"],
             'bootstrap': [True]
             }
model = stat_mod(model_ty=model_ty, ndlev=ndlev, class_weight=class_weight, hyperp=hypp_set, grid_search=False,
                 train_x=train_x_all, train_y=train_y_all, cv_score="f1")


#%% train the model
model.fit(train_x, train_y)


#%% predict
pred_test = model.predict(test_x)
pred_test_all = model.predict(test_x_all)


#%% model accuracy
acc_test = accuracy_score(test_y, pred_test)
print(f'Accuracy balanced test data:     {(acc_test * 100)} %')

acc_test_all = accuracy_score(test_y_all, pred_test_all)
print(f'Accuracy all test data:          {(acc_test_all * 100)} %\n')


#%% print the classification_report
class_rep = classification_report(test_y_all, pred_test_all)
cr_vals = precision_recall_fscore_support(test_y_all, pred_test_all)
print(class_rep)


#%% investigate feature importance
importances = model.feature_importances_

print("\nFeature importances:", importances, "\n")


#%% compute the SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(train_x)


#%% generate a waterfall plot of some instance
shap.plots.waterfall(shap_values[1000, :, 0], max_display=10)


#%% bar plot
shap.plots.bar(shap_values[:, :, 1], max_display=10)


#%% beeswarm plot
shap.plots.beeswarm(shap_values[:, :, 1])


#%% plot the average of features for AvDs and for non-AvDs
y_lim = None  # (-1, 1)

var_bpl = "s7_emin"

fig = pl.figure()
ax00 = fig.add_subplot(111)
ax00.boxplot(train_x[var_bpl][train_y == 0], positions=[0])
ax00.boxplot(train_x[var_bpl][train_y == 1], positions=[1])

ax00.set_ylabel(var_bpl)
ax00.set_xticks([0, 1])
ax00.set_xticklabels(["non-AvD", "AvD"])

ax00.set_ylim(y_lim)

pl.show()
pl.close()


print(f"{var_bpl} non-AvD mean +- std: {np.mean(train_x[var_bpl][train_y == 0]):.2f} +- " +
      f"{np.std(train_x[var_bpl][train_y == 0]):.3f}")
print(f"{var_bpl} AvD mean +- std:     {np.mean(train_x[var_bpl][train_y == 1]):.2f} +- " +
      f"{np.std(train_x[var_bpl][train_y == 1]):.3f}")


#%% plot feature importances

# get the features
sel_feats = np.array(list(test_x.columns))

imp_sort = np.argsort(importances)[::-1]

fig = pl.figure(figsize=(4, 18))
ax00 = fig.add_subplot(111)
# ax01 = fig.add_subplot(122)

ax00.barh(sel_feats[imp_sort[::-1]], importances[imp_sort[::-1]])
ax00.set_xlabel("Importance")
ax00.set_title(f"Feature importance ndlev={ndlev}")

ax00.tick_params(axis='x', labelrotation=0)

ax00.set_ylim((-1, len(sel_feats)))

os.makedirs(fi_path, exist_ok=True)
pl.savefig(fi_path + f"{model_ty}_AllFeatImportance_{a_p_str}_{ndlev}Levels{sp_str}.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot feature importances -- only the first few
a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

n_feats = 10

# get the features
sel_feats = np.array(list(test_x.columns))

imp_sort = np.argsort(importances)[::-1]

fig = pl.figure(figsize=(2, 3.5))
ax00 = fig.add_subplot(111)
# ax01 = fig.add_subplot(122)

ax00.barh(sel_feats[imp_sort][:n_feats][::-1], importances[imp_sort][:n_feats][::-1])
ax00.set_xlabel("Importance")
ax00.set_title(f"{a_ps[a_p]}")

ax00.tick_params(axis='x', labelrotation=0)

ax00.set_ylim((-1, n_feats))

os.makedirs(fi_path, exist_ok=True)
pl.savefig(fi_path + f"{model_ty}_{n_feats}BestFeatImportance_{a_p_str}_{ndlev}Levels{sp_str}.png",
           bbox_inches="tight", dpi=200)

pl.show()
pl.close()



#%% prepare the confusion matrix
conf_test = confusion_matrix(test_y, pred_test)
conf_test_all = confusion_matrix(test_y_all, pred_test_all)

labels = conf_helper(test_y_all, conf_test_all)
conf_data = conf_test_all / np.tile(np.sum(conf_test_all , axis=1), reps=(2, 1)).T


#%% plot the confusion matrix
fig = pl.figure(figsize=(3, 1.75))
ax00 = fig.add_subplot()
hm = sns.heatmap(conf_data, annot=labels, fmt="", cmap="Blues", ax=ax00)

ax00.set_xlabel("Predicted danger")
ax00.set_ylabel("True danger")
# ax00.set_title("Confusion matrix test_all")
ax00.set_title(f"{a_ps[a_p]}")

os.makedirs(modev_path, exist_ok=True)
pl.savefig(modev_path + f"{model_ty}_ConvMat_{a_p_str}_{ndlev}Levels{sp_str}.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% generate a suffix for the number of features
nbest_suff = ""
if feats == "best":
    nbest_suff = f"_{n_best:02}best"
# end if


#%% set up the model name
mod_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}{bal_suff}{nbest_suff}_" + \
                                                                                    f"{cw_str}_{a_p_str}{sp_str}.joblib"
bundle_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}{bal_suff}{nbest_suff}_" + \
                                                                              f"{cw_str}_{a_p_str}{sp_str}_wData.joblib"


#%% store the model
os.makedirs(mod_path, exist_ok=True)
dump(model, f"{mod_path}/{mod_name}")

# generate the bundle
bundle = {"model":model,
          "train_x":train_x, "train_y":train_y,
          "test_x":test_x, "test_y":test_y,
          "train_x_all":train_x_all, "train_y_all":train_y_all,
          "test_x_all":test_x_all, "test_y_all":test_y_all}
dump(bundle, f"{mod_path}/{bundle_name}")


#%% NEW PROCEDURE: If we use the full set of best features (i.e., feats = best and n_best = 0), generate a new best
#                  features list, since the order of features likely changes if the model is trained again after the
#                  iterative feature search. For this purpose create a new file.
if gen_new_best_feats:
    f_sort_perm = np.argsort(model.feature_importances_)[::-1]
    f_imp_sorted = model.feature_importances_[f_sort_perm]
    f_sorted = model.feature_names_in_[f_sort_perm]

    best_path = f"{data_path}/Feature_Selection/{elev_dir}/"
    best_name = f"BestFeats_{model_ty}_{ndlev}DL_{reg_code_save}_{agg_str}" + \
                                                           f"{elev_n}_{sea}{bal_suff}_{cw_str}_{a_p_str}_PostSearch.csv"


    pd.DataFrame({"feature":f_sorted, "importances":f_imp_sorted}).set_index("feature").to_csv(best_path + best_name)

    print("\nNew best features list produced to be found in:\n")
    print(best_path)
    print(best_name)

# end if


#%% experiment with the "distribution-oriented" measures suggested in Murphy (1993)

# reliability
reli_part = []
for lev_pred in np.arange(ndlev):
    # the average true level for a given predicted level
    lev_true = np.mean(test_y[pred_test == lev_pred])
    print(f"\nThe average true level for predicted level {lev_pred} is {lev_true}")
    reli_part.append(lev_true - lev_pred)
# end for lev
print(f"\nThe reliablity is {np.mean(reli_part)}")


# discrimination 1
disc1_part = []
for lev_true in np.arange(ndlev):
    # the average predicted level for a given true level
    lev_pred = np.mean(pred_test[test_y == lev_true])
    print(f"\nThe average predicted level for true level {lev_true} is {lev_pred}")
    disc1_part.append(lev_true - lev_pred)
# end for lev
print(f"\nThe discrimination 1 is {np.mean(disc1_part)}")


#%% calculate some of the distribution-oriented metrics suggested in Murphy (1993)
dist_metrics_test = dist_metrics(test_y, pred_test, verbose=True)

