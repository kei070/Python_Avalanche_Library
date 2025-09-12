#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform a leave-one-out validation of random forest model.
"""


#%% imports
import os
import sys
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from matplotlib import gridspec
from timeit import default_timer as timer
from joblib import dump, Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from ava_functions.Data_Loading import load_features2, load_snowpack_stab, load_agg_feats_adl
from ava_functions.Helpers_Load_Data import extract_sea
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.StatMod import stat_mod
from ava_functions.ConfMat_Helper import conf_helper
from ava_functions.Model_Fidelity_Metrics import mod_metrics, dist_metrics
from ava_functions.Lists_and_Dictionaries.Features import se_norge_feats, nora3_clean
from ava_functions.Lists_and_Dictionaries.Paths import path_par, obs_path


#%% set parameters
n_best = 0  # set to 0 to use all best features -- this is required to generate a new best features list
feats = "all"
a_p = "wind"

class_weight = {0:1, 1:1}
cw_str = f"CW{'_'.join([str(k).replace('.', 'p') for k in class_weight.values()])}"
print(f"\nUsing the following class weights: {class_weight}\n")

n_estimators = [500]  # tested: [200, 500, 800]
max_depths = [50]  # tested: [20, 50, 80]

balance_meth = "SMOTE"
gen_feats_list = True  # set to true if new best features list should be created (only if feats=best and n_best=0)

min_samps = [5, 15, 22, 30, 50, 60]

# do not use 2021, 2023 in the model optimisation because they comprise the test data
# splits = [[2017, 2020], [2019, 2022], [2018, 2024]] #  [2021, 2022], [2023, 2024]]
splits = [[2018], [2019], [2020], [2022], [2024]]

with_snowpack = True
slope_angle = 0  # "agg"
slope_azi = 0  # "agg"
model_ty = "RF"
sea = "full"
balancing = "external"
add_feats = [] # ["snow_depth3_emax"]
drop_feats = []  # ["r3_emax"]
ndlev = 2
reg_code = 0
agg_type = "mean"
perc = 75
h_low = -1
h_hi = -1

a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

try:
    test_pers = [f"{k[0]}_{k[1]}" for k in splits]
except:
    test_pers = [f"{k[0]}" for k in splits]
# end try except

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
data_dict = {}
for split, test_per in zip(splits, test_pers):
    # test_per = "_".join([str(splits[0]), str(splits[1])])
    # print(f"{test_per}...")
    data_dict[test_per] =\
                       extract_sea(all_df=feats_df, drop=drop, split=split, balance_meth=balance_meth,
                                   target_n=a_p, ord_vars=["ftc_emax", "ava_clim"],
                                   ord_lims={"ftc_emax":[0, 1], "ava_clim":[1, 3]})

# end for splits


#%% def function
def calc_metrics(test_per, data_dict, ndlev, class_weight, hypp_set):

    result = {}

    train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all = data_dict[test_per]

    model = stat_mod(model_ty=model_ty, ndlev=ndlev, verbose=False, class_weight=class_weight, hyperp=hypp_set)

    #% train the model
    model.fit(train_x, train_y)

    #% predict
    pred_test_all = model.predict(test_x_all)

    #% model accuracy
    acc_test_all = accuracy_score(test_y_all, pred_test_all)
    result["acc_test"] = acc_test_all

    #% calculate precision, recall, F1, support
    result["cr_vals"] = precision_recall_fscore_support(test_y_all, pred_test_all)

    #% prepare the confusion matrix
    conf_test_all = confusion_matrix(test_y_all, pred_test_all)

    if ndlev == 2:
        result["metrics"] = mod_metrics(conf_test_all)
        result["score"] = (2*result["metrics"]["POD"] + result["metrics"]["PON"] + 2*(1-result["metrics"]["FAR"]) +
                           result["metrics"]["RPC"]) / 6
        result["dist_met"] = dist_metrics(true_y=test_y_all, pred_y=pred_test_all)
    # end if

    return result
# end def


#%% set up the model
n_iter = len(min_samps) * len(data_dict)

min_samp_dict = {}
count = 0
start_ti = timer()
for max_depth in max_depths:
    min_samp_dict[max_depth] = {}
    for n_estimator in n_estimators:
        min_samp_dict[max_depth][n_estimator] = {}
        for min_samp in min_samps:

            hypp_set = {'n_estimators':n_estimator,  # [100, 200, 300],
                        'max_depth':max_depth,  # [2, 5, 8, 11, 14, 17, 20],
                        'min_samples_leaf':min_samp,
                        'min_samples_split':min_samp,
                        'max_features':"sqrt",
                        'bootstrap':True
                        }

            min_samp_dict[max_depth][n_estimator][min_samp] = []

            min_samp_dict[max_depth][n_estimator][min_samp].append(Parallel(n_jobs=-1)(delayed(calc_metrics)(test_per,
                                                                                                             data_dict,
                                                                                                  ndlev, class_weight,
                                                                                   hypp_set) for test_per in test_pers))

        # end for min_samp
    # end for n_estimator
# end for max_depth
end_ti = timer()
elapsed_ti = end_ti - start_ti
print(f"Model training time: {elapsed_ti:.2f} seconds.")


#%% lines plot -- 4 panels
if len(splits) == 4:
    width = 0.1
    x1, x2 = np.array([-0.15, -0.05]), np.array([0.05, 0.15])

    a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

    fig = pl.figure(figsize=(6, 4.5))

    ax00 = fig.add_subplot(221)
    ax01 = fig.add_subplot(222)
    ax10 = fig.add_subplot(223)
    ax11 = fig.add_subplot(224)
    axes = [ax00, ax01, ax10, ax11]

    for j in np.arange(len(test_pers)):
        i = 0

        # precision non-AvD
        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][0][0] for min_samp in min_samps],
                     color="black", marker="o")
        # precision AvD
        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][0][1] for min_samp in min_samps],
                     color="black", marker="x")
        # recall non-AvD
        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][1][0] for min_samp in min_samps],
                     color="red", marker="o")
        # recall AvD
        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][1][1] for min_samp in min_samps],
                     color="red", marker="x")

        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["acc_test"] for min_samp in min_samps],
                     color="blue", marker="s")

        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["metrics"]["FAR"] for min_samp in min_samps],
                     color="gray", marker="d")
        axes[j].set_title("-".join(test_pers[j].split("_")))
    # end for j

    ax00.plot([], [], marker="o", c="black", label="PR non-AvD")
    ax00.plot([], [], marker="x", c="black", label="PR AvD")
    ax10.plot([], [], marker="o", c="red", label="RC non-AvD")
    ax10.plot([], [], marker="x", c="red", label="RC AvD")
    ax11.plot([], [], marker="s", c="blue", label="ACC")
    ax11.plot([], [], marker="d", c="gray", label="FAR")

    ax00.legend(ncol=1)
    ax10.legend(ncol=1)
    ax11.legend(ncol=2)

    ax00.set_xticklabels([])
    ax01.set_xticklabels([])
    ax01.set_yticklabels([])
    ax11.set_yticklabels([])

    ax10.set_xticks(min_samps)
    ax10.set_xticklabels(min_samps)
    ax11.set_xticks(min_samps)
    ax11.set_xticklabels(min_samps)

    ax00.set_ylabel("Metric")
    ax10.set_ylabel("Metric")
    # ax10.set_xlabel("min_sample_leaf & _split")
    # ax11.set_xlabel("min_sample_leaf & _split")
    ax10.set_xlabel("MSL & MSF")
    ax11.set_xlabel("MSL & MSF")

    ax00.set_ylim(0, 1)
    ax01.set_ylim(0, 1)
    ax10.set_ylim(0, 1)
    ax11.set_ylim(0, 1)

    fig.suptitle(a_ps[a_p] + " problem")
    fig.subplots_adjust(wspace=0.1)

    pl_path = f"{obs_path}/IMPETUS/Publishing/The Cryosphere/Avalanche_Paper_2/00_Figures/"
    cw_str = "_".join([str(k).replace(".", "p") for k in class_weight.values()])
    feats_str = f"{n_best}_{feats}Feats" if feats == "best" else f"{feats}Feats"
    pl.savefig(pl_path + f"Scores_AllSeasons_{a_p}_{feats_str}_CW{cw_str}.pdf", bbox_inches="tight", dpi=200)

    pl.show()
    pl.close()
# end if


#%% lines plot -- 3 panels
if len(splits) == 3:
    width = 0.1
    x1, x2 = np.array([-0.15, -0.05]), np.array([0.05, 0.15])

    a_ps = {"wind_slab":"Wind slab", "pwl_slab":"PWL slab", "wet":"Wet", "y":"General"}

    fig = pl.figure(figsize=(8, 3))

    ax00 = fig.add_subplot(131)
    ax01 = fig.add_subplot(132)
    ax02 = fig.add_subplot(133)
    axes = [ax00, ax01, ax02]

    for j in np.arange(len(test_pers)):
        i = 0
        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][0][0] for min_samp in min_samps],
                     color="black", marker="o")
        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][0][1] for min_samp in min_samps],
                     color="black", marker="x")

        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][1][0] for min_samp in min_samps],
                     color="red", marker="o")
        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["cr_vals"][1][1] for min_samp in min_samps],
                     color="red", marker="x")

        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["acc_test"] for min_samp in min_samps],
                     color="blue", marker="s")

        axes[j].plot(min_samps, [min_samp_dict[min_samp][0][j]["metrics"]["FAR"] for min_samp in min_samps],
                     color="gray", marker="d")
        axes[j].set_title(" & ".join(test_pers[j].split("_")))
    # end for j

    ax00.plot([], [], marker="o", c="black", label="PR non-AvD")
    ax00.plot([], [], marker="x", c="black", label="PR AvD")
    ax01.plot([], [], marker="o", c="red", label="RC non-AvD")
    ax01.plot([], [], marker="x", c="red", label="RC AvD")
    ax02.plot([], [], marker="s", c="blue", label="ACC")
    ax02.plot([], [], marker="d", c="gray", label="FAR")

    ax00.legend(ncol=1)
    ax01.legend(ncol=1)
    ax02.legend(ncol=1)

    ax00.set_xlabel("MSL & MSF")
    ax01.set_xlabel("MSL & MSF")
    ax02.set_xlabel("MSL & MSF")

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


#%% plot my "weighted score" together with sum other more sophisticated scores such as true skill score
colors1 = ["black", "gray", "red"]
colors2 = ["orange", "blue", "violet"]

fig = pl.figure(figsize=(6, 3))
ax00 = fig.add_subplot(111)
i = 0
for max_depth in max_depths:
    for n_estimator in n_estimators:
        wss = []
        tss = []
        hss = []
        pss = []
        acc = []
        far = []
        recall0 = []
        precis0 = []
        recall1 = []
        precis1 = []
        f1mac = []
        avg_true0 = []
        avg_true1 = []
        avg_pred0 = []
        avg_pred1 = []
        for j in np.arange(len(test_pers)):

            recall0.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["cr_vals"][0][0] for min_samp in
                                                                                                             min_samps])
            recall1.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["cr_vals"][0][1] for min_samp in
                                                                                                             min_samps])

            precis0.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["cr_vals"][1][0] for min_samp in
                                                                                                             min_samps])
            precis1.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["cr_vals"][1][1] for min_samp in
                                                                                                             min_samps])

            f1mac.append([np.mean(min_samp_dict[max_depth][n_estimator][min_samp][0][j]["cr_vals"][2][:]) for min_samp
                                                                                                          in min_samps])

            acc.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["acc_test"] for min_samp in min_samps])
            wss.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["score"] for min_samp in min_samps])
            tss.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["metrics"]["TSS"] for min_samp in
                                                                                                             min_samps])
            hss.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["metrics"]["HSS"] for min_samp in
                                                                                                             min_samps])
            pss.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["metrics"]["PSS"] for min_samp in
                                                                                                             min_samps])
            far.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["metrics"]["FAR"] for min_samp in
                                                                                                             min_samps])

            avg_true0.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["dist_met"]\
                                                                           ["avg_true"]["0"] for min_samp in min_samps])

            ax00.scatter(min_samps, [min_samp_dict[max_depth][n_estimator][min_samp][0][j]["score"] for min_samp in
                                                                                                             min_samps],
                         facecolor="none", edgecolor=colors1[i], marker="d")
        # end for j

        recall0 = np.array(recall0)
        recall0_means = [np.mean(recall0[:, k]) for k in np.arange(len(min_samps))]
        recall0_stds = [np.std(recall0[:, k]) for k in np.arange(len(min_samps))]

        recall1 = np.array(recall1)
        recall1_means = [np.mean(recall1[:, k]) for k in np.arange(len(min_samps))]
        recall1_stds = [np.std(recall1[:, k]) for k in np.arange(len(min_samps))]

        precis0 = np.array(precis0)
        precis0_means = [np.mean(precis0[:, k]) for k in np.arange(len(min_samps))]
        precis0_stds = [np.std(precis0[:, k]) for k in np.arange(len(min_samps))]

        precis1 = np.array(precis1)
        precis1_means = [np.mean(precis1[:, k]) for k in np.arange(len(min_samps))]
        precis1_stds = [np.std(precis1[:, k]) for k in np.arange(len(min_samps))]

        acc = np.array(acc)
        acc_means = [np.mean(acc[:, k]) for k in np.arange(len(min_samps))]
        acc_stds = [np.std(acc[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, acc_means, facecolor="none", edgecolor="black", marker="o", s=80)
        ax00.errorbar(min_samps, acc_means, yerr=acc_stds, color="black", linestyle="-", capsize=5)

        far = np.array(far)
        far_means = [np.mean(far[:, k]) for k in np.arange(len(min_samps))]
        far_stds = [np.std(far[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, far_means, facecolor="none", edgecolor="black", marker="o", s=80)
        ax00.errorbar(min_samps, far_means, yerr=far_stds, color="black", linestyle="--", capsize=5)

        f1mac = np.array(f1mac)
        f1mac_means = [np.mean(f1mac[:, k]) for k in np.arange(len(min_samps))]
        f1mac_stds = [np.std(f1mac[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, f1mac_means, facecolor="none", edgecolor="black", marker="o", s=80)
        ax00.errorbar(min_samps, f1mac_means, yerr=f1mac_stds, color="black", linestyle="-.", capsize=5)

        tss = np.array(tss)
        tss_means = [np.mean(tss[:, k]) for k in np.arange(len(min_samps))]
        tss_stds = [np.std(tss[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, tss_means, facecolor="none", edgecolor="black", marker="o", s=80)
        ax00.errorbar(min_samps, tss_means, yerr=tss_stds, color="black", linestyle=":", capsize=5)


        """
        wss = np.array(wss)
        means = [np.mean(wss[:, k]) for k in np.arange(len(min_samps))]
        stds = [np.std(wss[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, means, facecolor="none", edgecolor="black", marker="o", s=80)
        ax00.errorbar(min_samps, means, yerr=stds, color="black", capsize=5)

        hss = np.array(hss)
        hss_means = [np.mean(hss[:, k]) for k in np.arange(len(min_samps))]
        hss_stds = [np.std(hss[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, hss_means, facecolor="none", edgecolor="blue", marker="o", s=80)
        ax00.errorbar(min_samps, hss_means, yerr=hss_stds, color="blue", capsize=5)

        pss = np.array(pss)
        pss_means = [np.mean(pss[:, k]) for k in np.arange(len(min_samps))]
        pss_stds = [np.std(pss[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, pss_means, facecolor="none", edgecolor="orange", marker="o", s=80)
        ax00.errorbar(min_samps, pss_means, yerr=pss_stds, color="orange", capsize=5)
        """
        if len(n_estimators) > 1:
            i += 1
        # end if
    # end for n_estimator
    if len(max_depths) > 1:
        i += 1
    # end if
# end for max_depth

ax00.plot([], [], c="black", linestyle="-", label="PC")
ax00.plot([], [], c="black", linestyle="--", label="FAR")
ax00.plot([], [], c="black", linestyle="-.", label="F1-macro")
ax00.plot([], [], c="black", linestyle=":", label="TSS")

"""
ax00.plot([], [], c="black", label="WSS")
ax00.plot([], [], c="blue", label="HSS")
ax00.plot([], [], c="orange", label="PSS")
"""

ax00.legend(ncol=5)

if len(max_depths) != len(n_estimators):
    for i in np.arange(np.max([len(max_depths), len(n_estimators)])):
        if len(max_depths) > len(n_estimators):
            ax00.plot([], [], c=colors2[i], label=max_depths[i])
        else:
            ax00.plot([], [], c=colors2[i], label=n_estimators[i])
        # end if else
    # end for i
    l_title = "max_depth" if len(max_depths) > len(n_estimators) else "n_estimators"
    ax00.legend(title=l_title)
# end if

ax00.set_ylabel("Score")
ax00.set_xlabel("MSL & MSF")
ax00.set_title(f"Scores {a_ps[a_p]} problem")

ax00.set_ylim(0, 1)

if len(max_depths) != len(n_estimators):
    p_add = "max_depth" if len(max_depths) < len(n_estimators) else "n_estimators"
    p_add += str(max_depths[0]) if len(max_depths) < len(n_estimators) else str(n_estimators[0])
# end if

pl_path = f"{obs_path}/IMPETUS/NORA3/Plots/Model_Evaluation/With_SNOWPACK/WeightedScore_HyppTest/"
# pl.savefig(pl_path + f"WeightedScore_{a_p}_{p_add}.png", dpi=200, bbox_inches="tight")
pl.savefig(pl_path + f"WSS_TSS_HSS_PSS_{a_p}.png", dpi=200, bbox_inches="tight")
pl.savefig(pl_path + f"WSS_TSS_HSS_PSS_{a_p}.pdf", dpi=200, bbox_inches="tight")

pl.show()
pl.close()


#%% plot the precision and recall scores separately for the AvDs and non-AvDs
fig = pl.figure(figsize=(6, 3))
ax00 = fig.add_subplot(111)

ax00.scatter(min_samps, recall0_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(min_samps, recall0_means, yerr=recall0_stds, color="black", linestyle="-", capsize=5, label="R nAvD")

ax00.scatter(min_samps, recall1_means, facecolor="none", edgecolor="black", marker="o", s=80)
ax00.errorbar(min_samps, recall1_means, yerr=recall1_stds, color="black", linestyle="--", capsize=5, label="R AvD")

ax00.scatter(min_samps, precis0_means, facecolor="none", edgecolor="gray", marker="o", s=80)
ax00.errorbar(min_samps, precis0_means, yerr=precis0_stds, color="gray", linestyle="-", capsize=5, label="P nAvD")

ax00.scatter(min_samps, precis1_means, facecolor="none", edgecolor="gray", marker="o", s=80)
ax00.errorbar(min_samps, precis1_means, yerr=precis1_stds, color="gray", linestyle="--", capsize=5, label="P AvD")

ax00.legend()

ax00.set_ylabel("Score")
ax00.set_xlabel("MSL & MSF")
ax00.set_title(f"Scores {a_ps[a_p]} problem")

pl.show()
pl.close()


#%% plot only some Murphy (1993)-inspired distribution-oriented measures

colors1 = ["black", "gray", "red"]
colors2 = ["orange", "blue", "violet"]

lw2 = 0.65

fig = pl.figure(figsize=(6, 3))
ax00 = fig.add_subplot(111)
i = 0
for max_depth in max_depths:
    for n_estimator in n_estimators:
        avg_true0 = []
        avg_true1 = []
        reliab = []
        avg_pred0 = []
        avg_pred1 = []
        discri = []
        for j in np.arange(len(test_pers)):
            avg_true0.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["dist_met"]\
                                                                           ["avg_true"]["0"] for min_samp in min_samps])
            avg_true1.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["dist_met"]\
                                                                           ["avg_true"]["1"] for min_samp in min_samps])
            reliab.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["dist_met"]\
                                                                             ["reliability"] for min_samp in min_samps])

            avg_pred0.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["dist_met"]\
                                                                           ["avg_pred"]["0"] for min_samp in min_samps])
            avg_pred1.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["dist_met"]\
                                                                           ["avg_pred"]["1"] for min_samp in min_samps])
            discri.append([min_samp_dict[max_depth][n_estimator][min_samp][0][j]["dist_met"]\
                                                                         ["discrimination1"] for min_samp in min_samps])

            # ax00.scatter(min_samps, [min_samp_dict[max_depth][n_estimator][min_samp][0][j]["score"] for min_samp in
            #                                                                                               min_samps],
            #              facecolor="none", edgecolor=colors1[i], marker="d")
        # end for j

        avg_true0 = np.array(avg_true0)
        avg_true0_means = [np.mean(avg_true0[:, k]) for k in np.arange(len(min_samps))]
        avg_true0_stds = [np.std(avg_true0[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, avg_true0_means, facecolor="none", edgecolor="gray", marker="o", s=80)
        ax00.errorbar(min_samps, avg_true0_means, yerr=avg_true0_stds, color="gray", capsize=5, linewidth=lw2)

        avg_true1 = np.array(avg_true1)
        avg_true1_means = [np.mean(avg_true1[:, k]) for k in np.arange(len(min_samps))]
        avg_true1_stds = [np.std(avg_true1[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, avg_true1_means, facecolor="none", edgecolor="gray", marker="d", s=80)
        ax00.errorbar(min_samps, avg_true1_means, yerr=avg_true1_stds, color="gray", capsize=5, linewidth=lw2)

        reliab = np.array(reliab)
        reliab_means = [np.mean(reliab[:, k]) for k in np.arange(len(min_samps))]
        reliab_stds = [np.std(reliab[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, reliab_means, facecolor="none", edgecolor="blue", marker="s", s=80)
        ax00.errorbar(min_samps, reliab_means, yerr=reliab_stds, color="blue", capsize=5)

        avg_pred0 = np.array(avg_pred0)
        avg_pred0_means = [np.mean(avg_pred0[:, k]) for k in np.arange(len(min_samps))]
        avg_pred0_stds = [np.std(avg_pred0[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, avg_pred0_means, facecolor="none", edgecolor="black", marker="o", s=80)
        ax00.errorbar(min_samps, avg_pred0_means, yerr=avg_pred0_stds, color="black", capsize=5, linewidth=lw2)

        avg_pred1 = np.array(avg_pred1)
        avg_pred1_means = [np.mean(avg_pred1[:, k]) for k in np.arange(len(min_samps))]
        avg_pred1_stds = [np.std(avg_pred1[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, avg_pred1_means, facecolor="none", edgecolor="black", marker="d", s=80)
        ax00.errorbar(min_samps, avg_pred1_means, yerr=avg_pred1_stds, color="black", capsize=5, linewidth=lw2)

        discri = np.array(discri)
        discri_means = [np.mean(discri[:, k]) for k in np.arange(len(min_samps))]
        discri_stds = [np.std(discri[:, k]) for k in np.arange(len(min_samps))]
        ax00.scatter(min_samps, discri_means, facecolor="none", edgecolor="red", marker="s", s=80)
        ax00.errorbar(min_samps, discri_means, yerr=discri_stds, color="red", capsize=5)


        if len(n_estimators) > 1:
            i += 1
        # end if
    # end for n_estimator
    if len(max_depths) > 1:
        i += 1
    # end if
# end for max_depth

ax00.plot([], [], c="gray", marker="o", label="AvgTrue0", linewidth=lw2)
ax00.plot([], [], c="gray", marker="d", label="AvgTrue1", linewidth=lw2)
ax00.plot([], [], c="blue", marker="s", label="Reliability")
ax00.plot([], [], c="black", marker="o", label="AvgPred0", linewidth=lw2)
ax00.plot([], [], c="black", marker="d", label="AvgPred1", linewidth=lw2)
ax00.plot([], [], c="red", marker="s", label="Discrimination")
ax00.legend(ncol=2)


if len(max_depths) != len(n_estimators):
    for i in np.arange(np.max([len(max_depths), len(n_estimators)])):
        if len(max_depths) > len(n_estimators):
            ax00.plot([], [], c=colors2[i], label=max_depths[i])
        else:
            ax00.plot([], [], c=colors2[i], label=n_estimators[i])
        # end if else
    # end for i
    l_title = "max_depth" if len(max_depths) > len(n_estimators) else "n_estimators"
    ax00.legend(title=l_title)
# end if

ax00.set_ylabel("Score")
ax00.set_xlabel("MSL & MSF")
ax00.set_title(f"Distribution measures {a_ps[a_p]} problem")

ax00.set_ylim((0, 1))

if len(max_depths) != len(n_estimators):
    p_add = "max_depth" if len(max_depths) < len(n_estimators) else "n_estimators"
    p_add += str(max_depths[0]) if len(max_depths) < len(n_estimators) else str(n_estimators[0])
# end if
