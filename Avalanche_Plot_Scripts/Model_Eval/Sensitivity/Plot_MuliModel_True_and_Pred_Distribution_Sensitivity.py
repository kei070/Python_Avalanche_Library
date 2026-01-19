#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENSITIVITY ANALYSIS FOR RF MODEL UNCERTAINTY

Plot a multi-panel of the distributions of true and predicted avalanche days; one panel per random forest model.
"""

#%% imports
import sys
import numpy as np
import seaborn as sns
import pylab as pl
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import gridspec

from ava_functions.Model_Fidelity_Metrics import mod_metrics
from ava_functions.ConfMat_Helper import conf_helper
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Features import feats_all


#%% set some parameters
model_ty = "RF"
ndlev = 2
sea = "full"
agg_type = "mean"
perc = 0
with_snowpack = True


#%% generate a name prefix/suffix depending on the gridcell aggregation

# make sure that the percentile is 0 if the type is not percentile
if agg_type != "percentile":
    perc = 0
# end if

agg_str = f"{agg_type.capitalize()}{perc if perc != 0 else ''}"


#%% add a string to the file names depending on the inclusion of SNOWPACK-derived data
if with_snowpack:
    sp_str = ""
else:
    sp_str = "_wo_SNOWP"
# end if else


#%% set paths
mod_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/{agg_str}/Elev_Agg/00_Final/"
pl_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/Mean/Elev_Agg/00_Final/Plots/"


#%% generate the model names
n_feats = {"21_23":[20, 20, 35, 20], "18_22":[60, 40, 40, 30], "19_24":[40, 60, 35, 60], "20_22":[50, 30, 30, 30]}

win_buns = {}
pwl_buns = {}
wet_buns = {}
adl_buns = {}

win_preds = {}
pwl_preds = {}
wet_preds = {}
adl_preds = {}

win_confs = {}
pwl_confs = {}
wet_confs = {}
adl_confs = {}

win_freqs = {}
pwl_freqs = {}
wet_freqs = {}
adl_freqs = {}

for test_yrs in ["21_23", "18_22", "19_24", "20_22"]:
    test_yrs_str = f"wo_{test_yrs}"

    # set the names
    win_name = f"{test_yrs_str}/{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_{n_feats[test_yrs][0]}best_CW1_1_" + \
                                                                        f"wind_slab{sp_str}_wData_{test_yrs_str}.joblib"
    pwl_name = f"{test_yrs_str}/{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_{n_feats[test_yrs][1]}best_CW1_1_" + \
                                                                         f"pwl_slab{sp_str}_wData_{test_yrs_str}.joblib"
    wet_name = f"{test_yrs_str}/{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_{n_feats[test_yrs][2]}best_CW1_1_" + \
                                                                              f"wet{sp_str}_wData_{test_yrs_str}.joblib"
    adl_name = f"{test_yrs_str}/{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_{n_feats[test_yrs][3]}best_CW1_1_" + \
                                                                          f"general{sp_str}_wData_{test_yrs_str}.joblib"

    # load the models
    win_bun = load(mod_path + win_name)
    pwl_bun = load(mod_path + pwl_name)
    wet_bun = load(mod_path + wet_name)
    adl_bun = load(mod_path + adl_name)

    win_buns[test_yrs] = win_bun
    pwl_buns[test_yrs] = pwl_bun
    wet_buns[test_yrs] = wet_bun
    adl_buns[test_yrs] = adl_bun

    # generate the prediction data
    win_pred = win_bun["model"].predict(win_bun["test_x_all"])
    pwl_pred = pwl_bun["model"].predict(pwl_bun["test_x_all"])
    wet_pred = wet_bun["model"].predict(wet_bun["test_x_all"])
    adl_pred = adl_bun["model"].predict(adl_bun["test_x_all"])

    win_preds[test_yrs] = win_pred
    pwl_preds[test_yrs] = pwl_pred
    wet_preds[test_yrs] = wet_pred
    adl_preds[test_yrs] = adl_pred

    # generate the confusion matrices
    win_conf = confusion_matrix(win_bun["test_y_all"], win_pred)
    pwl_conf = confusion_matrix(pwl_bun["test_y_all"], pwl_pred)
    wet_conf = confusion_matrix(wet_bun["test_y_all"], wet_pred)
    adl_conf = confusion_matrix(adl_bun["test_y_all"], adl_pred)

    win_confs[test_yrs] = win_conf
    pwl_confs[test_yrs] = pwl_conf
    wet_confs[test_yrs] = wet_conf
    adl_confs[test_yrs] = adl_conf

    # calculate the relative frequencies
    win_freqs[test_yrs] = {}
    pwl_freqs[test_yrs] = {}
    wet_freqs[test_yrs] = {}
    adl_freqs[test_yrs] = {}
    win_freqs[test_yrs]["true"] = np.sum(win_conf, axis=1) / np.sum(win_conf)
    win_freqs[test_yrs]["predicted"] = np.sum(win_conf, axis=0) / np.sum(win_conf)
    pwl_freqs[test_yrs]["true"] = np.sum(pwl_conf, axis=1) / np.sum(pwl_conf)
    pwl_freqs[test_yrs]["predicted"] = np.sum(pwl_conf, axis=0) / np.sum(pwl_conf)
    wet_freqs[test_yrs]["true"] = np.sum(wet_conf, axis=1) / np.sum(wet_conf)
    wet_freqs[test_yrs]["predicted"] = np.sum(wet_conf, axis=0) / np.sum(wet_conf)
    adl_freqs[test_yrs]["true"] = np.sum(adl_conf, axis=1) / np.sum(adl_conf)
    adl_freqs[test_yrs]["predicted"] = np.sum(adl_conf, axis=0) / np.sum(adl_conf)
# end for test_yrs

# sys.exit()


#%% plot
test_yr_cs = {"21_23":"black", "18_22":"red", "19_24":"blue", "20_22":"orange"}
test_yr_wd = {"21_23":0.5, "18_22":0.6, "19_24":0.7, "20_22":0.8}

fig = pl.figure(figsize=(5, 5))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)

for test_yrs in test_yr_cs.keys():
    ax00.bar(x=np.array([0, 1]), height=win_freqs[test_yrs]["true"], width=test_yr_wd[test_yrs], facecolor="none",
             edgecolor=test_yr_cs[test_yrs])
# end for test_yrs

for test_yrs in test_yr_cs.keys():
    ax01.bar(x=np.array([0, 1]), height=pwl_freqs[test_yrs]["true"], width=test_yr_wd[test_yrs], facecolor="none",
             edgecolor=test_yr_cs[test_yrs])
# end for test_yrs

for test_yrs in test_yr_cs.keys():
    ax10.bar(x=np.array([0, 1]), height=wet_freqs[test_yrs]["true"], width=test_yr_wd[test_yrs], facecolor="none",
             edgecolor=test_yr_cs[test_yrs])
# end for test_yrs

for test_yrs in test_yr_cs.keys():
    ax11.bar(x=np.array([0, 1]), height=adl_freqs[test_yrs]["true"], width=test_yr_wd[test_yrs], facecolor="none",
             edgecolor=test_yr_cs[test_yrs])
# end for test_yrs

pl.show()
pl.close()


#%% plot 2
y_lims = (0, 0.49)
width = 0.4
bar_cols = ["black", "red"]

fig = pl.figure(figsize=(5, 5))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)

ax00.bar([-0.2, 0.2], [win_freqs["21_23"]["true"][1], win_freqs["21_23"]["predicted"][1]], width=width,
         color=bar_cols)
ax00.bar([0.8, 1.2], [win_freqs["18_22"]["true"][1], win_freqs["18_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax00.bar([1.8, 2.2], [win_freqs["19_24"]["true"][1], win_freqs["19_24"]["predicted"][1]], width=width,
         color=bar_cols)
ax00.bar([2.8, 3.2], [win_freqs["20_22"]["true"][1], win_freqs["20_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax00.set_title("(a) Wind slab")

ax01.bar([-0.2, 0.2], [pwl_freqs["21_23"]["true"][1], pwl_freqs["21_23"]["predicted"][1]], width=width,
         color=bar_cols)
ax01.bar([0.8, 1.2], [pwl_freqs["18_22"]["true"][1], pwl_freqs["18_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax01.bar([1.8, 2.2], [pwl_freqs["19_24"]["true"][1], pwl_freqs["19_24"]["predicted"][1]], width=width,
         color=bar_cols)
ax01.bar([2.8, 3.2], [pwl_freqs["20_22"]["true"][1], pwl_freqs["20_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax01.set_title("(b) PWL slab")

ax10.bar([-0.2, 0.2], [wet_freqs["21_23"]["true"][1], wet_freqs["21_23"]["predicted"][1]], width=width,
         color=bar_cols)
ax10.bar([0.8, 1.2], [wet_freqs["18_22"]["true"][1], wet_freqs["18_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax10.bar([1.8, 2.2], [wet_freqs["19_24"]["true"][1], wet_freqs["19_24"]["predicted"][1]], width=width,
         color=bar_cols)
ax10.bar([2.8, 3.2], [wet_freqs["20_22"]["true"][1], wet_freqs["20_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax10.set_title("(c) Wet")

ax11.bar([-0.2, 0.2], [adl_freqs["21_23"]["true"][1], adl_freqs["21_23"]["predicted"][1]], width=width,
         color=bar_cols)
ax11.bar([0.8, 1.2], [adl_freqs["18_22"]["true"][1], adl_freqs["18_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax11.bar([1.8, 2.2], [adl_freqs["19_24"]["true"][1], adl_freqs["19_24"]["predicted"][1]], width=width,
         color=bar_cols)
ax11.bar([2.8, 3.2], [adl_freqs["20_22"]["true"][1], adl_freqs["20_22"]["predicted"][1]], width=width,
         color=bar_cols)
ax11.set_title("(d) General")

p00 = ax10.scatter([], [], color=bar_cols[0], label="True", marker="s")
p01 = ax10.scatter([], [], color=bar_cols[1], label="Predicted", marker="s")
ax10.legend(handles=[p00, p01])

ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax01.set_yticklabels([])
ax11.set_yticklabels([])

ax00.set_ylim(y_lims)
ax01.set_ylim(y_lims)
ax10.set_ylim(y_lims)
ax11.set_ylim(y_lims)

ax00.set_xticks([0, 1, 2, 3])
ax01.set_xticks([0, 1, 2, 3])

ax10.set_xticks([0, 1, 2, 3])
ax10.set_xticklabels(["21/23", "18/22", "19/24", "20/22"])
ax11.set_xticks([0, 1, 2, 3])
ax11.set_xticklabels(["21/23", "18/22", "19/24", "20/22"])

ax00.set_ylabel("Fraction of days")
ax10.set_ylabel("Fraction of days")

ax10.set_xlabel("Excluded years")
ax11.set_xlabel("Excluded years")

fig.suptitle("True and predicted fraction of AvDs")

fig.subplots_adjust(wspace=0.05)

pl.savefig(pl_path + "True_and_Predicted_AvD_Fraction.pdf", bbox_inches="tight", dpi=150)

pl.show()
pl.close()


