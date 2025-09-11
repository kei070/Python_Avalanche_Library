#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a multi-panel confusion matrix plot for the different avalanche problems.
"""

#%% imports
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


#%% set a dictionary for the feature names
"""
feats_m = {'t_mean':"t1", 't_max':"tmax", 't_min':"tmin", 't_range':"dtr",
               'ftc':"ftc",
               't3':"t3", 't7':"t7",
               'tmax3':'tmax3', 'tmax7':'tmax7',
               'dtemp1':'dtr1', 'dtemp2':'dtr2', 'dtemp3':'dtr3',
               'dtempd1':'dtrd1', 'dtempd2':'dtrd2', 'dtempd3':'dtrd3',
               'pdd':"pdd",
               'ws_mean':"w1", 'ws_max':"wmax", 'ws_min':"wmin", "ws_range":"dws",
               "dws1":"dws1", "dws2":"dws2", "dws3":"dws3",
               "dwsd1":"dwsd1", "dwsd2":"dwsd2", "dwsd3":"dwsd3",
               'wind_direction':"w_dir", "dwdir":"dwdir",
               "dwdir1":"dwdir1", "dwdir2":"dwdir2", "dwdir3":"dwdir3",
               'w3':"w3",
               'wmax3':'wmax3', 'wmax7':'wmax7',
               'total_prec_sum':"Ptot",
               's1':'s1', 'r1':'r1',
               'r3':'r3', 'r7':'r7',
               's3':'s3', 's7':'s7',
               'wdrift':'wdrift', 'wdrift_3':'wdrift_3',
               'wdrift3':'wdrift3', 'wdrift3_3':'wdrift3_3',
               'RH':"rh",
               'NLW':"nlw", 'NSW':"nsw",
               'RH3':"rh3", 'RH7':"rh7",
               'NLW3':"nlw3", 'NLW7':"nlw7",
               'NSW3':"nsw3", 'NSW7':"nsw7",
               "ava_clim":"aci"}


feats_sp = {"snow_depth":"SD1", "snow_depth3":"SD3", "snow_depth7":"SD7",
            "t_top":"t_top",
            "lwc_i":"lwc_i", "lwc_max":"lwc_max", "lwc_sum":"lwc_sum", "lwc_s_top":"lwc_s_top",
            "RTA_100":"SSI_100", "RTA_2":"SSI_2",
            "Sk38_100":"Sk38_100", "Sk38_2":"Sk38_2",
            "Sn38_100":"Sn38_100", "Sn38_2":"Sn38_2"}

feats_sp = {**feats_sp, **{k + "_d1":feats_sp[k] + "_d1" for k in feats_sp.keys()},
            **{k + "_d2":feats_sp[k] + "_d2" for k in feats_sp.keys()},
            **{k + "_d3":feats_sp[k] + "_d3" for k in feats_sp.keys()}}

feats_m = {**feats_m, **feats_sp}

feats_emin = {k + "_emin":feats_m[k] + "_emin" for k in feats_m.keys()}
feats_emax = {k + "_emax":feats_m[k] + "_emax" for k in feats_m.keys()}

feats_all = {**feats_m, **feats_emin, **feats_emax}
"""

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
mod_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/{agg_str}/Elev_Agg/"


#%% generate the model names
# win_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SVMSMOTE_20best_wind_slab{sp_str}_wData.joblib"
# pwl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_ADASYN_10best_pwl_slab{sp_str}_wData.joblib"
# wet_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SVMSMOTE_30best_wet{sp_str}_wData.joblib"
# adl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SVMSMOTE_20best_general{sp_str}_wData.joblib"
# win_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_30best_wind_slab{sp_str}_wData.joblib"
# pwl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_15best_pwl_slab{sp_str}_wData.joblib"
# wet_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_15best_wet{sp_str}_wData.joblib"
# adl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_30best_general{sp_str}_wData.joblib"

"""
win_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_30best_CW1_1_wind_slab{sp_str}_wData.joblib"
pwl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_30best_CW1_1_pwl_slab{sp_str}_wData.joblib"
wet_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_35best_CW1_1_wet{sp_str}_wData.joblib"
adl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_30best_CW1_1_general{sp_str}_wData.joblib"
"""

win_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_20best_CW1_1_wind_slab{sp_str}_wData.joblib"
pwl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_50best_CW1_1_pwl_slab{sp_str}_wData.joblib"
wet_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_50best_CW1_1_wet{sp_str}_wData.joblib"
adl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_15best_CW1_1_general{sp_str}_wData.joblib"


#%% load the models
win_bun = load(mod_path + win_name)
pwl_bun = load(mod_path + pwl_name)
wet_bun = load(mod_path + wet_name)
adl_bun = load(mod_path + adl_name)


#%% generate the prediction data
win_pred = win_bun["model"].predict(win_bun["test_x_all"])
pwl_pred = pwl_bun["model"].predict(pwl_bun["test_x_all"])
wet_pred = wet_bun["model"].predict(wet_bun["test_x_all"])
adl_pred = adl_bun["model"].predict(adl_bun["test_x_all"])


#%% generate the confusion matrices
win_conf = confusion_matrix(win_bun["test_y_all"], win_pred)
pwl_conf = confusion_matrix(pwl_bun["test_y_all"], pwl_pred)
wet_conf = confusion_matrix(wet_bun["test_y_all"], wet_pred)
adl_conf = confusion_matrix(adl_bun["test_y_all"], adl_pred)

win_metr = mod_metrics(win_conf)
pwl_metr = mod_metrics(pwl_conf)
wet_metr = mod_metrics(wet_conf)
adl_metr = mod_metrics(adl_conf)
metr_l = [win_metr, pwl_metr, wet_metr, adl_metr]

win_conf_dat = win_conf / np.tile(np.sum(win_conf, axis=1), reps=(2, 1)).T
pwl_conf_dat = pwl_conf / np.tile(np.sum(pwl_conf, axis=1), reps=(2, 1)).T
wet_conf_dat = wet_conf / np.tile(np.sum(wet_conf, axis=1), reps=(2, 1)).T
adl_conf_dat = adl_conf / np.tile(np.sum(adl_conf, axis=1), reps=(2, 1)).T


#%% generate the confusion matrix heatmap labels
win_labels = conf_helper(win_bun["test_y_all"], win_conf)
pwl_labels = conf_helper(pwl_bun["test_y_all"], pwl_conf)
wet_labels = conf_helper(wet_bun["test_y_all"], wet_conf)
adl_labels = conf_helper(adl_bun["test_y_all"], adl_conf)


#%% plot the confusion matrix
vmin = 0
vmax = 1

fig = pl.figure(figsize=(6.2, 4.2))
gs = gridspec.GridSpec(nrows=42, ncols=62)
ax00 = fig.add_subplot(gs[0:16, 0:28])
ax01 = fig.add_subplot(gs[0:16, 30:58])
ax10 = fig.add_subplot(gs[26:, 0:28])
ax11 = fig.add_subplot(gs[26:, 30:58])

ax_cbar = fig.add_subplot(gs[:, 59:])

hm00 = sns.heatmap(win_conf_dat, annot=win_labels, fmt="", cmap="Blues", ax=ax00, vmin=vmin, vmax=vmax, cbar=False)
hm01 = sns.heatmap(pwl_conf_dat, annot=pwl_labels, fmt="", cmap="Blues", ax=ax01, vmin=vmin, vmax=vmax, cbar=False)
hm10 = sns.heatmap(wet_conf_dat, annot=wet_labels, fmt="", cmap="Blues", ax=ax10, vmin=vmin, vmax=vmax, cbar=False)
hm11 = sns.heatmap(adl_conf_dat, annot=adl_labels, fmt="", cmap="Blues", ax=ax11, vmin=vmin, vmax=vmax, cbar=True,
                   cbar_ax=ax_cbar)

ax00.set_xlabel("Predicted danger")
ax00.set_ylabel("True danger")
ax00.set_title("(a) Wind slab")

ax01.set_xlabel("Predicted danger")
# ax01.set_ylabel("True danger")
ax01.set_title("(b) PWL slab")

ax10.set_xlabel("Predicted danger")
ax10.set_ylabel("True danger")
ax10.set_title("(c) Wet")

ax11.set_xlabel("Predicted danger")
# ax11.set_ylabel("True danger")
ax11.set_title("(d) General")

ax00.set_xticks([0.5, 1.5])
ax00.set_yticks([0.5, 1.5])
ax00.set_xticklabels(["non-AvD", "AvD"])
ax00.set_yticklabels(["non-AvD", "AvD"])

ax01.set_xticks([0.5, 1.5])
ax01.set_yticks([0.5, 1.5])
ax01.set_xticklabels(["non-AvD", "AvD"])
ax01.set_yticklabels(["", ""])

ax10.set_xticks([0.5, 1.5])
ax10.set_yticks([0.5, 1.5])
ax10.set_xticklabels(["non-AvD", "AvD"])
ax10.set_yticklabels(["non-AvD", "AvD"])

ax11.set_xticks([0.5, 1.5])
ax11.set_yticks([0.5, 1.5])
ax11.set_xticklabels(["non-AvD", "AvD"])
ax11.set_yticklabels(["", ""])


fig.subplots_adjust(wspace=0.4, hspace=0.7)

pl_path = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Avalanche_Paper_2/00_Figures/"
# pl.savefig(pl_path + "ConfMats_AllProblems.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% prepare the feature importances plot
win_imp = win_bun["model"].feature_importances_
win_feat = win_bun["model"].feature_names_in_
win_sort = np.argsort(win_imp)

pwl_imp = pwl_bun["model"].feature_importances_
pwl_feat = pwl_bun["model"].feature_names_in_
pwl_sort = np.argsort(pwl_imp)

wet_imp = wet_bun["model"].feature_importances_
wet_feat = wet_bun["model"].feature_names_in_
wet_sort = np.argsort(wet_imp)

adl_imp = adl_bun["model"].feature_importances_
adl_feat = adl_bun["model"].feature_names_in_
adl_sort = np.argsort(adl_imp)


#%% select the feature names to plot
win_feat_pl = np.array([feats_all[k] for k in win_feat])
pwl_feat_pl = np.array([feats_all[k] for k in pwl_feat])
wet_feat_pl = np.array([feats_all[k] for k in wet_feat])
adl_feat_pl = np.array([feats_all[k] for k in adl_feat])


#%% plot the feature importances
n_feats = 15
x_max = 0.17

text_x = 0.075

fig = pl.figure(figsize=(6.3, 6.3))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)

ax00.barh(win_feat_pl[win_sort][-n_feats:], win_imp[win_sort][-n_feats:])
# ax00.set_xlabel("Importance")
# ax00.set_title("(a) Wind slab")
ax00.text(text_x, 0, "(a) Wind slab")

ax00.tick_params(axis='x', labelrotation=0)
ax00.set_xticklabels([])

ax00.set_ylim((-1, n_feats))
ax00.set_xlim(0, x_max)

ax01.barh(pwl_feat_pl[pwl_sort][-n_feats:], pwl_imp[pwl_sort][-n_feats:])
# ax01.set_xlabel("Importance")
# ax01.set_title("(b) PWL slab")
ax01.text(text_x, 0, "(b) PWL slab")

ax01.tick_params(axis='x', labelrotation=0)
ax01.set_xticklabels([])

ax01.set_ylim((-1, n_feats))
ax01.set_xlim(0, x_max)

ax10.barh(wet_feat_pl[wet_sort][-n_feats:], wet_imp[wet_sort][-n_feats:])
ax10.set_xlabel("Importance")
# ax10.set_title("(c) Wet")
ax10.text(text_x, 0, "(c) Wet")

ax10.tick_params(axis='x', labelrotation=0)

ax10.set_ylim((-1, n_feats))
ax10.set_xlim(0, x_max)

ax11.barh(adl_feat_pl[adl_sort][-n_feats:], adl_imp[adl_sort][-n_feats:])
ax11.set_xlabel("Importance")
# ax11.set_title("(d) General")
ax11.text(text_x, 0, "(d) General")

ax11.tick_params(axis='x', labelrotation=0)

ax11.set_ylim((-1, n_feats))
ax11.set_xlim(0, x_max)

fig.subplots_adjust(wspace=0.75, hspace=0.04)

pl_path = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Avalanche_Paper_2/00_Figures/"
pl.savefig(pl_path + "Importances_AllProblems.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% calculate the relative frequencies of AvD and non-AvD per AP
print("Relative frequencies non-AvD  AvD")
print(f"wind slab: {np.sum(win_conf, axis=1) / np.sum(win_conf)}")
print(f"PWL slab:  {np.sum(pwl_conf, axis=1) / np.sum(pwl_conf)}")
print(f"wet:       {np.sum(wet_conf, axis=1) / np.sum(wet_conf)}")
print(f"general:   {np.sum(adl_conf, axis=1) / np.sum(adl_conf)}")


#%% plot model metrics as bar plots
metr_c = {"POD":"red", "PON":"orange", "FAR":"violet", "MFR":"blue", "RPC":"black", "TSS":"gray"}
width = 0.2
fig = pl.figure(figsize=(6.5, 3))
ax00 = fig.add_subplot(111)

x = 0
for x in [0, 1, 2, 3]:
    ax00.bar(x=x-0.3, height=metr_l[x]["POD"], width=width, color=metr_c["POD"])
    ax00.bar(x=x-0.1, height=metr_l[x]["PON"], width=width, color=metr_c["PON"])
    ax00.bar(x=x+0.1, height=metr_l[x]["FAR"], width=width, color=metr_c["FAR"])
    ax00.bar(x=x+0.3, height=metr_l[x]["MFR"], width=width, color=metr_c["MFR"])
    ax00.bar(x=x, height=metr_l[x]["RPC"], width=0.6, facecolor="none", edgecolor=metr_c["RPC"])
    ax00.bar(x=x, height=metr_l[x]["TSS"], width=0.2, facecolor="none", edgecolor=metr_c["TSS"])
# end for x

ax00.bar(x=0, height=0, width=width, color="red", label="POD")
ax00.bar(x=0, height=0, width=width, color="orange", label="PON")
ax00.bar(x=0, height=0, width=width, color="violet", label="TSS")
ax00.bar(x=0, height=0, width=width, color="blue", label="MFR")
ax00.plot([], [], c="black", label="RPC")
ax00.plot([], [], c="gray", label="FAR")

ax00.legend(ncol=3)

ax00.set_ylim(0, 1.1)

ax00.set_xticks([0, 1, 2, 3])
ax00.set_xticklabels(["Wind slab", "PWL slab", "Wet", "General"])

ax00.set_ylabel("Metric")
ax00.set_xlabel("Problem")
ax00.set_title("Model performance metrics")

pl.show()
pl.close()
