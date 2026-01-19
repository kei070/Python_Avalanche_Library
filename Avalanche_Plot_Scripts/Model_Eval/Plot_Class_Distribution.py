#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the class distribution used in RF model training.
"""

#%% imports
import numpy as np
import pylab as pl
from joblib import load

from ava_functions.Lists_and_Dictionaries.Paths import path_par


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
pl_path = f"{path_par}/IMPETUS/NORA3/Stored_Models/{agg_str}/Elev_Agg/00_Final/Plots/"


#%% generate the model names

# models used for the paper
win_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_60best_CW1_1_wind_slab{sp_str}_wData.joblib"
pwl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_50best_CW1_1_pwl_slab{sp_str}_wData.joblib"
wet_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_35best_CW1_1_wet{sp_str}_wData.joblib"
adl_name = f"{model_ty}_{ndlev}DL_AllReg_{agg_str}_{sea}_SMOTE_20best_CW1_1_general{sp_str}_wData.joblib"


#%% load the models
win_bun = load(mod_path + win_name)
pwl_bun = load(mod_path + pwl_name)
wet_bun = load(mod_path + wet_name)
adl_bun = load(mod_path + adl_name)


#%% generate the prediction
win_pred_te_y = win_bun["model"].predict(win_bun["test_x_all"])
pwl_pred_te_y = pwl_bun["model"].predict(pwl_bun["test_x_all"])
wet_pred_te_y = wet_bun["model"].predict(wet_bun["test_x_all"])
adl_pred_te_y = adl_bun["model"].predict(adl_bun["test_x_all"])

win_pred_tr_y = win_bun["model"].predict(win_bun["train_x_all"])
pwl_pred_tr_y = pwl_bun["model"].predict(pwl_bun["train_x_all"])
wet_pred_tr_y = wet_bun["model"].predict(wet_bun["train_x_all"])
adl_pred_tr_y = adl_bun["model"].predict(adl_bun["train_x_all"])

win_pred_teb_y = win_bun["model"].predict(win_bun["test_x"])
pwl_pred_teb_y = pwl_bun["model"].predict(pwl_bun["test_x"])
wet_pred_teb_y = wet_bun["model"].predict(wet_bun["test_x"])
adl_pred_teb_y = adl_bun["model"].predict(adl_bun["test_x"])

win_pred_trb_y = win_bun["model"].predict(win_bun["train_x"])
pwl_pred_trb_y = pwl_bun["model"].predict(pwl_bun["train_x"])
wet_pred_trb_y = wet_bun["model"].predict(wet_bun["train_x"])
adl_pred_trb_y = adl_bun["model"].predict(adl_bun["train_x"])


#%% plot a histogram of the occurrences
train_all_c = "black"
train_c = "red"
test_all_c = "black"
test_c = "red"

y_max = np.max([np.sum(win_bun["train_y"] == 0), np.sum(pwl_bun["train_y"] == 0), np.sum(wet_bun["train_y"] == 0),
                np.sum(adl_bun["train_y"] == 0)])
y_max = y_max + 0.05 * y_max

fig = pl.figure(figsize=(7, 3.5))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)

# upper left
ax00.bar(x=0, height=np.sum(win_bun["train_y_all"] == 0), facecolor="none", edgecolor=train_all_c)
ax00.bar(x=1, height=np.sum(win_bun["train_y_all"] == 1), facecolor="none", edgecolor=train_all_c)
ax00.bar(x=0, height=np.sum(win_bun["train_y"] == 0), facecolor="none", edgecolor=train_c, width=0.5)
ax00.bar(x=1, height=np.sum(win_bun["train_y"] == 1), facecolor="none", edgecolor=train_c, width=0.5)

ax00.bar(x=2, height=np.sum(win_bun["test_y_all"] == 0), facecolor="none", edgecolor=test_all_c)
ax00.bar(x=3, height=np.sum(win_bun["test_y_all"] == 1), facecolor="none", edgecolor=test_all_c)
ax00.bar(x=2, height=np.sum(win_bun["test_y"] == 0), facecolor="none", edgecolor=test_c, width=0.5)
ax00.bar(x=3, height=np.sum(win_bun["test_y"] == 1), facecolor="none", edgecolor=test_c, width=0.5)

# upper right
ax01.bar(x=0, height=np.sum(pwl_bun["train_y_all"] == 0), facecolor="none", edgecolor=train_all_c, label="original")
ax01.bar(x=1, height=np.sum(pwl_bun["train_y_all"] == 1), facecolor="none", edgecolor=train_all_c)
ax01.bar(x=0, height=np.sum(pwl_bun["train_y"] == 0), facecolor="none", edgecolor=train_c, width=0.5, label="balanced")
ax01.bar(x=1, height=np.sum(pwl_bun["train_y"] == 1), facecolor="none", edgecolor=train_c, width=0.5)

ax01.bar(x=2, height=np.sum(pwl_bun["test_y_all"] == 0), facecolor="none", edgecolor=test_all_c)
ax01.bar(x=3, height=np.sum(pwl_bun["test_y_all"] == 1), facecolor="none", edgecolor=test_all_c)
ax01.bar(x=2, height=np.sum(pwl_bun["test_y"] == 0), facecolor="none", edgecolor=test_c, width=0.5)
ax01.bar(x=3, height=np.sum(pwl_bun["test_y"] == 1), facecolor="none", edgecolor=test_c, width=0.5)

# lower left
ax10.bar(x=0, height=np.sum(wet_bun["train_y_all"] == 0), facecolor="none", edgecolor=train_all_c)
ax10.bar(x=1, height=np.sum(wet_bun["train_y_all"] == 1), facecolor="none", edgecolor=train_all_c)
ax10.bar(x=0, height=np.sum(wet_bun["train_y"] == 0), facecolor="none", edgecolor=train_c, width=0.5)
ax10.bar(x=1, height=np.sum(wet_bun["train_y"] == 1), facecolor="none", edgecolor=train_c, width=0.5)

ax10.bar(x=2, height=np.sum(wet_bun["test_y_all"] == 0), facecolor="none", edgecolor=test_all_c)
ax10.bar(x=3, height=np.sum(wet_bun["test_y_all"] == 1), facecolor="none", edgecolor=test_all_c)
ax10.bar(x=2, height=np.sum(wet_bun["test_y"] == 0), facecolor="none", edgecolor=test_c, width=0.5)
ax10.bar(x=3, height=np.sum(wet_bun["test_y"] == 1), facecolor="none", edgecolor=test_c, width=0.5)

# lower right
ax11.bar(x=0, height=np.sum(adl_bun["train_y_all"] == 0), facecolor="none", edgecolor=train_all_c)
ax11.bar(x=1, height=np.sum(adl_bun["train_y_all"] == 1), facecolor="none", edgecolor=train_all_c)
ax11.bar(x=0, height=np.sum(adl_bun["train_y"] == 0), facecolor="none", edgecolor=train_c, width=0.5)
ax11.bar(x=1, height=np.sum(adl_bun["train_y"] == 1), facecolor="none", edgecolor=train_c, width=0.5)

ax11.bar(x=2, height=np.sum(adl_bun["test_y_all"] == 0), facecolor="none", edgecolor=test_all_c)
ax11.bar(x=3, height=np.sum(adl_bun["test_y_all"] == 1), facecolor="none", edgecolor=test_all_c)
ax11.bar(x=2, height=np.sum(adl_bun["test_y"] == 0), facecolor="none", edgecolor=test_c, width=0.5)
ax11.bar(x=3, height=np.sum(adl_bun["test_y"] == 1), facecolor="none", edgecolor=test_c, width=0.5)

# legend
ax01.legend(loc="center right")

# text
ax00.text(x=2, y=4600, s="(a) Wind slab")
ax01.text(x=2, y=4600, s="(b) PWL slab")
ax10.text(x=2, y=4600, s="(c) Wet")
ax11.text(x=2, y=4600, s="(d) General")

# adjust ticks
ax01.set_yticklabels([])
ax11.set_yticklabels([])
ax00.set_xticklabels([])
ax01.set_xticklabels([])

ax00.set_ylabel("Number of\noccurrences")
ax10.set_ylabel("Number of\noccurrences")

ax10.set_xticks([0, 1, 2, 3])
ax11.set_xticks([0, 1, 2, 3])
ax10.set_xticklabels(["Train\nnon-AvD", "Train\nAvD", "Test\nnon-AvD", "Test\nAvD"])
ax11.set_xticklabels(["Train\nnon-AvD", "Train\nAvD", "Test\nnon-AvD", "Test\nAvD"])

# set y limits
ax00.set_ylim((0, y_max))
ax01.set_ylim((0, y_max))
ax10.set_ylim((0, y_max))
ax11.set_ylim((0, y_max))

# title
fig.suptitle("Class imbalance and class balancing")

fig.subplots_adjust(wspace=0.065, hspace=0.065, top=0.9)

pl.savefig(pl_path + "Class_Imbalance.pdf", bbox_inches="tight", dpi=200)
pl.savefig(pl_path + "Class_Imbalance.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot a histogram of the prediction v the truth for the TEST data
pred_c = "red"
true_c = "black"
pred_w = 0.7
true_w = 0.4

y_max = np.max([np.sum(win_bun["test_y_all"] == 0), np.sum(pwl_bun["test_y_all"] == 0),
                np.sum(wet_bun["test_y_all"] == 0), np.sum(adl_bun["test_y_all"] == 0),
                np.sum(win_pred_te_y == 0), np.sum(pwl_pred_te_y == 0), np.sum(wet_pred_te_y == 0),
                np.sum(adl_pred_te_y == 0)])
y_max = y_max + 0.05 * y_max

fig = pl.figure(figsize=(7, 3.5))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)


# upper left
ax00.bar(x=0, height=np.sum(win_bun["test_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=1, height=np.sum(win_bun["test_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=0, height=np.sum(win_pred_te_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax00.bar(x=1, height=np.sum(win_pred_te_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# upper right
ax01.bar(x=0, height=np.sum(pwl_bun["test_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=1, height=np.sum(pwl_bun["test_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=0, height=np.sum(pwl_pred_te_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax01.bar(x=1, height=np.sum(pwl_pred_te_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower left
ax10.bar(x=0, height=np.sum(wet_bun["test_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w, label="true")
ax10.bar(x=1, height=np.sum(wet_bun["test_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax10.bar(x=0, height=np.sum(wet_pred_te_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w, label="predicted")
ax10.bar(x=1, height=np.sum(wet_pred_te_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower right
ax11.bar(x=0, height=np.sum(adl_bun["test_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=1, height=np.sum(adl_bun["test_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=0, height=np.sum(adl_pred_te_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax11.bar(x=1, height=np.sum(adl_pred_te_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# legend
ax10.legend(loc="center right")

# adjust ticks
ax01.set_yticklabels([])
ax11.set_yticklabels([])
ax00.set_xticklabels([])
ax01.set_xticklabels([])

ax00.set_ylabel("Number of\noccurrences")
ax10.set_ylabel("Number of\noccurrences")

ax10.set_xticks([0, 1])
ax11.set_xticks([0, 1])
ax10.set_xticklabels(["non-AvD", "AvD"])
ax11.set_xticklabels(["non-AvD", "AvD"])

# set y limits
ax00.set_ylim((0, y_max))
ax01.set_ylim((0, y_max))
ax10.set_ylim((0, y_max))
ax11.set_ylim((0, y_max))

# text
ax00.text(x=0.7, y=1450, s="(a) Wind slab")
ax01.text(x=0.7, y=1450, s="(b) PWL slab")
ax10.text(x=0.7, y=1450, s="(c) Wet")
ax11.text(x=0.7, y=1450, s="(d) General")

# title
fig.suptitle("Class imbalance predicted and true")

fig.subplots_adjust(wspace=0.065, hspace=0.065, top=0.9)

pl.savefig(pl_path + "Class_Imbalance_Predicted_v_True.pdf", bbox_inches="tight", dpi=200)
pl.savefig(pl_path + "Class_Imbalance_Predicted_v_True.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot a histogram of the prediction v the truth for the TRAINING data
pred_c = "red"
true_c = "black"
pred_w = 0.7
true_w = 0.4

y_max = np.max([np.sum(win_bun["train_y_all"] == 0), np.sum(pwl_bun["train_y_all"] == 0),
                np.sum(wet_bun["train_y_all"] == 0), np.sum(adl_bun["train_y_all"] == 0),
                np.sum(win_pred_te_y == 0), np.sum(pwl_pred_te_y == 0), np.sum(wet_pred_te_y == 0),
                np.sum(adl_pred_te_y == 0)])
y_max = y_max + 0.05 * y_max

fig = pl.figure(figsize=(7, 3.5))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)


# upper left
ax00.bar(x=0, height=np.sum(win_bun["train_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=1, height=np.sum(win_bun["train_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=0, height=np.sum(win_pred_tr_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax00.bar(x=1, height=np.sum(win_pred_tr_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# upper right
ax01.bar(x=0, height=np.sum(pwl_bun["train_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=1, height=np.sum(pwl_bun["train_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=0, height=np.sum(pwl_pred_tr_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax01.bar(x=1, height=np.sum(pwl_pred_tr_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower left
ax10.bar(x=0, height=np.sum(wet_bun["train_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w,
         label="true")
ax10.bar(x=1, height=np.sum(wet_bun["train_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax10.bar(x=0, height=np.sum(wet_pred_tr_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w, label="predicted")
ax10.bar(x=1, height=np.sum(wet_pred_tr_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower right
ax11.bar(x=0, height=np.sum(adl_bun["train_y_all"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=1, height=np.sum(adl_bun["train_y_all"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=0, height=np.sum(adl_pred_tr_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax11.bar(x=1, height=np.sum(adl_pred_tr_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# legend
ax10.legend(loc="center right")

# adjust ticks
ax01.set_yticklabels([])
ax11.set_yticklabels([])
ax00.set_xticklabels([])
ax01.set_xticklabels([])

ax00.set_ylabel("Number of\noccurrences")
ax10.set_ylabel("Number of\noccurrences")

ax10.set_xticks([0, 1])
ax11.set_xticks([0, 1])
ax10.set_xticklabels(["non-AvD", "AvD"])
ax11.set_xticklabels(["non-AvD", "AvD"])

# set y limits
ax00.set_ylim((0, y_max))
ax01.set_ylim((0, y_max))
ax10.set_ylim((0, y_max))
ax11.set_ylim((0, y_max))

# text
ax00.text(x=0.7, y=4450, s="(a) Wind slab")
ax01.text(x=0.7, y=4450, s="(b) PWL slab")
ax10.text(x=0.7, y=4450, s="(c) Wet")
ax11.text(x=0.7, y=4450, s="(d) General")

# title
fig.suptitle("Class imbalance predicted and true (training data)")

fig.subplots_adjust(wspace=0.065, hspace=0.065, top=0.9)

pl.show()
pl.close()


#%% plot a histogram of the prediction v the truth for the TEST data BALANCED
pred_c = "red"
true_c = "black"
pred_w = 0.7
true_w = 0.4

y_max = np.max([np.sum(win_bun["test_y"] == 0), np.sum(pwl_bun["test_y"] == 0),
                np.sum(wet_bun["test_y"] == 0), np.sum(adl_bun["test_y"] == 0),
                np.sum(win_pred_teb_y == 0), np.sum(pwl_pred_teb_y == 0), np.sum(wet_pred_teb_y == 0),
                np.sum(adl_pred_teb_y == 0)])
y_max = y_max + 0.05 * y_max

fig = pl.figure(figsize=(7, 3.5))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)


# upper left
ax00.bar(x=0, height=np.sum(win_bun["test_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=1, height=np.sum(win_bun["test_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=0, height=np.sum(win_pred_teb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax00.bar(x=1, height=np.sum(win_pred_teb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# upper right
ax01.bar(x=0, height=np.sum(pwl_bun["test_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=1, height=np.sum(pwl_bun["test_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=0, height=np.sum(pwl_pred_teb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax01.bar(x=1, height=np.sum(pwl_pred_teb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower left
ax10.bar(x=0, height=np.sum(wet_bun["test_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w, label="true")
ax10.bar(x=1, height=np.sum(wet_bun["test_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax10.bar(x=0, height=np.sum(wet_pred_teb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w, label="predicted")
ax10.bar(x=1, height=np.sum(wet_pred_teb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower right
ax11.bar(x=0, height=np.sum(adl_bun["test_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=1, height=np.sum(adl_bun["test_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=0, height=np.sum(adl_pred_teb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax11.bar(x=1, height=np.sum(adl_pred_teb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# legend
ax10.legend(loc="center right")

# adjust ticks
ax01.set_yticklabels([])
ax11.set_yticklabels([])
ax00.set_xticklabels([])
ax01.set_xticklabels([])

ax00.set_ylabel("Number of\noccurrences")
ax10.set_ylabel("Number of\noccurrences")

ax10.set_xticks([0, 1])
ax11.set_xticks([0, 1])
ax10.set_xticklabels(["non-AvD", "AvD"])
ax11.set_xticklabels(["non-AvD", "AvD"])

# set y limits
ax00.set_ylim((0, y_max))
ax01.set_ylim((0, y_max))
ax10.set_ylim((0, y_max))
ax11.set_ylim((0, y_max))

# text
ax00.text(x=0.7, y=1450, s="(a) Wind slab")
ax01.text(x=0.7, y=1450, s="(b) PWL slab")
ax10.text(x=0.7, y=1450, s="(c) Wet")
ax11.text(x=0.7, y=1450, s="(d) General")

# title
fig.suptitle("Class imbalance predicted and true (balanced)")

fig.subplots_adjust(wspace=0.065, hspace=0.065, top=0.9)

pl.show()
pl.close()


#%% plot a histogram of the prediction v the truth for the TRAINING data BALANCED
pred_c = "red"
true_c = "black"
pred_w = 0.7
true_w = 0.4

y_max = np.max([np.sum(win_bun["train_y"] == 0), np.sum(pwl_bun["train_y"] == 0),
                np.sum(wet_bun["train_y"] == 0), np.sum(adl_bun["train_y"] == 0),
                np.sum(win_pred_trb_y == 0), np.sum(pwl_pred_trb_y == 0), np.sum(wet_pred_trb_y == 0),
                np.sum(adl_pred_trb_y == 0),
                np.sum(win_bun["train_y"] == 1), np.sum(pwl_bun["train_y"] == 1),
                np.sum(wet_bun["train_y"] == 1), np.sum(adl_bun["train_y"] == 1),
                np.sum(win_pred_trb_y == 1), np.sum(pwl_pred_trb_y == 1), np.sum(wet_pred_trb_y == 1),
                np.sum(adl_pred_trb_y == 1)])
y_max = y_max + 0.05 * y_max

fig = pl.figure(figsize=(7, 3.5))
ax00 = fig.add_subplot(221)
ax01 = fig.add_subplot(222)
ax10 = fig.add_subplot(223)
ax11 = fig.add_subplot(224)


# upper left
ax00.bar(x=0, height=np.sum(win_bun["train_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=1, height=np.sum(win_bun["train_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax00.bar(x=0, height=np.sum(win_pred_trb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax00.bar(x=1, height=np.sum(win_pred_trb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# upper right
ax01.bar(x=0, height=np.sum(pwl_bun["train_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=1, height=np.sum(pwl_bun["train_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax01.bar(x=0, height=np.sum(pwl_pred_trb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax01.bar(x=1, height=np.sum(pwl_pred_trb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower left
ax10.bar(x=0, height=np.sum(wet_bun["train_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w, label="true")
ax10.bar(x=1, height=np.sum(wet_bun["train_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax10.bar(x=0, height=np.sum(wet_pred_trb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w, label="predicted")
ax10.bar(x=1, height=np.sum(wet_pred_trb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# lower right
ax11.bar(x=0, height=np.sum(adl_bun["train_y"] == 0), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=1, height=np.sum(adl_bun["train_y"] == 1), facecolor="none", edgecolor=true_c, width=true_w)
ax11.bar(x=0, height=np.sum(adl_pred_trb_y == 0), facecolor="none", edgecolor=pred_c, width=pred_w)
ax11.bar(x=1, height=np.sum(adl_pred_trb_y == 1), facecolor="none", edgecolor=pred_c, width=pred_w)

# legend
ax10.legend(loc="center right")

# adjust ticks
ax01.set_yticklabels([])
ax11.set_yticklabels([])
ax00.set_xticklabels([])
ax01.set_xticklabels([])

ax00.set_ylabel("Number of\noccurrences")
ax10.set_ylabel("Number of\noccurrences")

ax10.set_xticks([0, 1])
ax11.set_xticks([0, 1])
ax10.set_xticklabels(["non-AvD", "AvD"])
ax11.set_xticklabels(["non-AvD", "AvD"])

# set y limits
ax00.set_ylim((0, y_max))
ax01.set_ylim((0, y_max))
ax10.set_ylim((0, y_max))
ax11.set_ylim((0, y_max))

# text
ax00.text(x=0.7, y=1450, s="(a) Wind slab")
ax01.text(x=0.7, y=1450, s="(b) PWL slab")
ax10.text(x=0.7, y=1450, s="(c) Wet")
ax11.text(x=0.7, y=1450, s="(d) General")

# title
fig.suptitle("Class imbalance predicted and true (training, balanced)")

fig.subplots_adjust(wspace=0.065, hspace=0.065, top=0.9)

pl.show()
pl.close()