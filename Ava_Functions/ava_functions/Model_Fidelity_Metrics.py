#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for computing the model fidelity metrics.

UPDATE (24 June 2025): Implementation of some of the "distibution-oriented" metrics presented and suggested in Murphy
                       (1993) https://doi.org/10.1175/1520-0434(1993)008<0281:WIAGFA>2.0.CO;2
"""

# imports
import numpy as np


def mod_metrics(con_mat):

    """
    Function for computing the model fidelity metrics.
    a = con_mat[0, 0]  # correct non-event
    b = con_mat[1, 0]  # miss
    c = con_mat[0, 1]  # false alarm
    d = con_mat[1, 1]  # hit
    """
    a = con_mat[0, 0]  # correct non-event
    b = con_mat[1, 0]  # miss
    c = con_mat[0, 1]  # false alarm
    d = con_mat[1, 1]  # hit

    # metrics as presented in Hendrikx et al. (2014) and see also Wilks (2011)
    rpc = 0.5 * (a / (a + c) + d / (b + d))  # unweighted average accuracy
    tss = d / (b + d) - c / (a + c)  # true skill score
    far = c / (c + d)  # false alarm ration
    pod = d / (b + d)  # probability of detection
    pon = a / (a + c)  # probability of non-events
    hss = 2 * (a*d - c*b) / ((a+b)*(b+d) + (a+c)*(c+d))  # Heidke Skill Score
    pss = (a*d - c*b) / ((a+b)*(c+d))  # Peirce Skill Score
    mfr = b / (a + b + c + d)  # fraction of misses

    output = {"RPC":rpc, "TSS":tss, "FAR":far, "POD":pod, "PON":pon, "HSS":hss, "PSS":pss, "MFR":mfr}

    return(output)
# end def


# distribution-oriented metrics
def dist_metrics(true_y, pred_y, abs_val=True, verbose=False):

    # we assume here that the forecast is a discrete one
    ndlev = len(np.unique(true_y))

    # reliability
    reli_part = []
    avg_true = {}
    for lev_pred in np.arange(ndlev):
        # the average true level for a given predicted level
        lev_true = np.mean(true_y[pred_y == lev_pred])
        if verbose:
            print(f"\nThe average true level for predicted level {lev_pred} is {lev_true}")
        # end if
        avg_true[str(lev_pred)] = lev_true
        reli_part.append(lev_true - lev_pred)
    # end for lev

    # discrimination 1
    disc1_part = []
    avg_pred = {}
    for lev_true in np.arange(ndlev):
        # the average predicted level for a given true level
        lev_pred = np.mean(pred_y[true_y == lev_true])
        if verbose:
            print(f"\nThe average predicted level for true level {lev_true} is {lev_pred}")
        # end if
        avg_pred[str(lev_true)] = lev_pred
        disc1_part.append(lev_true - lev_pred)
    # end for lev

    # calculate the reliability and discrimination 1 values and take the absolute value if requested
    reliab = np.abs(np.mean(reli_part)) if abs_val else np.mean(reli_part)
    disc1 = np.abs(np.mean(disc1_part)) if abs_val else np.mean(disc1_part)

    if verbose:
        print(f"\nThe reliablity is {np.mean(reli_part)}")
        print(f"\nThe discrimination 1 is {np.mean(disc1_part)}")
    # end if

    return {"avg_true":avg_true, "reliability":reliab,
            "avg_pred":avg_pred, "discrimination1":disc1}

# end def
