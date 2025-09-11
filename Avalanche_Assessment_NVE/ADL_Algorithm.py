#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An attempt at implementing an algorithm that converts the avalanche problem information into a danger level PER
avalanche problem.
--> see Karsten Mueller's 2016a and 2023 ISSW proceedings papers.
"""

#%% imports
import numpy as np
import pandas as pd


#%% define the likelihood matrix function
def likel_mat(sens, dist):
    if ( (sens == 1) | ((sens == 2) & (dist == 1)) ):
        likel = 1
    if ( ((sens == 2) & (dist != 1)) | ((sens == 3) & (dist != 3)) | ((sens == 4) & (dist == 1)) ):
        likel = 2
    if ( ((sens == 3) & (dist == 3)) | ((sens == 4) & (dist == 2)) ):
        likel = 3
    if ( (sens == 4) & (dist == 3) ):
        likel = 4
    # end if
    return likel
# end def


#%% define the danger matrix function
def danger_mat(likel, size):
    if ( ((likel == 1) & (size < 3)) | ((likel == 2) & ((size == 1))) ):
        adl = 1
    elif ( ((likel == 1) & (size == 3)) | ((likel == 2) & (size == 2)) | ((likel == 3) & (size == 1)) ):
        adl = 2
    elif ( ((likel == 1) & (size == 4)) | ((likel == 2) & (size == 3)) | ((likel == 3) & (size == 2)) |\
                                                                                         ((likel == 4) & (size == 1)) ):
        adl = 3
    elif ( ((likel == 1) & (size == 5)) | ((likel == 2) & (size > 3)) | ((likel == 3) & (size in [3, 4])) |\
                                                                                      ((likel == 4) & size in [2, 3]) ):
        adl = 4
    else:
        adl = 5
    # end if elif else
    return adl
# end def


#%% combine both functions to work on arrays
def ap_to_adl(sens_s, dist_s, size_s):

    adl_s = []
    likel_s = []
    for i in np.arange(len(sens_s)):
        likel = likel_mat(sens_s[i], dist_s[i])
        likel_s.append(likel)
        adl = danger_mat(likel, size_s[i])
        adl_s.append(adl)
    # end for
    return adl_s

# end def


#%% set up a test data frame
"""
ap_df = pd.DataFrame({"size":[1, 2, 3, 4, 5], "sensitivity":[1, 1, 2, 3, 4], "distribution":[1, 2, 3, 1, 2]})


#%% apply the "likelihood matrix" --> i.e., from distribution and sensitivity derive the likelihood

sens = 3
dist = 2

likel = likel_mat(sens, dist)


#%% apply the "danger matrix" --> i.e., use likelihood and size to derive the danger level
size = 3

adl = danger_mat(likel, size)

print(adl)


#%%

sens_s = np.array(ap_df["sensitivity"])
dist_s = np.array(ap_df["distribution"])
size_s = np.array(ap_df["size"])

adls = np.array(ap_to_adl(sens_s, dist_s, size_s))
"""