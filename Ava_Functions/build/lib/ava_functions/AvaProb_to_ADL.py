#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An attempt at implementing an algorithm that converts the avalanche problem information into a danger level PER
avalanche problem.
--> see Karsten Mueller's 2016a and 2023 ISSW proceedings papers.
"""


#%% imports
import numpy as np


#%% define the likelihood matrix function
def likel_mat(sens, dist):

    # make sure sens and dist are zero-dimensional
    if np.ndim(sens) == 0:
        sens = np.array([sens])
    if np.ndim(dist) == 0:
        dist = np.array([dist])
    # end if

    # set up the likelihood array containing only NaNs so far
    likel = np.zeros(len(sens))
    likel[:] = np.nan

    # set the conditions for the individual danger levels
    con1 = ( (sens == 1) | ((sens == 2) & (dist == 1)) )
    con2 = ( ((sens == 2) & (dist != 1)) | ((sens == 3) & (dist != 3)) | ((sens == 4) & (dist == 1)) )
    con3 = ( ((sens == 3) & (dist == 3)) | ((sens == 4) & (dist == 2)) )
    con4 = ( (sens == 4) & (dist == 3) )

    # set the likelihood according to the conditions
    likel[con1] = 1
    likel[con2] = 2
    likel[con3] = 3
    likel[con4] = 4

    """ OLD VERSION
    if ( (sens == 1) | ((sens == 2) & (dist == 1)) ):
        likel = 1
    if ( ((sens == 2) & (dist != 1)) | ((sens == 3) & (dist != 3)) | ((sens == 4) & (dist == 1)) ):
        likel = 2
    if ( ((sens == 3) & (dist == 3)) | ((sens == 4) & (dist == 2)) ):
        likel = 3
    if ( (sens == 4) & (dist == 3) ):
        likel = 4
    # end if
    """

    return likel
# end def


#%% define the danger matrix function
def danger_mat(likel, size):

    # make sure likel and size are zero-dimensional
    if np.ndim(likel) == 0:
        likel = np.array([likel])
    if np.ndim(size) == 0:
        size = np.array([size])
    # end if

    # set up the ADL array containing only NaNs so far
    adl = np.zeros(len(likel))
    adl[:] = np.nan

    # set the conditions for the individual danger levels
    con1 = ( ((likel == 1) & (size < 3)) | ((likel == 2) & ((size == 1))) )
    con2 = ( ((likel == 1) & (size == 3)) | ((likel == 2) & (size == 2)) | ((likel == 3) & (size == 1)) )
    con3 = ( ((likel == 1) & (size == 4)) | ((likel == 2) & (size == 3)) | ((likel == 3) & (size == 2)) |\
                                                                                          ((likel == 4) & (size == 1)) )
    con4 = ( ((likel == 1) & (size == 5)) | ((likel == 2) & (size > 3)) | ((likel == 3) & np.isin(size, [3, 4])) |\
                                                                                ((likel == 4) & np.isin(size, [2, 3])) )
    con5 = ( ((likel == 4) & (size == 5)) | ((likel == 4) & (size > 3)) )

    # set the ADL according to the conditions
    adl[con1] = 1
    adl[con2] = 2
    adl[con3] = 3
    adl[con4] = 4
    adl[con5] = 5

    """ OLD VERSION
    if ( ((likel == 1) & (size < 3)) | ((likel == 2) & ((size == 1))) ):
        adl = 1
    elif ( ((likel == 1) & (size == 3)) | ((likel == 2) & (size == 2)) | ((likel == 3) & (size == 1)) ):
        adl = 2
    elif ( ((likel == 1) & (size == 4)) | ((likel == 2) & (size == 3)) | ((likel == 3) & (size == 2)) |\
                                                                                         ((likel == 4) & (size == 1)) ):
        adl = 3
    elif ( ((likel == 1) & (size == 5)) | ((likel == 2) & (size > 3)) | ((likel == 3) & (size in [3, 4])) |\
                                                                                    ((likel == 4) & (size in [2, 3])) ):
        adl = 4
    elif ( ((likel == 4) & (size == 5)) | ((likel == 4) & (size > 3)) ):
        adl = 5
    # end if elif
    """

    return adl
# end def


#%% combine both functions to work on arrays
def ap_to_adl(sens, dist, size):

    # determine the likelihood
    likel = likel_mat(sens, dist)

    # determine the ADL
    adl = danger_mat(likel, size)

    """ OLD VERSION
    adl_s = []
    likel_s = []
    for i in np.arange(len(sens_s)):
        likel = likel_mat(sens_s[i], dist_s[i])
        likel_s.append(likel)
        adl = danger_mat(likel, size_s[i])
        adl_s.append(adl)
    # end for
    """

    return adl
# end def


#%% test functionality
"""
c1 = np.array([1, 2, 3, 4])
c2 = np.array([2, 2, 1, 3])
c3 = np.array([2, 5, 1, 4])

print(ap_to_adl(c1, c2, c3))
"""