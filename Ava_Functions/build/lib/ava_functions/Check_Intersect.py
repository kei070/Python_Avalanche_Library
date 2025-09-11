#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check two numpy arrays the elments NOT in their intersect.
"""

#%% imports
import numpy as np

# function -- mostly taken from ChatGPT
def non_intersect(array1, array2):

    intersection = np.intersect1d(array1, array2)
    # Find elements in array1 but not in the intersection
    not_in_intersection_1 = np.setdiff1d(array1, intersection)
    # Find elements in array2 but not in the intersection
    not_in_intersection_2 = np.setdiff1d(array2, intersection)

    return not_in_intersection_1, not_in_intersection_2

# end def

