#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to perform a lead-lag correlation. Mostly taken from ChatGPT.
"""


# imports
import pandas as pd
from scipy.stats import linregress as linr

# import proprietary functions
from .Correlation_Helpers import t_test_adj

# function
def lead_lag_corr(a, b, max_lag, df_fac=1):

    """
    Series b is here shifted by [-N, -(N-1), ..., -1, 0, 1, ..., N-1, N], where N=max_lag.

    A sensible way to understand the result might be: Say the correlation is highest at -3. That means that if series b
    is shifted is such a way that what used to be 1990 is now at 1987 the correlation is best, which might be expressed
    as "series a leads series b by 3 years".

    Parameters:
        a        Pandas Series.
        b        Pandas Series.
        max_lag  Integer. The maximum lead/lag to be considered.
        df_fac   Integer. A correction factor for the degrees of freedom in the significance test. The degrees of
                          freedom are calculated as df = n / df_fac - p - 1, where p = 1 and n = number of sample pairs
                          in the linear regression. This comes into effect via the functions imported from the
                          Correlation_Helpers module. Defaults to 1 (i.e., having no effect).

    Output:
        correlation, ps

        correlation    Dictionary containing the Pearson R for each lead/lag.
        ps             Dictionart containing the corresponding p values.
    """

    lags = range(-max_lag, max_lag + 1)

    # calculate lead-lag correlations
    correlations = {}
    ps = {}
    for lag in lags:
        b_shifted = b.shift(lag)  # shift series_b by 'lag'

        # combine the two series into a DataFrame and drop rows with NaN values
        comb = pd.concat([a, b_shifted], axis=1).dropna()

        # calculate correlation using non-NaN values
        corr = comb.iloc[:, 0].corr(comb.iloc[:, 1])
        sl, y_i, corr_r, p_v, err = linr(comb.iloc[:, 0], comb.iloc[:, 1])
        correlations[lag] = corr

        if df_fac != 1:
            p_v = t_test_adj(X=comb.iloc[:, 0], Y=comb.iloc[:, 1], df_fac=df_fac)
        # end if

        ps[lag] = p_v

    # end for lag

    return correlations, ps
# end def