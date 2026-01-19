#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to help the correlation analysis that includes the possibility of changing the degrees of freedom in the
significance test. This appears to be necessary when running means are correlated as the individual values now strongly
depend on each other (or in other words, they are strongly autocorrelated).
"""

# imports
import numpy as np
from scipy.stats import t
from scipy.stats import linregress as lr


def calc_t_stat(X, Y, df_fac=1):
    """
    Calculates the t statistic

    Formulas taken from
    https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Mostly_Harmless_Statistics_(Webb)/12%3A_Correlation_and_Regression/12.02%3A_Simple_Linear_Regression/12.2.01%3A_Hypothesis_Test_for_Linear_Regression

    """

    n = np.round(len(X) / df_fac)

    p = 1

    lr_test = lr(X, Y)
    slope = lr_test.slope

    dX = X - np.mean(X)
    dY = Y - np.mean(Y)

    ss_xx = np.sum(dX**2)
    ss_yy = np.sum(dY**2)
    ss_xy = np.sum(dX*dY)

    ssr = ss_xy**2 / ss_xx

    sse = ss_yy - ssr

    df = (n - p - 1)  # degrees of freedom

    mse = sse / df

    t_stat = slope / (mse / ss_xx)**0.5

    return {"t_stat":t_stat, "df":df}

# end def


def p_from_t(t_stat, df):

    """
    Calculate the p value from a t statistic given the degrees of freedom. Assuming a two-sided t test.
    """

    return 2 * (1 - t.cdf(t_stat, df=df))

# end def


def t_test_adj(X, Y, df_fac=1):

    """
    Combinatation of calc_t_stat and p_from_t.
    """

    t_test_vals = calc_t_stat(X=X, Y=Y, df_fac=df_fac)

    return p_from_t(t_stat=t_test_vals["t_stat"], df=t_test_vals["df"])

# end def