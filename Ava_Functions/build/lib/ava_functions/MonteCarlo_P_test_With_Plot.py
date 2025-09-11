#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for running a Monte-Carlo style p-value test. Generated with the help of ChatUiT.
"""

# imports
import numpy as np
import pylab as pl

# test statistic --> here we use the difference in means
def test_statistic(sample_a, sample_b):
    return np.mean(sample_a) - np.mean(sample_b)
# end def

# the Monte-Carlo style function
def monte_carlo_p(sample_a, sample_b, num_permutations=500000, plot_hist=False, hist_title="", bins=50, h_range=None,
                  dens=True, verbose=False):

    """
    This function "tests" the hypothesis that it is only by chance that sample a is larger than sample b. That is, if
    the resulting p-value is < 0.05 one may accept on th 95% percent level that sample a is larger than sample b.

    Explanation:
       --> the observed statisitic is mean(sample_a) - mean(sample_b)
       --> if mean(sample_a) > mean(sample_b) this is positive
       --> we then shuffle the data around and extract two new (now random) samples and calculate their difference
       --> we then check if the difference between this random sample is LARGER than our observed statistic
       --> only if it is larger, we increase the counter
       --> we repeat this a larger number of times (e.g., 500000) = number of permuations
       --> if the statistic based on the random samples is OFTEN larger than our observed statistic, our count variable
           will also be quite large and vice versa
       --> we then calculate the p-value as the count variable divided by the number of perumtations
       --> if this p-value is smaller than 0.05 we may say that our result is significant on the 95% level
    """

    # calculate the test statistic for the original samples
    observed_statistic = test_statistic(sample_a, sample_b)

    # Permutation test
    count = 0

    combined_samples = np.concatenate((sample_a, sample_b))
    n1 = len(sample_a)

    new_ss = []
    for _ in range(num_permutations):
        np.random.shuffle(combined_samples)
        new_sample1 = combined_samples[:n1]
        new_sample2 = combined_samples[n1:]
        new_statistic = test_statistic(new_sample1, new_sample2)

        new_ss.append(new_statistic)

        # Check if the permuted test statistic is at least as extreme as the observed
        if new_statistic >= observed_statistic:
            count += 1
        # end if
    # end for _

    new_ss = np.array(new_ss)

    # if requested generate the histogram
    if plot_hist:

        if hist_title == "":
            hist_title = "Histogram Monte Carlo test"
        # end if

        fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 5))
        axes.hist(new_ss, histtype="step", bins=bins, range=h_range, density=dens, edgecolor="black")
        axes.axvline(x=0, c="black")
        axes.axvline(x=np.std(new_ss), c="gray", label="$\pm$1$\sigma$")
        axes.axvline(x=-np.std(new_ss), c="gray")
        axes.axvline(x=2*np.std(new_ss), c="gray", linestyle="--", label="$\pm$2$\sigma$")
        axes.axvline(x=-2*np.std(new_ss), c="gray", linestyle="--")
        axes.axvline(x=0, c="black")
        axes.axvline(x=observed_statistic, c="red", label="observed\ndifference")
        axes.legend()
        if dens:
            axes.set_ylabel("Density")
        else:
            axes.set_ylabel("Number in bin")
        # end if
        axes.set_xlabel("Difference")
        axes.set_title(hist_title)
        pl.show()
        pl.close()
    # end if

    # Calculate the p-value
    p_value = count / num_permutations

    # Print the result
    # print(f"Observed statistic: {observed_statistic}")
    if verbose: print(f"P-value: {p_value}")

    return p_value

# end def


# the Monte-Carlo style function -- two-sided
def monte_carlo_p2(sample_a, sample_b, num_permutations=500000, return_smaller=True, p_sign=True, verbose=False):

    """
    This function executes both:
        pv1 = monte_carlo_p(sample_a, sample_b)
        pv2 = monte_carlo_p(sample_b, sample_a)
    to account for the fact, that b might be larger than a. Per default only the smaller p value is returned. In case
    pv2 is smaller than pv1, the value is returned as negative to indicate that a is SMALLER and not larger than b. If
    this is not requested, set p_sign to False.
    """

    pv1 = monte_carlo_p(sample_a, sample_b, num_permutations=num_permutations)
    pv2 = monte_carlo_p(sample_b, sample_a, num_permutations=num_permutations)

    # print the result
    ## print(f"Observed statistic: {observed_statistic}")
    if verbose: print(f"P-value 1: {pv1}  |  P-value 2: {pv2}")

    p_fac = -1 if p_sign else 1

    if return_smaller:
        return pv1 if pv1 < pv2 else pv2*p_fac
    else:
        return {"a>b":pv1, "a<b":pv2}
    # end if else

# end def