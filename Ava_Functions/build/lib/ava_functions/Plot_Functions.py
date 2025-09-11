#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a collection of helpful plot functions.
"""

#%% imports
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pylab as pl
from matplotlib.colors import ListedColormap


#%% heat map function that only plots the lower triangle
def heat_map(diffs, p_vals, ax, mask, title, ticks, title_size=None, tick_size=None, vmin=None, vmax=None,
             annot_size=None, rects=None, gray_text_mask=None, gray_mask_color="gray"):

    """
    Plots a heat map where some of the entries are masked. This means that the diffs and p_vals parameters of the
    function must be mask numpy arrays. This is intended for heat maps where only either the upper or lower triangle
    (preferably the latter) is plotted.

    The rects is a list of tuples such as e.g. rects = [(0, 1), (0, 6)] which each indicate the grid cells that are
    supposed to have a black frame (e.g. indicating cells to focus on).

    Note that this for convenience allows the p_vals input to have negative p values: In the two-sided Monte-Carlo test
    [monte_carlo_p2(a, b)] a negative p value indicates that b was larger than a.

    The gray_text_mask must be a 1d array or list with a length corresponding to the number of cells filled with a text.
    The text in the cells where the gray_text_mask is > 0 will be coloured gray.
    """

    hm00 = sns.heatmap(diffs, annot=True, fmt=".1f", cmap='coolwarm', cbar=False, mask=mask,
                       vmin=vmin, vmax=vmax, ax=ax, annot_kws={"size":annot_size})

    # iterate over the text annotations and make the text bold for p-values < 0.05
    for text, p_value in zip(hm00.texts, p_vals.flatten()[~p_vals.flatten().mask]):
        if ((p_value < 0.05) & (p_value > -0.05)):
            text.set_weight('bold')
        # end if
    # end for text, p_value

    if type(gray_text_mask) != type(None):
        for text, m in zip(hm00.texts, gray_text_mask):
            print(text)
            if m > 0:
                text.set_color(gray_mask_color)
            # end if
        # end for text, m
    # end if

    hm00.set_xlim((0, len(ticks)-1))
    hm00.set_ylim((len(ticks), 1))

    hm00.set_xticks(np.arange(0, len(ticks)-1) + 0.5)
    hm00.set_xticklabels(ticks[:-1], fontsize=tick_size)
    hm00.set_yticks(np.arange(1, len(ticks)) + 0.5)
    hm00.set_yticklabels(ticks[1:], rotation=0, fontsize=tick_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    hm00.set_title(title, fontsize=title_size)

    if type(rects) != type(None):
        for rect in rects:
            highlight = Rectangle(rect, 1, 1, fill=False, edgecolor='black', lw=2)
            hm00.add_patch(highlight)
        # end for rect
    # end if

    return hm00

# end def


#%% function for plotting the predicted danger-level classes in the predictor space.
def plot_2predictor_space(data_x, model, test_x, test_y, sel_feats, h=0.5):

    """
    Plots the predictor space in a 2d-meshgrid as well as the locations target variable in this space.
    The function is based on a suggestion by ChatUiT.

    Parameters:
        data_x     The full predictor data set. If load_xlevel_preds or apply_stat_mod is used to load the data, just
                   concatenate train_x_all and test_x_all.
        model      The statistical model used for the prediction. Use e.g. the output of apply_stat_mod.
        test_x     The test predictors. These will be used to plot the locations of the target variable in the predictor
                   space.
        test_y     The target variable corresponding to the predictor values given in test_x.
        sel_feats  The selected features (i.e., predictors).
        h          Float. The step size in the predictor space mesh.
    """

    # get the number of classes
    ndlev = np.unique(test_y)

    # create a mesh grid
    x_min, x_max = data_x[:, 0].min() - 1, data_x[:, 0].max() + 1
    y_min, y_max = data_x[:, 1].min() - 1, data_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # predict class labels for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # define the colors
    # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Light red, light green, light blue, golden rod
    background_colors = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#DAA520'])

    # Red, green, blue, gold
    data_point_colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFD700'])


    # plot the decision boundary
    fig = pl.figure(figsize=(8, 5))
    ax00 = fig.add_subplot(111)

    # span up the predictor space
    ax00.pcolormesh(xx, yy, Z, cmap=background_colors)

    # plot also the target locations
    p00 = ax00.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap=data_point_colors, edgecolor='k', s=20)
    ax00.legend(*p00.legend_elements(), title="Classes")
    ax00.set_xlim(xx.min(), xx.max())
    ax00.set_ylim(yy.min(), yy.max())
    ax00.set_title(f"{len(ndlev)}-Class classification")
    try:
        ax00.set_xlabel(sel_feats[0])
        ax00.set_ylabel(sel_feats[1])
    except:
        print("\nNo feature names given.\n")
    # end try
    pl.show()
    pl.close()
# end def

