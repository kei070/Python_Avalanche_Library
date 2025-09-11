#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare statistics of ADF/AvD based on the NorCP historical and on the NORA3 data from the same period.
"""

#%% imports
import pylab as pl
import numpy as np
from joblib import load

from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set paths
no3_adf_path = f"{path_par}/IMPETUS/NORA3/ML_Predictions/"
ncp_adf_path = f"{path_par}/IMPETUS/NorCP/ML_Predictions/"


#%% load the data
no3_adf_dict = {}
ec_adf_dict = {}
gf_adf_dict = {}
try:
    for reg_code in regions.keys():
        no3_adf_dict[reg_code] = load(no3_adf_path + f"ADF_{reg_code}_{regions[reg_code]}_NORA3.joblib")
        ec_adf_dict[reg_code] = load(ncp_adf_path + f"ADF_EC-Earth_{reg_code}_{regions[reg_code]}.joblib")
        gf_adf_dict[reg_code] = load(ncp_adf_path + f"ADF_GFDL-CM3_{reg_code}_{regions[reg_code]}.joblib")
    # end for reg_code
except:
    print("\nData loading failed. Have you generated the ADF data with Test_Model_With_SNOWPACK_fullNORA3.py?\n")
# end try except


#%% plot some values
reg_code = 3009
a_p = "wind_slab"

fig = pl.figure(figsize=(6, 3.5))
ax00 = fig.add_subplot(111)

ax00.plot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], c="black", linestyle="-", label="EC-Earth")
ax00.plot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], c="black", linestyle="--")

ax00.plot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], c="grey", linestyle="-", label="GFDL-CM3")
ax00.plot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], c="grey", linestyle="--")

ax00.plot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], c="red", linestyle="-", label="NORA3")
ax00.plot(no3_adf_dict[reg_code]["adf"][a_p]["full"][0].loc[1985:2005], c="red", linestyle="--")

pl.show()
pl.close()


#%% boxplot
reg_code = 3009

patch_artist = True

fig = pl.figure(figsize=(6, 3.5))
ax00 = fig.add_subplot(111)

a_p = "wind_slab"
x = 0
ax00.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], positions=[x-0.2],
             patch_artist=patch_artist, medianprops={"color":"blue"})
ax00.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][0].loc[1985:2005], positions=[x],
             patch_artist=patch_artist, medianprops={"color":"grey"})
ax00.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], positions=[x+0.2],
             patch_artist=patch_artist, medianprops={"color":"red"})

ax00.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1-0.2],
             patch_artist=patch_artist, medianprops={"color":"blue"})
ax00.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], positions=[x+1],
             patch_artist=patch_artist, medianprops={"color":"grey"})
ax00.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1+0.2],
             patch_artist=patch_artist, medianprops={"color":"red"})

a_p = "pwl_slab"
x = 2
ax00.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], positions=[x-0.2],
             patch_artist=patch_artist, medianprops={"color":"blue"})
ax00.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][0].loc[1985:2005], positions=[x],
             patch_artist=patch_artist, medianprops={"color":"grey"})
ax00.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], positions=[x+0.2],
             patch_artist=patch_artist, medianprops={"color":"red"})

ax00.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1-0.2],
             patch_artist=patch_artist, medianprops={"color":"blue"})
ax00.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], positions=[x+1],
             patch_artist=patch_artist, medianprops={"color":"grey"})
ax00.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1+0.2],
             patch_artist=patch_artist, medianprops={"color":"red"})

a_p = "wet"
x = 4
ax00.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], positions=[x-0.2],
             patch_artist=patch_artist, medianprops={"color":"blue"})
ax00.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][0].loc[1985:2005], positions=[x],
             patch_artist=patch_artist, medianprops={"color":"grey"})
ax00.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][0], positions=[x+0.2],
             patch_artist=patch_artist, medianprops={"color":"red"})

ax00.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1-0.2],
             patch_artist=patch_artist, medianprops={"color":"blue"})
ax00.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], positions=[x+1],
             patch_artist=patch_artist, medianprops={"color":"grey"})
ax00.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1+0.2],
             patch_artist=patch_artist, medianprops={"color":"red"})

ax00.set_ylabel("Number of AvDs/non-AvDs")

pl.show()
pl.close()


#%% boxplot
ylims = (0, 120)
patch_artist = True
dx = 0.25

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(231)
ax01 = fig.add_subplot(232)
ax02 = fig.add_subplot(233)
ax10 = fig.add_subplot(234)
ax11 = fig.add_subplot(235)

axes = [ax00, ax01, ax02, ax10, ax11]

for ax, reg_code in zip(axes, regions.keys()):
    a_p = "wind_slab"
    x = 0
    bpl000 = ax.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1-dx],
                          patch_artist=patch_artist, medianprops={"color":"blue"})
    bpl001 = ax.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], positions=[x+1],
                          patch_artist=patch_artist, medianprops={"color":"black"})
    bpl002 = ax.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1+dx],
                          patch_artist=patch_artist, medianprops={"color":"red"})

    for patch0, patch1, patch2 in zip(bpl000['boxes'], bpl001['boxes'], bpl002['boxes']):
        patch0.set_facecolor("red")
        patch1.set_facecolor("grey")
        patch2.set_facecolor("blue")
    # end for

    a_p = "pwl_slab"
    x = 1
    bpl003 = ax.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1-dx],
                          patch_artist=patch_artist, medianprops={"color":"blue"})
    bpl004 = ax.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], positions=[x+1],
                          patch_artist=patch_artist, medianprops={"color":"black"})
    bpl005 = ax.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1+dx],
                          patch_artist=patch_artist, medianprops={"color":"red"})

    for patch0, patch1, patch2 in zip(bpl003['boxes'], bpl004['boxes'], bpl005['boxes']):
        patch0.set_facecolor("red")
        patch1.set_facecolor("grey")
        patch2.set_facecolor("blue")
    # end

    a_p = "wet"
    x = 2
    bpl006 = ax.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1-dx],
                          patch_artist=patch_artist, medianprops={"color":"blue"})
    bpl007 = ax.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], positions=[x+1],
                          patch_artist=patch_artist, medianprops={"color":"black"})
    bpl008 = ax.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1+dx],
                          patch_artist=patch_artist, medianprops={"color":"red"})

    for patch0, patch1, patch2 in zip(bpl006['boxes'], bpl007['boxes'], bpl008['boxes']):
        patch0.set_facecolor("red")
        patch1.set_facecolor("grey")
        patch2.set_facecolor("blue")
    # end

    a_p = "y"
    x = 3
    bpl009 = ax.boxplot(ec_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1-dx],
                          patch_artist=patch_artist, medianprops={"color":"blue"})
    bpl010 = ax.boxplot(no3_adf_dict[reg_code]["adf"][a_p]["full"][1].loc[1985:2005], positions=[x+1],
                          patch_artist=patch_artist, medianprops={"color":"black"})
    bpl011 = ax.boxplot(gf_adf_dict[reg_code]["adf"]["historical"][a_p]["full"][1], positions=[x+1+dx],
                          patch_artist=patch_artist, medianprops={"color":"red"})

    for patch0, patch1, patch2 in zip(bpl009['boxes'], bpl010['boxes'], bpl011['boxes']):
        patch0.set_facecolor("red")
        patch1.set_facecolor("grey")
        patch2.set_facecolor("blue")
    # end
    ax.set_ylim(ylims)
    ax.set_title(f"{regions_pl[reg_code]}")
# end for ax, reg_code

# xticks
for ax in [ax02, ax10, ax11]:
    ax.set_xticks(sorted(np.concatenate([np.arange(1, 5, 1), np.arange(1, 5, 1)-dx, np.arange(1, 5, 1)+dx])))
    ax.set_xticklabels(np.tile(["EC-Earth", "NORA3", "GFDL-CM3"], reps=4), rotation=90)
# end for ax
for ax in [ax00, ax01]:
    ax.set_xticklabels([])
# end for ax

# yticks
for ax in [ax01, ax02, ax11]:
    ax.set_yticklabels([])

# ylabel
for ax in [ax00, ax10]:
    ax.set_ylabel("Number of AvDs/non-AvDs")
    ax.set_ylabel("Number of AvDs/non-AvDs")
# end for ax

fig.subplots_adjust(hspace=0.175, wspace=0.05)

pl.show()
pl.close()
