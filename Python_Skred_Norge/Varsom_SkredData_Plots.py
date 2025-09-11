#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot some data from avalanches.
"""

#%% imports
import numpy as np
import pandas as pd
import pylab as pl

from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set paths
data_path = f"{path_par}/IMPETUS/Skred_Norge/Varsom_Skredulykker/"
pl_path = f"{path_par}/IMPETUS/Skred_Norge/Varsom_Skredulykker/Plots/"

fn_akt = "Varsom_Aktivitet.csv"
fn_off = "Varsom_AntallOffer.csv"
fn_maa = "Varsom_EtterMaaned.csv"
fn_omr = "Varsom_Omraade.csv"


#%% load data
df_akt = pd.read_csv(data_path + fn_akt, sep=";", header=2)
df_off = pd.read_csv(data_path + fn_off, sep=";", header=2)
df_maa = pd.read_csv(data_path + fn_maa, sep=";", header=2)
df_omr = pd.read_csv(data_path + fn_omr, sep=";", header=2)


#%% some lists
regions = ["Østfold ", "Akersh.", "Hedm.", "Oppl.", "Busker.", "Telem.", "Agder ", "Rogal.", "Hordal.", "S. og F.",
           "M. og R.", "Trøndel.", "Nordl.", "Troms ", "Finnm.", "Svalb.", "Jan M."]

activities = ["Skiing", "Walking", "Snow\nscooter", "Snow\ngroomer", "Road", "Train", "Building"]

season = [f"{k:02}" for k in np.arange(9, 25+1, 1)]

season2 = [f"20{k:02}/{k+1}" for k in np.arange(8, 24+1, 1)]


#%% plot -- for the project proposal
fig = pl.figure(figsize=(9, 5))
ax00 = fig.add_subplot(221)
ax00_1 = ax00.twinx()
ax01 = fig.add_subplot(222)
ax01_1 = ax01.twinx()
ax10 = fig.add_subplot(223)
ax10_1 = ax10.twinx()
ax11 = fig.add_subplot(224)
ax11_1 = ax11.twinx()

ax00.bar(df_off.index-0.2, df_off["Døde"], width=0.4, color="red")
ax00_1.bar(df_off.index+0.2, df_off["Skredtatte"], width=0.4, color="gray")
ax00.set_xticks(df_off.index)
ax00.set_xticklabels(season, rotation=0)
ax00.set_ylabel("Number of Fatalities")
# ax00_1.set_ylabel("Taken")
ax00.set_title("(a) Avalanche season")

ax01.bar(df_maa.index-0.2, df_maa["Døde"], width=0.4, color="red")
ax01_1.bar(df_maa.index+0.2, df_maa["Skredtatte"], width=0.4, color="gray")
ax01.set_xticks(df_maa.index)
ax01.set_xticklabels(["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"], rotation=0)
# ax01.set_ylabel("Fatalities")
ax01_1.set_ylabel("Number of people seized")
ax01.set_title("(b) Month")

ax10.bar(df_omr.index-0.2, df_omr["Døde"], width=0.4, color="red")
ax10_1.bar(df_omr.index+0.2, df_omr["Skredtatte"], width=0.4, color="gray")
ax10.set_xticks(df_omr.index)
ax10.set_xticklabels(regions, rotation=90)
ax10.set_ylabel("Number of Fatalities")
# ax10_1.set_ylabel("Taken")
ax10.set_title("(c) Region")

ax11.bar(df_akt.index-0.2, df_akt["Døde"], width=0.4, color="red")
ax11_1.bar(df_akt.index+0.2, df_akt["Skredtatte"], width=0.4, color="gray")
ax11.set_xticks(df_akt.index)
ax11.set_xticklabels(activities, rotation=90)
# ax11.set_ylabel("Fatalities")
ax11_1.set_ylabel("Number of people seized")
ax11.set_title("(d) Activity")

fig.subplots_adjust(wspace=0.25, hspace=0.35)

pl.savefig(pl_path + "Varsom_Skredulykker.png", dpi=200, bbox_inches="tight")

pl.show()
pl.close()


#%% for the adaptation pathways paper
fig = pl.figure(figsize=(6, 2.5))
ax00 = fig.add_subplot(111)
ax00_1 = ax00.twinx()

ax00.bar(df_off.index-0.2, df_off["Døde"], width=0.4, color="red")
ax00_1.bar(df_off.index+0.2, df_off["Skredtatte"], width=0.4, color="black")
ax00.set_xticks(df_off.index)
ax00.set_xticklabels(season2, rotation=45)
ax00.set_ylabel("Number of fatalities", color='red')
ax00_1.set_ylabel("Number of people seized")

ax00.spines['left'].set_color('red')
ax00_1.spines['left'].set_color('red')
ax00.tick_params(axis='y', colors='red')

pl.savefig(pl_path + "Varsom_Skredulykker_AP_WP5_Paper.png", dpi=200, bbox_inches="tight")

pl.show()
pl.close()




