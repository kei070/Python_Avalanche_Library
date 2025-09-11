#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Take a first look into the "skred data" from Norway.

Problem: There appear to be a strong inhomogeneities in these data, especially after 2010, with a strong (exponential)
         increase in the number of events. That is, before 2010 in most years fewer than 100 events are recorded, with
         individual years exhibiting about 200 events. However, after about 2015 more than 200 events are recorded per
         year, with especially the years after 2020 exhibiting up to 1000 events.

According to the following document skredType >= 130 and <= 139 and == 151 are SNOW avalanches.

https://register.geonorge.no/data/documents/Tegneregler_Skredhendelser_v5_presentasjonsregler-skredhendelser-20180530_.pdf

The data were downloaded from https://nedlasting.nve.no/gis/ with
   -- format: ESRI .shp
   -- coords: Geografiske koordinateer WGS84
   -- utvalg: Overlapper
   -- area:   fylke --> Troms
"""


#%% imports
import numpy as np
import pandas as pd
import geopandas as gpd
import pylab as pl
from ava_functions.Assign_Winter_Year import assign_winter_year
from ava_functions.Lists_and_Dictionaries.Paths import path_par, obs_path


#%% set paths
# path = f"/{path_par}/IMPETUS/Skred_Norge/SHP/"
# fn = "Skred_Skredhendelse_Troms.shp"

path = f"/{path_par}/IMPETUS/Skred_Norge/NVE_47551B14_1754397250973_15500/NVEKartdata/NVEData/Skred_Skredhendelse/"
fn = "Skred_Skredhendelse.shp"

warn_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
pl_path = f"/{path_par}/IMPETUS/Skred_Norge/Plots/"


#%% load the list of Norwegian mainland warning regions with daily ADLs
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg")


#%% exclude the Svalbard regions and extract A and B
warn_info = warn_info[warn_info["reg_code"] > 3004]
warn_a_info = warn_info[warn_info["regionType"] == "A"]
warn_b_info = warn_info[warn_info["regionType"] == "B"]


#%% extract the North
warn_a_info_north = warn_a_info[(warn_a_info.reg_code < 3014) & (warn_a_info.reg_code > 3008)]
warn_b_info_north = warn_b_info[(warn_b_info.reg_code < 3014) & (warn_b_info.reg_code > 3008)]


#%% load the shp file
skred = gpd.read_file(path + fn)


#%% filter the snow avalanches
snow_skred = skred[(((skred["skredType"] >= 130) & (skred["skredType"] <= 139)) | (skred["skredType"] == 151))]
snow_skred = snow_skred[~snow_skred["Tidspunkt"].isnull()]


#%% re-format the "Tidspunkt" column
dates = pd.to_datetime([k[:8] for k in snow_skred["Tidspunkt"]], format='%Y%m%d')


#%% add the dates column to the snow_skred dataframe
snow_skred["dates"] = dates


#%% assign the winter year
snow_skred["winter_season"] = assign_winter_year(snow_skred["dates"], start_month=12)


#%% get the indices of the winter (Dec-Feb) and spring (Mar-May) events
winter_inds = snow_skred["dates"].dt.month == 0
spring_inds = snow_skred["dates"].dt.month == 0
for mon in [12, 1, 2]:
    winter_inds = winter_inds | (snow_skred["dates"].dt.month == mon)
# end for mon
for mon in [3, 4, 5]:
    spring_inds = spring_inds | (snow_skred["dates"].dt.month == mon)
# end for mon


#%% count the number of events per
u_year = np.unique(snow_skred["winter_season"])

event_count = np.array([np.sum(snow_skred["winter_season"] == k) for k in u_year])


#%% count the events in winter and spring
u_year_wi = np.unique(snow_skred[winter_inds]["winter_season"])
u_year_sp = np.unique(snow_skred[spring_inds]["winter_season"])

event_count_wi = np.array([np.sum(snow_skred[winter_inds]["winter_season"] == k) for k in u_year_wi])
event_count_sp = np.array([np.sum(snow_skred[spring_inds]["winter_season"] == k) for k in u_year_sp])


#%% plot the number of events per winter season
sta_yr = 1980

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.scatter(u_year[u_year >= sta_yr], event_count[u_year >= sta_yr])

ax00.set_xlabel("Year")
ax00.set_ylabel("Number of snow avalanches")

ax00.set_title("NVE 'Snøskredhendelser'")

pl.show()
pl.close()


#%% plot the number of events per winter season -- winter and spring separately
sta_yr = 1980

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.scatter(u_year_wi[u_year_wi >= sta_yr], event_count_wi[u_year_wi >= sta_yr], c="blue", label="winter")
ax00.scatter(u_year_sp[u_year_sp >= sta_yr], event_count_sp[u_year_sp >= sta_yr], c="red", label="spring")
ax00.legend()

ax00.set_xlabel("Year")
ax00.set_ylabel("Number of snow avalanches")

ax00.set_title("NVE 'Snøskredhendelser'")

pl.show()
pl.close()



#%% count the number of deaths due to snow avalanches
tot_deaths = snow_skred["antPersOmk"].sum()


#%% count the deaths per season
deaths_per_season = np.array([snow_skred[snow_skred["winter_season"] == k]["antPersOmk"].sum() for k in u_year])


#%% plot the events on a map
fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(111)
warn_a_info_north.plot(ax=ax00, facecolor="none")
skred.plot(ax=ax00)

pl.show()
pl.close()


#%% plot the fatalities per season
sta_yr = 1700

fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

ax00.scatter(u_year[u_year >= sta_yr], deaths_per_season[u_year >= sta_yr])

ax00.set_xlabel("Year")
ax00.set_ylabel("Number of snow-avalanche related fatalities")

ax00.set_title("NVE 'Snøskredhendelser'")

pl.show()
pl.close()


#%% plot events and deaths
sta_yr = 1700

fig = pl.figure(figsize=(5, 3))
ax00 = fig.add_subplot(111)
ax00_1 = ax00.twinx()

p00 = ax00.scatter(u_year[u_year >= sta_yr], deaths_per_season[u_year >= sta_yr], facecolor="none", edgecolor="red",
                   label=f"Fatalities (total: {tot_deaths.astype(int)})")
p01 = ax00_1.scatter(u_year[u_year >= sta_yr], event_count[u_year >= sta_yr], facecolor="none", edgecolor="black",
                     marker="d", label="Obeserved avalanches")
l00 = ax00.legend(handles=[p00], loc=(0.01, 0.75), labelcolor="red")
l01 = ax00.legend(handles=[p01], loc=(0.01, 0.638), labelcolor="black")
ax00.add_artist(l00)

ax00.set_xlabel("Year")
ax00.set_ylabel("Number of snow-avalanche\nrelated fatalities", color='red')
ax00_1.set_ylabel("Number of observed\nsnow avalanches")

ax00.set_title("Snow avalanches and associated fatalities Troms")

ax00.set_ylim(0, 24.5)
ax00_1.set_ylim(0, 1100)

ax00.spines['left'].set_color('red')
ax00_1.spines['left'].set_color('red')
ax00.tick_params(axis='y', colors='red')

pl.savefig(pl_path + "SnowAva_and_Fatalities.pdf", bbox_inches="tight", dpi=150)

pl.show()
pl.close()
