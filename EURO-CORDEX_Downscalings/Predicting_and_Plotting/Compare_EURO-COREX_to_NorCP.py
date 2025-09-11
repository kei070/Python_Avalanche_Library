#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the EURO-CORDEX data to the NorCP data.
"""


#%% imports
import numpy as np
import pandas as pd
import pylab as pl

from ava_functions.Lists_and_Dictionaries.Paths import path_par, path_par3


#%% set parameters
ncp_mods = ["EC-Earth", "GFDL-CM3"]
ncp_scen_0 = "historical"
ncp_scen_a = "rcp85_MC"
ncp_scen_b = "rcp85_LC"
euc_mods = ["CNRM_RCA", "IPSL_RCA", "MPI_RCA"]
euc_scen_0 = "hist"
euc_scen_a = "rcp85"

h_lo = 300
h_hi = 600


#%% set paths
ncp_path = f"{path_par}/IMPETUS/NorCP/Avalanche_Region_Predictors/Mean/"  # "EC-EARTH_rcp45_LC/Between300_and_600m/"
euc_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Avalanche_Region_Predictors/"  # "rcp45/IPSL_RCA/Between300_and_600m"


#%% load the data
ncp_dfs = {}
ncp_a_dfs = {}
ncp_as_dfs = {}
for ncp_mod in ncp_mods:
    ncp_dir_0 = f"{ncp_mod}_{ncp_scen_0}/Between{h_lo}_and_{h_hi}m/"
    ncp_dir_a = f"{ncp_mod}_{ncp_scen_a}/Between{h_lo}_and_{h_hi}m/"
    ncp_dir_b = f"{ncp_mod}_{ncp_scen_b}/Between{h_lo}_and_{h_hi}m/"

    ncp_df_0 = pd.read_csv(ncp_path + ncp_dir_0 +
                           f"Lyngen_NorCP_Predictors_MultiCellMean_Between{h_lo}_and_{h_hi}m.csv",
                           index_col=0, parse_dates=True)
    ncp_df_a = pd.read_csv(ncp_path + ncp_dir_a +
                           f"Lyngen_NorCP_Predictors_MultiCellMean_Between{h_lo}_and_{h_hi}m.csv",
                           index_col=0, parse_dates=True)
    ncp_df_b = pd.read_csv(ncp_path + ncp_dir_b +
                           f"Lyngen_NorCP_Predictors_MultiCellMean_Between{h_lo}_and_{h_hi}m.csv",
                           index_col=0, parse_dates=True)
    ncp_df = pd.concat([ncp_df_0, ncp_df_a, ncp_df_b], axis=0)

    ncp_a_df = ncp_df.groupby(ncp_df.index.year).mean()
    ncp_as_df = ncp_df.groupby(ncp_df.index.year).sum()

    # ncp_a_df["total_prec_sum"][ncp_a_df["total_prec_sum"] == 0] = np.nan
    # ncp_as_df["total_prec_sum"][ncp_as_df["total_prec_sum"] == 0] = np.nan

    ncp_a_dfs[ncp_mod] = ncp_a_df
    ncp_as_dfs[ncp_mod] = ncp_as_df

# end for ncp_mod

euc_dfs = {}
euc_a_dfs = {}
euc_as_dfs = {}
for euc_mod in euc_mods:
    euc_dir_0 = f"{euc_scen_0}/{euc_mod}/Between{h_lo}_and_{h_hi}m/"
    euc_dir_a = f"{euc_scen_a}/{euc_mod}/Between{h_lo}_and_{h_hi}m/"
    euc_df_0 = pd.read_csv(euc_path + euc_dir_0 +
                           f"Lyngen_{euc_mod}_Predictors_MultiCellMean_Between{h_lo}_and_{h_hi}m.csv",
                           index_col=0, parse_dates=True)
    euc_df_a = pd.read_csv(euc_path + euc_dir_a +
                           f"Lyngen_{euc_mod}_Predictors_MultiCellMean_Between{h_lo}_and_{h_hi}m.csv",
                           index_col=0, parse_dates=True)
    euc_df = pd.concat([euc_df_0, euc_df_a], axis=0)
    euc_dfs[euc_mod] = euc_df

    euc_a_dfs[euc_mod] = euc_df.groupby(euc_df.index.year).mean()
    euc_as_dfs[euc_mod] = euc_df.groupby(euc_df.index.year).sum()
# end for euc_mod


#%% plot temperature
fig = pl.figure(figsize=(8, 4))
ax00 = fig.add_subplot(111)

ax00.plot(ncp_a_dfs["EC-Earth"].index, ncp_a_dfs["EC-Earth"]["t_mean"], c="black")
ax00.plot(ncp_a_dfs["GFDL-CM3"]["t_mean"], c="gray")
ax00.plot(euc_a_dfs["CNRM_RCA"]["t_mean"], c="red")
ax00.plot(euc_a_dfs["IPSL_RCA"]["t_mean"], c="orange")
ax00.plot(euc_a_dfs["MPI_RCA"]["t_mean"], c="green")

p00, = ax00.plot([], c="black", label="EC-Earth")
p01, = ax00.plot([], c="gray", label="GFDL-CM3")
p02, = ax00.plot([], c="red", label="CNRM-RCA")
p03, = ax00.plot([], c="orange", label="IPSL-RCA")
p04, = ax00.plot([], c="green", label="MPI-RCA")

l00 = ax00.legend(handles=[p00, p01], title="NorCP", loc="upper center")
ax00.legend(handles=[p02, p03, p04], title="EURO-CORDEX", loc="upper left")
ax00.add_artist(l00)

ax00.set_xlabel("Year")
ax00.set_ylabel("Temperature in K")

ax00.set_title("Mean temperature Lyngen (300-600 m)")

pl.show()
pl.close()


#%% plot precipitation
fig = pl.figure(figsize=(8, 4))
ax00 = fig.add_subplot(111)

ax00.plot(ncp_as_dfs["EC-Earth"]["total_prec_sum"], c="black")
ax00.plot(ncp_as_dfs["GFDL-CM3"]["total_prec_sum"], c="gray")
ax00.plot(euc_as_dfs["CNRM_RCA"]["rr"], c="red")
ax00.plot(euc_as_dfs["IPSL_RCA"]["rr"], c="orange")
ax00.plot(euc_as_dfs["MPI_RCA"]["rr"], c="green")

p00, = ax00.plot([], c="black", label="EC-Earth")
p01, = ax00.plot([], c="gray", label="GFDL-CM3")
p02, = ax00.plot([], c="red", label="CNRM-RCA")
p03, = ax00.plot([], c="orange", label="IPSL-RCA")
p04, = ax00.plot([], c="green", label="MPI-RCA")

l00 = ax00.legend(handles=[p00, p01], title="NorCP", loc="upper center")
ax00.legend(handles=[p02, p03, p04], title="EURO-CORDEX", loc="upper left")
ax00.add_artist(l00)

ax00.set_xlabel("Year")
ax00.set_ylabel("Precipitation in mm")

ax00.set_title("Precipitation Lyngen (300-600 m)")

pl.show()
pl.close()