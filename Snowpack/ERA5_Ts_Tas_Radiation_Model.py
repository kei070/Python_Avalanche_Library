#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to generate a (linear?) model to infer the surface temperature from 2m temperature, radiation, and potentially
some other quantities like RH etc.
"""


#%% imports
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import pylab as pl
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set paths
model_path = f"{path_par}/IMPETUS/ERA5/Stat_Models/"
e5_path = f"{path_par}/IMPETUS/ERA5/"
e5_fnl1 = ["ERA5_hourly_data_1980_90_1.nc", "ERA5_hourly_data_2000_1.nc"]
e5_fnl2 = ["ERA5_hourly_data_1980_90_2.nc", "ERA5_hourly_data_2000_2.nc"]


#%% load the file
e5_nc1 = xr.open_mfdataset([e5_path + e5_fn1 for e5_fn1 in e5_fnl1])
e5_nc2 = xr.open_mfdataset([e5_path + e5_fn2 for e5_fn2 in e5_fnl2])


#%% load data for plot on map
ts_pl = np.array(e5_nc1["skt"][10, :, :])
lat = np.array(e5_nc1["latitude"][:])
lon = np.array(e5_nc1["longitude"][:])


#%% plot the data on a map
# j, i = 23, 58
j, i = 24, 58

ts_pl[j, i] = 500

x, y = np.meshgrid(lon, lat)

fig = pl.figure(figsize=(6, 4))
ax00 = fig.add_subplot(111, projection=ccrs.PlateCarree())

ax00.pcolormesh(x, y, ts_pl, transform=ccrs.PlateCarree())

ax00.add_feature(cfeature.COASTLINE, zorder=101)
ax00.add_feature(cfeature.BORDERS, edgecolor="black")

pl.show()
pl.close()


#%% extract individual variables for a random point
ts = np.array(e5_nc1["skt"][:, j, i])
tas = np.array(e5_nc1["t2m"][:, j, i])
ws = (np.array(e5_nc1["u10"][:, j, i])**2 + np.array(e5_nc1["v10"][:, j, i])**2)**0.5
snlw = np.array(e5_nc2["str"][:, j, i])
snsw = np.array(e5_nc2["ssr"][:, j, i])


#%% plot tas v. ts
fig = pl.figure(figsize=(5, 3))
ax00 = fig.add_subplot(111)
ax00.scatter(tas, ts, s=10, facecolor="none", edgecolor="black")
ax00.set_xlabel("Tas in K")
ax00.set_ylabel("Ts in K")

pl.show()
pl.close()


#%% set up the data for a linear model
data_x = pd.DataFrame({"tas":tas, "ws":ws, "snlw":snlw, "snsw":snsw}, index=e5_nc1["valid_time"])
data_y = ts


#%% set up a training and test data split
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=42)


#%% plot the distributions of the training and test target data (i.e., the Ts)
fig = pl.figure(figsize=(5, 3))
ax00 = fig.add_subplot(111)
sns.kdeplot(y_train, fill=False, label="traning", ax=ax00)
sns.kdeplot(y_test, fill=False, label="test", ax=ax00)
sns.kdeplot(data_y, fill=False, label="full", ax=ax00)
ax00.legend()
pl.show()
pl.close()


#%% generate the model
model = LinearRegression()
model.fit(X_train, y_train)


#%% perform the prediction
y_pred = model.predict(X_test)
y_pred_tr = model.predict(X_train)


#%% assess the model quality
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


#%% inspect the coefficients
coefficients = model.coef_
intercept = model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')


#%% plot the values against each other
ylim = (235, 300)
xlim = (235, 300)

fig = pl.figure(figsize=(8, 3))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

ax00.scatter(y_test, X_test["tas"], s=10, facecolor="none", edgecolor="gray", label="Tas")
ax00.scatter(y_test, y_pred, s=10, facecolor="none", edgecolor="black", label="Predicted")
ax00.axline(xy1=(275, 275), slope=1, c="orange")
ax00.set_xlabel("True Ts in K")
ax00.set_ylabel("Predicted or Tas in K")
ax00.set_title("Test data")
ax00.legend()

ax01.scatter(y_train, X_train["tas"], s=10, facecolor="none", edgecolor="gray")
ax01.scatter(y_train, y_pred_tr, s=10, facecolor="none", edgecolor="black")
ax01.axline(xy1=(275, 275), slope=1, c="orange")
ax01.set_xlabel("True Ts in K")
ax01.set_title("Traning data")
ax01.set_yticklabels([])

ax00.set_ylim(ylim)
ax01.set_ylim(ylim)
ax00.set_xlim(xlim)
ax01.set_xlim(xlim)

fig.subplots_adjust(wspace=0.05)

pl.show()
pl.close()


#%% dump the model
dump(model, model_path + "Linear_Ts_Model_ERA5.joblib")
