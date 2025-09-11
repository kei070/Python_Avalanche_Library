#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the avalanche warning regions from the NVE server.

For help see https://gis4.nve.no/map/sdk/rest/index.html#/Query_Map_Service_Layer/02ss0000000r000000/
"""

#%% imports
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% set the key
url = "https://gis4.nve.no/map/rest/services/Mapservices/XgeoStatic1/MapServer/22/"
query = "query?where=areal_km2<360000&outFields=areal_km2,regionType,safeName,OMRAADEID,OMRAADENAVN"
form = "f=geojson"
key = f"{url}{query}&{form}"


#%% submit the request
data = requests.get(key)


#%% convert to JSON
data_j = data.json()


#%% loop over the dictionary's keys and extract the elements
coords = []
names = []
rtype = []
reg_code = []
reg_name = []
area = []

for elem in data_j["features"]:
    coords.append(Polygon(elem["geometry"]["coordinates"][0]))
    names.append(elem["properties"]["safeName"])
    rtype.append(elem["properties"]["regionType"])
    reg_code.append(elem["properties"]["OMRAADEID"])
    # reg_name.append(elem["properties"]["OMRAADENAVN"])
    area.append(elem["properties"]["areal_km2"])
# end for elem


#%% convert to dataframe
data_df = gpd.GeoDataFrame({"reg_code":reg_code, "safeName":names, "regionType":rtype,
                            "areal_km2":area, "geometry":coords})
# data_df.set_geometry(col="geometry")


#%% attempt a plot
data_df.plot()


#%% store
store_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
data_df.to_file(store_path + "Warning_Region_Info.gpkg", driver="GPKG")

