"""
Plot the avalanche-region shape files.
"""

#%% load modules
import sys
import xarray as xr
import geopandas as gpd
import pylab as pl
import matplotlib as mpl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import shapely
from datetime import datetime
# import earthpy.spatial as es
from matplotlib.colors import LightSource

from ava_functions.Lists_and_Dictionaries.Paths import obs_path, path_par


#%% set paths
shp_path = f"{obs_path}/IMPETUS/Debmita/AvalancheRegion_shapefiles/"
nora3_path = f"{path_par}/IMPETUS/NORA3/"
warn_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"


#%% load the nc file
f_name = "nora3_topo.nc"
nc = Dataset(nora3_path + f_name)
xr_nc = xr.open_dataset(nora3_path + f_name)


#%% get the topography info from xarray
xr_sz = xr_nc["ZS"].values.squeeze()

# Create a light source object
ls = LightSource(azdeg=315, altdeg=45)

# Generate hillshade
hillshade = ls.hillshade(xr_sz)


#%% extract the projection
crs = nc.variables["projection_lambert"].proj4


#%% load the surface geopotential
print(nc.variables["ZS"])
sz = nc.variables["ZS"][:].squeeze()

lons_sz_all = nc.variables["longitude"]
lats_sz_all = nc.variables["latitude"]


#%% reduce the size
x1, x2 = 380, 520
y1, y2 = 840, 960
sz_pl = sz[y1:y2, x1:x2]
lons_sz_pl = lons_sz_all[y1:y2, x1:x2]
lats_sz_pl = lats_sz_all[y1:y2, x1:x2]
# hillshade = es.hillshade(sz, azimuth=270, altitude=1)
# sz_pl[sz_pl < 10] = np.nan


#%% load the shape file and the NORA3 file for the grid
# shp = gpd.read_file(shp_path + "avalanche_regions.shp")
nora3 = Dataset(nora3_path + "fc2021030100_003_fp.nc")


#%% load the Tromsoe shape
"""
shp_tro = gpd.read_file(shp_path + "Tromsoe_Shape.shp")
shp_tro["N&R"] = "Tromsø (3011)"
shp_tro["region"] = 3011


#%% extract only the necessary columns
shp_tro = shp_tro[["region", "geometry", "N&R"]]
"""

#%% load the lat and lon from NORA3
x_n3 = nora3.variables["x"][:]
y_n3 = nora3.variables["y"][:]


#%% load the Lyngen gridcell coordinates as .csv
coords_path = f"{obs_path}/IMPETUS/Coords_Files/"

coords_df = pd.read_csv(coords_path + "Lyngen_Nora3_Coords.csv")


#%% set up discrete colorbar
cmap = mpl.cm.viridis
# bounds = np.sort(np.array(shp.region))
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
# mapp = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  --> apparently does not work with geopandas


#%% load the avalanche warning region info
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg")


#%% exclude the Svalbard regions
warn_info = warn_info[warn_info["reg_code"] > 3005]
warn_a_info = warn_info[warn_info["regionType"] == "A"]


#%%  add a new column to the dataframe, combining the name and region code
"""
n_and_r = []
for i, j in zip(shp.omrNavn, shp.region):
    element = f"{i} ({j})"
    n_and_r.append(element)
# end for i, j

shp["N&R"] = n_and_r


#%% extract only the necessary columns
shp = shp[["region", "geometry", "N&R"]]


#%% concatenate the north and tro shape files
shp = pd.concat([shp, shp_tro], axis=0)
"""

#%% calculate the centroid of the polygons
"""
centroids = []
for i in shp.geometry:
    cen = i.centroid
    centroids.append(cen)
# end for i, j

centroids = gpd.GeoSeries(centroids, crs=shp.crs)
centroids = centroids.to_crs(shp.crs)
shp["centroid"] = centroids
"""

#%% extract the northern regions
# shp_north = shp[(shp.region < 3014) & (shp.region > 3008)]


#%% add in the avalanche risk change info
table_path = (f"{path_par}/IMPETUS/NORA3/Avalanche_Risk_Tables/Auto_TrainTest_Split/" +
              "Between400_and_900m/s3-wspeed_max/")
full_df = pd.read_csv(table_path +
                      "AvalancheRisk_HighRiskDay_Change_Between400_and_900m_1985-2023_PerLen15Yr_AllReg.csv")
# win_df = pd.read_csv(table_path + "AvalancheRisk_Change_Winter_1985-2022_AllReg.csv")
# spr_df = pd.read_csv(table_path + "AvalancheRisk_Change_Spring_1985-2022_AllReg.csv")


#%% join the dataframes
# shp_north_f = pd.merge(shp_north, full_df, on="region")
# shp_north_w = pd.merge(shp_north, win_df, on="region")
# shp_north_s = pd.merge(shp_north, spr_df, on="region")


#%% set up new discrete colorbar
cmap = mpl.cm.viridis
# bounds = np.sort(np.array(shp_north.region))
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
# mapp = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


#%% extract the projection
crs = nora3.variables["projection_lambert"].proj4


#%% gridcell extent: 3000x3000m2
dx = 3000
dy = 3000


#%% reduce the size of x and y for the NORA3 grid
x_red = x_n3[400:450]
y_red = y_n3[870:920]

x_red = x_n3[350:480]
y_red = y_n3[820:970]

# x_red = x_n3[380:410]
# y_red = y_n3[850:880]


#%% set up the NORA3 grid
grid_cells = []
for x0 in np.arange(np.min(x_red), np.max(x_red)+dx, dx):
    for y0 in np.arange(np.min(y_red), np.max(y_red)+dy, dy):
        # bounds
        x1 = x0 - dx
        y1 = y0 + dy
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    # for y0
# for x0

cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)


#%% reproject the shp_north to Orthographic
orth = ccrs.Orthographic(central_latitude=69, central_longitude=20)
warn_info_orth = warn_a_info.to_crs(crs=orth.proj4_init)
# shp_north_f_orth = shp_north_f.to_crs(crs=orth.proj4_init)
# shp_orth = shp.to_crs(crs=orth.proj4_init)
# shp_north_w_orth = shp_north_w.to_crs(crs=orth.proj4_init)
# shp_north_s_orth = shp_north_s.to_crs(crs=orth.proj4_init)
# shp_tro_orth = shp_tro.to_crs(crs=orth.proj4_init)


#%% extract the North
warn_info_north = warn_info_orth[(warn_info_orth.reg_code < 3014) & (warn_info_orth.reg_code > 3008)]


#%% set Lyngen centroid
lon = 19.80548
lat = 69.58474
# sys.exit()


#%% plot the northern avalanche regions --> Cryosphere paper Fig. 1
pl_path = f"{obs_path}/IMPETUS/Plots/"
pl_path_pub = f"{obs_path}/IMPETUS/Publishing/The Cryosphere/Copernicus_LaTeX_Package/"

nora3_crs = ccrs.LambertConformal(central_longitude=-42, central_latitude=66.3, standard_parallels=(66.3, 66.3))

fig = pl.figure(figsize=(8, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)

axes = fig.add_subplot(gs[0, 0], projection=ccrs.Orthographic(central_latitude=69, central_longitude=20))

axes.axis("off")

p_topo = axes.pcolormesh(lons_sz_pl, lats_sz_pl, sz_pl, transform=ccrs.PlateCarree(), cmap="Grays")
cb = fig.colorbar(p_topo, ax=axes, shrink=0.7)
cb.set_label("Elevation in m")

axes.add_feature(cfeature.OCEAN, zorder=101)
axes.add_feature(cfeature.COASTLINE, zorder=101)
axes.add_feature(cfeature.BORDERS, edgecolor="black")

ext = [16, 22.7, 68.25, 70.45]
axes.set_extent(ext)


# add the regions shp
"""
shp_north_f_orth.plot(ax=axes, column="N&R", legend=True, zorder=101, linewidth=1.8,
                      legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                      edgecolor="black", cmap="Set1")
"""
warn_info_north.plot(ax=axes, legend=False, zorder=101, linewidth=1.8,
                    legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                    edgecolor="black", cmap="Set1")

fweight = "heavy"
axes.text(15.8, 69.6, "Sør-Troms", transform=ccrs.PlateCarree(), zorder=101, fontweight=fweight, fontsize=15)
axes.text(17.18, 70.1, "Tromsø", transform=ccrs.PlateCarree(), zorder=101, fontweight=fweight, fontsize=15)
axes.text(18.3, 68.8, "Indre Troms", transform=ccrs.PlateCarree(), zorder=101, fontweight=fweight, fontsize=15)
axes.text(20.7, 69.55, "Nord-Troms", transform=ccrs.PlateCarree(), zorder=101, fontweight=fweight, fontsize=15)
axes.text(19.1, 69.5, "Lyngen", transform=ccrs.PlateCarree(), zorder=101, fontweight=fweight, fontsize=15)

# add the NORA3 grid mesh
# cell.plot(ax=axes, facecolor="none", edgecolor='gray', linewidth=0.1, transform=nora3_crs, zorder=101)
gl = axes.gridlines(draw_labels=True, linewidth=0.5, zorder=101)

gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True

# Create an inset axes with a different projection in the lower right corner
# The rect parameter [left, bottom, width, height] specifies the location and size of the inset
ax_inset = pl.axes([0.577, 0.2255, 0.2, 0.2], facecolor='white',
                   projection=ccrs.LambertConformal(central_longitude=0, central_latitude=60))
ax_inset.gridlines(draw_labels=False, linewidth=0.2)
ax_inset.set_extent([4.5, 22.5, 57.75, 71.5])

# ax_inset.axis("off")

x11, x12 = 16, 23  # 15, 21
x21, x22 = 15.9, 22.9  # 19, 25
y11, y12 = 68.2, 68.2  # 69.5, 68
y21, y22 = 70.4, 70.4  # 69.5+2, 68+2

# horizontal lines
ax_inset.plot([x21, x22], [y21, y22], transform=ccrs.PlateCarree(), c="black")  # northern line
ax_inset.plot([x11, x12], [y11, y12], transform=ccrs.PlateCarree(), c="black")  # southern line

# vertical lines
ax_inset.plot([x11, x21], [y11, y21], transform=ccrs.PlateCarree(), c="black")  # western line
ax_inset.plot([x12, x22], [y12, y22], transform=ccrs.PlateCarree(), c="black")  # eastern line

# Add features to the inset map
ax_inset.add_feature(cfeature.LAND)
ax_inset.add_feature(cfeature.OCEAN)
ax_inset.add_feature(cfeature.COASTLINE)
ax_inset.add_feature(cfeature.BORDERS, linestyle=':')

axes.set_title("Avalanche danger assessment regions")

pl_path = f"{path_par}/IMPETUS/NORA3/Plots/NORA3_Maps/"
pl.savefig(pl_path + "Study_Region_with_Inset.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()



#%% plot the northern avalanche regions with hillshade
"""
pl_path = "/home/kei070/Documents/IMPETUS/Plots/"
pl_path_pub = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Copernicus_LaTeX_Package/"

nora3_crs = ccrs.LambertConformal(central_longitude=-42, central_latitude=66.3, standard_parallels=(66.3, 66.3))

fig = pl.figure(figsize=(6, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)

axes = fig.add_subplot(gs[0, 0], projection=ccrs.Orthographic(central_latitude=69, central_longitude=20))

axes.axis("off")

axes.imshow(hillshade, origin='upper', extent=[16, 22.7, 68.25, 70.45], transform=ccrs.PlateCarree(),
            cmap='gray', alpha=0.7)

axes.add_feature(cfeature.OCEAN, zorder=101)
axes.add_feature(cfeature.COASTLINE, zorder=101)
axes.add_feature(cfeature.BORDERS, edgecolor="black")

ext = [16, 22.7, 68.25, 70.45]
axes.set_extent(ext)


# cell.plot(ax=axes, facecolor="none", edgecolor='gray', linewidth=0.1, transform=nora3_crs)
gl = axes.gridlines(draw_labels=True, linewidth=0.2)

gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True

# Create an inset axes with a different projection in the lower right corner
# The rect parameter [left, bottom, width, height] specifies the location and size of the inset
ax_inset = pl.axes([0.555, 0.295, 0.2, 0.2], facecolor='white',
                   projection=ccrs.LambertConformal(central_longitude=0, central_latitude=60))
ax_inset.gridlines(draw_labels=False, linewidth=0.2)
ax_inset.set_extent([4.5, 22.5, 57.75, 71.5])

# ax_inset.axis("off")

x11, x12 = 15, 21
x21, x22 = 19, 25
y11, y12 = 69.5, 68
y21, y22 = 69.5+2, 68+2
ax_inset.plot([x11, x12], [y11, y12], transform=ccrs.PlateCarree(), c="black")
ax_inset.plot([x21, x22], [y21, y22], transform=ccrs.PlateCarree(), c="black")

ax_inset.plot([x11, x21], [y11, y21], transform=ccrs.PlateCarree(), c="black")
ax_inset.plot([x12, x22], [y12, y22], transform=ccrs.PlateCarree(), c="black")

# Add features to the inset map
ax_inset.add_feature(cfeature.LAND)
ax_inset.add_feature(cfeature.OCEAN)
ax_inset.add_feature(cfeature.COASTLINE)
ax_inset.add_feature(cfeature.BORDERS, linestyle=':')

axes.set_title("Avalanche danger assessment regions")

# pl_path = "/media/kei070/One Touch/IMPETUS/NORA3/Plots/NORA3_Maps/"
# pl.savefig(pl_path + "Study_Region_with_Inset.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% test hillshade

# lon_start, lon_end, lat_start, lat_end = 16, 22.7, 68.25, 70.45
lon_start, lon_end = np.min(lons_sz_all), np.max(lons_sz_all)
lat_start, lat_end = np.min(lats_sz_all), np.max(lats_sz_all)

# Define the projection
projection = ccrs.LambertConformal(central_longitude=-42, central_latitude=66.3, standard_parallels=(66.3, 66.3))
projection = ccrs.PlateCarree()

# Create a figure and add a GeoAxes with the specified projection
fig, ax = pl.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

# Set the extent (modify this based on your data's geographic extent)
ax.set_extent([lon_start, lon_end, lat_start, lat_end], crs=ccrs.PlateCarree())

# Plot the hillshade data
ax.imshow(hillshade.T, origin='upper', extent=[lon_start, lon_end, lat_start, lat_end],
          transform=ccrs.PlateCarree(), cmap='gray', alpha=0.7)

# Add features like coastlines for context
# ax.coastlines()

# Show the plot
pl.show()
"""