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

from ava_functions.Lists_and_Dictionaries.Region_Codes import regions, regions_pl
from ava_functions.Lists_and_Dictionaries.Paths import obs_path, path_par


#%% set paths
shp_path = f"{obs_path}/IMPETUS/Debmita/AvalancheRegion_shapefiles/"
nora3_path = f"{path_par}/IMPETUS/NORA3/"
warn_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"


#%% load the nc files
f_name = "nora3_topo.nc"
nc = Dataset(nora3_path + f_name)
xr_nc = xr.open_dataset(nora3_path + f_name)
warn_nc = Dataset(nora3_path + "Warn_Region_Raster.nc")


#%% get the topography info from xarray
xr_sz = xr_nc["ZS"].values.squeeze()


#%% extract the projection
crs = nc.variables["projection_lambert"].proj4


#%% load the surface geopotential
print(nc.variables["ZS"])
sz = nc.variables["ZS"][:].squeeze()

lons_sz_all = nc.variables["longitude"]
lats_sz_all = nc.variables["latitude"]


#%% load the warning regions raster
warn = warn_nc.variables["warn_regions"][:]


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


#%% load the lat and lon from NORA3
x_n3 = nora3.variables["x"][:]
y_n3 = nora3.variables["y"][:]


#%% load the Lyngen gridcell coordinates as .csv
coords_path = "/home/kei070/Documents/IMPETUS/Coords_Files/"

coords_df = pd.read_csv(coords_path + "Lyngen_Nora3_Coords.csv")


#%% set up discrete colorbar
cmap = mpl.cm.viridis


#%% load the avalanche warning region info
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg")


#%% exclude the Svalbard regions
warn_info = warn_info[warn_info["reg_code"] > 3005]
warn_a_info = warn_info[warn_info["regionType"] == "A"]


#%% set up new discrete colorbar
cmap = mpl.cm.viridis


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


#%% extract the North
warn_info_north = warn_info_orth[(warn_info_orth.reg_code < 3014) & (warn_info_orth.reg_code > 3008)]


#%% set Lyngen centroid
lon = 19.80548
lat = 69.58474
# sys.exit()


#%% get region heights
reg_height = {}

for reg_code in regions.keys():

    reg_height[reg_code] = {}
    reg_height[reg_code]["mean"] = np.mean(sz[warn == reg_code])
    reg_height[reg_code]["std"] = np.std(sz[warn == reg_code])

# end for reg_code


#%% plot the northern avalanche regions --> Cryosphere paper Fig. 2
pl_path = "/home/kei070/Documents/IMPETUS/Plots/"
pl_path_pub = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Copernicus_LaTeX_Package/"

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

# pl_path = "/media/kei070/One_Touch/IMPETUS/NORA3/Plots/NORA3_Maps/"
# pl.savefig(pl_path + "Study_Region_with_Inset.png", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


#%% plot the northern avalanche regions --> Cryosphere paper 2 Fig. 2 -- Second attempt
pl_path = "/home/kei070/Documents/IMPETUS/Plots/"
pl_path_pub = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Copernicus_LaTeX_Package/"

reg_colors = {3009:"red", 3010:"green", 3011:"blue", 3012:"orange", 3013:"black"}
reg_colors = {3009:"black", 3010:"black", 3011:"black", 3012:"black", 3013:"black"}
reg_markers = {3009:"o", 3010:"s", 3011:"v", 3012:"^", 3013:"d"}

nora3_crs = ccrs.LambertConformal(central_longitude=-42, central_latitude=66.3, standard_parallels=(66.3, 66.3))

fig = pl.figure(figsize=(10, 3))
gs = gridspec.GridSpec(nrows=1, ncols=555)

axes = fig.add_subplot(gs[0, :300], projection=ccrs.Orthographic(central_latitude=69, central_longitude=20))
axes1 = fig.add_subplot(gs[0, 360:555])
axes2 = axes1.twinx()

axes.axis("off")

p_topo = axes.pcolormesh(lons_sz_pl, lats_sz_pl, sz_pl, transform=ccrs.PlateCarree(), cmap="Grays")
cb = fig.colorbar(p_topo, ax=axes, shrink=1, pad=0.01)
cb.set_label("Elevation in m")

axes.add_feature(cfeature.OCEAN, zorder=101)
axes.add_feature(cfeature.COASTLINE, zorder=101)
axes.add_feature(cfeature.BORDERS, edgecolor="black")

ext = [16, 22.7, 68.25, 70.45]
axes.set_extent(ext)


# add the regions shp
warn_info_north.plot(ax=axes, legend=False, zorder=102, linewidth=1.8,
                    legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                    edgecolor="black", cmap="Set1")

fweight = "heavy"
f_size = 13
axes.text(15.8, 69.6, "Sør-\nTroms", transform=ccrs.PlateCarree(), zorder=102, fontweight=fweight, fontsize=f_size)
axes.text(17., 70.3, "Tromsø", transform=ccrs.PlateCarree(), zorder=102, fontweight=fweight, fontsize=f_size)
axes.text(18.5, 68.7, "Indre\nTroms", transform=ccrs.PlateCarree(), zorder=102, fontweight=fweight, fontsize=f_size)
axes.text(20.55, 69.35, "Nord-\nTroms", transform=ccrs.PlateCarree(), zorder=102, fontweight=fweight, fontsize=f_size)
axes.text(19., 69.4, "Lyngen", transform=ccrs.PlateCarree(), zorder=102, fontweight=fweight, fontsize=f_size,
          rotation=45)

# add the NORA3 grid mesh
# cell.plot(ax=axes, facecolor="none", edgecolor='gray', linewidth=0.1, transform=nora3_crs, zorder=101)
gl = axes.gridlines(draw_labels=True, linewidth=0.5, zorder=101)

gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True

# Create an inset axes with a different projection in the lower right corner
# The rect parameter [left, bottom, width, height] specifies the location and size of the inset
ax_inset = pl.axes([0.314, 0.1325, 0.27, 0.27], facecolor='white',
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

axes.set_title("(a) Avalanche warning regions")


# right panel
for reg_code in regions.keys():
    axes1.scatter(reg_code-0.1, warn_info_north["areal_km2"][warn_info_north["reg_code"] == reg_code],
                  marker=reg_markers[reg_code], c="red", s=70)
    axes2.scatter(reg_code+0.1, reg_height[reg_code]["mean"], marker=reg_markers[reg_code],
                  c=reg_colors[reg_code], s=70)
    axes2.errorbar(x=reg_code+0.1, y=reg_height[reg_code]["mean"], yerr=reg_height[reg_code]["std"],
                   xerr=None, fmt='', ecolor=reg_colors[reg_code], elinewidth=None)
# end for reg_code

p00 = axes1.scatter([], [], linewidth=5, c="red", label="Size")
p01 = axes1.scatter([], [], linewidth=5, c="black", label="Elevation")
l00 = axes1.legend(handles=[p00], loc=(0.15, 0.87), labelcolor="red")
l01 = axes1.legend(handles=[p01], loc=(0.15, 0.76))
axes1.add_artist(l00)

axes1.set_xticks(list(regions.keys()))
axes1.set_xticklabels(["Nord-\nTroms", "Lyngen", "Tromsø", "Sør-\nTroms", "Indre\nTroms"])
axes2.set_xticks(list(regions.keys()))
axes2.set_xticklabels(["Nord-\nTroms", "Lyngen", "Tromsø", "Sør-\nTroms", "Indre\nTroms"])

axes1.set_ylabel("Size in km$^2$", color='red')
axes2.set_ylabel("Average region elevation")
# axes1.set_xlabel("Region code")
# axes2.set_xlabel("Region code")
axes1.set_title("(b) Warning region size and elevation")
# axes2.set_title("(c) Warning region elevation")

axes1.spines['left'].set_color('red')
axes2.spines['left'].set_color('red')
axes1.tick_params(axis='y', colors='red')

# pl_path = "/media/kei070/One_Touch/IMPETUS/NORA3/Plots/NORA3_Maps/"
pl_path = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Avalanche_Paper_2/00_Figures/"
pl.savefig(pl_path + "Study_Region_with_Inset_Size_Elev.png", bbox_inches="tight", dpi=100)

pl.show()
pl.close()