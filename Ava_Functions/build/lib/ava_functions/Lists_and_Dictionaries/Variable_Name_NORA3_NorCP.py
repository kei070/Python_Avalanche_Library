#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lists/dictionaries for the NORA3 and NorCP variable names
"""

# general keys
gen_var_name = ["t2m", "prec", "snow", "rain", "rh", "w10m", "wdir", "nlw", "nsw", "ts"]

# NORA3 names
varn_nora3_l = ["air_temperature_2m", "precipitation_amount_hourly", "snow", "rain", "relative_humidity_2m",
                "wind_speed", "wind_direction", "surface_net_longwave_radiation", "surface_net_shortwave_radiation",
                "ts"]

# NorCP names
varn_norcp_l = ["tas", "pr", "snow", "rain", "hurs", "wspeed", "wdir", "rlns", "rsns", "rlds", "rsds"]

# NORA3 dictionary
varn_nora3 = {k:v for k, v in zip(gen_var_name, varn_nora3_l)}

# NorCP dictionary
varn_norcp = {k:v for k, v in zip(gen_var_name, varn_norcp_l)}

# NorCP file name dictionary
fn_norcp = {"t2m":"tas", "prec":"pr", "snow":"Rain_Snow", "rain":"Rain_Snow", "rh":"hurs", "w10m":"Wind_Speed_Dir",
            "wdir":"Wind_Speed_Dir", "rlds":"rlds", "rsds":"rsds", "nlw":"rlns", "nsw":"rsns"}

# predictor names
pred_names = {"t2m":{"mean":"t", "min":"tmin", "max":"tmax"}, "rh":{"mean":"rh"},
              "w10m":{"mean":"w", "min":"wmin", "max":"wmax"},
              "wdir":{"mean":"wdir"}, "nlw":{"mean":"nlw"}, "nsw":{"mean":"nsw"}}

# set the 3h variables and the 1h variables (NorCP)
var1h = ["tas", "pr", "snow", "rain", "wspeed", "wdir", "rlds", "rsds"]
var3h = ["rlns", "rsns", "hurs"]