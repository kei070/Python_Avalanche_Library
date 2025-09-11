#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to calculate relative humidity from air temperature, pressure, and specific humidity.
"""

# imports
import numpy as np

# absolute vapour pressure
def abs_vap_pres(spec_hum, air_pres):

    """
    See, e.g., Meteorologisch-klimatologisches Grundwissen, p. 71 and some rearranging equation 3.8.

    Returns the absolute vapour pressure in the same units in which the air pressure (air_pres) is given.
    """

    # return (spec_hum * air_pres) / (0.622 + 0.378 * spec_hum)

    return spec_hum * air_pres / 0.622

# end def

# saturation vapour pressure
def sat_vap_pres(air_tem, method="Magnus", kelvin=True):

    """
    See, e.g., Meteorologisch-klimatologisches Grundwissen, p. 70.

    There are different approximations; two are given here.

    The air_tem parameter can be either a numpy array (numpy.ndarray) or a float (or integer).

    Note that the saturation vapour pressure is calculated in hPa.
    """

    if kelvin:
        air_tem = air_tem - 273.15
    # end if

    if type(air_tem) in [type(np.array([]))]:

        # air_tem_return = np.zeros(np.shape(air_tem))

        if method == "Magnus":
            return np.where(air_tem > 0, 6.1 * 10**((7.5 * air_tem) / (air_tem + 237.2)),
                            6.1 * 10**((9.5 * air_tem) / (air_tem + 265.5)))

        elif method == "Teten":
            return 6.112 * np.exp((17.67 * air_tem) / (air_tem + 243.5))
        # end if elif
    else:
        if method == "Magnus":
            if air_tem > 0:
                return 6.1 * 10**((7.5 * air_tem) / (air_tem + 237.2))
            else:
                return 6.1 * 10**((9.5 * air_tem) / (air_tem + 265.5))
            # end if

        elif method == "Teten":
            return 6.112 * np.exp((17.67 * air_tem) / (air_tem + 243.5))
        # end if elif
    # end if else

# end def


# relative humiditiy
def rel_hum(spec_hum, air_pres, air_tem, method="Magnus", e_s_fac=1, e_fac=1):

    """
    Parameters:

        e_s_fac  Factor for multiplying the e_s to make sure that e_s and e have the same units.
        e_s      Factor for multiplying the e to make sure that e_s and e have the same units.

    """

    e_s = sat_vap_pres(air_tem=air_tem, method=method) * e_s_fac

    e = abs_vap_pres(spec_hum=spec_hum, air_pres=air_pres) * e_fac

    return e / e_s

# end def


