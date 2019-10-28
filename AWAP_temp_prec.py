#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-21"


import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata

lat = 691
lon = 886
T_ave = np.zeros((12,lat,lon))
R_ave = np.zeros((12,lat,lon))
T_AU_ave = np.zeros(30)
R_AU_ave = np.zeros(30)

# load landmask file
fmask     = "/srv/ccrc/data02/z3236814/data/AWAP/awap_landmask.nc"
data_mask = nc.Dataset(fmask,'r')
landmask  = pd.DataFrame(data_mask.variables['landmask'][:,:])#, columns=['latitude', 'longitude'])
#print(landmask)

# calculate
for i in np.arange(1979,2020,1):
    ftmax = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/tmax/tmax.%s.nc" %(i)
    ftmin = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/tmin/tmin.%s.nc" %(i)
    frain = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/rainfall_calib/pre.%s.nc" %(i)

    data_tmax = xr.open_dataset(ftmax)
    data_tmin = xr.open_dataset(ftmin)
    data_rain = xr.open_dataset(frain)
    print(data_tmax.time.values)
    Time      = pd.to_datetime(data_tmax.time.values,format="%Y%j",infer_datetime_format=False)
    tmax      = data_tmax.variable_name.values
    print(tmax)
    #tmax      = pd.DataFrame(data=data_tmax.variables['tmax'][:,:,:], columns=['time','lat', 'lon'])
    #tmin      = pd.DataFrame(data=data_tmin.variables['tmin'][:,:,:], columns=['time','lat', 'lon'])
    #rain      = pd.DataFrame(data=data_rain.variables['rain'][:,:,:], columns=['time','lat', 'lon'])



    tmax['dates'] = Time
    tmax = tmax.set_index('dates')
    tmax = tmax.resample("M").agg('mean')

    tmin['dates'] = Time
    tmin = tmin.set_index('dates')
    tmin = tmin.resample("M").agg('mean')

    rain['dates'] = Time
    rain = rain.set_index('dates')
    rain = rain.resample("M").agg('mean')

    tmax = tmax.where(landmask == 1, float(nan))
    tmin = tmin.where(landmask == 1, float(nan))
    rain = rain.where(landmask == 1, float(nan))

    print(Time)
    print(tmax)

    if i == 1979:
        T_ave[11,:,:] = (tmax[11,:,:] + tmin[11,:,:])/2.
        R_ave[11,:,:] = rain[11,:,:]
        tmax_former = tmax[11,:,:]
        tmin_former = tmin[11,:,:]
        rain_former = rain[11,:,:]
    elif i in np.arange(1980,2019):
        T_ave    = T_ave + (tmax+tmin)/2.
        R_ave    = R_ave + rain
        T_AU_ave[i] = ( tmax_former.mean() + tmin_former.mean() \
                       +tmax[0,:,:].mean() + tmin[0,:,:].mean() \
            	       +tmax[1,:,:].mean() + tmin[1,:,:].mean() )/6.
        R_AU_ave[i] = ( rain_former.mean() \
                        +rain[0,:,:].mean() \
                        +rain[1,:,:].mean() )/3.
        tmax_former = tmax[11,:,:]
        tmin_former = tmin[11,:,:]
        rain_former = rain[11,:,:]
    elif i == 2019:
        T_ave[0:1,:,:] = T_ave[0:1,:,:] + (tmax[0:1,:,:] + tmin[0:1,:,:])/2.
        R_ave[0:1,:,:] = R_ave[0:1,:,:] + rain[0:1,:,:]
        T_AU_ave[i] =( tmax_former.mean() + tmin_former.mean() \
                      +tmax[0,:,:].mean() + tmin[0,:,:].mean() \
                      +tmax[1,:,:].mean() + tmin[1,:,:].mean() )/6.
        R_AU_ave[i] =( rain_former.mean() \
                      +rain[0,:,:].mean() \
                      +rain[1,:,:].mean() )/3.

T_ave = T_ave/30.
R_ave = R_ave/30.

T_mean = np.zeros(30)
R_mean = np.zeros(30)
for i in np.arange(0,30):
    T_mean[i] = T_AU_ave.mean()
    R_mean[i] = R_AU_ave.mean()

# ____________________ Plot obs _______________________
fig = plt.figure(figsize=[15,10])
fig.subplots_adjust(hspace=0.1)
fig.subplots_adjust(wspace=0.05)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

almost_black = '#262626'
# change the tick colors also to the almost black
plt.rcParams['ytick.color'] = almost_black
plt.rcParams['xtick.color'] = almost_black

# change the text colors also to the almost black
plt.rcParams['text.color'] = almost_black

# Change the default axis colors from black to a slightly lighter black,
# and a little thinner (0.5 instead of 1)
plt.rcParams['axes.edgecolor'] = almost_black
plt.rcParams['axes.labelcolor'] = almost_black

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

width = 1.
x = np.arange(1980,2020)
ax1.plot(x, T_AU_ave,   c="red", lw=1.0, ls="-") #, label="")
ax1.plot(x, T_mean , c="black", lw=1.0, ls=".")#, label="swc")

ax2.plot(x, R_AU_ave,   c="red", lw=1.0, ls="-") #, label="")
ax2.plot(x, R_mean , c="black", lw=1.0, ls=".")#, label="swc")

cleaner_dates = ["1980","1985","1990","1995","2000","2005","2010","2015","2019"]
xtickslocs    = [1980,1985,1990,1995,2000,2005,2010,2015,2019]

# plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
ax1.set_ylabel("Temperature (oC)")
ax1.axis('tight')
ax1.set_ylim(25.,35.)
# ax1.set_xlim(367,2739)
# ax1.legend()



# plt.setp(ax1.get_xticklabels(), visible=False)
ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
ax2.set_ylabel("Rainfall (mm/day)")
ax2.axis('tight')
ax2.set_ylim(0.,5.)
# ax2.set_xlim(367,2739)
# ax2.legend()


fig.savefig("ge_jun.png", bbox_inches='tight', pad_inches=0.1)
