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
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata


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


lat = 691
lon = 886
T_ave = np.zeros((12,lat,lon))
R_ave = np.zeros((12,lat,lon))
T_AU_ave = np.zeros(30)
R_AU_ave = np.zeros(30)

# load landmask file
fmask     = "/srv/ccrc/data02/z3236814/data/AWAP/awap_landmask.nc"
data_mask = nc.Dataset(fmask,'r')
Lat       = data_mask.variables['latitude'][:]
Lon       = data_mask.variables['longitude'][:]
landmask  = xr.DataArray(data_mask.variables['landmask'][:,:], dims=('lat','lon'), coords={'lat': Lat, 'lon': Lon})
print(landmask)
data_mask.close()

# calculate
for i in np.arange(1979,2020,1):
    ftmax = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/tmax/tmax.%s.nc" %(i)
    ftmin = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/tmin/tmin.%s.nc" %(i)
    frain = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/rainfall_calib/pre.%s.nc" %(i)

    data_tmax = nc.Dataset(ftmax, 'r')
    data_tmin = nc.Dataset(ftmin, 'r')
    data_rain = nc.Dataset(frain, 'r')

    # set coords
    Time      = pd.to_datetime(data_tmax.variables['time'][:],format="%Y%j",infer_datetime_format=False)

    # read 3D data
    tmax      = xr.DataArray(data_tmax.variables['tmax'][:,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})
    tmin      = xr.DataArray(data_tmin.variables['tmin'][:,:,:], dims=('time','lat','lon'),coords={'time':Time, 'lat': Lat, 'lon': Lon})
    rain      = xr.DataArray(data_rain.variables['pre'][:,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})

    #xarray.DataArray.set_index

    data_tmax.close()
    data_tmin.close()
    data_rain.close()

    tmax = tmax.resample(time="M").mean()
    tmin = tmin.resample(time="M").mean()
    rain = rain.resample(time="M").mean()

    for j in np.arange(0,12):
        tmax[j,:,:] = tmax[j,:,:].where(landmask == 1)
        tmin[j,:,:] = tmin[j,:,:].where(landmask == 1)
        rain[j,:,:] = rain[j,:,:].where(landmask == 1)

    #img = ax1.imshow(tmax[0,:,:], interpolation='nearest')
    #plt.show()

    if i == 1979:
        print("i=1979")
        T_ave[11,:,:] = (tmax[11,:,:] + tmin[11,:,:])/2.
        R_ave[11,:,:] = rain[11,:,:]
        temp_former = (tmax[11,:,:] + tmin[11,:,:])/2.
        rain_former = rain[11,:,:]
    elif i in np.arange(1980,2019):
        print("1979 < i < 2019 ")
        T_ave    = T_ave + (tmax+tmin)/2.
        R_ave    = R_ave + rain
        T_AU_ave[i-1979] = ( temp_former.nanmean() \
                       +(tmax[0,:,:].nanmean() + tmin[0,:,:].nanmean())/2. \
            	       +(tmax[1,:,:].nanmean() + tmin[1,:,:].nanmean())/2. )/3.
        R_AU_ave[i-1979] = ( rain_former.nanmean() \
                        +rain[0,:,:].nanmean() \
                        +rain[1,:,:].nanmean() )/3.
        temp_former = (tmax[11,:,:] + tmin[11,:,:])/2.
        rain_former = rain[11,:,:]
        '''
        img1 = ax1.imshow(temp_former, interpolation='nearest')
        img2 = ax2.imshow(rain_former, interpolation='nearest')


        cbar = fig.colorbar(img1, orientation="vertical", pad=0.1, shrink=.6) #"horizontal"
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

        cbar = fig.colorbar(img2, orientation="vertical", pad=0.1, shrink=.6) #"horizontal"
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        plt.show()
        '''
    elif i == 2019:
        print("i = 2019")
        T_ave[0:1,:,:] = T_ave[0:1,:,:] + (tmax[0:1,:,:] + tmin[0:1,:,:])/2.
        R_ave[0:1,:,:] = R_ave[0:1,:,:] + rain[0:1,:,:]
        T_AU_ave[i-1979] =( temp_former.nanmean() \
                       +(tmax[0,:,:].nanmean() + tmin[0,:,:].nanmean())/2. \
            	       +(tmax[1,:,:].nanmean() + tmin[1,:,:].nanmean())/2. )/3.
        R_AU_ave[i-1979] =( rain_former.nanmean() \
                      +rain[0,:,:].nanmean() \
                      +rain[1,:,:].nanmean() )/3.
T_ave = T_ave/30.
R_ave = R_ave/30.

#plt.plot(T_AU_ave)
#plt.show()
T_mean = np.zeros(30)
R_mean = np.zeros(30)
for i in np.arange(0,30):
    T_mean[i] = T_AU_ave.mean()
    R_mean[i] = R_AU_ave.mean()

'''

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
'''
