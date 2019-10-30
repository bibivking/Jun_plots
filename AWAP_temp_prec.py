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

# load landmask file
fmask     = "/srv/ccrc/data02/z3236814/data/AWAP/awap_landmask.nc"
data_mask = nc.Dataset(fmask,'r')
Lat       = data_mask.variables['latitude'][:]
Lon       = data_mask.variables['longitude'][:]
landmask  = xr.DataArray(data_mask.variables['landmask'][:,:], dims=('lat','lon'), coords={'lat': Lat, 'lon': Lon})
data_mask.close()

# calculate
lat = 691
lon = 886
tot_step = 39 * 12 + 3

T   = np.zeros((tot_step,lat,lon))
R   = np.zeros((tot_step,lat,lon))

for i in np.arange(1979,2020,1):
    ftmax = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/tmax/tmax.%s.nc" %(i)
    ftmin = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/tmin/tmin.%s.nc" %(i)
    frain = "/srv/ccrc/data02/z3236814/data/AWAP/DAILY/netcdf/rainfall_calib/pre.%s.nc" %(i)

    data_tmax = nc.Dataset(ftmax, 'r')
    data_tmin = nc.Dataset(ftmin, 'r')
    data_rain = nc.Dataset(frain, 'r')

    if i in np.arange(1979,2019):
        # set coords
        Time      = pd.to_datetime(data_tmax.variables['time'][:],format="%Y%j",infer_datetime_format=False)
        # read 3D data
        tmax      = xr.DataArray(data_tmax.variables['tmax'][:,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})
        tmin      = xr.DataArray(data_tmin.variables['tmin'][:,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})
        rain      = xr.DataArray(data_rain.variables['pre'][:,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})
        end_month = 12
    else:
        # set coords
        Time      = pd.to_datetime(data_tmax.variables['time'][0:90],format="%Y%j",infer_datetime_format=False)
        # read 3D data
        tmax      = xr.DataArray(data_tmax.variables['tmax'][0:90,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})
        tmin      = xr.DataArray(data_tmin.variables['tmin'][0:90,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})
        rain      = xr.DataArray(data_rain.variables['pre'][0:90,:,:], dims=('time','lat','lon'), coords={'time':Time, 'lat': Lat, 'lon': Lon})
        end_month = 3

    data_tmax.close()
    data_tmin.close()
    data_rain.close()

    tmax = tmax.resample(time="M").mean()
    tmin = tmin.resample(time="M").mean()
    rain = rain.resample(time="M").mean()

    for j in np.arange(0,end_month):
        tmax[j,:,:] = tmax[j,:,:].where(landmask == 1)
        tmin[j,:,:] = tmin[j,:,:].where(landmask == 1)
        rain[j,:,:] = rain[j,:,:].where(landmask == 1)

    if i == 1979:
        T[0,:,:] = (tmax[11,:,:] + tmin[11,:,:])/2.
        R[0,:,:] = rain[11,:,:]
    elif i in np.arange(1980,2019):
        stp_s = (i-1980)*12+1
        stp_e = (i-1980)*12+13
        T[stp_s:stp_e,:,:] = (tmax[:,:,:] + tmin[:,:,:])/2.
        R[stp_s:stp_e,:,:] = rain[:,:,:]
    else:
        T[-2:,:,:] = (tmax[0:2,:,:] + tmin[0:2,:,:])/2.
        R[-2:,:,:] = rain[0:2,:,:]

time_month = np.arange(np.datetime64('1979-12-31','M'), np.datetime64('2019-03-31','M'))
Temp = xr.DataArray(T, dims=('month','lat','lon'), coords={'month':time_month, 'lat': Lat, 'lon': Lon})
Rain = xr.DataArray(R, dims=('month','lat','lon'), coords={'month':time_month, 'lat': Lat, 'lon': Lon})

T_AU_ave = np.zeros(40)
R_AU_ave = np.zeros(40)

stp_s = 0
stp_e = 3
for i in np.arange(1980,2020):
    T_AU_ave[i-1980] = Temp[stp_s:stp_e,:,:].mean(dim=['month','lat','lon'])
    R_AU_ave[i-1980] = Rain[stp_s:stp_e,:,:].mean(dim=['month','lat','lon'])
    stp_s += 12
    stp_e += 12

T_30_ave = [T_AU_ave[10:].mean()]*40
R_30_ave = [R_AU_ave[10:].mean()]*40

stp_s = 420
stp_e = 423
T_Sum = Temp[stp_s:stp_e,:,:].mean(dim=['month'])
R_Sum = Rain[stp_s:stp_e,:,:].mean(dim=['month'])
T_Dec = Temp[stp_s,:,:]
R_Dec = Rain[stp_s,:,:]
T_Jan = Temp[stp_s+1,:,:]
R_Jan = Rain[stp_s+1,:,:]
T_Feb = Temp[stp_s+2,:,:]
R_Feb = Rain[stp_s+2,:,:]

for i in np.arange(2016,2019):
    stp_s += 12
    stp_e += 12
    T_Sum = T_Sum + Temp[stp_s:stp_e,:,:].mean(dim=['month'])
    R_Sum = R_Sum + Rain[stp_s:stp_e,:,:].mean(dim=['month'])
    T_Dec = T_Dec + Temp[stp_s,:,:]
    R_Dec = R_Dec + Rain[stp_s,:,:]
    T_Jan = T_Jan + Temp[stp_s+1,:,:]
    R_Jan = R_Jan + Rain[stp_s+1,:,:]
    T_Feb = T_Feb + Temp[stp_s+2,:,:]
    R_Feb = R_Feb + Rain[stp_s+2,:,:]

T_Sum = T_Sum/4.
R_Sum = R_Sum/4.
T_Dec = T_Dec/4.
R_Dec = R_Dec/4.
T_Jan = T_Jan/4.
R_Jan = R_Jan/4.
T_Feb = T_Feb/4.
R_Feb = R_Feb/4.

T_Sum_19 = Temp[-3:,:,:].mean(dim=['month']) - T_Sum
R_Sum_19 = Rain[-3:,:,:].mean(dim=['month']) - R_Sum
T_Dec_19 = Temp[-3,:,:] - T_Dec
R_Dec_19 = Rain[-3,:,:] - R_Dec
T_Jan_19 = Temp[-2,:,:] - T_Jan
R_Jan_19 = Rain[-2,:,:] - R_Jan
T_Feb_19 = Temp[-1,:,:] - T_Feb
R_Feb_19 = Rain[-1,:,:] - R_Feb

#T_summer = Temp.resample(month="4Q-FEB").mean()


# ____________________ Plot obs _______________________
fig1 = plt.figure(figsize=[15,10],constrained_layout=True)
fig1.subplots_adjust(hspace=0.1)
fig1.subplots_adjust(wspace=0.05)
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

ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

width = 1.
x = np.arange(1980,2020)
ax1.plot(x, T_AU_ave,  c="red", lw=2.0, ls='-', marker='o') #, label="")
ax1.plot(x, T_30_ave,  c="black", lw=2.0, ls='-')#, label="swc")

ax2.plot(x, R_AU_ave,  c="blue", lw=2.0, ls='-', marker='o') #, label="")
ax2.plot(x, R_30_ave,  c="black", lw=2.0, ls='-')#, label="swc")

cleaner_dates = ["1980","1990","2000","2010","2020"]
xtickslocs    = [1980,1990,2000,2010,2020]

# plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
ax1.set_ylabel("Temperature (oC)")
ax1.axis('tight')
ax1.set_ylim(26.,30.)
ax1.set_xlim(1980,2020)
# ax1.legend()

# plt.setp(ax1.get_xticklabels(), visible=False)
ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
ax2.set_ylabel("Rainfall (mm/day)")
ax2.axis('tight')
ax2.set_ylim(1.,5.)
ax2.set_xlim(1980,2020)
# ax2.legend()
fig1.savefig("figure1.png", bbox_inches='tight', pad_inches=0.1)

# Figure 2
fig2 = plt.figure(constrained_layout=True)
fig2.subplots_adjust(hspace=0.15)
fig2.subplots_adjust(wspace=0.1)
axs1 = fig2.add_subplot(221)
axs2 = fig2.add_subplot(222)
axs3 = fig2.add_subplot(223)
axs4 = fig2.add_subplot(224)

cmap = plt.cm.jet # nipy_spectral #
axs1.set_title("(a) Dec")
axs2.set_title("(b) Jan")
axs3.set_title("(c) Feb")
axs4.set_title("(d) Summer")

img1 = axs1.imshow(T_Dec_19[::-1,:], cmap=cmap, vmin=-3.5, vmax=3.5, origin="upper", interpolation='nearest')
img2 = axs2.imshow(T_Jan_19[::-1,:], cmap=cmap, vmin=-3.5, vmax=3.5, origin="upper", interpolation='nearest')
img3 = axs3.imshow(T_Feb_19[::-1,:], cmap=cmap, vmin=-3.5, vmax=3.5, origin="upper", interpolation='nearest')
img4 = axs4.imshow(T_Sum_19[::-1,:], cmap=cmap, vmin=-3.5, vmax=3.5, origin="upper", interpolation='nearest')
fig2.colorbar(img1, ax=[axs3, axs4],location='bottom')
fig2.savefig("figure2.png", bbox_inches='tight', pad_inches=0.1)

# Figure 3
fig3 = plt.figure(constrained_layout=True)
fig3.subplots_adjust(hspace=0.15)
fig3.subplots_adjust(wspace=0.1)
axs1 = fig3.add_subplot(221)
axs2 = fig3.add_subplot(222)
axs3 = fig3.add_subplot(223)
axs4 = fig3.add_subplot(224)

cmap = plt.cm.jet #nipy_spectral
axs1.set_title("(a) Dec")
axs2.set_title("(b) Jan")
axs3.set_title("(c) Feb")
axs4.set_title("(d) Summer")

img1 = axs1.imshow(R_Dec_19[::-1,:], cmap=cmap, vmin=-9., vmax=9., origin="upper", interpolation='nearest')
img2 = axs2.imshow(R_Jan_19[::-1,:], cmap=cmap, vmin=-9., vmax=9., origin="upper", interpolation='nearest')
img3 = axs3.imshow(R_Feb_19[::-1,:], cmap=cmap, vmin=-9., vmax=9., origin="upper", interpolation='nearest')
img4 = axs4.imshow(R_Sum_19[::-1,:], cmap=cmap, vmin=-9., vmax=9., origin="upper", interpolation='nearest')
fig3.colorbar(img1, ax=[axs3, axs4],location='bottom')
fig3.savefig("figure3.png", bbox_inches='tight', pad_inches=0.1)

f = nc.Dataset("AWAP_2015-2019_Summer_Temp_Rain.nc","w",format='NETCDF4')
f.description = 'AWAP summer season temperature and rainfall data during 2015-2019'
f.creation_date = "%s" % (dt.datetime.now())

# set dimensions
f.createDimension('time', None)
f.createDimension('year', 40)
f.createDimension('lat', lat)
f.createDimension('lon', lon)
f.Conventions = "CF-1.0"

# create variables
time = f.createVariable('time', 'i4', ('time',))
time.units = "years since 2015"
time.long_name = "time"
time.calendar = "standard"
time[:] = [2015, 2016, 2017, 2018, 2019]

year = f.createVariable('year', 'i4', ('year',))
year.units = "years since 1980"
year.long_name = "year"
year[:] = np.arange(1980,2020,1)

latitude = f.createVariable('lat', 'f4', ('lat',))
latitude.units = "degrees_north"
latitude.missing_value = -9999.
latitude.long_name = "Latitude"
latitude[:] = Lat

longitude = f.createVariable('lon', 'f4', ('lon',))
longitude.units = "degrees_east"
longitude.missing_value = -9999.
longitude.long_name = "Longitude"
longitude[:] = Lon

temp_ave = f.createVariable('temp_ave', 'f4', ('year',))
temp_ave.units = "deg C"
temp_ave.long_name = "summer temperature"
temp_ave[:] = T_AU_ave

rain_ave = f.createVariable('rain_ave', 'f4', ('year',))
rain_ave.units = "mm/day"
rain_ave.long_name = "summer rainfall"
rain_ave[:] = R_AU_ave

temp_sum= f.createVariable('temp_sum', 'f4', ('time','lat','lon'))
temp_sum.units = "deg C"
temp_sum.long_name = "summer temperature"
temp_sum.missing_value = float('nan')

temp_dec = f.createVariable('temp_dec', 'f4', ('time','lat','lon'))
temp_dec.units = "deg C"
temp_dec.long_name = "December temperature"
temp_dec.missing_value = float('nan')

temp_jan = f.createVariable('temp_jan', 'f4', ('time','lat','lon'))
temp_jan.units = "deg C"
temp_jan.long_name = "January temperature"
temp_jan.missing_value = float('nan')

temp_feb = f.createVariable('temp_feb', 'f4', ('time','lat','lon'))
temp_feb.units = "deg C"
temp_feb.long_name = "February temperature"
temp_feb.missing_value = float('nan')

rain_sum= f.createVariable('rain_sum', 'f4', ('time','lat','lon'))
rain_sum.units = "mm/day"
rain_sum.long_name = "summer rainfall"
rain_sum.missing_value = float('nan')

rain_dec = f.createVariable('rain_dec', 'f4', ('time','lat','lon'))
rain_dec.units = "mm/day"
rain_dec.long_name = "December rainfall"
rain_dec.missing_value = float('nan')

rain_jan = f.createVariable('rain_jan', 'f4', ('time','lat','lon'))
rain_jan.units = "mm/day"
rain_jan.long_name = "January rainfall"
rain_jan.missing_value = float('nan')

rain_feb = f.createVariable('rain_feb', 'f4', ('time','lat','lon'))
rain_feb.units = "mm/day"
rain_feb.long_name = "February rainfall"
rain_feb.missing_value = float('nan')


stp_s = 420
stp_e = 423

for i in np.arange(0,5,1):
    temp_sum[i,:,:] = Temp[stp_s:stp_e,:,:].mean(dim=['month'])
    rain_sum[i,:,:] = Rain[stp_s:stp_e,:,:].mean(dim=['month'])
    temp_dec[i,:,:] = Temp[stp_s,:,:]
    rain_dec[i,:,:] = Rain[stp_s,:,:]
    temp_jan[i,:,:] = Temp[stp_s+1,:,:]
    rain_jan[i,:,:] = Rain[stp_s+1,:,:]
    temp_feb[i,:,:] = Temp[stp_s+2,:,:]
    rain_feb[i,:,:] = Rain[stp_s+2,:,:]
    stp_s += 12
    stp_e += 12

f.close()
