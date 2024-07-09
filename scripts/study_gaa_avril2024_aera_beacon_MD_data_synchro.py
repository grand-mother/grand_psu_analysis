import sys
sys.path.append('./')
import uproot
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import grand_psu_lib.utils.utils as utils
import grand_psu_lib.utils.filtering as filt
import grand_psu_lib.utils.utils_gp13 as u13
import argparse
import glob
import datetime
import pymap3d

from scipy.signal import butter, sosfilt, sosfiltfilt

from scipy import signal

from grand import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
    topography,
)
from grand import ECEF, Geodetic, GRANDCS, LTP



site = 'gaa'

plot_path = '/Users/ab212678/Documents/GRAND/data/plots/study_aera_gaa_avril24/'
os.makedirs(plot_path, exist_ok=True)



f1 = 58.887
f2 = 61.523
f3 = 68.555
f4 = 71.191

file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/03/gaa_20240329_053214_RUN003002_MD_phys.bin.root'

file_dict_gaa= {file_gaa: "tadc"}
file_dict_gaa_trawv= {file_gaa: "trawvoltage"}

tadc_gaa = uproot.concatenate(file_dict_gaa)
trawv_gaa = uproot.concatenate(file_dict_gaa_trawv)


list_gaa = utils.get_dulist(tadc_gaa)


du_id = 83
traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', du_id)
traces_gaa = traces_gaa.to_numpy()

psd_gaa = filt.return_psd(traces_gaa, 500)

sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]

n_samples_gaa = traces_gaa.shape[-1]

fft_freq__gaa = np.fft.rfftfreq(n_samples_gaa) * sample_freq  # [MHz]

arg_freq1 = np.argmin(abs(fft_freq__gaa - f1))
arg_freq2 = np.argmin(abs(fft_freq__gaa - f2))
arg_freq3 = np.argmin(abs(fft_freq__gaa - f3))
arg_freq4 = np.argmin(abs(fft_freq__gaa - f4))


plt.figure(56)
plt.plot(date_array_gaa, psd_gaa[:, :, 0, arg_freq1])
plt.plot(date_array_gaa, psd_gaa[:, :, 0, arg_freq2])
plt.plot(date_array_gaa, psd_gaa[:, :, 0, arg_freq3])
plt.plot(date_array_gaa, psd_gaa[:, :, 0, arg_freq4])


plt.figure(1, figsize=(15, 8))
plt.clf()
plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')

plt.figure(2, figsize=(15, 8))
plt.clf()
plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')

plt.figure(3, figsize=(15, 8))
plt.clf()
plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')







### Beacon reconstruction

list_freqs = [f1, f2, f3, f4]
arg_freqs = [np.argmin(abs(fft_freq__gaa - f)) for f in list_freqs]








iev = 21
tadc_ev = tadc_gaa[iev]
trawv_ev = trawv_gaa[iev]
trace_ = tadc_ev.trace_ch.to_numpy()

du_list = tadc_ev.du_id

arg_freqs2 = [965, 1008, 1123, 1166, 964, 1007, 1122, 1165, 966, 1009, 1124, 1167]

beacon_lon = -69.599922933 + 360
beacon_lat = -35.114365909
beacon_alt = 1719.803


du_lat = trawv_ev.gps_lat.to_numpy()
du_lon = trawv_ev.gps_long.to_numpy()+360
du_alt = trawv_ev.gps_alt.to_numpy()

# Set the local frame origin
beacon = Geodetic(latitude=beacon_lat, longitude=beacon_lon, height=beacon_alt)
du_s = Geodetic(latitude=du_lat, longitude=du_lon, height=du_alt)

xyz_pos = GRANDCS(du_s, location=beacon)
xyz_beacon = GRANDCS(beacon, location=beacon)

plt.figure(345)
plt.clf()
plt.plot(xyz_pos.x, xyz_pos.y, 'k.')
[plt.text(xyz_pos.x[i], xyz_pos.y[i], "{}".format(du_list[i]) ) for i in range(len(du_list)) ]
plt.plot(0, 0, 'ro')




dist = np.sqrt(xyz_pos.x**2 + xyz_pos.y**2 + xyz_pos.z**2 )
c  = 3e8 
delta_t = dist / c
delta_t_ns = np.array(delta_t) * 1e9

t = 1/(500e6) * np.arange(8192) * 1e9




#t1 = 1/(500e6) * np.arange(1024) * 1e9
ev_second = tadc_ev.time_seconds
ev_du_sec = tadc_ev.du_seconds.to_numpy()
ev_du_nanosec = tadc_ev.du_nanoseconds.to_numpy()
ev_du_dnanosec = ev_du_nanosec - tadc_ev.time_nanoseconds  ## offset in nsec wrt to the fisrt du. so the first du has dnano = 0
ev_tri_pos = tadc_ev.trigger_position.to_numpy()
timex = []
[timex.append(  t - t[ev_tri_pos[i]]+ ev_du_dnanosec[i] ) for i in range(len(tadc_ev.du_id))]



tr_filtered = []
fig, ax = plt.subplots(len(du_list), 1, sharex=True, sharey=False ,figsize=(10, 10))
for i in range(len(du_list)):
    id_channel = 0
    if du_list[i] == 59:
        id_channel = 1
    print(id_channel)
    tr2, _ = filt.harmonic_filter(trace_[i, id_channel], arg_freqs2)
    tr_filtered.append(tr2)
    ax[i].plot(timex[i],  tr2, label='du{}'.format(du_list[i]))
    #ax[i].plot(t,  trace_[i, id_channel], label='du{}'.format(du_list[i]))
    ax[i].legend(loc=1)
#[a.set_xlim(2000, 5000) for a in ax]
ax[-1].set_xlabel('time [ns]')








plt.figure()
n_du = len(du_list)
lags = signal.correlation_lags(8192, 8192)
lag_list = []
for i in arange(0, n_du):
    correl = signal.correlate(tr_filtered[i], tr_filtered[0])
    lag = lags[np.argmax(correl)]
    lag_list.append(lag)
    plt.plot(correl, label='corr({}, {})'.format(du_list[0], du_list[i]))

plt.legend()


fig, ax = plt.subplots(len(du_list), 1, figsize=(10, 10))
for i in range(len(du_list)):
    id_channel = 0
    if du_list[i] == 59:
        id_channel = 1
    tr2 = filter2(trace_[i, id_channel], f1, f2, f3, f4, df, 1)
    tr_filterd.append(tr2)
    ax[i].plot(t + lag_list[i]*2 , tr2, label='du{}'.format(du_list[i]))
    ax[i].legend(loc=1)

[a.set_xlim(2000, 5000) for a in ax]

