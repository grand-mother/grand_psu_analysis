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

##work on a CD file with 1024 traces.

f1 = 58.887
f2 = 61.523
f3 = 68.555
f4 = 71.191

#file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/gaa_20240405_074134_RUN003002_CD_phys.root'


filegaa = "gaa_20240410_121220_RUN003002_CD_phys.root"
file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/{}'.format(filegaa)

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
plt.yscale('log')






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


for du_id in list_gaa:

    traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', du_id)
    traces_gaa = traces_gaa.to_numpy()
    print('du {}, {} hrs'.format(du_id, (date_array_gaa[-1] - date_array_gaa[0]).seconds/60/60))
    psd_gaa = filt.return_psd(traces_gaa, 500)
    plt.figure(1)
    plt.plot(fft_freq__gaa, psd_gaa.mean(axis=0)[0, 0], label='DU {}'.format(du_id))
    plt.figure(2)
    plt.plot(fft_freq__gaa, psd_gaa.mean(axis=0)[0, 1], label='DU {}'.format(du_id))
    plt.figure(3)
    plt.plot(fft_freq__gaa, psd_gaa.mean(axis=0)[0, 2], label='DU {}'.format(du_id))






plt.figure(1)
plt.yscale('log')
plt.ylim(1e-2, 1e3)
plt.title('Mean PSD x-axis')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.savefig(os.path.join(plot_path, "psd_mean_chX.png"))
plt.xlim(55, 75)
plt.savefig(os.path.join(plot_path, "psd_mean_chX_zoom.png"))


plt.figure(2)
plt.yscale('log')
plt.ylim(1e-2, 1e3)
plt.title('Mean PSD y-axis')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.savefig(os.path.join(plot_path, "psd_mean_chY.png"))
plt.xlim(55, 75)
plt.savefig(os.path.join(plot_path, "psd_mean_chY_zoom.png"))




plt.figure(3)
plt.yscale('log')
plt.ylim(1e-2, 1e3)
plt.title('Mean PSD z-axis')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.savefig(os.path.join(plot_path, "psd_mean_chZ.png"))
plt.xlim(55, 75)
plt.savefig(os.path.join(plot_path, "psd_mean_chZ_zoom.png"))







### Beacon reconstruction

idu = 151
traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', idu)
traces_gaa = traces_gaa.to_numpy()

list_freqs = [f1, f2, f3, f4]

arg_freqs = [np.argmin(abs(fft_freq__gaa - f)) for f in list_freqs]


def harmonic_filter(trace_in,  arg_freqs_list):

    rfft = np.fft.rfft(trace_in)
    rfft_copy = rfft.copy() * 0
    rfft_copy[arg_freqs_list] = rfft[arg_freqs_list]

    trace_out = np.fft.irfft(rfft_copy)
    rfft_out = np.fft.rfft(trace_out)
    return trace_out, rfft_out



def harmonic_filter_old(trace_in, n_df1, fft_freq, deltaf):
    ind1 = np.where(
        (fft_freq >= f1 - n_df1 * deltaf *2 ) *( fft_freq <= f1+n_df1 * 2* deltaf) +
        (fft_freq >= f2 - n_df1 * deltaf ) *( fft_freq <= f2+n_df1 * deltaf) + 
        (fft_freq >= f3 - n_df1 * deltaf *2 ) *( fft_freq <= f3+n_df1 * 2*deltaf) +
        (fft_freq >= f4 - n_df1 * deltaf ) *( fft_freq <= f4+n_df1 * deltaf)
    )

    rfft = np.fft.rfft(trace_in)
    rfft_copy = rfft.copy() * 0
    rfft_copy[ind1] = rfft[ind1]

    trace_out = np.fft.irfft(rfft_copy)
    rfft_out = np.fft.rfft(trace_out)
    return trace_out, rfft_out

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filt(data, lowcut, highcut, fs, order):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    y = sosfiltfilt(sos, data)
    return y, sos




def filter1(tr, f1, f2, f3, f4, order):
    out, sos = butter_bandpass_filt(tr, f1-0.5, f4+0.5, 500, order)
    return out


def filter2(tr, f1, f2, f3, f4, df, order):

    tr1_, _ = butter_bandpass_filt(tr, f1-df, f1+df, 500, order)
    tr2_, _ = butter_bandpass_filt(tr, f2-df, f2+df, 500, order)
    tr3_, _ = butter_bandpass_filt(tr, f3-df, f3+df, 500, order)
    tr4_, _ = butter_bandpass_filt(tr, f4-df, f4+df, 500, order)
    out = tr1_ + tr2_ + tr3_ + tr4_
    return out







ev = tadc_gaa[4]

tr = traces_gaa[57, 0]
#arg_freqs2 = [965, 1008, 1123, 1166, 964, 1007, 1122, 1165, 966, 1009, 1124, 1167]
tr1, _ = harmonic_filter(tr[0], arg_freqs)
tr2, _ = harmonic_filter(tr[0], arg_freqs2)





##### filtering the data with filter2

t = 1/(500e6) * np.arange(8192) * 1e9
tr_fl2_o1_df0p1 = filter2(tr[0], f1, f2, f3, f4, 0.1, 1)
tr_fl2_o1_df0p2 = filter2(tr[0], f1, f2, f3, f4, 0.2, 1)



fig, ax = plt.subplots(3, 1)
ax[0].plot(t, tr[0], label='Full trace')
ax[1].plot(t, tr1, label='4 freqs.')
ax[2].plot(t, tr2, label='12 freqs.')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('AERA beacon reconstuction DU{}'.format(idu))





fig, ax = plt.subplots(3, 1)
ax[0].plot(t, tr[0], label='Full trace')
ax[1].plot(t, tr1, label='4 freqs.')
ax[2].plot(t, tr2, label='12 freqs.')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('AERA beacon reconstuction DU{}'.format(idu))
[a.set_xlim(1000, 4000) for a in ax]





iev = 4
tadc_ev = tadc_gaa[iev]
trawv_ev = trawv_gaa[iev]
trace_ = tadc_ev.trace_ch.to_numpy()

du_list = tadc_ev.du_id

arg_freqs2 =arg_freqs # [965, 1008, 1123, 1166, 964, 1007, 1122, 1165, 966, 1009, 1124, 1167]

# fig, ax = plt.subplots(len(du_list), 1)


# for i in range(10):

#     tr2, _ = harmonic_filter(trace_[i, 0], arg_freqs2)
#     ax[i].plot(t, tr2, label='du{}'.format(du_list[i]))
#     ax[i].legend(loc=1)

# [a.set_xlim(1000, 4000) for a in ax]

### get positions of dus.

#! /usr/bin/env python

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

plt.figure(345)
plt.clf()
plt.plot(xyz_pos.x, xyz_pos.y, 'k.')

[plt.text(xyz_pos.x[i], xyz_pos.y[i], "{}".format(du_list[i]) ) for i in range(len(du_list)) ]
plt.plot(0, 0, 'ro')




dist = np.sqrt(xyz_pos.x**2 + xyz_pos.y**2)
c  = 3e8 
delta_t = dist / c
delta_t_ns = np.array(delta_t) * 1e9

t = 1/(500e6) * np.arange(1024) * 1e9

fig, ax = plt.subplots(len(du_list), 1, figsize=(10, 10))
for i in range(len(du_list)):
    id_channel = 0
    if du_list[i] == 59:
        id_channel = 1
    tr2, _ = harmonic_filter(trace_[i, id_channel], arg_freqs2)
    ax[i].plot(t - 0*tadc_ev.du_nanoseconds[i]+ 0* (delta_t_ns[i]-delta_t_ns[0]), tr2, label='du{}'.format(du_list[i]))
    ax[i].legend(loc=1)

#[a.set_xlim) for a in ax]



#dunhuang = Geodetic(latitude=40.902317, longitude=94.054550, height=0)


radius = 1000  # m
topography.update_data(beacon, radius=radius)



