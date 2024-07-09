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


def sin_wave(t, f):
    w = 2*np.pi * f
    return np.sin(w * t)


site = 'gaa'



f1 = 58.887
f2 = 61.523
f3 = 68.555
f4 = 71.191




f1_ = 58.887-3
f2_ = 61.523-3
f3_ = 68.555-3
f4_ = 71.191-3

file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/03/gaa_20240329_053214_RUN003002_MD_phys.bin.root'
file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/GatA_20240308_1859_003001_CD_coin.root'
file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/gaa_20240405_074134_RUN003002_CD_phys.root'

file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/gaa_20240413_070637_RUN003002_CD_phys.root'

file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/gaa_20240412_133542_RUN003002_CD_phys.root'

plot_path = '/Users/ab212678/Documents/GRAND/data/plots/study_aera_gaa_avril24/{}'.format(os.path.splitext(os.path.basename(file_gaa))[0])
os.makedirs(plot_path, exist_ok=True)


file_dict_gaa= {file_gaa: "tadc"}
file_dict_gaa_trawv= {file_gaa: "trawvoltage"}

tadc_gaa = uproot.concatenate(file_dict_gaa)
trawv_gaa = uproot.concatenate(file_dict_gaa_trawv)


list_gaa = utils.get_dulist(tadc_gaa)

for iii in range(len(tadc_gaa)):
    iev = iii

    title_str = 'GatA_20240308_1859_003001_CD_coin, event {}'.format(iev)

    tadc_ev = tadc_gaa[iev]
    trace_ = tadc_ev.trace_ch.to_numpy()

    du_list = tadc_ev.du_id

    if (len(np.unique(du_list)) == len(du_list)) * (len(du_list) == 4):
        print(iev)
        n_du = len(du_list)
        fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
        [axs[i].plot(trace_[i, 0,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
        [ax.legend(loc=1) for ax in axs]
        [axs[i].axvline(tadc_ev.trigger_position[i]) for i in range(len(tadc_ev.du_id))]
        [ax.set_ylabel('ADC counts') for ax in axs]
        axs[-1].set_xlabel('time bin')
        axs[0].set_title(title_str)
        plt.savefig(os.path.join(plot_path, 'gaa_CD_ev{}_traces1.png'.format(iev)))

        t1 = 1/(500e6) * np.arange(1024) * 1e9
        ev_second = tadc_ev.time_seconds
        ev_du_sec = tadc_ev.du_seconds.to_numpy()
        ev_du_nanosec = tadc_ev.du_nanoseconds.to_numpy()
        ev_du_dnanosec = ev_du_nanosec - tadc_ev.time_nanoseconds  ## offset in nsec wrt to the fisrt du. so the first du has dnano = 0
        ev_tri_pos = tadc_ev.trigger_position.to_numpy()
        timex = []
        [timex.append(  t1 - t1[ev_tri_pos[i]]+ ev_du_dnanosec[i] ) for i in range(len(tadc_ev.du_id))]

        fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
        [axs[i].plot(timex[i], trace_[i, 0,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
        [ax.legend(loc=1) for ax in axs]
        axs[-1].set_xlabel('time [ns]')
        axs[0].set_title(title_str)
        plt.savefig(os.path.join(plot_path, 'gaa_CD_ev{}_traces2.png'.format(iev)))




du_id = 83
traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', du_id)
traces_gaa = traces_gaa.to_numpy()

sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]
n_samples_gaa = traces_gaa.shape[-1]
fft_freq__gaa = np.fft.rfftfreq(n_samples_gaa) * sample_freq  # [MHz]

arg_freq1 = np.argmin(abs(fft_freq__gaa - f1))
arg_freq2 = np.argmin(abs(fft_freq__gaa - f2))
arg_freq3 = np.argmin(abs(fft_freq__gaa - f3))
arg_freq4 = np.argmin(abs(fft_freq__gaa - f4))


t8 = 1/(500e6) * np.arange(8192) * 1e9
t4 = 1/(500e6) * np.arange(4096) * 1e9
t2 = 1/(500e6) * np.arange(2048) * 1e9
t1 = 1/(500e6) * np.arange(1024) * 1e9


trace1 = sin_wave(t8/1e9, f1*1e6)
trace2 = sin_wave(t8/1e9, f2*1e6)
trace3 = sin_wave(t8/1e9, f3*1e6)
trace4 = sin_wave(t8/1e9, f4*1e6)
tr_sim = trace1 + trace2 + trace3 + trace4


trace1_ = sin_wave(t8/1e9, f1_*1e6)
trace2_ = sin_wave(t8/1e9, f2_*1e6)
trace3_ = sin_wave(t8/1e9, f3_*1e6)
trace4_ = sin_wave(t8/1e9, f4_*1e6)
tr_sim_ = trace1_ + trace2_ + trace3_ + trace4_



plt.figure(1, figsize=(12, 6))
plt.clf()
plt.plot(t8, tr_sim)
plt.xlabel('time [ns]')
plt.ylabel('A. U.')
plt.xlim(0, 3000)
plt.savefig(os.path.join(plot_path, 'simulated_beacon.png'))


idu = 151
traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', idu)
traces_gaa = traces_gaa.to_numpy()




trace_8192 = traces_gaa
trace_4096 = traces_gaa[:, :, :, 0:4096]
trace_2048 = traces_gaa[:, :, :, 0:2048]
trace_1024 = traces_gaa[:, :, :, 0:1024]

n_samples_8192 = trace_8192.shape[-1]
n_samples_4096 = trace_4096.shape[-1]
n_samples_2048 = trace_2048.shape[-1]
n_samples_1024 = trace_1024.shape[-1]

fft_freq_8192 = np.fft.rfftfreq(n_samples_8192) * sample_freq  # [MHz]
fft_freq_4096 = np.fft.rfftfreq(n_samples_4096) * sample_freq  # [MHz]
fft_freq_2048 = np.fft.rfftfreq(n_samples_2048) * sample_freq  # [MHz]
fft_freq_1024 = np.fft.rfftfreq(n_samples_1024) * sample_freq  # [MHz]



tr8 = trace_8192[5, 0, 0]
tr4 = trace_4096[5, 0, 0]
tr2 = trace_2048[5, 0, 0]
tr1 = trace_1024[5, 0, 0]



list_freqs = [f1, f2, f3, f4]

arg_freqs_8192 = [np.argmin(abs(fft_freq_8192 - f)) for f in list_freqs]
arg_freqs_4096 = [np.argmin(abs(fft_freq_4096 - f)) for f in list_freqs]
arg_freqs_2048 = [np.argmin(abs(fft_freq_2048 - f)) for f in list_freqs]
arg_freqs_1024 = [np.argmin(abs(fft_freq_1024 - f)) for f in list_freqs]



arg_freqs_8192_v2 = arg_freqs_8192 + [arg+1 for arg in arg_freqs_8192] + [arg-1 for arg in arg_freqs_8192] 
arg_freqs_4096_v2 = arg_freqs_4096 + [arg+1 for arg in arg_freqs_4096] + [arg-1 for arg in arg_freqs_4096] 
arg_freqs_2048_v2 = arg_freqs_2048 + [arg+1 for arg in arg_freqs_2048] + [arg-1 for arg in arg_freqs_2048] 
arg_freqs_1024_v2 = arg_freqs_1024 + [arg+1 for arg in arg_freqs_1024] + [arg-1 for arg in arg_freqs_1024] 







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




tr8_sim, _ = harmonic_filter(tr_sim[0:8192], arg_freqs_8192)
tr4_sim, _ = harmonic_filter(tr_sim[0:4096], arg_freqs_4096)
tr2_sim, _ = harmonic_filter(tr_sim[0:2048], arg_freqs_2048)
tr1_sim, _ = harmonic_filter(tr_sim[0:1024], arg_freqs_1024)


tr8_sim_v2, _ = harmonic_filter(tr_sim[0:8192], arg_freqs_8192_v2)
tr4_sim_v2, _ = harmonic_filter(tr_sim[0:4096], arg_freqs_4096_v2)
tr2_sim_v2, _ = harmonic_filter(tr_sim[0:2048], arg_freqs_2048_v2)
tr1_sim_v2, _ = harmonic_filter(tr_sim[0:1024], arg_freqs_1024_v2)


psd8_sim = filt.return_psd(tr_sim[0:8192], 500)
psd4_sim = filt.return_psd(tr_sim[0:4096], 500)
psd2_sim = filt.return_psd(tr_sim[0:2048], 500)
psd1_sim = filt.return_psd(tr_sim[0:1024], 500)


plt.figure(458, figsize=(15, 8))
plt.clf()
plt.plot(fft_freq_8192, psd8_sim, '.-', label='8192')
plt.plot(fft_freq_4096, psd4_sim, '.-', label='4096')
plt.plot(fft_freq_2048, psd2_sim, '.-', label='2048')
plt.plot(fft_freq_1024, psd1_sim, '.-', label='1024')

plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')
plt.yscale('log')
plt.title('Simulated beacon (4 sin waves)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(loc=0)
plt.savefig(os.path.join(plot_path, 'PSD_beaconsim.png'))
plt.xlim((57, 75))
plt.savefig(os.path.join(plot_path, 'PSD_beaconsim_zoom.png'))


##### filtering the data with filter2


plt.figure(45)
plt.clf()
plt.plot(t8, tr8_)
plt.plot(t4, tr4_)
plt.plot(t2, tr2_)
plt.plot(t1, tr1_)

plt.figure(44)
plt.clf()
plt.plot(t8, tr8)
plt.plot(t4, tr4)
plt.plot(t2, tr2)
plt.plot(t1, tr1)




fig, ax = plt.subplots(4, 1)
ax[0].plot(t8, tr8_, label='8192')
ax[1].plot(t4, tr4_, label='4096')
ax[2].plot(t2, tr2_, label='2048')
ax[3].plot(t1, tr1_, label='1024')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('Fourier filtering AERA beacon reconstuction DU{}'.format(idu))
[a.set_xlim(t1[0], t2[-1]) for a in ax]


fig, ax = plt.subplots(4, 1)
ax[0].plot(t8, tr8_v2, label='8192')
ax[1].plot(t4, tr4_v2, label='4096')
ax[2].plot(t2, tr2_v2, label='2048')
ax[3].plot(t1, tr1_v2, label='1024')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('Fourier v2 filtering AERA beacon reconstuction DU{}'.format(idu))
[a.set_xlim(t1[0], t2[-1]) for a in ax]


fig, ax = plt.subplots(5, 1, figsize=(15, 8))
ax[0].plot(t8, tr_sim, label='sim')
ax[1].plot(t8, tr8_sim, label='8192')
ax[2].plot(t4, tr4_sim, label='4096')
ax[3].plot(t2, tr2_sim, label='2048')
ax[4].plot(t1, tr1_sim, label='1024')
[a.legend(loc=1) for a in ax]
ax[0].set_title('Fourier filtering simulated beacon (4 frequencies)')
[a.set_xlim(t1[0], t2[-1]) for a in ax]
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel('ADC [A. U.]')
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'Fourier_sim_recons_4freq.png'))



fig, ax = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(15, 8))
ax[0].plot(t8, tr_sim, label='sim')
ax[1].plot(t8, tr8_sim_v2, label='8192')
ax[2].plot(t4, tr4_sim_v2, label='4096')
ax[3].plot(t2, tr2_sim_v2, label='2048')
ax[4].plot(t1, tr1_sim_v2, label='1024')
[a.legend(loc=1) for a in ax]
ax[0].set_title('Fourier filtering simulated beacon (12 frequencies)')
[a.set_xlim(t1[0], t2[-1]) for a in ax]
ax[4].set_xlabel('time [ns]')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel('ADC [A. U.]')
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'Fourier_sim_recons_12freq.png'))





#### Real time filters


df = 0.5
tr8_f2_sim = filter2(tr_sim[0:8192], f1, f2, f3, f4, df, 1)
tr4_f2_sim = filter2(tr_sim[0:4096], f1, f2, f3, f4, df, 1)
tr2_f2_sim = filter2(tr_sim[0:2048], f1, f2, f3, f4, df, 1)
tr1_f2_sim = filter2(tr_sim[0:1024], f1, f2, f3, f4, df, 1)



fig, ax = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(15, 8))
ax[0].plot(t8, tr_sim, label='sim')
ax[1].plot(t8, tr8_f2_sim, label='8192')
ax[2].plot(t4, tr4_f2_sim, label='4096')
ax[3].plot(t2, tr2_f2_sim, label='2048')
ax[4].plot(t1, tr1_f2_sim, label='1024')
[a.legend(loc=1) for a in ax]
ax[0].set_title('Butterworth bandpass filters simulated beacon')
[a.set_xlim(t1[0], t2[-1]) for a in ax]
ax[4].set_xlabel('time [ns]')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel('ADC [A. U.]')
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'butterworth_sim_recons.png'))




#### Real time filters with  noise


tr_sim_noise =  tr_sim +  np.random.randn(8192) *2

df = 0.5
tr8_f2_sim = filter2(tr_sim_noise[0:8192], f1, f2, f3, f4, df, 1)
tr4_f2_sim = filter2(tr_sim_noise[0:4096], f1, f2, f3, f4, df, 1)
tr2_f2_sim = filter2(tr_sim_noise[0:2048], f1, f2, f3, f4, df, 1)
tr1_f2_sim = filter2(tr_sim_noise[0:1024], f1, f2, f3, f4, df, 1)



fig, ax = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(15, 8))
ax[0].plot(t8, tr_sim, label='sim')
ax[0].plot(t8, tr_sim_noise, label='sim')
ax[1].plot(t8, tr8_f2_sim, label='8192')
ax[2].plot(t4, tr4_f2_sim, label='4096')
ax[3].plot(t2, tr2_f2_sim, label='2048')
ax[4].plot(t1, tr1_f2_sim, label='1024')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('Filter2 simulated beacon')
[a.set_xlim(t1[0], t2[-1]) for a in ax]



###########################
#### work on data!!!!
########################



idu = 151
traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', idu)
traces_gaa = traces_gaa.to_numpy()




trace_8192 = traces_gaa
trace_4096 = traces_gaa[:, :, :, 0:4096]
trace_2048 = traces_gaa[:, :, :, 0:2048]
trace_1024 = traces_gaa[:, :, :, 0:1024]

n_samples_8192 = trace_8192.shape[-1]
n_samples_4096 = trace_4096.shape[-1]
n_samples_2048 = trace_2048.shape[-1]
n_samples_1024 = trace_1024.shape[-1]

fft_freq_8192 = np.fft.rfftfreq(n_samples_8192) * sample_freq  # [MHz]
fft_freq_4096 = np.fft.rfftfreq(n_samples_4096) * sample_freq  # [MHz]
fft_freq_2048 = np.fft.rfftfreq(n_samples_2048) * sample_freq  # [MHz]
fft_freq_1024 = np.fft.rfftfreq(n_samples_1024) * sample_freq  # [MHz]

id_trace = 986

tr8 = trace_8192[id_trace, 0, 0]
tr4 = trace_4096[id_trace, 0, 0]
tr2 = trace_2048[id_trace, 0, 0]
tr1 = trace_1024[id_trace, 0, 0]


psd8 = filt.return_psd(tr8, 500)
psd4 = filt.return_psd(tr4, 500)
psd2 = filt.return_psd(tr2, 500)
psd1 = filt.return_psd(tr1, 500)


plt.figure(4590, figsize=(15, 8))
plt.clf()
plt.plot(fft_freq_8192, psd8, '.-', label='8192')
plt.plot(fft_freq_4096, psd4, '.-', label='4096')
plt.plot(fft_freq_2048, psd2, '.-', label='2048')
plt.plot(fft_freq_1024, psd1, '.-', label='1024')

plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')
plt.yscale('log')
plt.title('Simulated beacon (4 sin waves)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(loc=0)
plt.savefig(os.path.join(plot_path, 'PSD_151_id{}.png'.format(id_trace)))
plt.xlim((57, 75))
plt.savefig(os.path.join(plot_path, 'PSD_151_id{}_zoom.png'.format(id_trace)))



fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(15, 8))
ax[0].plot(t8, tr8, label='8192')
ax[1].plot(t4, tr4, label='4096')
ax[2].plot(t2, tr2, label='2048')
ax[3].plot(t1, tr1, label='1024')
[a.legend(loc=1) for a in ax]
ax[3].set_xlabel('time [mus]')
ax[0].set_title('DU151 random trace x-axis')
[a.set_xlim(t1[0], t8[-1]) for a in ax]






#arg_freqs2 = [965, 1008, 1123, 1166, 964, 1007, 1122, 1165, 966, 1009, 1124, 1167]
tr8_, _ = harmonic_filter(tr8, arg_freqs_8192)
tr4_, _ = harmonic_filter(tr4, arg_freqs_4096)
tr2_, _ = harmonic_filter(tr2, arg_freqs_2048)
tr1_, _ = harmonic_filter(tr1, arg_freqs_1024)

tr8_v2, _ = harmonic_filter(tr8, arg_freqs_8192_v2)
tr4_v2, _ = harmonic_filter(tr4, arg_freqs_4096_v2)
tr2_v2, _ = harmonic_filter(tr2, arg_freqs_2048_v2)
tr1_v2, _ = harmonic_filter(tr1, arg_freqs_1024_v2)




df = 0.5
tr8_f2 = filter2(tr8, f1, f2, f3, f4, df, 1)
tr4_f2 = filter2(tr4, f1, f2, f3, f4, df, 1)
tr2_f2 = filter2(tr2, f1, f2, f3, f4, df, 1)
tr1_f2 = filter2(tr1, f1, f2, f3, f4, df, 1)




fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(15, 8))
ax[0].plot(t8, tr8_f2, label='8192')
ax[1].plot(t4, tr4_f2, label='4096')
ax[2].plot(t2, tr2_f2, label='2048')
ax[3].plot(t1, tr1_f2, label='1024')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('Butterworth filter AERA beacon DU{} id{}'.format(idu, id_trace))
[a.set_xlim(t1[0], t2[-1]) for a in ax]
plt.savefig(os.path.join(plot_path, 'recons_butter_151_id{}.png'.format(id_trace)))


fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(15, 8))
ax[0].plot(t8, tr8_, label='8192')
ax[1].plot(t4, tr4_, label='4096')
ax[2].plot(t2, tr2_, label='2048')
ax[3].plot(t1, tr1_, label='1024')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('Fourier AERA beacon DU{} id{} 4 frequencies'.format(idu, id_trace))
[a.set_xlim(t1[0], t2[-1]) for a in ax]
plt.savefig(os.path.join(plot_path, 'recons_fourier4_151_id{}.png'.format(id_trace)))

fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(15, 8))
ax[0].plot(t8, tr8_v2, label='8192')
ax[1].plot(t4, tr4_v2, label='4096')
ax[2].plot(t2, tr2_v2, label='2048')
ax[3].plot(t1, tr1_v2, label='1024')
[a.legend(loc=1) for a in ax]
ax[2].set_xlabel('time [mus]')
ax[0].set_title('Fourier AERA beacon DU{} id{} 12 frequencies'.format(idu, id_trace))
[a.set_xlim(t1[0], t2[-1]) for a in ax]
plt.savefig(os.path.join(plot_path, 'recons_fourier12_151_id{}.png'.format(id_trace)))








iev = 21
tadc_ev = tadc_gaa[iev]
trawv_ev = trawv_gaa[iev]
trace_ = tadc_ev.trace_ch.to_numpy()

du_list = tadc_ev.du_id

arg_freqs2 = [965, 1008, 1123, 1166, 964, 1007, 1122, 1165, 966, 1009, 1124, 1167]

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

t = 1/(500e6) * np.arange(8192) * 1e9

fig, ax = plt.subplots(len(du_list), 1, figsize=(10, 10))
for i in range(len(du_list)):
    id_channel = 0
    if du_list[i] == 59:
        id_channel = 1
    tr2, _ = harmonic_filter(trace_[i, id_channel], arg_freqs2)
    ax[i].plot(t + tadc_ev.du_nanoseconds[i]+ 0* (delta_t_ns[i]-delta_t_ns[0]), tr2, label='du{}'.format(du_list[i]))
    ax[i].legend(loc=1)

[a.set_xlim(2000, 5000) for a in ax]

tr_filterd = []
fig, ax = plt.subplots(len(du_list), 1, figsize=(10, 10))
for i in range(len(du_list)):
    id_channel = 0
    if du_list[i] == 59:
        id_channel = 1
    tr2 = filter2(trace_[i, id_channel], f1, f2, f3, f4, df, 1)
    tr_filterd.append(tr2)
    ax[i].plot(t + tadc_ev.du_nanoseconds[i]+ 0* (delta_t_ns[i]-delta_t_ns[0]), tr2, label='du{}'.format(du_list[i]))
    ax[i].legend(loc=1)

[a.set_xlim(2000, 5000) for a in ax]


plt.figure()
n_du = len(du_list)
lags = signal.correlation_lags(8192, 8192)
lag_list = []
for i in arange(0, n_du):
    correl = signal.correlate(tr_filterd[0], tr_filterd[i])
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


#dunhuang = Geodetic(latitude=40.902317, longitude=94.054550, height=0)


radius = 1000  # m
topography.update_data(beacon, radius=radius)



