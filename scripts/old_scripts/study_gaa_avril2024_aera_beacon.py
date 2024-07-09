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



site = 'gaa'

plot_path = '/Users/ab212678/Documents/GRAND/data/plots/study_aera_gaa_avril24/'
os.makedirs(plot_path, exist_ok=True)



f1 = 58.887
f2 = 61.523
f3 = 68.555
f4 = 71.191

file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/03/gaa_20240329_053214_RUN003002_MD_phys.bin.root'

file_dict_gaa= {file_gaa: "tadc"}



tadc_gaa = uproot.concatenate(file_dict_gaa)

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


for du_id in [58, 59, 151]: #list_gaa:

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




array_to_bins = psd_gaa[:, 0, 0, arg_freq1]


def bin_timed_data(date_array, array_to_bin, delta_t):

    ts_array = np.array([t.timestamp() for t in date_array])*1.0


    ts_array2 = ts_array - ts_array[0]
    ts_array3 = ts_array2 / delta_t

    n_bins = int(ts_array3[-1])+1

    ts_array4 = np.int32(ts_array3)

    binned_value = []
    std_value = []
    n_values = []
    time_bin_value = []

    for ibin in np.arange(n_bins):
        indd = np.where(ts_array4 == ibin)[0]
        print(indd)
        binned_value.append(array_to_bin[indd].mean())
        time_bin_value.append(int(ts_array[indd].mean()))
        std_value.append(array_to_bin[indd].std())
        n_values.append(len(indd))

    date_bin_value = [datetime.datetime.fromtimestamp(ts) for ts in time_bin_value]

    return date_bin_value, np.array(binned_value), np.array(std_value), np.array(n_values)








#### X-axis
plt.figure(11, figsize=(15, 8))
plt.clf()
plt.figure(12, figsize=(15, 8))
plt.clf()
plt.figure(13, figsize=(15, 8))
plt.clf()
plt.figure(14, figsize=(15, 8))
plt.clf()

for idu in list_gaa:

    traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', idu)
    traces_gaa = traces_gaa.to_numpy()

    psd_gaa = filt.return_psd(traces_gaa, 500)


    plt.figure(11)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 0, arg_freq1], ms=1, label='DU {}'.format(idu))
    plt.figure(12)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 0, arg_freq2], label='DU {}'.format(idu))
    plt.figure(13)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 0, arg_freq3], label='DU {}'.format(idu))
    plt.figure(14)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 0, arg_freq4], ms=1, label='DU {}'.format(idu))


    #plt.title('AREA Beacon f1={} MHz'.format(f1))
    #plt.yscale('log')

plt.figure(11)
plt.title('AERA beacon PSD of frequency 1 (X-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq1_xaxis.png"))

plt.figure(12)
plt.title('AERA beacon PSD of frequency 2 (X-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq2_xaxis.png"))

plt.figure(13)
plt.title('AERA beacon PSD of frequency 3 (X-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq3_xaxis.png"))

plt.figure(14)
plt.title('AERA beacon PSD of frequency 4 (X-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq4_xaxis.png"))





#### YY-axis
plt.figure(11, figsize=(15, 8))
plt.clf()
plt.figure(12, figsize=(15, 8))
plt.clf()
plt.figure(13, figsize=(15, 8))
plt.clf()
plt.figure(14, figsize=(15, 8))
plt.clf()

for idu in list_gaa:

    traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', idu)
    traces_gaa = traces_gaa.to_numpy()

    psd_gaa = filt.return_psd(traces_gaa, 500)


    plt.figure(11)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 1, arg_freq1], ms=1, label='DU {}'.format(idu))
    plt.figure(12)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 1, arg_freq2], label='DU {}'.format(idu))
    plt.figure(13)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 1, arg_freq3], label='DU {}'.format(idu))
    plt.figure(14)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 1, arg_freq4], ms=1, label='DU {}'.format(idu))


    #plt.title('AREA Beacon f1={} MHz'.format(f1))
    #plt.yscale('log')

plt.figure(11)
plt.title('AERA beacon PSD of frequency 1 (Y-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq1_yaxis.png"))

plt.figure(12)
plt.title('AERA beacon PSD of frequency 2 (Y-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq2_yaxis.png"))

plt.figure(13)
plt.title('AERA beacon PSD of frequency 3 (Y-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq3_yaxis.png"))

plt.figure(14)
plt.title('AERA beacon PSD of frequency 4 (Y-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq4_yaxis.png"))



#### ZZ-axis
plt.figure(11, figsize=(15, 8))
plt.clf()
plt.figure(12, figsize=(15, 8))
plt.clf()
plt.figure(13, figsize=(15, 8))
plt.clf()
plt.figure(14, figsize=(15, 8))
plt.clf()

for idu in list_gaa:

    traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', idu)
    traces_gaa = traces_gaa.to_numpy()

    psd_gaa = filt.return_psd(traces_gaa, 500)


    plt.figure(11)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 2, arg_freq1], ms=1, label='DU {}'.format(idu))
    plt.figure(12)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 2, arg_freq2], label='DU {}'.format(idu))
    plt.figure(13)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 2, arg_freq3], label='DU {}'.format(idu))
    plt.figure(14)
    plt.plot(date_array_gaa, psd_gaa[:, 0, 2, arg_freq4], ms=1, label='DU {}'.format(idu))


    #plt.title('AREA Beacon f1={} MHz'.format(f1))
    #plt.yscale('log')

plt.figure(11)
plt.title('AERA beacon PSD of frequency 1 (Z-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq1_zaxis.png"))

plt.figure(12)
plt.title('AERA beacon PSD of frequency 2 (Z-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq2_zaxis.png"))

plt.figure(13)
plt.title('AERA beacon PSD of frequency 3 (Z-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq3_zaxis.png"))

plt.figure(14)
plt.title('AERA beacon PSD of frequency 4 (Z-axis)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.grid()
plt.savefig(os.path.join(plot_path, "aera_beacon_freq4_zaxis.png"))








### Beacon reconstruction
traces_gaa, date_array_gaa = utils.get_column_for_given_du(tadc_gaa, 'trace_ch', 151)
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



tr = traces_gaa[57, 0]
arg_freqs = [965, 1008, 1123, 1166, 964, 1007, 1122, 1165, 966, 1009, 1124, 1167]
tr2, _ = harmonic_filter(tr[0], arg_freqs)




##### filtering the data with filter2

t = 1/(500e6)* arange(8192)*1e9
tr_fl2_o1_df0p1 = filter2(tr[0], f1, f2, f3, f4, 0.1, 1)
tr_fl2_o1_df0p2 = filter2(tr[0], f1, f2, f3, f4, 0.2, 1)



fig, ax = plt.subplots(3, 1)
plt.figure(51)
plt.clf()
ax[0].plot(t, tr_fl2_o1_df0p1)
ax[1].plot(t, tr_fl2_o1_df0p2)
ax[2].plot(t, tr2, label='data1a, fl2, o3')
plt.legend(loc=0)
plt.title('data filter2')


fig, ax = plt.subplots(3, 1)
plt.figure(51)
plt.clf()
ax[0].plot(t, tr_fl2_o1_df0p1)
ax[1].plot(t, tr_fl2_o1_df0p2)
ax[2].plot(t, tr2, label='data1a, fl2, o3')
plt.legend(loc=0)
[a.set_xlim(0, 2500) for a in ax]
plt.title('data filter2')


