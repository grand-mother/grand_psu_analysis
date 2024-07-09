import sys
sys.path.append('./')
import uproot
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import grand_psu_lib.utils.utils as utils
import grand_psu_lib.utils.filtering as filt

import argparse
import glob
import datetime

from scipy import signal


sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]

n_samples1 = 2048
n_samples2 = n_samples1 // 2
n_samples3 = n_samples2


freq1 = np.fft.rfftfreq(n_samples1) * sample_freq  # [MHz]
freq2 = np.fft.rfftfreq(n_samples2) * sample_freq  # [MHz]
freq3 = np.fft.rfftfreq(n_samples3) * sample_freq/2


df1 = np.diff(freq1)[0]
df2 = np.diff(freq2)[0]
df3 = np.diff(freq3)[0]


t1 = np.arange(n_samples1)* sample_period
t2 = np.arange(n_samples2)* sample_period
t3 = t1[::2]







file3 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f0003.root'
files = np.sort(glob.glob(file3))

file_dict_tadc = {}
for fi in files:
    file_dict_tadc[fi] = 'tadc'

file_dict_trawv = {}
for fi in files:
    file_dict_trawv[fi] = 'trawvoltage'


plot_path = './plots/galaxy/'

site = 'gaa'

base = 'run24_3-4-5'

plot_path = os.path.join(plot_path, '{}'.format(base))
trace_plot_path = os.path.join(plot_path, 'random_traces_and_their_spectra')

os.makedirs(plot_path, exist_ok=True)
os.makedirs(trace_plot_path, exist_ok=True)



tadc = uproot.concatenate(file_dict_tadc)
trawv = uproot.concatenate(file_dict_trawv)


#tadc = uproot.concatenate({file3:'tadc', file4:'tadc', file5:'tadc' })
#trawv = uproot.concatenate({file3:'trawvoltage', file4:'trawvoltage', file5:'trawvoltage'})


du_list = utils.get_dulist(tadc)



tz_gmt = utils.TZ_GMT()
tz_auger = utils.TZ_auger()
tz_gp13 = utils.TZ_GP13()


if site == 'gaa':
    tz = tz_auger



idu = 84


request = 'trace_ch'
result, date_array = utils.get_column_for_given_du(tadc, request, idu)
traces_np = result[:, 0, 0:3].to_numpy()



t2048 = np.arange(n_samples1)* sample_period
t1024 = np.arange(n_samples2)* sample_period
t1024_downsampled = t2048[::2]


tr_2048x = traces_np[0, 0]

fft2048 = np.fft.rfft(tr_2048x)

fft2048[512:] *= 0
tr_2048_filtered = np.fft.irfft(fft2048)

tr_1024_filtered_downsampled = tr_2048_filtered[::2]




tr_1024_x1 = tr_2048x[0:1024]
tr_1024_x2 = tr_2048x[1024:]
tr_1024_x_downsampled = tr_2048x[::2]
tr_1024_decimated = signal.decimate(tr_2048x, 2)

plt.figure(1, figsize=(20, 5))
plt.clf()
plt.plot(t2048, tr_2048x)
plt.ylabel('ADC')
plt.xlabel('time [ms]')
plt.savefig('downsample_fig1.png')


plt.figure(11, figsize=(20, 5))
plt.clf()
plt.plot(t2048, tr_2048x, label='Initial trace')
plt.plot(t1024_downsampled, tr_1024_x_downsampled, 'k.', label='simple downsampling')
plt.ylabel('ADC')
plt.xlabel('time [ms]')
plt.xlim(0.100, 0.400)
plt.legend(loc=0)
plt.savefig('downsample_fig1_zoom.png')




plt.figure(12, figsize=(20, 5))
plt.clf()
plt.plot(t2048, tr_2048x, label='Initial trace')
plt.plot(t1024_downsampled, tr_1024_x_downsampled, 'k.', label='simple downsampling')
plt.plot(t1024_downsampled, tr_1024_filtered_downsampled, 'g.', label='filter then downsample')

plt.ylabel('ADC')
plt.xlabel('time [ms]')
plt.xlim(0.100, 0.400)
plt.legend(loc=0)
plt.savefig('downsample_fig2_zoom.png')





plt.figure(13, figsize=(20, 5))
plt.clf()
plt.plot(t2048, tr_2048x, label='Initial trace')
plt.plot(t1024_downsampled, tr_1024_x_downsampled, 'k.', label='simple downsampling')
plt.plot(t1024_downsampled, tr_1024_filtered_downsampled, 'g.', label='filter then downsample')

plt.plot(t1024_downsampled, tr_1024_decimated, 'r.', label='scipy decimate')

plt.ylabel('ADC')
plt.xlabel('time [ms]')
plt.xlim(0.100, 0.400)
plt.legend(loc=0)
plt.savefig('downsample_fig3_zoom.png')



plt.plot(t1024_downsampled, tr_1024_decimated, 'r.')
plt.plot(t1024_downsampled, tr_1024_filtered_downsampled, 'g.')



psd2048 = filt.return_psd(tr_2048x, sample_freq)
psd1024_downsampled = filt.return_psd(tr_1024_x_downsampled, sample_freq//2)
psd1024_decimated = filt.return_psd(tr_1024_decimated, sample_freq//2)
psd1024_filtered_downsampled = filt.return_psd(tr_1024_filtered_downsampled, sample_freq//2)




plt.figure(2, figsize=(20, 5))
plt.clf()
plt.plot(freq1,  psd2048, label='Initial trace')
plt.plot(freq3,  psd1024_downsampled, 'k-', label='simple downsampling')
#plt.plot(freq3,  psd1024_decimated, label='tr3')
#plt.plot(freq3,  psd1024_filtered_downsampled, 'g.', label='filter then downsample')

plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2/MHz] ')
plt.yscale('log')
plt.title('PSD')
plt.ylim(1e-3, 10)
plt.legend()
plt.savefig('psd_downsample_fig1.png')




plt.figure(21, figsize=(20, 5))
plt.clf()
plt.plot(freq1,  psd2048, label='Initial trace')
plt.plot(freq3,  psd1024_downsampled, 'k-', label='simple downsampling')
#plt.plot(freq3,  psd1024_decimated, label='tr3')
plt.plot(freq3,  psd1024_filtered_downsampled, 'g-', label='filter then downsample')

plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2/MHz] ')
plt.yscale('log')
plt.title('PSD')
plt.ylim(1e-3, 10)
plt.legend()
plt.savefig('psd_downsample_fig2.png')



plt.figure(22, figsize=(20, 5))
plt.clf()
plt.plot(freq1,  psd2048, label='Initial trace')
plt.plot(freq3,  psd1024_downsampled, 'k-', label='simple downsampling')
plt.plot(freq3,  psd1024_decimated, 'r-',  label='scipy decimate')
plt.plot(freq3,  psd1024_filtered_downsampled, 'g-', label='filter then downsample')

plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2/MHz] ')
plt.yscale('log')
plt.title('PSD')
plt.ylim(1e-3, 10)
plt.legend()
plt.savefig('psd_downsample_fig3.png')
#### same thing with an array of traces




fft = np.fft.rfft(traces_np)
fft[:, :, 512:] *= 0
traces_np_ = np.fft.irfft(fft)

n_tr = 2000

tr1 = traces_np[0:n_tr]
tr2 = traces_np[0:n_tr, :, 0:1024]
tr3 = traces_np[0:n_tr, :, 0::2]
tr4 = signal.decimate(tr1, 2, axis=-1)
tr5 = traces_np_[0:n_tr, :, 0::2]




psd1 = filt.return_psd(tr1, sample_freq)
psd2 = filt.return_psd(tr2, sample_freq)
psd3 = filt.return_psd(tr3, sample_freq/2)
psd4 = filt.return_psd(tr4, sample_freq/2)
psd5 = filt.return_psd(tr5, sample_freq/2)




ff1 = np.fft.rfft(tr1)
ff2 = np.fft.rfft(tr2)
ff3 = np.fft.rfft(tr3)



ff1_sq = np.abs(ff1)**2
ff2_sq = np.abs(ff2)**2
ff3_sq = np.abs(ff3)**2




plt.figure(55, figsize=(20, 5))
plt.clf()
plt.plot(freq1,  psd1.mean(axis=0)[0], label='Initial trace')
#plt.plot(freq2,  psd2.mean(axis=0)[0], label='tr2')
plt.plot(freq3,  psd3.mean(axis=0)[0], 'k-', label='simple downsampling')
plt.plot(freq3,  psd4.mean(axis=0)[0], 'r-', label='scipy decimate')
plt.plot(freq3,  psd5.mean(axis=0)[0], 'g-', label='filter then downsample')


plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2/MHz] ')
plt.yscale('log')
plt.title('PSD')
plt.ylim(1e-3, 10)
plt.legend()

plt.savefig('psd_mean_dowsample_fig1.png')










plt.figure(56)
plt.clf()
plt.plot(freq1,  ff1_sq.mean(axis=0)[0], label='tr1')
plt.plot(freq2,  ff2_sq.mean(axis=0)[0], label='tr2')
plt.plot(freq3,  ff3_sq.mean(axis=0)[0], label='tr3')
plt.xlabel('Frequency [MHz]')
plt.ylabel('DFT$^2$ [ADC$^2$]')
plt.xlim(0, freq1[512-1])
plt.yscale('log')
plt.ylim(1e3, 1e7)
plt.title('Square of DFT')
plt.legend()
plt.savefig('psd_study_fig1.png')


ps1 = ff1_sq / n_samples1 * sample_period
ps2 = ff2_sq / n_samples2 * sample_period
ps3 = ff3_sq / n_samples3 * (sample_period*2)



def return_psd(trace_array, sampling_rate):
    # make sure the trace array has the samples vbalues in the last dimension
    # units of the psd are [trace_array]^2/ [sampling_rate]
    fft = np.fft.rfft(trace_array)
    psd = np.abs(fft)**2
    psd[..., 1:-1] *= 2
    N = trace_array.shape[-1]
    print(N)
    psd = psd / N / sampling_rate
    return psd
    


plt.figure(57)
plt.clf()
plt.plot(freq1, ps1.mean(axis=0)[0], label='tr1')
plt.plot(freq2, ps2.mean(axis=0)[0], label='tr2')
plt.plot(freq3, ps3.mean(axis=0)[0], label='tr3')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC$^2$/Hz]')
plt.xlim(0, freq1[512-1])
plt.yscale('log')
plt.ylim(1e-2, 1e1)
plt.title('Power Spectrum Density')
plt.legend()
plt.savefig('psd_study_fig2.png')



plt.figure(58)
plt.clf()
plt.plot(freq1, ps1.mean(axis=0)[0]*df1, label='tr1')
plt.plot(freq2, ps2.mean(axis=0)[0]*df2, label='tr2')
plt.plot(freq3, ps3.mean(axis=0)[0]*df3, label='tr3')
plt.legend()



psdd1 = return_psd(tr1, sample_freq)
psdd2 = return_psd(tr2, sample_freq)
psdd3 = return_psd(tr3, sample_freq/2)



plt.figure(59)
plt.clf()
plt.plot(freq1, psdd1.mean(axis=0)[0], label='tr1')
plt.plot(freq2, psdd2.mean(axis=0)[0], label='tr2')
plt.plot(freq3, psdd3.mean(axis=0)[0], label='tr3')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC$^2$/Hz]')
plt.xlim(0, freq1[512-1])
plt.yscale('log')
plt.ylim(1e-2, 1e1)
plt.title('Power Spectrum Density')
plt.legend()
plt.savefig('psd_study_fig3.png')

