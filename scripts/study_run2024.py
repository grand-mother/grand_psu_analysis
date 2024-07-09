import sys
sys.path.append('./')
import uproot
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import grand_psu_lib.utils.utils as utils
import argparse
import glob
import datetime


file1 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f0001.root'
file2 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f0002.root'
file3 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f0003.root'
file4 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f0004.root'
file5 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f0005.root'



plot_path = './plots/run24/'

site = 'gaa'

base = 'td002024_f0003.root'

plot_path = os.path.join(plot_path, '{}'.format(base))
trace_plot_path = os.path.join(plot_path, 'random_traces_and_their_spectra')

os.makedirs(plot_path, exist_ok=True)
os.makedirs(trace_plot_path, exist_ok=True)




tadc = uproot.concatenate({file3:'tadc', file4:'tadc', file5:'tadc' })
trawv = uproot.concatenate({file3:'trawvoltage'})


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



sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]
n_samples = len(tadc.trace_ch[0][0][0])
n_samples_ab =n_samples // 2

fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]



fft_freqab = np.fft.rfftfreq(n_samples_ab) * sample_freq  # [MHz]


df = np.diff(fft_freq)[0]
dfab = np.diff(fft_freqab)[0]

trace = traces_np[0:200]
tracea = traces_np[0:200, :, 0:1024]
traceb = traces_np[0:200, :, 1024:]



def return_psd(trace_array, sampling_rate):
    # make sure the trace array has the samples vbalues in the last dimension
    # units of the psd are [trace_array]^2/ [sampling_rate]
    fft = np.fft.rfft(trace_array)
    fft[..., 1:-1] *= 2
    N = trace_array.shape[-1]
    psd = np.abs(fft)**2 / N / sampling_rate
    return psd
    


psd = return_psd(trace, 500)
psda = return_psd(tracea, 500)
psdb = return_psd(traceb, 500)



plt.figure(435)
plt.clf()
plt.plot(fft_freq, psd.mean(axis=0)[0])
plt.plot(fft_freqab, psdb.mean(axis=0)[0])
plt.plot(fft_freqab, psda.mean(axis=0)[0])
plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.tight_layout()



fft = np.fft.rfft(trace)
ffta = np.fft.rfft(tracea)
fftb = np.fft.rfft(traceb)




m_fft_0 = (np.abs(fft)**2).mean(axis=0)[0] 
m_ffta_0 = (np.abs(ffta)**2).mean(axis=0)[0]
m_fftb_0 = (np.abs(fftb)**2).mean(axis=0)[0] 

plt.figure(1)
plt.clf()
plt.plot(fft_freq, m_fft_0)
plt.plot(fft_freqab, m_ffta_0 )
plt.plot(fft_freqab, m_fftb_0)
plt.yscale('log')



plt.figure(2)
plt.clf()
plt.plot(fft_freq, m_fft_0/df)
plt.plot(fft_freqab, m_ffta_0/dfab )
plt.plot(fft_freqab, m_fftb_0/dfab)
plt.yscale('log')




plt.figure(3)
plt.clf()
plt.plot(fft_freq, m_fft_0/df/n_samples**2)
#plt.plot(fft_freqab, m_ffta_0/dfab/n_samples_ab**2 )
#plt.plot(fft_freqab, m_fftb_0/dfab/n_samples_ab**2)
plt.yscale('log')



axs = plt.gca()




TV_transmitter = 67.25  # [MHz]
TV_audio_carrier = 71.75  # [MHz]


f1 = 58.887
f2 = 61.523
f3 = 68.555
f4 = 71.191

axs.axvline(f1, color='k', ls=(0, (5, 10)))
axs.axvline(f2, color='k', ls=(0, (5, 10)))
axs.axvline(f3, color='k', ls=(0, (5, 10)))
axs.axvline(f4, color='k', ls=(0, (5, 10)))
axs.axvline(TV_transmitter, color='m', ls=(0, (5, 10)))
axs.axvline(TV_audio_carrier, color='m', ls=(0, (5, 10)))
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC$^2$/Hz]')
plt.title('Run 2024')
