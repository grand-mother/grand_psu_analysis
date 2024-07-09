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





file1 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f0004.root'
file2 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run25-28/td002028_f0001.root'



plot_path = './plots/'
site = 'gaa'

base = 'study_runs25-28'

plot_path = os.path.join(plot_path, '{}'.format(base))
trace_plot_path = os.path.join(plot_path, 'random_traces_and_their_spectra')

os.makedirs(plot_path, exist_ok=True)
os.makedirs(trace_plot_path, exist_ok=True)



tz_gmt = utils.TZ_GMT()
tz_auger = utils.TZ_auger()
tz_gp13 = utils.TZ_GP13()



if site == 'gaa':
    tz = tz_auger



sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]
n_samples = 2048
fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]



## load run202 file6-9 for reference


file_2024 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run24/td002024_f000[6-8].root'

tadc_24 = uproot.concatenate({file_2024:'tadc',})
#trawv_24 = uproot.concatenate({file_2024:'trawvoltage'})





idu = 84
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_24, request, idu)
traces_np_24 = result1[:, 0, 0:3].to_numpy()
psd_2024 = filt.return_psd(traces_np_24, sample_freq)



plt.figure(435)
plt.clf()
plt.plot(fft_freq, psd_2024.mean(axis=0)[0], label='run2024 ch0')


plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('Frequency [MHz]')
plt.legend()
plt.tight_layout()





file_2025 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run25-28/td002025_f000*.root'
tadc_25 = uproot.concatenate({file_2025:'tadc',})
du_list_25 = utils.get_dulist(tadc_25)


idu = 184
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_25, request, idu)
traces_np_run25_du184 = result1[:, 0, 0:3].to_numpy()
psd_run25_du184 = filt.return_psd(traces_np_run25_du184, sample_freq)


idu = 183
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_25, request, idu)
traces_np_run25_du183 = result1[:, 0, 0:3].to_numpy()
psd_run25_du183 = filt.return_psd(traces_np_run25_du183, sample_freq)



file_2026 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run25-28/td002026_f000*.root'
tadc_26 = uproot.concatenate({file_2026:'tadc',})
du_list_26 = utils.get_dulist(tadc_26)




idu = 184
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_26, request, idu)
traces_np_run26_du184 = result1[:, 0, 0:3].to_numpy()
psd_run26_du184 = filt.return_psd(traces_np_run26_du184, sample_freq)


idu = 159
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_26, request, idu)
traces_np_run26_du159 = result1[:, 0, 0:3].to_numpy()
psd_run26_du159 = filt.return_psd(traces_np_run26_du159, sample_freq)

idu = 244
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_26, request, idu)
traces_np_run26_du244 = result1[:, 0, 0:3].to_numpy()
psd_run26_du244 = filt.return_psd(traces_np_run26_du244, sample_freq)







plt.plot(fft_freq, psd_run25_du184.mean(axis=0)[0], label='run25 du184 ch0')
plt.plot(fft_freq, psd_run25_du183.mean(axis=0)[0], label='run25 du183 ch0')

plt.plot(fft_freq, psd_run26_du184.mean(axis=0)[0], label='run26 du184 ch0')
plt.plot(fft_freq, psd_run26_du159.mean(axis=0)[0], label='run26 du159 ch0')
plt.plot(fft_freq, psd_run26_du244.mean(axis=0)[0], label='run26 du244 ch0')






file_2027 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run25-28/td002027_f000*.root'
tadc_27 = uproot.concatenate({file_2027:'tadc',})
du_list_27 = utils.get_dulist(tadc_27)





idu = 59
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_27, request, idu)
traces_np_run27_du59 = result1[:, 0, 0:3].to_numpy()
psd_run27_du59 = filt.return_psd(traces_np_run27_du59, sample_freq)

idu = 60
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_27, request, idu)
traces_np_run27_du60 = result1[:, 0, 0:3].to_numpy()
psd_run27_du60 = filt.return_psd(traces_np_run27_du60, sample_freq)


idu = 69
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_27, request, idu)
traces_np_run27_du69 = result1[:, 0, 0:3].to_numpy()
psd_run27_du69 = filt.return_psd(traces_np_run27_du69, sample_freq)



idu = 84
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_27, request, idu)
traces_np_run27_du84 = result1[:, 0, 0:3].to_numpy()
psd_run27_du84 = filt.return_psd(traces_np_run27_du84, sample_freq)

idu = 144
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_27, request, idu)
traces_np_run27_du144 = result1[:, 0, 0:3].to_numpy()
psd_run27_du144 = filt.return_psd(traces_np_run27_du144, sample_freq)




## plot du run 27
plt.figure(456, figsize=(15, 8))
plt.clf()

plt.plot(fft_freq, psd_2024.mean(axis=0)[0], 'k-', label='run2024 du84 ch0')

plt.plot(fft_freq, psd_run27_du59.mean(axis=0)[0], label='run27 du59 ch0')
plt.plot(fft_freq, psd_run27_du60.mean(axis=0)[0], label='run27 du60 ch0')
plt.plot(fft_freq, psd_run27_du69.mean(axis=0)[0], label='run27 du69 ch0')
plt.plot(fft_freq, psd_run27_du84.mean(axis=0)[0], label='run27 du84 ch0')
plt.plot(fft_freq, psd_run27_du144.mean(axis=0)[0], label='run27 du144 ch0')

plt.legend()

plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('Frequency [MHz]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'comparisation_run27_ch0.png'))




plt.figure(457, figsize=(15, 8))
plt.clf()

plt.plot(fft_freq, psd_2024.mean(axis=0)[1], 'k-', label='run2024 du84 ch1')

plt.plot(fft_freq, psd_run27_du59.mean(axis=0)[1], label='run27 du59 ch1')
plt.plot(fft_freq, psd_run27_du60.mean(axis=0)[1], label='run27 du60 ch1')
plt.plot(fft_freq, psd_run27_du69.mean(axis=0)[1], label='run27 du69 ch1')
plt.plot(fft_freq, psd_run27_du84.mean(axis=0)[1], label='run27 du84 ch1')
plt.plot(fft_freq, psd_run27_du144.mean(axis=0)[1], label='run27 du144 ch1')

plt.legend()

plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('Frequency [MHz]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'comparisation_run27_ch1.png'))




plt.figure(458, figsize=(15, 8))
plt.clf()

plt.plot(fft_freq, psd_2024.mean(axis=0)[2], 'k-', label='run2024 du84 ch2')

plt.plot(fft_freq, psd_run27_du59.mean(axis=0)[2], label='run27 du59 ch2')
plt.plot(fft_freq, psd_run27_du60.mean(axis=0)[2], label='run27 du60 ch2')
plt.plot(fft_freq, psd_run27_du69.mean(axis=0)[2], label='run27 du69 ch2')
plt.plot(fft_freq, psd_run27_du84.mean(axis=0)[2], label='run27 du84 ch2')
plt.plot(fft_freq, psd_run27_du144.mean(axis=0)[2], label='run27 du144 ch2')

plt.legend()

plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('Frequency [MHz]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'comparisation_run27_ch2.png'))














file_2028 = '/Users/ab212678/Documents/GRAND/data/auger/TD/run25-28/td002028_f000*.root'
tadc_28 = uproot.concatenate({file_2028:'tadc',})
du_list_28 = utils.get_dulist(tadc_28)




idu = 84
request = 'trace_ch'
result1, date_array1 = utils.get_column_for_given_du(tadc_28, request, idu)
traces_np_run28_du84 = result1[:, 0, 0:3].to_numpy()
psd_run28_du84 = filt.return_psd(traces_np_run28_du84, sample_freq)


plt.plot(fft_freq, psd_run28_du84.mean(axis=0)[0], label='run28 du84 ch0')





#### Plot 84 accross runs



plt.figure(4367, figsize=(15, 8))
plt.clf()
plt.plot(fft_freq, psd_2024.mean(axis=0)[0], label='run2024 du84 ch0')
plt.plot(fft_freq, psd_run25_du184.mean(axis=0)[0], label='run25 du184 ch0')
#plt.plot(fft_freq, psd_run26_du184.mean(axis=0)[0], label='run26 du184 ch0')
plt.plot(fft_freq, psd_run27_du84.mean(axis=0)[0], label='run27 du84 ch0')
plt.plot(fft_freq, psd_run28_du84.mean(axis=0)[0], label='run28 du84 ch0')
plt.legend()

plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('Frequency [MHz]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'comparisation_du84_184_ch0.png'))




plt.figure(4368, figsize=(15, 8))
plt.clf()
plt.plot(fft_freq, psd_2024.mean(axis=0)[1], label='run2024 du84 ch1')
plt.plot(fft_freq, psd_run25_du184.mean(axis=0)[1], label='run25 du184 ch1')
#plt.plot(fft_freq, psd_run26_du184.mean(axis=0)[1], label='run26 du184 ch1')
plt.plot(fft_freq, psd_run27_du84.mean(axis=0)[1], label='run27 du84 ch1')
plt.plot(fft_freq, psd_run28_du84.mean(axis=0)[1], label='run28 du84 ch1')
plt.legend()

plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('Frequency [MHz]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'comparisation_du84_184_ch1.png'))



plt.figure(4369, figsize=(15, 8))
plt.clf()
plt.plot(fft_freq, psd_2024.mean(axis=0)[2], label='run2024 du84 ch2')
plt.plot(fft_freq, psd_run25_du184.mean(axis=0)[2], label='run25 du184 ch2')
#plt.plot(fft_freq, psd_run26_du184.mean(axis=0)[2], label='run26 du184 ch2')
plt.plot(fft_freq, psd_run27_du84.mean(axis=0)[2], label='run27 du84 ch2')
plt.plot(fft_freq, psd_run28_du84.mean(axis=0)[2], label='run28 du84 ch2')
plt.legend()

plt.yscale('log')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('Frequency [MHz]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'comparisation_du84_184_ch2.png'))
