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


site = 'gp13'

plot_path = '/Users/ab212678/Documents/GRAND/data/gp13/mar24/plots/'
os.makedirs(plot_path, exist_ok=True)




file125 = '/Users/ab212678/Documents/GRAND/data/gp13/mars24/RUN125/GP13_20240312_065733_RUN125_MD_RAW_10s_ChanXYZ_20dB_11DUs_test_330_dat.root'
file129 = '/Users/ab212678/Documents/GRAND/data/gp13/mars24/RUN129/GP13_20240318_075843_RUN129_MD_RAW_10s_ChanXYZ_20dB_DU10_DU13_DU85_4096points_test_028_dat.root'

file_dict_125 = {file125: "tadc"}
file_dict_129 = {file129: "tadc"}


tadc125 = uproot.concatenate(file_dict_125)
tadc129 = uproot.concatenate(file_dict_129)

list_125 = utils.get_dulist(tadc125)
list_129 = utils.get_dulist(tadc129)

traces125, _ = utils.get_column_for_given_du_gp13(tadc125, 'trace_ch', 1010)
traces125 = traces125.to_numpy()
traces129, _ = utils.get_column_for_given_du_gp13(tadc129, 'trace_ch', 1010)
traces129 = traces129.to_numpy()

traces129 = traces129[:, :, :, 0:1024]

psd125 = filt.return_psd(traces125, 500)
psd129 = filt.return_psd(traces129, 500)

sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]

n_samples125 = 1024
n_samples129 = 1024

fft_freq_125 = np.fft.rfftfreq(n_samples125) * sample_freq  # [MHz]
fft_freq_129 = np.fft.rfftfreq(n_samples129) * sample_freq  # [MHz]

plt.figure(23)
plt.clf()
plt.plot(fft_freq_125, psd125.mean(axis=0)[0, 1], label='RUN125 x-axis')
plt.plot(fft_freq_129, psd129.mean(axis=0)[0, 1], label='RUN129 x-axis')
plt.yscale('log')
plt.ylabel('PSD [ADC^2/MHz] ')
plt.xlabel('Frequency [MHz]')
plt.legend(loc=0)
plt.savefig('comp_gp13_run125_run129.png')


