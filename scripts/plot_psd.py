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



data_path = '/Users/ab212678/Documents/GRAND/data/gp13/2024/10'

fname = 'GP13_20241005_210704_RUN127_UD_RAW_ChanXYZ_20dB_12DUs_109.root'

input_root_file = os.path.join(data_path, fname)




tadc = uproot.concatenate({input_root_file: 'tadc'})
trawv = uproot.concatenate({input_root_file: 'trawvoltage'})
du_list = utils.get_dulist(tadc)



sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]

n_samples = 1024

fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]
all_data_alldu = []
idu = du_list[0]

tp10, d1 = utils.get_column_for_given_du_gp13(tadc, 'trigger_pattern_10s', idu)
print('du={}, {} traces, {} are MD'.format(idu, len(tp10), np.sum(tp10)))
traces, date_list = utils.get_column_for_given_du_gp13(tadc, 'trace_ch', idu)

traces_MD_np = traces.to_numpy()[tp10]
date_arr = np.array(date_list)[tp10.to_numpy()[:, 0]]
date_arr = np.expand_dims(date_arr, axis=1)
mean_psd = filt.return_psd(traces_MD_np, 500).mean(axis=0)


plt.figure(1)
plt.clf()
plt.plot(fft_freq, mean_psd[0], label='ch 0')
plt.plot(fft_freq, mean_psd[1], label='ch 1')
plt.plot(fft_freq, mean_psd[2], label='ch 2')
plt.plot(fft_freq, mean_psd[3], label='ch 3')
plt.yscale('log')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2/MHz] ')
plt.legend()
plt.tight_layout()