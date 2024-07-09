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


filegaa = 'gaa_20240321_093928_RUN003002_CD_phys.bin.root'
filegaa = 'gaa_20240320_125935_RUN003002_CD_phys.bin.root'
filegaa = 'gaa_20240319_121904_RUN003002_CD_phys.bin.root'

filegaa = "gaa_20240410_121220_RUN003002_CD_phys.root"
file = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/{}'.format(filegaa)

tadc = uproot.concatenate({file:'tadc'})
trawv = uproot.concatenate({file:'trawvoltage'})

du_list = utils.get_dulist(tadc)


sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]

n_samples = len(tadc.trace_ch[0][0][0])
fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]

tz_gmt = utils.TZ_GMT()
tz_auger = utils.TZ_auger()
tz_gp13 = utils.TZ_GP13()

site = 'gaa'

if site == 'gaa':
    tz = tz_auger


plot_path = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/plots_new_gaa/{}'.format(filegaa)
os.makedirs(plot_path, exist_ok=True)






n_events = len(tadc)

n_du_per_event = []
for i in range(n_events):
    n_du_per_event.append(len(tadc[i].event_id))

n_du_per_event = np.array(n_du_per_event)

plt.figure()
plt.clf()
plt.hist(n_du_per_event)




for n_coin in range(4, 7):
    ids_coin = np.where(np.array(n_du_per_event)==n_coin)[0]
    for id_coin in ids_coin:
        plot_path2 = os.path.join(plot_path, 'CD{}'.format(n_coin), 'id_{}'.format(id_coin))
        os.makedirs(plot_path2, exist_ok=True)


        t_coin = tadc[id_coin]
        print(t_coin.du_id)
        for k, idu in enumerate(t_coin.du_id):
            utils.plot_trace_and_psd(
                t_coin.trace_ch.to_numpy()[k],
                'DU {}'.format(idu),
                os.path.join(plot_path2, './id{}_{}_du{}.png'.format(id_coin,k, idu)),
                tadc_or_voltage='tadc'
            )



du_list = utils.get_dulist(tadc)


for idu in du_list:

    request = 'trace_ch'
    result, date_array_trace = utils.get_column_for_given_du(tadc, request, idu)

    yrs = [dt.year for dt in date_array_trace]
    days = [dt.day for dt in date_array_trace]
    hrs =  np.array([dt.hour for dt in date_array_trace])


    id = np.where(
        (np.array(yrs)> 2023)
    )[0]
    traces_np = result[:, 0, 0:3].to_numpy()[id]

    fft_freq = np.fft.rfftfreq(traces_np.shape[-1]) * 500  # [MHz]
   
    psd = filt.return_psd(traces_np, 500)
    plt.figure(567)
    plt.clf()
    plt.plot(fft_freq, psd.mean(axis=0)[0], label='{}'.format(idu))

    plt.figure(568)
    plt.clf()
    plt.plot(fft_freq, psd.mean(axis=0)[1], label='{}'.format(idu))

    plt.figure(569)
    plt.clf()
    plt.plot(fft_freq, psd.mean(axis=0)[2], label='{}'.format(idu))

    plt.figure(567)
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)
    #plt.title('Mean PSD x-axis for March 9th {}h-{}h'.format(hmin-3, hmax-3))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('PSD [ADC^2 / MHz]')
    plt.legend(ncol=2, loc=0)
    plt.savefig(os.path.join(plot_path,  "psd_mean_chX_{}.png".format(idu)))

    plt.figure(568)
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)
    #plt.title('Mean PSD y-axis for March 9th {}h-{}h'.format(hmin-3, hmax-3))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('PSD [ADC^2 / MHz]')
    plt.legend(ncol=2, loc=0)
    plt.savefig(os.path.join(plot_path, "psd_mean_chY_{}.png".format(idu)))

    plt.figure(569)
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)
    #plt.title('Mean PSD z-axis for March 9th {}h-{}h'.format(hmin-3, hmax-3))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('PSD [ADC^2 / MHz]')
    plt.legend(ncol=2, loc=0)
    plt.savefig(os.path.join(plot_path, "psd_mean_chZ_{}.png".format(idu)))

