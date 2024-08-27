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


from scipy.signal import butter, sosfilt, sosfiltfilt, hilbert


from grand import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
    topography,
)
from grand import ECEF, Geodetic, GRANDCS, LTP

from matplotlib import rc
rc('font', size = 20.0)


"""
Script used to extract the CD event in the CD files for the coincidence analisis of GAA (e.g planes)
"""




def extract_and_plot_CD_data(tadc, trawv, n_CD_du, do_plots_, recons_data_path_, plot_path_):

    for iii in range(len(tadc_gaa)):
        if do_plots_:
            do_plots = True
            if n_CD_du >= 5:
                random_rate = 1
            else:
                random_rate = 0.3
            if np.random.rand(1) > random_rate:
                do_plots = False

        iev = iii

        tadc_ev = tadc_gaa[iev]
        trawv_ev = trawv_gaa[iev]

        du_list = tadc_ev.du_id

        if (len(np.unique(du_list)) == len(du_list)) * (len(du_list) == n_CD_du) * (144 not in du_list) * (69 not in du_list):
            savethisone = False
            plot_path = os.path.join(plot_path_, '{}du'.format(int(n_CD_du)))
            recons_data_path = os.path.join(recons_data_path_, '{}du'.format(int(n_CD_du)))
            os.makedirs(plot_path, exist_ok=True)
            os.makedirs(recons_data_path, exist_ok=True)

            event_number = tadc_ev.event_number
            print(iev, du_list, event_number)
            trace_ = tadc_ev.trace_ch.to_numpy()
            trace_stdx = trace_[:, 0].std(axis=1)
            max_x = trace_[:, 0].max(axis=1)
            if np.min(max_x / trace_stdx) > 3:
     
                title_str = '{}, evid {}, evnum {}'.format(file_base, iev, event_number)

                n_du = len(du_list)
                if do_plots:
                    fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
                    [axs[i].plot(trace_[i, 0,], label='DU{} SN'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
                    [ax.legend(loc=1) for ax in axs]
                    [axs[i].axvline(tadc_ev.trigger_position[i]) for i in range(len(tadc_ev.du_id))]
                    [ax.set_ylabel('ADC counts') for ax in axs]
                    axs[-1].set_xlabel('time bin')
                    axs[0].set_title(title_str)
                    plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnum{}_traces1.png'.format(iev, event_number)))
                    plt.close(fig)
                t1 = 1/(500e6) * np.arange(1024) * 1e9
                ev_second = tadc_ev.time_seconds
                ev_du_sec = tadc_ev.du_seconds.to_numpy()
                ev_du_nanosec = tadc_ev.du_nanoseconds.to_numpy()
                ev_du_dnanosec = ev_du_nanosec - tadc_ev.time_nanoseconds  ## offset in nsec wrt to the fisrt du. so the first du has dnano = 0
                ev_tri_pos = tadc_ev.trigger_position.to_numpy()
                timex = []
                [timex.append(t1 - t1[ev_tri_pos[i]]+ ev_du_dnanosec[i] ) for i in range(len(tadc_ev.du_id))]
                if do_plots:
                    fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
                    plt.xlabel("time [ns]")
                    fig.supylabel("ADC counts")
                    [axs[i].plot(timex[i], trace_[i, 0,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
                    [axs[i].plot(timex[i], trace_[i, 0,]*0 + 3*trace_[i, 0,].std(), 'r-') for i in range(len(tadc_ev.du_id))]
                    [ax.legend(loc=1) for ax in axs]
                    #[ax.set_ylabel('ADC counts') for ax in axs]
                    #axs[-1].set_xlabel('time [ns]')
                    axs[0].set_title(title_str + ' SN axis')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnmu{}_traces2_SNaxis.png'.format(iev, event_number)))
                    plt.close(fig)

                    fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
                    [axs[i].plot(timex[i], trace_[i, 1,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
                    [ax.legend(loc=1) for ax in axs]
                    axs[-1].set_xlabel('time [ns]')
                    axs[0].set_title(title_str + ' EW axis')
                    plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnum{}_traces2_EWaxis.png'.format(iev, event_number)))
                    plt.close(fig)

                    fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
                    [axs[i].plot(timex[i], trace_[i, 2,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
                    [ax.legend(loc=1) for ax in axs]
                    axs[-1].set_xlabel('time [ns]')
                    axs[0].set_title(title_str + ' Vert axis')
                    plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnum{}_traces2_Vertaxis.png'.format(iev, event_number)))
                    plt.close(fig)

                    fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
                    [axs[i].plot(timex[i], np.abs(hilbert(trace_[i, 0,])), label='DU{} SN'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
                    [axs[i].plot(timex[i], np.abs(hilbert(trace_[i, 1,])), label='DU{} EW'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
                    [ax.legend(loc=1) for ax in axs]
                    axs[-1].set_xlabel('time [ns]')
                    axs[0].set_title(title_str + ' SN axis')
                    plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnum{}_traces3_SNaxis.png'.format(iev, event_number)))
                    plt.close(fig)


                i_max_x = [np.argmax(np.abs(hilbert(trace_[i, 0,]))) for i in range(len(tadc_ev.du_id))]
                i_max_y = [np.argmax(np.abs(hilbert(trace_[i, 1,]))) for i in range(len(tadc_ev.du_id))]
                du_id_ev = [tadc_ev.du_id[i] for i in range(len(tadc_ev.du_id)) ]
                du_gps_lat = [trawv_ev.gps_lat[i] for i in range(len(tadc_ev.du_id))]
                du_gps_long = [trawv_ev.gps_long[i] for i in range(len(tadc_ev.du_id))]
                du_gps_alt = [trawv_ev.gps_alt[i] for i in range(len(tadc_ev.du_id))]

                t_max_x = [timex[i][i_max_x[i]] for  i in range(len(tadc_ev.du_id))]
                t_max_y = [timex[i][i_max_y[i]] for  i in range(len(tadc_ev.du_id))]

                data_ev = np.vstack([du_id_ev, du_gps_lat, du_gps_long, du_gps_alt, ev_du_sec, t_max_x, t_max_y]).T

                fname = os.path.join(recons_data_path, 'recons_data_evnum_{}.npy'.format(event_number))
                fname_trace = os.path.join(recons_data_path, 'trace_data_evnum_{}.npy'.format(event_number))
                np.save(fname, data_ev)
                np.save(fname_trace, trace_)





site = 'gaa'

file_list = glob.glob('/Users/ab212678/Documents/GRAND/data/auger/2024/08/*CD*.root')

output_path = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons/august2024_v1/'

for file_gaa in file_list:

    print('working of file ', file_gaa)
    #file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/gaa_20240427_224428_RUN003002_CD_phys.root'

    file_base = os.path.splitext(os.path.basename(file_gaa))[0]

    plot_path_ = os.path.join(output_path, '{}/plots/'.format(file_base))
    recons_data_path_ = os.path.join(output_path, '{}/recons_data/'.format(file_base))
    os.makedirs(plot_path_, exist_ok=True)

    do_plots_ = True

    file_dict_gaa = {file_gaa: "tadc"}
    file_dict_gaa_trawv = {file_gaa: "trawvoltage"}

    tadc_gaa = uproot.concatenate(file_dict_gaa)
    trawv_gaa = uproot.concatenate(file_dict_gaa_trawv)

    list_gaa = utils.get_dulist(tadc_gaa)
    #for n_CD_du in [3, 4, 5, 6, 7, 8, 9, 10]:
    for n_CD_du in [4, 5, 6, 7, 8, 9, 10]:
    
        # n_CD_du = 5
        extract_and_plot_CD_data(tadc_gaa, trawv_gaa, n_CD_du, do_plots_, recons_data_path_, plot_path_)