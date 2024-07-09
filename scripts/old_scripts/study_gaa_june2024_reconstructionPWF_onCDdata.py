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



site = 'gaa'



file_gaa = '/Users/ab212678/Documents/GRAND/data/auger/2024/04/gaa_20240427_224428_RUN003002_CD_phys.root'

plot_path = '/Users/ab212678/Documents/GRAND/data/plots/study_gaa_recons_june24/5du/{}'.format(os.path.splitext(os.path.basename(file_gaa))[0])
os.makedirs(plot_path, exist_ok=True)


file_dict_gaa= {file_gaa: "tadc"}
file_dict_gaa_trawv= {file_gaa: "trawvoltage"}

tadc_gaa = uproot.concatenate(file_dict_gaa)
trawv_gaa = uproot.concatenate(file_dict_gaa_trawv)


list_gaa = utils.get_dulist(tadc_gaa)


header_lonlat = 'du_id longitude_[deg] latitude_[deg] altitude_[m]'

with open('/Users/ab212678/Documents/GRAND/Codes/dc2_code/gaa_position_lonlat_april2024.txt', 'w') as f:
    f.write('{} \n'.format(header_lonlat))

    for idu in list_gaa:
        longitude = utils.get_column_for_given_du(trawv_gaa, 'gps_long', idu)[0][76].to_numpy()[0]
        latitude = utils.get_column_for_given_du(trawv_gaa, 'gps_lat', idu)[0][76].to_numpy()[0]
        altitude = utils.get_column_for_given_du(trawv_gaa, 'gps_alt', idu)[0][76].to_numpy()[0]
        f.write('{} {} {} {} \n'.format(int(idu), longitude, latitude, altitude))



for iii in range(len(tadc_gaa)):

    iev = iii

    tadc_ev = tadc_gaa[iev]
    trace_ = tadc_ev.trace_ch.to_numpy()
    event_number = tadc_ev.event_number
    title_str = 'gaa_20240427_224428_RUN003002_CD_phys, evid {}, evnum {}'.format(iev, event_number)
    du_list = tadc_ev.du_id

    if (len(np.unique(du_list)) == len(du_list)) * (len(du_list) == 5):
        print(iev, du_list, event_number)

        n_du = len(du_list)
        fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
        [axs[i].plot(trace_[i, 0,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
        [ax.legend(loc=1) for ax in axs]
        [axs[i].axvline(tadc_ev.trigger_position[i]) for i in range(len(tadc_ev.du_id))]
        [ax.set_ylabel('ADC counts') for ax in axs]
        axs[-1].set_xlabel('time bin')
        axs[0].set_title(title_str)
        plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnum{}_traces1.png'.format(iev, event_number)))

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
        axs[0].set_title(title_str + ' SN axis')
        plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnmu{}_traces2_SNaxis.png'.format(iev, event_number)))

        fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
        [axs[i].plot(timex[i], trace_[i, 1,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
        [ax.legend(loc=1) for ax in axs]
        axs[-1].set_xlabel('time [ns]')
        axs[0].set_title(title_str + ' EW axis')
        plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnum{}_traces2_EWaxis.png'.format(iev, event_number)))

        fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
        [axs[i].plot(timex[i], trace_[i, 2,], label='DU{}'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
        [ax.legend(loc=1) for ax in axs]
        axs[-1].set_xlabel('time [ns]')
        axs[0].set_title(title_str + ' Vert axis')
        plt.savefig(os.path.join(plot_path, 'gaa_CD_evid{}_evnum{}_traces2_Vertaxis.png'.format(iev, event_number)))

        fig, axs = plt.subplots(n_du, 1, sharex=True, sharey=True, figsize=(15, 8))
        [axs[i].plot(timex[i], np.abs(hilbert(trace_[i, 0,])), label='DU{} SN'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]
        [axs[i].plot(timex[i], np.abs(hilbert(trace_[i, 1,])), label='DU{} EW'.format(tadc_ev.du_id[i])) for i in range(len(tadc_ev.du_id))]

        i_max_x = [np.argmax(np.abs(hilbert(trace_[i, 0,]))) for i in range(len(tadc_ev.du_id))]
        i_max_y = [np.argmax(np.abs(hilbert(trace_[i, 1,]))) for i in range(len(tadc_ev.du_id))]
        du_id_ev = [tadc_ev.du_id[i] for i in range(len(tadc_ev.du_id)) ]
        t_max_x = [timex[i][i_max_x[i]] for  i in range(len(tadc_ev.du_id))]
        t_max_y = [timex[i][i_max_y[i]] for  i in range(len(tadc_ev.du_id))]

        data_ev = np.vstack([du_id_ev, t_max_x, t_max_y]).T

        fname = 'data_gaa_20240427_224428_RUN003002_CD_phys_evnum_{}.npy'.format(event_number)
        np.save(fname, data_ev)

        [ax.legend(loc=1) for ax in axs]
        axs[-1].set_xlabel('time [ns]')
        axs[0].set_title(title_str + ' SN axis')
        plt.savefig(os.path.join(plot_path, 'gaa_CD_ev{}_evnum{}_traces3_SNaxis.png'.format(iev, event_number)))





