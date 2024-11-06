
import uproot
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import grand_psu_lib.utils.utils as utils
import datetime
import scipy.signal as sg
from grand import Geodetic, GRANDCS
coord_1043 = Geodetic(latitude=40.98757751, longitude=93.95916533, height=1214.79775)


def get_DU_coord(lat, long, alt, obstime, origin=coord_1043):
    # From GPS to Cartisian coordinates
    geod = Geodetic(latitude=lat, longitude=long, height=alt)
    gcs = GRANDCS(geod, obstime=obstime, location=origin)
    return gcs


def find_coincs_in_udfile(input_root_file, n_DU_CD, do_plots=False):

    c_eff = sp.constants.speed_of_light / 1.001

    print('working on file {}'.format(input_root_file))

    tadc = uproot.open(input_root_file)['tadc']
    trawv = uproot.open(input_root_file)['trawvoltage']

    du_list = utils.get_dulist(tadc)
    n_du = len(du_list)

    today_as_timestamp = datetime.datetime.timestamp(datetime.datetime.today())
    geo_info = []
    # from the trawv tree get the lon, lat, alt of the du's
    for idu in du_list:
        print(idu)
        if idu != 10:
            gps_long = utils.get_column_for_given_du(trawv, 'gps_long', idu)[0].to_numpy()
            long_info = [gps_long.min(), gps_long.max(), gps_long.mean(), np.median(gps_long)]
            gps_lat = utils.get_column_for_given_du(trawv, 'gps_lat', idu)[0].to_numpy()
            lat_info = [gps_lat.min(), gps_lat.max(), gps_lat.mean(), np.median(gps_lat)]
            gps_alt = utils.get_column_for_given_du(trawv, 'gps_alt', idu)[0].to_numpy()
            alt_info = [gps_alt.min(), gps_alt.max(), gps_alt.mean(), np.median(gps_alt)]
            geo_info.append([idu, long_info[-1], lat_info[-1], alt_info[-1]])

    geo_info = np.array(geo_info)
    n_du = geo_info.shape[0]
    geo_info_xyz = geo_info.copy()
    for i in range(n_du):
        geo_info_xyz[i, 1:] = get_DU_coord(geo_info[i, 2], geo_info[i, 1], geo_info[i, 3], str(datetime.datetime.utcfromtimestamp(today_as_timestamp))[:10])[:, 0]

    distmat = np.array([[np.linalg.norm(geo_info_xyz[i, 1:4] - geo_info_xyz[j, 1:4])for i in range(n_du)] for j in range(n_du)])

    dmax = distmat.max()
    tmax = dmax / c_eff * 1e9  # in nsec

    du_sec = tadc['du_seconds'].array().to_numpy()[:, 0]
    delta_sec = du_sec - du_sec.min()
    delta_nsec = delta_sec*1e9 + tadc['du_nanoseconds'].array().to_numpy()[:, 0]

    cont = True
    i = 0
    k = 0

    i_to_avoid = list(np.where(tadc['trigger_pattern_10s'].array().to_numpy()[:, 0] == True)[0])+ list(np.where(du_sec > today_as_timestamp)[0])
    i_to_avoid = list(np.unique(i_to_avoid))
    coincs = []
    n_events_in_tadc = len(tadc['event_number'].array())
    while (cont):
        if i not in i_to_avoid:
            k += 1
            t0 = delta_nsec[i]
            id_close_in_time = np.where(np.abs(delta_nsec - t0) < tmax)[0]
            if len(id_close_in_time) >= n_DU_CD:
                # there might be a CD here. make sure that those trigger times correspond to different du'set
                [i_to_avoid.append(idcl) for idcl in id_close_in_time]
                _du_ids = [tadc['du_id'].array()[idcl] for idcl in id_close_in_time]
                if len(_du_ids) == len(np.unique(_du_ids)):
                    _event_numbers = [tadc['event_number'].array()[idcl] for idcl in id_close_in_time]
                    coincs.append(_event_numbers)
                    print('There is one coinc with {}dus!'.format(len(_du_ids)))
                pass
            else:
                if len(id_close_in_time) >= 2:
                    #print('not enough DU for this coincindence {} (only {})'.format(i, len(id_close_in_time)))
                    [i_to_avoid.append(idcl) for idcl in id_close_in_time]
        i += 1
        if i >= n_events_in_tadc:
            cont = False

    if do_plots:
        plot_path = input_root_file + '_plots'
        os.makedirs(plot_path, exist_ok=True)
        plt.figure()
        plt.plot(-geo_info_xyz[:, 2], geo_info_xyz[:, 1], 'k.')
        for ll in geo_info_xyz:
            plt.text(-ll[2], ll[1], '{}'.format(int(ll[0])))
        plt.ylabel('Northing [m]')
        plt.xlabel('Easting [m]')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'geometry.png'))

    return coincs, geo_info_xyz


def get_data_from_coincs(coincs, geo_info_xyz, fname, output_dir='./'):

    basename = os.path.basename(fname)
    output_path = os.path.join(output_dir, basename)

    tadc = uproot.open(fname)['tadc']
    tadc_du_ids = tadc['du_id'].array().to_numpy().squeeze()
    tadc_du_seconds = tadc['du_seconds'].array().to_numpy().squeeze()
    tadc_du_nanoseconds = tadc['du_nanoseconds'].array().to_numpy().squeeze()
    tadc_traces = tadc['trace_ch'].array().to_numpy().squeeze()

    def get_data_from_coinc(coinc, k):
        du_ids = tadc_du_ids[coinc]
        idus = [np.where(geo_info_xyz[:, 0] == idu)[0][0]for idu in du_ids]
        du_northing = geo_info_xyz[idus, 1]
        du_westing = geo_info_xyz[idus, 2]
        du_alt = geo_info_xyz[idus, 3]
        du_seconds = tadc_du_seconds[coinc]
        du_nanoseconds = tadc_du_nanoseconds[coinc]
        du_traces = tadc_traces[coinc]

        tadc_trigger_pattern_ch = tadc['trigger_pattern_ch'].array().to_numpy().squeeze()
        tadc_trigger_pattern_ch = [tadc_trigger_pattern_ch[c] for c in coinc]

        trigged_traces = [du_traces[i][tadc_trigger_pattern_ch[i]] for i in range(len(coinc))]
        abs_hilbert = [np.linalg.norm(sg.hilbert(trigged_traces[i]), axis=0) for i in range(len(coinc))]
        id_max_hilb = np.argmax(np.array(abs_hilbert), axis=1)

        du_delta_times = du_seconds - du_seconds.min() + 1e-9 * du_nanoseconds  # in seconds
        du_hilb_times = du_delta_times + 2e-9 * id_max_hilb  # in seconds

        xants = np.vstack([du_northing, du_westing, du_alt+0]).T
        tant1 = du_delta_times  # du_nanoseconds
        tant2 = du_hilb_times

        coinc_data_dir = output_path + '_recons'
        os.makedirs(coinc_data_dir, exist_ok=True)

        coinc_data_filename = os.path.join(coinc_data_dir, 'coinc_data_{}.npy'.format(k))
        coinc_data = np.vstack([du_ids, xants.T, tant1, tant2]).T
        np.save(coinc_data_filename, coinc_data)

    for k, coinc in enumerate(coincs):
        print(coinc)
        get_data_from_coinc(coinc, k)
