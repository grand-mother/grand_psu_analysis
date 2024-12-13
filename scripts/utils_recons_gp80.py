
import uproot
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import grand_psu_lib.utils.utils as utils
import datetime
import scipy.signal as sg
from grand import Geodetic, GRANDCS
import wavefronts_pwf as wpwf
import wavefronts_swf as wswf

#coord_1043 = Geodetic(latitude=40.98757751, longitude=93.95916533, height=1214.79775)
#daq_room = array([  93.94177,   40.99434, 1262.376  ])

coord_daq = Geodetic(
    latitude=40.99434,
    longitude=93.94177,
    height=1262.376)


def get_DU_coord(lat, long, alt, obstime, origin=coord_daq):
    # From GPS to Cartisian coordinates
    geod = Geodetic(latitude=lat, longitude=long, height=alt)
    gcs = GRANDCS(geod, obstime=obstime, location=origin)
    return gcs


obstime_ref = '2024-10-31'


def find_coincs_in_udfile_withvalid_du34(input_root_file, n_DU_CD, obstime=obstime_ref, do_plots=False):

    c_eff = sp.constants.speed_of_light / 1.001

    print('working on file {}'.format(input_root_file))

    tadc = uproot.open(input_root_file)['tadc']
    trawv = uproot.open(input_root_file)['trawvoltage']

    du_list = utils.get_dulist(tadc)
    gps_status_all = np.unique(tadc['gps_status'].array().to_numpy())
    print('gps_status_all =', gps_status_all )

    gps_warning_all = np.unique(tadc['gps_warnings'].array().to_numpy())
    print('gps_warning_all =', gps_warning_all )
    n_du = len(du_list)

    geo_info = []
    # from the trawv tree get the lon, lat, alt of the du's
    for idu in du_list:
        #print(idu)
        if idu != 10:
            gps_long = utils.get_column_for_given_du(trawv, 'gps_long', idu)[0].to_numpy()/57.3*180/np.pi
            long_info = [gps_long.min(), gps_long.max(), gps_long.mean(), np.median(gps_long)]
            gps_lat = utils.get_column_for_given_du(trawv, 'gps_lat', idu)[0].to_numpy()/57.3*180/np.pi
            lat_info = [gps_lat.min(), gps_lat.max(), gps_lat.mean(), np.median(gps_lat)]
            gps_alt = utils.get_column_for_given_du(trawv, 'gps_alt', idu)[0].to_numpy()
            alt_info = [gps_alt.min(), gps_alt.max(), gps_alt.mean(), np.median(gps_alt)]
            geo_info.append([idu, long_info[-1], lat_info[-1], alt_info[-1]])

    geo_info = np.array(geo_info)
    n_du = geo_info.shape[0]
    geo_info_xyz = geo_info.copy()

    today_as_timestamp = datetime.datetime.timestamp(datetime.datetime.today())

    # gp13_pos1_all = np.loadtxt('/Users/ab212678/Documents/GRAND/data/commissioning/gp13/GP13_pos_machine_readable.txt', skiprows=1, usecols=[ 1, 2, 3])
    # gp13_pos1_lonlat = gp13_pos1_all[14:, :]

    # correspondance_id = {}
    # correspondance_id[1] = {"new": 1023, "old": 1013}
    # correspondance_id[2] = {"new": 1000, "old": 1019}
    # correspondance_id[3] = {"new": 1001, "old": 1041}
    # correspondance_id[4] = {"new": 1048, "old": 1020}
    # correspondance_id[5] = {"new": 1035, "old": 1031}
    # correspondance_id[6] = {"new": 1043, "old": 1075}
    # correspondance_id[7] = {"new": 1058, "old": 1072}
    # correspondance_id[8] = {"new": 1018, "old": 1085}
    # correspondance_id[9] = {"new": 1045, "old": 1032}
    # correspondance_id[10] = {"new": 1059, "old": 1017}
    # correspondance_id[11] = {"new": 1065, "old": 1029}
    # correspondance_id[12] = {"new": 1040, "old": 1011}
    # correspondance_id[13] = {"new": 1038, "old": 1071}

    # correspondance_id2  = {}
    # for k, v in correspondance_id.items():
    #     correspondance_id2[v["new"]] = {'id':k, "old":v["old"]}


    # geo_info_v2 = geo_info.copy()

    # for k, new_idu in enumerate(geo_info_v2[:, 0]):
    #     if new_idu!= 1034:
    #         geo_info_v2[k, 1:4] = gp13_pos1_lonlat[correspondance_id2[int(new_idu)]["id"]]


    for i in range(n_du):
        geo_info_xyz[i, 1:] = get_DU_coord(geo_info[i, 2], geo_info[i, 1], geo_info[i, 3], obstime_ref)[:, 0]

    # geo_info_xyz_v2 = geo_info_xyz.copy()

    # for i in range(n_du):
    #     geo_info_xyz_v2[i, 1:] = get_DU_coord(geo_info_v2[i, 2], geo_info_v2[i, 1], geo_info_v2[i, 3], obstime_ref)[:, 0]
    if not (1034 in du_list):
        print('du34 is not in this file')
    if (1034 in du_list) * (len(gps_status_all) == 1):
        distmat = np.array([[np.linalg.norm(geo_info_xyz[i, 1:4] - geo_info_xyz[j, 1:4])for i in range(n_du)] for j in range(n_du)])

        dmax = distmat.max()
        tmax = dmax / c_eff * 1e9  # in nsec

        du_sec = tadc['du_seconds'].array().to_numpy()[:, 0]
        delta_sec = du_sec - du_sec.min()
        delta_nsec = delta_sec*1e9 + tadc['du_nanoseconds'].array().to_numpy()[:, 0]


        cont = True
        if dmax > 100000:
            cont = False
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
                    else:
                        print('DUPLICATES !!!!!!!')
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
    else:
        coincs = []
        return coincs, geo_info_xyz



def find_coincs_in_udfile_no34condition(input_root_file, n_DU_CD, obstime_ref=obstime_ref, do_plots=False, correct_gps=False):


    if correct_gps:
        gps_factor = 1.0/57.3*180/np.pi
    else:
        gps_factor = 1

    c_eff = sp.constants.speed_of_light / 1.01

    print('working on file {}'.format(input_root_file))

    tadc = uproot.open(input_root_file)['tadc']
    trawv = uproot.open(input_root_file)['trawvoltage']

    du_list = utils.get_dulist(tadc)
    gps_status_all = np.unique(tadc['gps_status'].array().to_numpy())
    gps_warning_all = np.unique(tadc['gps_warnings'].array().to_numpy())
    
    n_du = len(du_list)

    today_as_timestamp = datetime.datetime.timestamp(datetime.datetime.today())
    geo_info = []
    # from the trawv tree get the lon, lat, alt of the du's
    for idu in du_list:
        if idu != 10:
            gps_long = utils.get_column_for_given_du(trawv, 'gps_long', idu)[0].to_numpy() * gps_factor
            long_info = [gps_long.min(), gps_long.max(), gps_long.mean(), np.median(gps_long)]
            gps_lat = utils.get_column_for_given_du(trawv, 'gps_lat', idu)[0].to_numpy() * gps_factor
            lat_info = [gps_lat.min(), gps_lat.max(), gps_lat.mean(), np.median(gps_lat)]
            gps_alt = utils.get_column_for_given_du(trawv, 'gps_alt', idu)[0].to_numpy()
            alt_info = [gps_alt.min(), gps_alt.max(), gps_alt.mean(), np.median(gps_alt)]
            geo_info.append([idu, long_info[-1], lat_info[-1], alt_info[-1]])

    geo_info = np.array(geo_info)
    n_du = geo_info.shape[0]
    geo_info_xyz = geo_info.copy()

    for i in range(n_du):
        geo_info_xyz[i, 1:] = get_DU_coord(geo_info[i, 2], geo_info[i, 1], geo_info[i, 3], obstime_ref)[:, 0]


    print('len(gps_status_all) = ', len(gps_status_all) )
    print('len(gps_warning_all) = ', len(gps_warning_all) )
    if (len(gps_status_all) == 1) * (len(gps_warning_all) < 4):
        distmat = np.array([[np.linalg.norm(geo_info_xyz[i, 1:4] - geo_info_xyz[j, 1:4])for i in range(n_du)] for j in range(n_du)])

        dmax = distmat.max()
        print('dmax=', dmax)
        tmax = dmax / c_eff * 1e9  # in nsec
        print('tmax =', tmax )

        du_sec = tadc['du_seconds'].array().to_numpy()[:, 0]
        delta_sec = du_sec - du_sec.min()
        delta_nsec = delta_sec*1e9 + tadc['du_nanoseconds'].array().to_numpy()[:, 0]


        cont = True
        if dmax > 100000:
            cont = False
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
                    else:
                        print('DUPLICATES !!!!!!!')
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
    else:
        coincs = []
        return coincs, geo_info_xyz


def get_data_from_coincs(coincs, geo_info_xyz, fname, output_dir='./',  do_recons=False):

    basename = os.path.basename(fname)
    output_path = os.path.join(output_dir, basename)

    tadc = uproot.open(fname)['tadc']
    tadc_du_ids = tadc['du_id'].array().to_numpy().squeeze()
    tadc_du_seconds = tadc['du_seconds'].array().to_numpy().squeeze()
    tadc_du_nanoseconds = tadc['du_nanoseconds'].array().to_numpy().squeeze()
    tadc_traces = tadc['trace_ch'].array().to_numpy().squeeze()

    def get_data_from_coinc(coinc, k, do_recons):
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
        du_times = du_seconds + 1e-9 * du_nanoseconds 

        xants = np.vstack([du_northing, du_westing, du_alt+0]).T
        tant1 = du_delta_times
        tant2 = du_hilb_times
        tant3 = du_times

        if 1034 in du_ids:
            i34 = np.where(du_ids == 1034)[0]
            allids = list(np.arange(len(du_ids)))
            allids.remove(i34)
            xants_ = xants[allids]
            tant1_ = tant1[allids]
            tant2_ = tant2[allids]

        else:
            xants_ = xants
            tant1_ = tant1
            tant2_ = tant2
        if do_recons:
            pwf_recons1 = wpwf.PWF_minimize_alternate_loss(xants_, tant1_, nr=1.000)
            pwf_recons2 = wpwf.PWF_minimize_alternate_loss(xants_, tant2_, nr=1.000)

            #print(pwf_recons1*180/np.pi)

            swf_recons1, chi2_1, ndof1, flag1 = wswf.get_SWF_fit_v3(xants_, tant1_, sigma_t=5e-9, initial_guess=[0, 0, 0], ncall=4100)
            swf_recons2, chi2_2, ndof2, flag2 = wswf.get_SWF_fit_v3(xants_, tant2_, sigma_t=5e-9, initial_guess=[0, 0, 0], ncall=4100)

            #print(swf_recons2)

        coinc_data_dir = output_path + '_recons'
        os.makedirs(coinc_data_dir, exist_ok=True)

        coinc_data_filename = os.path.join(coinc_data_dir, 'coinc_data_{}.npy'.format(k))
        coinc_data = np.vstack([du_ids, xants.T, tant1, tant2, tant3]).T
        np.save(coinc_data_filename, coinc_data)
        if do_recons:
            recons_data = np.hstack([pwf_recons1, pwf_recons2, swf_recons1, flag1, swf_recons2, flag2])
            recons_data_filename = os.path.join(coinc_data_dir, 'recons_data_{}.npy'.format(k))
            np.save(recons_data_filename, recons_data)
            return coinc_data, recons_data
        else:
            return coinc_data

    for k, coinc in enumerate(coincs):

        if do_recons:
            coinc_data, recons_data = get_data_from_coinc(coinc, k, do_recons=do_recons)
        else:
            coinc_data = get_data_from_coinc(coinc, k, do_recons=do_recons)


def get_data_from_coincs_beacon(coincs, geo_info_xyz, fname, output_dir='./',  do_recons=False):

    basename = os.path.basename(fname)
    output_path = os.path.join(output_dir, basename)

    tadc = uproot.open(fname)['tadc']
    tadc_du_ids = tadc['du_id'].array().to_numpy().squeeze()
    tadc_du_seconds = tadc['du_seconds'].array().to_numpy().squeeze()
    tadc_du_nanoseconds = tadc['du_nanoseconds'].array().to_numpy().squeeze()
    tadc_traces = tadc['trace_ch'].array().to_numpy().squeeze()

    def get_data_from_coinc(coinc, k, do_recons):
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
        bitmask_trig_pattern = [np.sum(np.array([0, 1, 2, 4]) * ttpc) for ttpc in tadc_trigger_pattern_ch]

        trigged_traces = [du_traces[i][tadc_trigger_pattern_ch[i]] for i in range(len(coinc))]
        abs_hilbert = [np.linalg.norm(sg.hilbert(trigged_traces[i]), axis=0) for i in range(len(coinc))]
        id_max_hilb = np.argmax(np.array(abs_hilbert), axis=1)

        du_delta_times = du_seconds - du_seconds.min() + 1e-9 * du_nanoseconds  # in seconds
        du_hilb_times = du_delta_times + 2e-9 * id_max_hilb  # in seconds
        du_times = du_seconds + 1e-9 * du_nanoseconds 

        xants = np.vstack([du_northing, du_westing, du_alt+0]).T
        tant1 = du_delta_times
        tant2 = du_hilb_times
        tant3 = du_times

        if 1034 in du_ids:
            i34 = np.where(du_ids == 1034)[0]
            allids = list(np.arange(len(du_ids)))
            allids.remove(i34)
            xants_ = xants[allids]
            tant1_ = tant1[allids]
            tant2_ = tant2[allids]

        else:
            xants_ = xants
            tant1_ = tant1
            tant2_ = tant2
        if do_recons:
            pwf_recons1 = wpwf.PWF_minimize_alternate_loss(xants_, tant1_, nr=1.000)
            pwf_recons2 = wpwf.PWF_minimize_alternate_loss(xants_, tant2_, nr=1.000)

            #print(pwf_recons1*180/np.pi)

            swf_recons1, chi2_1, ndof1, flag1 = wswf.get_SWF_fit_v3(xants_, tant1_, sigma_t=5e-9, initial_guess=[0, 0, 0], ncall=4100)
            swf_recons2, chi2_2, ndof2, flag2 = wswf.get_SWF_fit_v3(xants_, tant2_, sigma_t=5e-9, initial_guess=[0, 0, 0], ncall=4100)

            #print(swf_recons2)

        coinc_data_dir = output_path + '_recons'
        os.makedirs(coinc_data_dir, exist_ok=True)

        coinc_data_filename = os.path.join(coinc_data_dir, 'coinc_data_{}.npy'.format(k))
        coinc_data = np.vstack([du_ids, xants.T, tant1, tant2, tant3, bitmask_trig_pattern]).T
        np.save(coinc_data_filename, coinc_data)
        if do_recons:
            recons_data = np.hstack([pwf_recons1, pwf_recons2, swf_recons1, flag1, swf_recons2, flag2])
            recons_data_filename = os.path.join(coinc_data_dir, 'recons_data_{}.npy'.format(k))
            np.save(recons_data_filename, recons_data)
            return coinc_data, recons_data
        else:
            return coinc_data

    for k, coinc in enumerate(coincs):

        if do_recons:
            coinc_data, recons_data = get_data_from_coinc(coinc, k, do_recons=do_recons)
        else:
            coinc_data = get_data_from_coinc(coinc, k, do_recons=do_recons)


def get_airplane_track(csv_file, obstime):
    airplane = np.loadtxt(csv_file, skiprows=1, delimiter=',', usecols=(0, 3, 4, 5), dtype=str)

    ap_ts = np.array([int(ap[0]) for ap in airplane])
    ap_lat = np.array([np.float32(ap[1][1:]) for ap in airplane])
    ap_long = np.array([np.float32(ap[2][:-1]) for ap in airplane])

    ap_alt = np.array([int(ap[3])*(0.3048) for ap in airplane])

    ap_xyz = get_DU_coord(ap_lat, ap_long, ap_alt, obstime)
    idd = np.where(np.linalg.norm(ap_xyz, axis=0)< 200000)[0]
    return ap_xyz[:, idd], ap_ts[idd]
