import numpy as np
import os
import grand_psu_lib.utils.utils as utils
import glob


def get_gps_temp_from_du_run(npz_dir, idu, run_string):

    gps_temp_file_list = np.sort(glob.glob(os.path.join(npz_dir, 'gps_temp_du{}_{}*'.format(idu, run_string))))
    if len(gps_temp_file_list) == 0:
        return []
    else:
        gps_temp = []
        gps_temp = [np.load(f, allow_pickle=True) for f in gps_temp_file_list]
        gps_temp = np.squeeze(np.vstack(gps_temp))
        return gps_temp


def get_battery_level_from_du_run(npz_dir, idu, run_string):

    bl_file_list = np.sort(glob.glob(os.path.join(npz_dir, 'battery_level_du{}_{}*'.format(idu, run_string))))
    if len(bl_file_list) == 0:
        return []
    else:
        bl = []
        bl = [np.load(f, allow_pickle=True) for f in bl_file_list]
        bl = np.squeeze(np.vstack(bl))
        return bl


def get_date_list_from_du_run(npz_dir, idu, run_string, tz=utils.TZ_GMT()):

    date_file_list = np.sort(glob.glob(os.path.join(npz_dir, 'date_array_du{}_{}*'.format(idu, run_string))))
    if len(date_file_list) == 0:
        return []
    else:
        date_list = []
        for f in date_file_list:
            date_list += list(np.load(f, allow_pickle=True))
        date_list = [d.astimezone(tz) for d  in date_list]
        return date_list


def get_trace_from_du_run(npz_dir, idu, run_string):
    trace_adc_file_list = np.sort(glob.glob(os.path.join(npz_dir, 'trace_adc_du{}_{}*'.format(idu, run_string))))
    if len(trace_adc_file_list) == 0:
        return np.array([0])
    else:
        trace_adc = [np.load(f, allow_pickle=True) for f in trace_adc_file_list]
        trace_adc = np.vstack(trace_adc)[:,0,:,:]

        return trace_adc
