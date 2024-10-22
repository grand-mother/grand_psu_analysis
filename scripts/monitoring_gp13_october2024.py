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



def make_reducedMDdata(input_root_file, output_dir):

    base_input_file = os.path.basename(input_root_file)

    tadc = uproot.concatenate({input_root_file: 'tadc'})
    trawv = uproot.concatenate({input_root_file: 'trawvoltage'})
    du_list = utils.get_dulist(tadc)



    sample_freq = 500   # [MHz]
    sample_period = 1/sample_freq # [us]

    n_samples = 1024

    fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]
    all_data_alldu = []

    for idu in du_list:
        all_data = []
        tp10, d1 = utils.get_column_for_given_du_gp13(tadc, 'trigger_pattern_10s', idu)
        print('du={}, {} traces, {} are MD'.format(idu, len(tp10), np.sum(tp10)))
        traces, date_list = utils.get_column_for_given_du_gp13(tadc, 'trace_ch', idu)

        traces_MD_np = traces.to_numpy()[tp10]
        date_arr = np.array(date_list)[tp10.to_numpy()[:, 0]]
        date_arr = np.expand_dims(date_arr, axis=1)
        mean_psd = filt.return_psd(traces_MD_np, 500).mean(axis=0)

        gps_temp, d2 = utils.get_column_for_given_du_gp13(trawv, 'gps_temp', idu)
        gps_temp = np.array(gps_temp)[tp10.to_numpy()[:, 0]]

        du_tab = gps_temp.copy() * 0 + idu
        all_data.append(date_arr)
        all_data.append(du_tab)
        all_data.append(gps_temp)
        all_data.append(traces_MD_np.mean(axis=2))
        all_data.append(traces_MD_np.std(axis=2))

        all_data = np.hstack(all_data)



        # plt.figure(idu)
        # plt.clf()
        # plt.plot(fft_freq, mean_psd[0], label='ch 0 ')
        # plt.plot(fft_freq, mean_psd[1], label='ch 1 ')
        # plt.plot(fft_freq, mean_psd[2], label='ch 2 ')
        # plt.plot(fft_freq, mean_psd[3], label='ch 3 ')
        # plt.xlabel('Frequency [MHz]')
        # plt.ylabel('PSD [ADC^2/MHz] ')
        # plt.title('Mean PSD of MD traces for du {}'.format(idu))
        # plt.yscale('log')
        # plt.legend()
        # plt.tight_layout()

        # utils.make_joint_plot4d(traces_MD_np, date_arr, 'mean', idu, './toto_mean_{}.png'.format(idu), tadc_or_voltage='tadc', tz=utils.TZ_GP13())
        # utils.make_joint_plot4d(traces_MD_np, date_arr, 'std', idu, './toto_std_{}.png'.format(idu), tadc_or_voltage='tadc', tz=utils.TZ_GP13())
        all_data_alldu.append(all_data)

    all_data_alldu = np.vstack(all_data_alldu)

    f1_sp = base_input_file.split('_')
    filename = f1_sp[0] + "_" + f1_sp[1] + "_" + f1_sp[2] + "_reducedMDdata.npy"
    filename = os.path.join(output_dir, filename)
    np.save(filename, all_data_alldu)



def str2bool(s):
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    ap = argparse.ArgumentParser(description='Open hourly GP13 UD files and extact summary .npy for further processing')
    ap.add_argument(
        "--dir_path",
        type=str,
        required=True,
        help="path of the root file directory"
    )
    ap.add_argument(
        "--glob_pattern",
        type=str,
        required=True,
        help="string sent to glob to select the files to work on"
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory"
    )
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    output_dir = args.output_dir
    dir_path = args.dir_path
    glob_pattern = args.glob_pattern

    data_path = os.path.normpath(dir_path) + '/'

    os.makedirs(output_dir, exist_ok=True)
    for file in glob.glob(data_path + glob_pattern):
        make_reducedMDdata(file, output_dir)
