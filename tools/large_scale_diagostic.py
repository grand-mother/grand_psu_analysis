import sys
sys.path.append('./')
import uproot
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import grand_psu_lib.utils.utils as utils
import argparse
import glob
import datetime


def str2bool(s):
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    ap = argparse.ArgumentParser(description='Run general analysis of one grand@auger root file')
    ap.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="root_file"
    )
    ap.add_argument(
        "--plot_path",
        type=str,
        required=True,
        help="plot path"
    )
    ap.add_argument(
        "--site",
        type=str,
        required=True,
        help="must be either gaa or gp13"
    )
    ap.add_argument(
        "--do_fourier_vs_time",
        type=str2bool,
        default=False,
        help="only plot masks [False]"
    )
    ap.add_argument(
        "--du_list",
        nargs="+",
        default=None,
        type=int,
        help="input DU list"
    )
    ap.add_argument(
        "--base",
        type=str,
        default='many_files',
        help="plot basename in case of more than one file"
    )
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    site = args.site
    if ((site != 'gaa') and (site != 'gp13')):
        print('site must be either gaa or gp13')    
        exit()

    file_str = args.file_path

    files = np.sort(glob.glob(file_str))

    file_dict_tadc = {}
    for fi in files:
        file_dict_tadc[fi] = 'tadc'

    file_dict_trawv = {}
    for fi in files:
        file_dict_trawv[fi] = 'trawvoltage'

    if len(files) == 1:
        base = os.path.basename(files[0])
    else:
        base = args.base

    plot_path = os.path.join(args.plot_path, '{}'.format(base))
    trace_plot_path = os.path.join(plot_path, 'random_traces_and_their_spectra')

    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(trace_plot_path, exist_ok=True)


    tadc = uproot.concatenate(file_dict_tadc)
    trawv = uproot.concatenate(file_dict_trawv)


    if args.du_list is None:
        du_list = utils.get_dulist(tadc)
    else:
        du_list = args.du_list

    # The following lines are hard coded. To be modified if those parameter vary

    sample_freq = 500   # [MHz]
    sample_period = 1/sample_freq # [us]

    n_samples = len(tadc.trace_ch[0][0][0])
    fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]

    tz_gmt = utils.TZ_GMT()
    tz_auger = utils.TZ_auger()
    tz_gp13 = utils.TZ_GP13()


    if site == 'gaa':
        tz = tz_auger


    if site == 'gp13':
        tz = tz_gp13
        duid = tadc.du_id.to_numpy()
        gps_time = tadc.gps_time.to_numpy().squeeze()
        battery_level = trawv.battery_level.to_numpy().squeeze()
        gps_temp = trawv.gps_temp.to_numpy().squeeze()
        uti, uti_index, uti_inverse, uti_counts = np.unique(
            tadc.gps_time.to_numpy(), return_index=True, return_inverse=True, return_counts=True)
        n_event = uti_inverse[-1]


    # Plot of battery level vs time
    do_battery_plot = True

    if do_battery_plot:
        request = "battery_level"

        fig, axs = plt.subplots()
        axs.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=tz))
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %Hh%M', tz=tz))

        for idx in du_list:
            if site == 'gaa':
                bl_, date_array = utils.get_column_for_given_du(trawv, request, idx)
                axs.plot(date_array, bl_, '.', ms='2', label='du{}'.format(idx))
                axs.grid()
  
            if site == 'gp13':
                idd = np.where(duid == idx)[0]
                gps_ti_idx = gps_time[idd]
                date_idx = [datetime.datetime.fromtimestamp(dt, tz=tz_gmt) for dt in gps_ti_idx]
                bl_idx = battery_level[idd]
                gps_temp_idx = gps_temp[idd]
                axs.plot(date_idx, bl_idx, '.', ms='2', label='du{}'.format(idx))

        for label in axs.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        plt.legend(loc=0, ncol=2)
        plt.title(request)
        plt.xlabel('Time [{}]'.format(tz.tzname()))
        plt.ylabel('Battery level [V]')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, '{}_batterylevel_vs_time.png'.format(base)))


    # Plot of the mean of the PSD in times slices
    # TO DO: checke the tz for the times slices. 
    nb_time_slices = 4
    if site == 'gaa':
        utils.plot_mean_psd_time_sliced(tadc, nb_time_slices, plot_path, du_list=du_list)
    if site == 'gp13':
        utils.plot_mean_psd_time_sliced_gp13(tadc, nb_time_slices, plot_path, du_list=du_list)



    # Plot fourier_vs_time plot. Those take time and space on disk so by default they are deactivated
    do_fourier_vs_time = args.do_fourier_vs_time
    if do_fourier_vs_time:
        for idx in du_list:
            if site == 'gaa':
                request = 'trace_ch'
                result, date_array_trace = utils.get_column_for_given_du(tadc, request, idx)
                traces_np = result[:, 0, 0:3].to_numpy()
                plot_lines = True

                #fft = np.fft.rfft(traces_np)
                date_array = date_array_trace 

            if site == 'gp13':
                idd = np.where(duid == idx)[0]

                gps_ti_idx = gps_time[idd]
                date_idx = [datetime.datetime.fromtimestamp(dt, tz=tz_gmt) for dt in gps_ti_idx]
                bl_idx = battery_level[idd]
                gps_temp_idx = gps_temp[idd]
                traces_np = tadc.trace_ch.to_numpy()[idd, 0, 1:4]
                date_array = date_idx
                plot_lines = False

            for i in range(0, 3):
                if site == 'gaa':
                    id_ch = i
                if site == 'gp13':
                    id_ch = i+1

                plot_filename = os.path.join(plot_path, 'fourier_vs_time_du{}_ch{}.png'.format(idx, id_ch))
                traces_array = traces_np[:, i]

                gps_lon, _ = utils.get_column_for_given_du(trawv, 'gps_long', idx)
                plot_title = 'DU{} {} channel {}'.format(idx, base, id_ch)
                utils.plot_fourier_vs_time(
                    traces_array,
                    date_array,
                    fft_freq,
                    gps_lon[0][0] / 180 * np.pi,
                    plot_title,
                    plot_filename,
                    tz=tz,
                    figsize=(10, 8), plot_lines=plot_lines
                )

    ### plots of the mean and std vs time, with batterylevel and gps_temp
    for idx in du_list:
        if site == 'gaa':
            request = "battery_level"
            bl_, date_array_bl = utils.get_column_for_given_du(trawv, request, idx)

            request = 'trace_ch'
            result, date_array_trace = utils.get_column_for_given_du(tadc, request, idx)
            traces_np = result[:, 0, 0:3].to_numpy()

            request = 'gps_temp'
            gps_temp_, date_array_gps_temp = utils.get_column_for_given_du(trawv, request, idx)

            plot_function = utils.make_joint_plot

        if site == 'gp13':
            idd = np.where(duid == idx)[0]
            gps_ti_idx = gps_time[idd]
            date_array_trace = [datetime.datetime.fromtimestamp(dt, tz=tz_gmt) for dt in gps_ti_idx]
            
            bl_ = battery_level[idd]
            gps_temp_ = gps_temp[idd]
            traces_np = tadc.trace_ch.to_numpy()[idd, 0, 0:4]
            date_array_bl = date_array_trace
            date_array_gps_temp = date_array_trace
            plot_function = utils.make_joint_plot4d

        plot_filename = os.path.join(plot_path, '{}_batterylevel_trace_mean_vs_time_{}_ADC.png'.format(base, idx))
        plot_function(
            traces_np, date_array_trace,
            'mean', idx,
            plot_filename,
            right_axis_qty=bl_, date_right_axis_qty=date_array_bl,
            right_ylabel='battery level [V]', tadc_or_voltage='tadc', tz=tz)

        plot_filename = os.path.join(plot_path, '{}_batterylevel_trace_std_vs_time_{}_ADC.png'.format(base, idx))
        plot_function(
            traces_np, date_array_trace,
            'std', idx,
            plot_filename,
            right_axis_qty=bl_, date_right_axis_qty=date_array_bl,
            right_ylabel='battery level [V]', tadc_or_voltage='tadc',tz=tz
        )

        plot_filename = os.path.join(plot_path, '{}_gps_temp_trace_mean_vs_time_{}_ADC.png'.format(base, idx))
        plot_function(
            traces_np, date_array_trace,
            'mean', idx,
            plot_filename,
            right_axis_qty=gps_temp_, date_right_axis_qty=date_array_gps_temp,
            right_ylabel='gps temperature [ºC]', tadc_or_voltage='tadc',tz=tz
        )

        plot_filename = os.path.join(plot_path, '{}_gps_temp_trace_std_vs_time_{}_ADC.png'.format(base, idx))
        plot_function(
            traces_np, date_array_trace,
            'std', idx,
            plot_filename,
            right_axis_qty=gps_temp_, date_right_axis_qty=date_array_gps_temp,
            right_ylabel='gps temperature [ºC]', tadc_or_voltage='tadc',tz=tz
        )


    # Plots 5 random traces for eacu DU in ADC
    for idx in du_list:

        if site == 'gaa':
            request = 'trace_ch'
            traces_, date_array_ = utils.get_column_for_given_du(tadc, request, idx)
            traces_, dtime_sec_ = utils.get_column_for_given_du(tadc, request, idx)
            traces_np = traces_[:, 0, 0:3].to_numpy()

            nb_events = traces_np.shape[0]
            idxs = np.random.permutation(nb_events)[0:200]

            for idd in idxs:
                title = '{}, event #{}, DU {}'.format(base, tadc.event_number[idd], idx)
                date_for_title = date_array_[idd].astimezone(tz)
                title += ' date {}'.format(date_for_title.strftime("%d-%m-%Y %Hh%Mm%Ss"))
                save_path = os.path.join(trace_plot_path, 'trace_event{}_du{}.png'.format(tadc.event_number[idd], idx))
                utils.plot_trace_and_psd(traces_np[idd], title, save_path, tadc_or_voltage='tadc')
        if site == 'gp13':
            idd = np.where(duid == idx)[0]
            gps_ti_idx = gps_time[idd]
            date_idx = [datetime.datetime.fromtimestamp(dt, tz=utils.TZ_GMT()) for dt in gps_ti_idx]
            bl_idx = battery_level[idd]
            gps_temp_idx = gps_temp[idd]

            traces_np4d = tadc.trace_ch.to_numpy()[idd, 0, 0:4]
            nb_events = traces_np4d.shape[0]
            idxs = np.random.permutation(nb_events)[0:20]

            for idd_ in idxs:
                title = '{},\n event #{}, DU {}'.format(base, tadc.event_number[idd_], idx)
                date_for_title = date_idx[idd_].astimezone(tz)
                title += ' date {}'.format(date_for_title.strftime("%d-%m-%Y %Hh%Mm%Ss"))
                save_path = os.path.join(trace_plot_path, 'trace_event{}_du{}.png'.format(tadc.event_number[idd_], idx))
                utils.plot_trace_and_psd4d(traces_np4d[idd_], title, save_path, tadc_or_voltage='tadc')
