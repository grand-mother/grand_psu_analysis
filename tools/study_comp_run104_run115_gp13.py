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

file_path = '/Users/ab212678/Documents/GRAND/data/gp13/nov2023/20dB/ROOTfiles/run81/'
rootfiles_path = os.path.join(file_path, 'root_files')


plot_path = '/Users/ab212678/Documents/GRAND/data/gp13/jan24/plots/'
os.makedirs(plot_path, exist_ok=True)

npz_dir = os.path.join(file_path, 'npz_files')



npz104 = '/Users/ab212678/Documents/GRAND/data/gp13/dec2023/20dB/ROOTfiles/run104/npz_files/'
npz115 = '/Users/ab212678/Documents/GRAND/data/gp13/jan24/run115/npz_files/'
rs104 = 'RUN104'
rs115 = 'RUN115'

label_list = ['run104', 'run115']



du_list_104 = [1010, 1013, 1017, 1019, 1020, 1021, 1029, 1031, 1032, 1035, 1041, 1075, 1085]
du_list_115 = [1010, 1017, 1019, 1020, 1021, 1032, 1035, 1041, 1075, 1085]



sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]

n_samples = 2048
fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]

tz_gp13 = utils.TZ_GP13()


def get_stuff_from_run_du(npz_dir, run_string, idu, tz):

    dl = u13.get_date_list_from_du_run(npz_dir, idu, run_string, tz=tz)
    trace_adc = u13.get_trace_from_du_run(npz_dir, idu, run_string)
    gps_temp = u13.get_gps_temp_from_du_run(npz_dir, idu, run_string)
    bl = u13.get_battery_level_from_du_run(npz_dir, idu, run_string)

    return dl, trace_adc, gps_temp, bl



### get the night data, then select the data in 1 or 2 hrs in lst 

#def plot_min_max_night_psd(idu, npz_dir, run_string, nb_traces, nb_traces_in_mean, plot_path_, tr_length=2048, fs=500):



run_string = rs104
idu = 1041
plot_path_ = plot_path

tr_length = 1024
fs = 500
npz_dir = npz104


def get_psd_night_lst(idu, npz_dir, run_string, plot_path_, tr_length, fs, time_min, time_max):

    plot_path2 = os.path.join(plot_path_, 'traces/{}/DU{}'.format(run_string, idu))
    os.makedirs(plot_path2, exist_ok=True)

    stuff = get_stuff_from_run_du(npz_dir, run_string, idu, tz_gp13)

    dl = np.array(stuff[0])
    trace_adc = stuff[1]
    gps_temp = stuff[2]
    stds = trace_adc.std(axis=-1)

    fft_freq = np.fft.rfftfreq(tr_length) * fs  # [MHz]

    tz_gmt = utils.TZ_GMT()

    hrs = [d.astimezone(tz_gp13).hour +d.astimezone(tz_gp13).minute/60  for d in dl]
    hrs = np.array(hrs)
    id_night = np.where(((hrs[:-1] > 23) + (hrs[:-1] < 5)))[0]

    s2 = stds[id_night]
    tr2 = trace_adc[id_night]
    gps_temp2 = gps_temp[id_night]
    dl2 = dl[id_night]
    hrs2 = hrs[id_night]

    id_time = np.where(((hrs2 > time_min) * (hrs2 < time_max)))[0]


    s3 = s2[id_time]
    tr3 = tr2[id_time][:, :, 0:tr_length]
    gps_temp3 = gps_temp2[id_time]
    dl3 = dl2[id_time]



    psd3 = filt.return_psd(tr3, fs)
    return psd3




psd1 = get_psd_night_lst(idu, npz104, rs104, plot_path_, 1024, fs, 2, 5)
psd1p = get_psd_night_lst(idu, npz104, rs104, plot_path_, 2048, fs, 2, 5)
psd1pp = get_psd_night_lst(idu, npz115, rs115, plot_path_, 1024, fs, 2, 5)

fft_freq_1024 = np.fft.rfftfreq(1024) * sample_freq  # [MHz]
fft_freq_2048 = np.fft.rfftfreq(2048) * sample_freq  # [MHz]



plt.figure()
plt.plot(fft_freq_1024, psd1.mean(axis=0)[1])
plt.plot(fft_freq_2048, psd1p.mean(axis=0)[1])
plt.plot(fft_freq_1024, psd1pp.mean(axis=0)[1])
plt.yscale('log')



psd1 = get_psd_night_lst(1020, npz104, rs104, plot_path_, 1024, fs, 2, 5)
psd1p = get_psd_night_lst(1020, npz104, rs104, plot_path_, 2048, fs, 2, 5)
psd1pp = get_psd_night_lst(1020, npz115, rs115, plot_path_, 1024, fs, 2, 5)

plt.figure()
plt.plot(fft_freq_1024, psd1.mean(axis=0)[1])
plt.plot(fft_freq_2048, psd1p.mean(axis=0)[1])
plt.plot(fft_freq_1024, psd1pp.mean(axis=0)[1])
plt.yscale('log')




    # ## ch2 select with max-min on ch1,ch2, ch3
    
    # plt.figure(46)
    # plt.clf()


    # #plt.plot(fft_freq, psd_low_var_axis_1.mean(axis=0)[2])
    # #plt.plot(fft_freq, psd_high_var_axis_1.mean(axis=0)[2])

    # plt.plot(fft_freq, psd_low_var_axis_2.mean(axis=0)[2])
    # plt.plot(fft_freq, psd_high_var_axis_2.mean(axis=0)[2])

    # #plt.plot(fft_freq, psd_low_var_axis_3.mean(axis=0)[2])
    # #plt.plot(fft_freq, psd_high_var_axis_3.mean(axis=0)[2])

    # plt.yscale('log')


    # ## ch3 select with max-min on ch1,ch2, ch3
    # plt.figure(47)
    # plt.clf()


    # #plt.plot(fft_freq, psd_low_var_axis_1.mean(axis=0)[3])
    # #plt.plot(fft_freq, psd_high_var_axis_1.mean(axis=0)[3])

    # #plt.plot(fft_freq, psd_low_var_axis_2.mean(axis=0)[3])
    # #plt.plot(fft_freq, psd_high_var_axis_2.mean(axis=0)[3])

    # plt.plot(fft_freq, psd_low_var_axis_3.mean(axis=0)[3])
    # plt.plot(fft_freq, psd_high_var_axis_3.mean(axis=0)[3])

    # plt.yscale('log')


sp = []
for idu in du_list:
    print(idu)
    sp.append(
        plot_min_max_night_psd(idu, npz104, "RUN104", 5, 40, plot_path, tr_length=2048, fs=500)
    )



du_list_115 = [1010, 1017, 1019, 1020, 1021, 1032, 1035, 1041, 1075, 1085]


sp = []
for idu in du_list_115:
    print(idu)
    sp.append(
        plot_min_max_night_psd(idu, npz115, "RUN115", 5, 40, plot_path, tr_length=1024, fs=500)
    )




fft_freq = np.fft.rfftfreq(1024) * 500  # [MHz]


plt.figure(789, figsize=(15, 6))
plt.clf()
plt.figure(790, figsize=(15, 6))
plt.clf()
plt.figure(791, figsize=(15, 6))
plt.clf()
plt.figure(792, figsize=(15, 6))
plt.clf()



for idu, s in zip(du_list_115, sp):

    plt.figure(789)
    plt.plot(fft_freq, s[0], label='{}'.format(idu), lw=1)
    plt.title('low variance ch1 night traces run 104')
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)

    
    plt.figure(790)
    plt.plot(fft_freq, s[2], label='{}'.format(idu), lw=1)
    plt.title('low variance ch2 night traces run 104')
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)


    plt.figure(791)
    plt.plot(fft_freq, s[4], label='{}'.format(idu), lw=1)
    plt.title('low variance ch3 night traces run 104')
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)

    # plt.figure(792)
    # plt.plot(fft_freq, s[3], label='{}'.format(idu), lw=0.7)
    # plt.title('low variance ch1 night traces run 104')
    # plt.yscale('log')
    # plt.ylim(1e-2, 1e3)


plt.figure(789)
plt.legend(ncol=3)
plt.ylabel('PSD [ADC^2 / MHz]')
plt.xlabel('frequency [MHz]')
plt.ylim(1e-2, 1e3) 


plt.figure(790)
plt.legend(ncol=3)






def plot_min_max_night_psd_filtered(idu, npz_dir, run_string, nb_traces, nb_traces_in_mean, fft_freq, f1, f2, plot_path_, tr_length=2048, fs=500):

    plot_path2 = os.path.join(plot_path_, 'traces/{}/DU{}'.format(run_string, idu))
    os.makedirs(plot_path2, exist_ok=True)

    stuff = get_stuff_from_run_du(npz_dir, run_string, idu, tz_gp13)


    fft_freq = np.fft.rfftfreq(tr_length) * fs  # [MHz]

    dl = np.array(stuff[0])
    trace_adc = stuff[1][:,:,0:1024]
    print(trace_adc.shape)


    trace_adc = filt.get_filtered_traces(trace_adc, fft_freq, f1, f2)
    gps_temp = stuff[2]
    stds = trace_adc.std(axis=-1)

    hrs = [d.astimezone(tz_gp13).hour +d.astimezone(tz_gp13).minute/60  for d in dl]
    hrs = np.array(hrs)
    id_night = np.where(((hrs[:-1] > 23) + (hrs[:-1] < 5)))[0]


    s2 = stds[id_night]

    tr2 = trace_adc[id_night]
    gps_temp2 = gps_temp[id_night]
    dl2 = dl[id_night]


    id_max_axis1 = np.argsort(s2[:, 1])[-nb_traces_in_mean:-1]
    id_min_axis1 = np.argsort(s2[:, 1])[:nb_traces_in_mean]

    id_max_axis2 = np.argsort(s2[:, 2])[-nb_traces_in_mean:-1]
    id_min_axis2 = np.argsort(s2[:, 2])[:nb_traces_in_mean]

    id_max_axis3 = np.argsort(s2[:, 3])[-nb_traces_in_mean:-1]
    id_min_axis3 = np.argsort(s2[:, 3])[:nb_traces_in_mean]


    for idd in id_max_axis1[0:nb_traces]:
        utils.plot_trace_and_psd4d(
            tr2[idd], '{}_{}'.format(run_string, idu), os.path.join(plot_path2, 'high_variance{}_{}_filtered_axis1.png'.format(idd, idu)), tadc_or_voltage='tadc'
        )

    for idd in id_min_axis1[0:nb_traces]:
        utils.plot_trace_and_psd4d(
            tr2[idd], '{}_{}'.format(run_string, idu), os.path.join(plot_path2, 'low_variance{}_{}_filtered_axis1.png'.format(idd, idu)), tadc_or_voltage='tadc'
        )


    for idd in id_max_axis2[0:nb_traces]:
        utils.plot_trace_and_psd4d(
            tr2[idd], '{}_{}'.format(run_string, idu), os.path.join(plot_path2, 'high_variance{}_{}_filtered_axis2.png'.format(idd, idu)), tadc_or_voltage='tadc'
        )

    for idd in id_min_axis2[0:nb_traces]:
        utils.plot_trace_and_psd4d(
            tr2[idd], '{}_{}'.format(run_string, idu), os.path.join(plot_path2, 'low_variance{}_{}_filtered_axis2.png'.format(idd, idu)), tadc_or_voltage='tadc'
        )

    for idd in id_max_axis3[0:nb_traces]:
        utils.plot_trace_and_psd4d(
            tr2[idd], '{}_{}'.format(run_string, idu), os.path.join(plot_path2, 'high_variance{}_{}_filtered_axis3.png'.format(idd, idu)), tadc_or_voltage='tadc'
        )

    for idd in id_min_axis3[0:nb_traces]:
        utils.plot_trace_and_psd4d(
            tr2[idd], '{}_{}'.format(run_string, idu), os.path.join(plot_path2, 'low_variance{}_{}_filtered_axis3.png'.format(idd, idu)), tadc_or_voltage='tadc'
        )



    low_var_trace_axis1 = tr2[id_min_axis1]
    high_var_trace_axis1 = tr2[id_max_axis1]

    psd_low_var_axis_1 = filt.return_psd(low_var_trace_axis1, 500)
    psd_high_var_axis_1 = filt.return_psd(high_var_trace_axis1, 500)

    low_var_trace_axis2 = tr2[id_min_axis2]
    high_var_trace_axis2 = tr2[id_max_axis2]


    psd_low_var_axis_2 = filt.return_psd(low_var_trace_axis2, 500)
    psd_high_var_axis_2 = filt.return_psd(high_var_trace_axis2, 500)


    low_var_trace_axis3 = tr2[id_min_axis3]
    high_var_trace_axis3 = tr2[id_max_axis3]

    psd_low_var_axis_3 = filt.return_psd(low_var_trace_axis3, 500)
    psd_high_var_axis_3 = filt.return_psd(high_var_trace_axis3, 500)


    ## ch1 select with max-min on ch1,ch2, ch3
    fig, axs = plt.subplots(3, 1,figsize=(16, 6))

    psd_low_axis1_mean = psd_low_var_axis_1.mean(axis=0)[1]
    psd_high_axis1_mean = psd_high_var_axis_1.mean(axis=0)[1]

    psd_low_axis2_mean = psd_low_var_axis_2.mean(axis=0)[2]
    psd_high_axis2_mean = psd_high_var_axis_2.mean(axis=0)[2]

    psd_low_axis3_mean = psd_low_var_axis_3.mean(axis=0)[2]
    psd_high_axis3_mean = psd_high_var_axis_3.mean(axis=0)[2]

    axs[0].plot(fft_freq, psd_low_axis1_mean, label='low')
    axs[0].plot(fft_freq, psd_high_axis1_mean, label='high')

    axs[1].plot(fft_freq, psd_low_axis2_mean, label='low')
    axs[1].plot(fft_freq, psd_high_axis2_mean, label='high')

    axs[2].plot(fft_freq, psd_low_axis3_mean, label='low')
    axs[2].plot(fft_freq, psd_high_axis3_mean, label='high')

    [ax.set_ylabel('PSD [ADC^2 / MHz]') for ax in axs]
    [ax.set_yscale('log') for ax in axs]

    [ax.set_ylim(1e-2, 1e3) for ax in axs]

    axs[2].set_xlabel('frequency [MHz]')
    [ax.legend(loc=1) for ax in axs]

    title = 'DU {}'.format(idu)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0) 
    fig.savefig(os.path.join(plot_path, 'psd_min_max_{}_{}_filtered_{}_{}_1024.png'.format(run_string, idu, f1, f2)))

    return psd_low_axis1_mean, psd_high_axis1_mean, psd_low_axis2_mean, psd_high_axis2_mean, psd_low_axis3_mean, psd_high_axis3_mean


sp_104 = []
for idu in du_list:
    print(idu)
    sp_104.append(
        plot_min_max_night_psd_filtered(idu, npz104, "RUN104", 1, 40, fft_freq, 100, 170, plot_path, tr_length=1024)
    )




sp_115 = []
for idu in du_list_115:
    print(idu)
    sp_115.append(
        plot_min_max_night_psd_filtered(idu, npz115, "RUN115", 1, 40, fft_freq, 100, 170, plot_path, tr_length=1024)
    )

