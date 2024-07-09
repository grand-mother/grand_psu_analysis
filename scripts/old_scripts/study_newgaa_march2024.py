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




file = '/Users/ab212678/Documents/GRAND/data/auger/GatA_20240308_1859_003001_MD_coin.root'

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


base = 'gaa_march8-10'
plot_path = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/plots_new_gaa/'

# Plot of battery level vs time



def make_plot_scalar_qty(trawv, request, tz, du_list, ylabel='', yrange=None):

    fig, axs = plt.subplots(figsize=(15, 8))
    axs.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=tz))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %Hh%M', tz=tz))

    for idu in du_list:

        res_, date_array = utils.get_column_for_given_du(trawv, request, idu)
        yrs = [dt.year for dt in date_array]
        id = np.where(np.array(yrs)> 2023)[0]
        res_ = res_[:, 0].to_numpy()
        res_ = res_[id]
        date_array = np.array(date_array)[id]

        axs.plot(date_array, res_, '.', ms='2', label='du{}'.format(idu))
        

    for label in axs.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    axs.grid()
    plt.legend(loc=0, ncol=2)
    plt.title(request)
    plt.xlabel('Time [{}]'.format(tz.tzname()))
    if yrange:
        plt.ylim(yrange)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, '{}_{}_vs_time.png'.format(base, request)))



make_plot_scalar_qty(trawv, 'battery_level', tz, du_list, 'Battery level [V]')
make_plot_scalar_qty(trawv, 'gps_temp', tz, du_list, 'GPS temperature [ºC]', yrange=[35, 70])

make_plot_scalar_qty(trawv, 'adc_temp', tz, du_list, 'ADC temperature [ºC]', yrange=[35, 70])

make_plot_scalar_qty(trawv, 'fpga_temp', tz, du_list, 'FPGA temperature [ºC]', yrange=[50, 90])

make_plot_scalar_qty(trawv, 'gps_alt', tz, du_list, 'GPS altitude [deg]', yrange = [1540, 1680])
make_plot_scalar_qty(trawv, 'gps_long', tz, du_list, 'GPS longitude [deg]' , yrange = [-69.54, -69.53])
make_plot_scalar_qty(trawv, 'gps_lat', tz, du_list, 'GPS latitude [m]', yrange = [-35.12, -35.11])

make_plot_scalar_qty(trawv, 'clock_tick', tz, du_list, 'clock tick [MHz]',)




if False:
    plt.figure()
    for idu in du_list:
        gps_temp, date_array = utils.get_column_for_given_du(trawv, 'gps_temp', idu)
        fpga_temp, date_array2 = utils.get_column_for_given_du(trawv, 'adc_temp', idu)
        yrs = [dt.year for dt in date_array]
        id = np.where(np.array(yrs)> 2023)[0]

        yrs2 = [dt.year for dt in date_array2]
        id2 = np.where(np.array(yrs2)> 2023)[0]

        gps_temp = gps_temp[:, 0].to_numpy()[id]
        fpga_temp = fpga_temp[:, 0].to_numpy()[id]

        plt.plot(gps_temp, fpga_temp, '.', ms=1)
        plt.xlim(35, 70)
    plt.plot(gps_temp, gps_temp, 'k.', ms=1)










### plots of the mean and std vs time, with batterylevel and gps_temp
for idu in du_list:

    request = "battery_level"
    bl_, date_array_bl = utils.get_column_for_given_du(trawv, request, idu)

    request = 'trace_ch'
    result, date_array_trace = utils.get_column_for_given_du(tadc, request, idu)

    traces_np = result[:, 0, 0:3].to_numpy()

    request = 'gps_temp'
    gps_temp_, date_array_gps_temp = utils.get_column_for_given_du(trawv, request, idu)


    yrs = [dt.year for dt in date_array_bl]
    id = np.where(np.array(yrs)> 2023)[0]
    bl_ = bl_[:, 0].to_numpy()
    bl_ = bl_[id]
    date_array_bl = list(np.array(date_array_bl)[id])
    date_array_trace = list(np.array(date_array_trace)[id])
    traces_np = traces_np[id]
    gps_temp_ = gps_temp_[:, 0].to_numpy()
    gps_temp_ = gps_temp_[id]


    # plot_function = utils.make_joint_plot

 
    # plot_filename = os.path.join(plot_path, '{}_batterylevel_trace_mean_vs_time_{}_ADC.png'.format(base, idu))
    # plot_function(
    #     traces_np, date_array_bl,
    #     'mean', idu,
    #     plot_filename,
    #     right_axis_qty=bl_, date_right_axis_qty=date_array_bl,
    #     right_ylabel='battery level [V]', tadc_or_voltage='tadc', tz=tz)

    # plot_filename = os.path.join(plot_path, '{}_batterylevel_trace_std_vs_time_{}_ADC.png'.format(base, idu))
    # plot_function(
    #     traces_np, date_array_bl,
    #     'std', idu,
    #     plot_filename,
    #     right_axis_qty=bl_, date_right_axis_qty=date_array_bl,
    #     right_ylabel='battery level [V]', tadc_or_voltage='tadc',tz=tz
    # )

    # plot_filename = os.path.join(plot_path, '{}_gps_temp_trace_mean_vs_time_{}_ADC.png'.format(base, idu))
    # plot_function(
    #     traces_np, date_array_bl,
    #     'mean', idu,
    #     plot_filename,
    #     right_axis_qty=gps_temp_, date_right_axis_qty=date_array_bl,
    #     right_ylabel='gps temperature [ºC]', tadc_or_voltage='tadc',tz=tz
    # )

    # plot_filename = os.path.join(plot_path, '{}_gps_temp_trace_std_vs_time_{}_ADC.png'.format(base, idu))
    # plot_function(
    #     traces_np, date_array_bl,
    #     'std', idu,
    #     plot_filename,
    #     right_axis_qty=gps_temp_, date_right_axis_qty=date_array_bl,
    #     right_ylabel='gps temperature [ºC]', tadc_or_voltage='tadc',tz=tz
    # )



    ts_list = [datetime.datetime.timestamp(dt) for dt in date_array_trace]
    ts_min = min(ts_list)
    ts_max = max(ts_list)

    ts_array = np.array(ts_list)
    n_slice = 10
    delta_ts = (ts_max - ts_min) / n_slice
    fig_num = idu * 3
    plt.figure(figsize=(15, 8))
    fft_freq = np.fft.rfftfreq(traces_np.shape[-1]) * 500  # [MHz]
    for i in range(n_slice):
        id_ts = np.where(((ts_min + i * delta_ts) < ts_array) *   (ts_array< (ts_min + (i+1)*delta_ts)))[0]
        tra = traces_np[id_ts]
        n = tra.shape[0]
        psd = filt.return_psd(tra, 500)
        if n > 0:

            bin_date_min_gmt = datetime.datetime.fromtimestamp(ts_min + i * delta_ts, tz=utils.TZ_GMT())
            bin_date_max_gmt = datetime.datetime.fromtimestamp(ts_min + (i+1)*delta_ts, tz=utils.TZ_GMT())

            bin_date_min_intz = bin_date_min_gmt.astimezone(tz)
            bin_date_max_intz = bin_date_max_gmt.astimezone(tz)
            
            plt.figure(fig_num, figsize=(15, 8))
            plt.plot(fft_freq, psd.mean(axis=0)[0], label='{} <t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )

            plt.figure(fig_num+1, figsize=(15, 8))
            plt.plot(fft_freq, psd.mean(axis=0)[1], label='{} <t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )


            plt.figure(fig_num+2, figsize=(15, 8))
            plt.plot(fft_freq, psd.mean(axis=0)[2], label='{} <t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )


    plt.figure(fig_num)        
    plt.title('du {} ch. X'.format(idu))
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('PSD [ADC^2 / MHz]')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chX.png".format(idu)))
    plt.close()

    plt.figure(fig_num+1)
    
    plt.title('du {} ch. Y'.format(idu))
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('PSD [ADC^2 / MHz]')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chY.png".format(idu)))
    plt.close()

    plt.figure(fig_num+2)
    
    plt.title('du {} ch. Z'.format(idu))
    plt.yscale('log')
    plt.ylim(1e-2, 1e3)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('PSD [ADC^2 / MHz]')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chZ.png".format( idu)))
    plt.close()






hmin = 4
hmax = 7

plt.figure(567, figsize=(15, 8))
plt.clf()

plt.figure(568, figsize=(15, 8))
plt.clf()

plt.figure(569, figsize=(15, 8))
plt.clf()


for idu in du_list:

    request = 'trace_ch'
    result, date_array_trace = utils.get_column_for_given_du(tadc, request, idu)

    fft_freq = np.fft.rfftfreq(traces_np.shape[-1]) * 500  # [MHz]

    yrs = [dt.year for dt in date_array_trace]
    days = [dt.day for dt in date_array_trace]
    hrs =  np.array([dt.hour for dt in date_array_trace])


    id = np.where(
        (np.array(yrs)> 2023)* (np.array(days)==9) * (hmin<hrs) * (hrs<hmax)
    )[0]
    traces_np = result[:, 0, 0:3].to_numpy()[id]
   
    psd = filt.return_psd(traces_np, 500)
    plt.figure(567)
    plt.plot(fft_freq, psd.mean(axis=0)[0], label='{}'.format(idu))

    plt.figure(568)
    plt.plot(fft_freq, psd.mean(axis=0)[1], label='{}'.format(idu))

    plt.figure(569)
    plt.plot(fft_freq, psd.mean(axis=0)[2], label='{}'.format(idu))

plt.figure(567)
plt.yscale('log')
plt.ylim(1e-2, 1e3)
plt.title('Mean PSD x-axis for March 9th {}h-{}h'.format(hmin-3, hmax-3))
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.savefig(os.path.join(plot_path, "psd_mean_march9th_{}h{}h_chX.png".format(hmin-3, hmax-3)))

plt.figure(568)
plt.yscale('log')
plt.ylim(1e-2, 1e3)
plt.title('Mean PSD y-axis for March 9th {}h-{}h'.format(hmin-3, hmax-3))
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.savefig(os.path.join(plot_path, "psd_mean_march9th_{}h{}h_chY.png".format(hmin-3, hmax-3)))

plt.figure(569)
plt.yscale('log')
plt.ylim(1e-2, 1e3)
plt.title('Mean PSD z-axis for March 9th {}h-{}h'.format(hmin-3, hmax-3))
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [ADC^2 / MHz]')
plt.legend(ncol=2, loc=0)
plt.savefig(os.path.join(plot_path, "psd_mean_march9th_{}h{}h_chZ.png".format(hmin-3, hmax-3)))












fig1, ax1 = plt.subplots(figsize=(15, 6))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh%M', tz=tz))

fig2, ax2 = plt.subplots(figsize=(15, 6))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh%M', tz=tz))

fig3, ax3 = plt.subplots(figsize=(15, 6))
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh%M', tz=tz))


for idu in du_list:

    request = 'trace_ch'
    result, date_array_trace = utils.get_column_for_given_du(tadc, request, idu)

    fft_freq = np.fft.rfftfreq(traces_np.shape[-1]) * 500  # [MHz]

    yrs = [dt.year for dt in date_array_trace]
    days = [dt.day for dt in date_array_trace]
    hrs =  np.array([dt.hour for dt in date_array_trace])


    id = np.where(
        (np.array(yrs)>2023)
    )[0]
    traces_np = result[:, 0, 0:3].to_numpy()[id]
    date_array_trace = np.array(date_array_trace)[id]

    iddd = np.argsort(date_array_trace)
    date_array_trace = date_array_trace[iddd]
    traces_np = traces_np[iddd]

    tr1_120_130 = filt.get_filtered_traces_inband(traces_np, fft_freq, 120, 130)

    ax1.plot(date_array_trace, np.convolve(tr1_120_130.std(axis=2)[:, 0], np.ones(1)/1, 'same'), '.', ms=1, label="{}".format(idu))
    ax2.plot(date_array_trace, np.convolve(tr1_120_130.std(axis=2)[:, 1], np.ones(1)/1, 'same'), '.', ms=1, label="{}".format(idu))
    ax3.plot(date_array_trace, np.convolve(tr1_120_130.std(axis=2)[:, 2], np.ones(1)/1, 'same'), '.', ms=1, label="{}".format(idu))



ax1.set_yscale('log')
ax1.legend(loc=0, ncol=5)
ax1.set_title('X-axis trace std filtered in 120-130 MHz')
ax1.set_xlabel('Time [{}]'.format(tz.tzname()))
ax1.set_ylabel('ADC trace std')

ax2.set_yscale('log')
ax2.legend(loc=0, ncol=5)
ax2.set_title('Y-axis trace std filtered in 120-130 MHz')
ax2.set_xlabel('Time [{}]'.format(tz.tzname()))
ax2.set_ylabel('ADC trace std')

ax3.set_yscale('log')
ax3.legend(loc=0, ncol=5)
ax3.set_title('Z-axis trace std filtered in 120-130 MHz')
ax3.set_xlabel('Time [{}]'.format(tz.tzname()))
ax3.set_ylabel('ADC trace std')


for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')


for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

for label in ax3.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')


fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

fig1.savefig(os.path.join(plot_path, "std123-130_x.png"))
fig2.savefig(os.path.join(plot_path, "std123-130_y.png"))
fig3.savefig(os.path.join(plot_path, "std123-130_z.png"))



idu =49

result, date_array_trace = utils.get_column_for_given_du(tadc, request, idu)

fft_freq = np.fft.rfftfreq(traces_np.shape[-1]) * 500  # [MHz]

yrs = [dt.year for dt in date_array_trace]
days = [dt.day for dt in date_array_trace]
hrs =  np.array([dt.hour for dt in date_array_trace])


id = np.where(
    (np.array(yrs)> 2023)
)[0]
traces_np = result[:, 0, 0:3].to_numpy()[id]
date_array_trace = np.array(date_array_trace)[id]

iddd = np.argsort(date_array_trace)
date_array_trace = date_array_trace[iddd]
traces_np = traces_np[iddd]

gps_lon, _ = utils.get_column_for_given_du(trawv, 'gps_long', 49)

plot_filename = os.path.join(plot_path, "fourier_49_temp")
 
utils.plot_fourier_vs_time(traces_np[:, 0], date_array_trace, fft_freq, gps_lon[0][0], "..tltl", plot_filename, tz=utils.TZ_auger(), figsize=(20, 35), plot_lines=False)
