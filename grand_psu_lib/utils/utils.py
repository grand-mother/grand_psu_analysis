import uproot
import awkward as ak
import numpy as np
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os

import pymap3d
from datetime import time, tzinfo, timedelta


class TZ_auger(tzinfo):

    def utcoffset(self, dt):
        return -timedelta(hours=3)

    def dst(self, dt):
        return timedelta(0)

    def tzname2(self, dt):
        return "-03:00"

    def tzname(self):
            return "GMT-3"

    def  __repr__(self):
        return f"{self.__class__.__name__}()"

class TZ_GMT(tzinfo):

    def utcoffset(self, dt):
        return timedelta(hours=0)

    def dst(self, dt):
        return timedelta(0)

    def tzname2(self,dt):
        return "00:00"

    def tzname(self):
        return "GMT"

    def  __repr__(self):
        return f"{self.__class__.__name__}()"

class TZ_GP13(tzinfo):

    def utcoffset(self, dt):
        return timedelta(hours=8)

    def dst(self, dt):
        return timedelta(0)

    def tzname2(self,dt):
        return "00:00"

    def tzname(self):
        return "GMT+8"

    def  __repr__(self):
        return f"{self.__class__.__name__}()"



def get_column_for_given_du(tree_, request, du_number):
    """
    Function that extract the values of a given column for a given du.
    Also outputs the corresponding arrays of dates.

    This function aims at working both for tacd "still on disk" (in case of a single file))
    and the tree_ stored in memory (in case of an concatenated tacd). So there is a test at the beginning
    """
    if type(tree_) == uproot.models.TTree.Model_TTree_v20:
        duid = tree_["du_id"].array()
        timestamp_array = tree_["time_seconds"].array()
        requested_array = tree_[request].array()

    elif type(tree_) == ak.highlevel.Array:
        duid = tree_["du_id"]
        timestamp_array = tree_["time_seconds"]
        requested_array = tree_[request]

    idx_du = (duid == du_number)

    requested_array_du = requested_array[idx_du]
    idx_dupresent = ak.where(ak.sum(idx_du, axis=1))
    result = requested_array_du[idx_dupresent]

    timestamp_array_du = timestamp_array.to_numpy()[idx_dupresent]
    date_array_du = [datetime.datetime.fromtimestamp(dt, tz=TZ_GMT()) for dt in timestamp_array_du]

    return result, date_array_du


def get_dulist(tadc):
    if type(tadc) == uproot.models.TTree.Model_TTree_v20:
        duid = tadc["du_id"].array()

    elif type(tadc) == ak.highlevel.Array:
        duid = tadc["du_id"]

    return np.unique(ak.flatten(duid))


def plot_trace_and_psd(trace, title, save_path, tadc_or_voltage='voltage'):
    n_samples = trace.shape[1]
    fft_freq = np.fft.rfftfreq(n_samples) * 500  # [MHz]
    fig, axs = plt.subplots(3, 2, figsize=(15, 8))
    # Remove vertical space between axes
    fig.subplots_adjust(hspace=0)

    sig_x = trace[0].std()
    sig_y = trace[1].std()
    sig_z = trace[2].std()

    axs[0, 0].plot(trace[0], label='x, std {}'.format(sig_x))
    axs[1, 0].plot(trace[1], label='y, std {}'.format(sig_y))
    axs[2, 0].plot(trace[2], label='z, std {}'.format(sig_z))

    axs[0, 0].plot(trace[0]*0 + sig_x, 'r-')
    axs[1, 0].plot(trace[1]*0 + sig_y, 'r-')
    axs[2, 0].plot(trace[2]*0 + sig_z, 'r-')

    axs[0, 0].plot(trace[0]*0 + 3*sig_x, 'r:')
    axs[1, 0].plot(trace[1]*0 + 3*sig_y, 'r:')
    axs[2, 0].plot(trace[2]*0 + 3*sig_z, 'r:')


    axs[0, 0].legend(loc=1)
    axs[1, 0].legend(loc=1)
    axs[2, 0].legend(loc=1)

    psd = np.abs(np.fft.rfft(trace)**2)

    axs[0, 1].plot(fft_freq, psd[0])
    axs[1, 1].plot(fft_freq, psd[1])
    axs[2, 1].plot(fft_freq, psd[2])

    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylim(5e3, 2e7)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_ylim(5e3, 2e7)
    axs[2, 1].set_yscale('log')
    axs[2, 1].set_ylim(5e3, 2e7)
    

    #axs[0].set_title('{}, event #{}, DU {} \n cluster {}'.format(base, trawv.event_number[ind_[i]], idx, lab))
    axs[2, 0].set_xlabel('sample id')
    axs[2, 1].set_xlabel('frequency [MHz]')
    fig.suptitle(title)
    if tadc_or_voltage == 'voltage':
        axs[1, 0].set_ylabel('Raw voltage [$\mu$V]')
        axs[1, 1].set_ylabel('PSD [tbd]')
    if tadc_or_voltage == 'tadc':
        axs[1, 0].set_ylabel('ADC')
        axs[1, 1].set_ylabel('PSD [tbd]')


    plt.savefig(save_path)
    plt.close()


def plot_trace_and_psd4d(trace, title, save_path, tadc_or_voltage='voltage'):
    n_samples = trace.shape[1]

    fft_freq = np.fft.rfftfreq(n_samples) * 500  # [MHz]
    fig, axs = plt.subplots(4, 2, figsize=(15, 8))
    # Remove vertical space between axes
    fig.subplots_adjust(hspace=0)

    sig_x = trace[0].std()
    sig_y = trace[1].std()
    sig_z = trace[2].std()
    sig_w = trace[3].std()

    axs[0, 0].plot(trace[0], label='ch0, std {}'.format(sig_x))
    axs[1, 0].plot(trace[1], label='ch1, std {}'.format(sig_y))
    axs[2, 0].plot(trace[2], label='ch2, std {}'.format(sig_z))
    axs[3, 0].plot(trace[3], label='ch3, std {}'.format(sig_w))

    axs[0, 0].plot(trace[0]*0 + sig_x, 'r-')
    axs[1, 0].plot(trace[1]*0 + sig_y, 'r-')
    axs[2, 0].plot(trace[2]*0 + sig_z, 'r-')
    axs[3, 0].plot(trace[2]*0 + sig_w, 'r-')

    axs[0, 0].plot(trace[0]*0 + 3*sig_x, 'r:')
    axs[1, 0].plot(trace[1]*0 + 3*sig_y, 'r:')
    axs[2, 0].plot(trace[2]*0 + 3*sig_z, 'r:')
    axs[3, 0].plot(trace[2]*0 + 3*sig_w, 'r:')


    axs[0, 0].legend(loc=1)
    axs[1, 0].legend(loc=1)
    axs[2, 0].legend(loc=1)
    axs[3, 0].legend(loc=1)

    psd = np.abs(np.fft.rfft(trace)**2)

    axs[0, 1].plot(fft_freq, psd[0])
    axs[1, 1].plot(fft_freq, psd[1])
    axs[2, 1].plot(fft_freq, psd[2])
    axs[3, 1].plot(fft_freq, psd[3])

    axs[0, 1].set_yscale('log')
    axs[1, 1].set_yscale('log')
    axs[2, 1].set_yscale('log')
    axs[3, 1].set_yscale('log')

    #axs[0].set_title('{}, event #{}, DU {} \n cluster {}'.format(base, trawv.event_number[ind_[i]], idx, lab))
    axs[3, 0].set_xlabel('sample id')
    axs[3, 1].set_xlabel('frequency [MHz]')
    fig.suptitle(title)
    if tadc_or_voltage == 'voltage':
        axs[1, 0].set_ylabel('Raw voltage [$\mu$V]')
        axs[1, 1].set_ylabel('PSD [tbd]')
    if tadc_or_voltage == 'tadc':
        axs[1, 0].set_ylabel('ADC')
        axs[1, 1].set_ylabel('PSD [tbd]')


    plt.savefig(save_path)
    plt.close()


def plot_traces_and_psds_in_time_interval(
    trace_np,
    time_secdu,
    time_min,
    time_max,
    commun_title,
    plot_path,
    tadc_or_voltage='voltage',
    tz=TZ_auger()):

    time_min_ = datetime.datetime.timestamp(time_min, tz=tz)
    time_max_ = datetime.datetime.timestamp(time_max, tz=tz)

    ind_time = np.where((time_min_ < time_secdu)*(time_secdu <= time_max_))[0]

    for i, idx in enumerate(ind_time):

        title = commun_title + ' {}'.format(datetime.datetime.fromtimestamp(time_secdu[idx], tz=tz).strftime("%d-%m-%Y_%H-%M-%S"))
        plot_filename = os.path.join(plot_path, 'trace_{}.png'.format(i))
        plot_trace_and_psd(
            trace_np[idx],
            title,
            plot_filename,
            tadc_or_voltage=tadc_or_voltage
        )


def make_joint_plot(
    traces_np, date_trace,
    mean_or_std, du_number,
    plot_filename,
    right_axis_qty=None, date_right_axis_qty=None,
    right_ylabel=None, tadc_or_voltage='voltage', tz=TZ_auger()):

    if mean_or_std == 'mean':
        arr = traces_np.mean(axis=2)
        plt_title = 'Mean of the traces for DU {}'.format(du_number)

    elif mean_or_std == 'std':
        arr = traces_np.std(axis=2)
        plt_title = 'std of the traces for DU {}'.format(du_number)

    if tadc_or_voltage == 'tadc':
        ylabel = '{} of each trace [ADC]'.format(mean_or_std)
    elif tadc_or_voltage == 'voltage':
        ylabel = '{} of each trace [$\mu$V]'.format(mean_or_std)

    fig, axs = plt.subplots(figsize=(20, 15))
    axs.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh%M', tz=tz))

    axs.plot(date_trace, arr[:, 0], 'r.', ms=4, label='ch 0')
    axs.plot(date_trace, arr[:, 1], 'g.', ms=4, label='ch 1')
    axs.plot(date_trace, arr[:, 2], 'b.', ms=4, label='ch 2')
    axs.legend(loc=0, ncol=2)
    if mean_or_std == 'std':
        axs.set_yscale('log')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('Time [{}]'.format(tz.tzname()))

    if right_axis_qty is not None:
        ax2 = axs.twinx()
        #ax2.plot(dtime_sec_bl, bl_)
        ax2.set_ylabel(right_ylabel) 
        ax2.plot(date_right_axis_qty, right_axis_qty, '-')
    ax2.grid()
    for label in axs.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    plt.title(plt_title)

    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()


def make_joint_plot4d(
    traces_np, date_trace,
    mean_or_std, du_number,
    plot_filename,
    right_axis_qty=None, date_right_axis_qty=None,
    right_ylabel=None, tadc_or_voltage='voltage', tz=None):

    if mean_or_std == 'mean':
        arr = traces_np.mean(axis=2)
        plt_title = 'Mean of the traces for DU {}'.format(du_number)

    elif mean_or_std == 'std':
        arr = traces_np.std(axis=2)
        plt_title = 'std of the traces for DU {}'.format(du_number)

    if tadc_or_voltage == 'tadc':
        ylabel = '{} of each trace [ADC]'.format(mean_or_std)
    elif tadc_or_voltage == 'voltage':
        ylabel = '{} of each trace [$\mu$V]'.format(mean_or_std)

    fig, axs = plt.subplots(figsize=(15, 8))
    axs.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh%M', tz=tz))

    axs.plot(date_trace, arr[:, 0], 'r.', ms=4, label='ch 0')
    axs.plot(date_trace, arr[:, 1], 'g.', ms=4, label='ch 1')
    axs.plot(date_trace, arr[:, 2], 'b.', ms=4, label='ch 2')
    axs.plot(date_trace, arr[:, 3], 'm.', ms=4, label='ch 3')

    axs.legend(loc=0, ncol=2)
    if mean_or_std == 'std':
        axs.set_yscale('log')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('Time [{}]'.format(tz.tzname()))

    if right_axis_qty is not None:
        ax2 = axs.twinx()
        #ax2.plot(dtime_sec_bl, bl_)
        ax2.set_ylabel(right_ylabel) 
        ax2.plot(date_right_axis_qty, right_axis_qty, '-')

    for label in axs.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    plt.title(plt_title)

    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()


def plot_mean_psd_time_sliced(tadc, nb_time_bins, plot_path, du_list=None, tz=TZ_auger()):

    n_samples = len(tadc.trace_ch[0][0][0])

    if du_list is None:
        du_list = get_dulist(tadc)
    else:
        du_list = du_list

    # The following is just toi get the smalles and largest timestamps in the tadc
    time_sec = tadc['time_seconds'].to_numpy()
    time_min = time_sec.min()
    time_max = time_sec.max()

    delta_time = time_max - time_min
    time_divider = delta_time / nb_time_bins

    ##### make mean psd spectrums plots for each du aeraged for a given period of time

    request = 'trace_ch'
    for du_number in du_list:
        result, date_array = get_column_for_given_du(tadc, request, du_number)
        traces_np = result[:, 0, 0:3].to_numpy()

        # reconstruct the ts_array
        ts_array_du = [d.timestamp() for d in date_array]

        t0 = time_min
        ts_du = ts_array_du - t0

        #n_time_bin = ts_du.max() // time_divider
        time_indices = []
        fig_num = du_number * 3
        plt.figure()
        plt.clf()
        fft_freq = np.fft.rfftfreq(n_samples) * 500  # [MHz]

        for i in range(nb_time_bins):
            idx = np.where((i*time_divider <= ts_du) * (ts_du < (i+1)*time_divider))[0]
            time_indices.append(idx)

            traces_np_timei = traces_np[time_indices[i]]
            n_traces = traces_np_timei.shape[0]
            fft_timei = np.fft.rfft(traces_np_timei)
            psd_timei = np.abs(fft_timei**2)
            if n_traces > 0:
                plt.figure(fig_num, figsize=(15, 15))
                
                bin_date_min_gmt = datetime.datetime.fromtimestamp(t0+i*time_divider, tz=TZ_GMT())
                bin_date_max_gmt = datetime.datetime.fromtimestamp(t0+(i+1)*time_divider, tz=TZ_GMT())

                bin_date_min_intz = bin_date_min_gmt.astimezone(tz)
                bin_date_max_intz = bin_date_max_gmt.astimezone(tz)

                plt.plot(fft_freq, psd_timei.mean(axis=0)[0], label='{} <t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )
                plt.figure(fig_num+1, figsize=(15, 15))
                
                plt.plot(fft_freq, psd_timei.mean(axis=0)[1], label='{}<t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )
            
                plt.figure(fig_num+2, figsize=(15, 15))
                plt.plot(fft_freq, psd_timei.mean(axis=0)[2], label='{}<t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )
                
            
        plt.figure(fig_num)
        
        plt.title('du {} ch. X'.format(du_number))
        plt.yscale('log')
        plt.ylim(5e3, 2e7)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Mean PSD [A. U.]')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chX.png".format(du_number)))
        plt.close()

        plt.figure(fig_num+1)
        
        plt.title('du {} ch. Y'.format(du_number))
        plt.yscale('log')
        plt.ylim(5e3, 2e7)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Mean PSD [A. U.]')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chY.png".format(du_number)))
        plt.close()

        plt.figure(fig_num+2)
        
        plt.title('du {} ch. Z'.format(du_number))
        plt.yscale('log')
        plt.ylim(5e3, 2e7)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Mean PSD [A. U.]')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chZ.png".format( du_number)))
        plt.close()


def plot_mean_psd_time_sliced_gp13(tadc, nb_time_bins, plot_path, du_list=None, tz=TZ_GP13()):
    
    n_samples = len(tadc.trace_ch[0][0][0])

    if du_list is None:
        du_list = get_dulist(tadc)
    else:
        du_list = du_list

    duid = tadc.du_id.to_numpy()
    gps_time = tadc.gps_time.to_numpy().squeeze()
    #time_sec = tadc['time_seconds'].to_numpy()

    time_min = gps_time.min()
    time_max = gps_time.max()

    delta_time = time_max - time_min
    time_divider = delta_time / nb_time_bins

    for du_number in du_list:
        idd = np.where(duid == du_number)[0]
       
        ts_array_du = gps_time[idd]
        #dtime_sec_du = [datetime.datetime.fromtimestamp(dt, tz=TZ_GMT()) for dt in time_secdu]

        traces_np = tadc.trace_ch.to_numpy()[idd, 0, 0:4]

        #t0 = time_secdu.min()
        t0 = time_min
        ts_du = ts_array_du - t0


        time_indices = []
        fig_num = du_number * 3
        plt.figure(figsize=(8, 15))
        plt.clf()
        fft_freq = np.fft.rfftfreq(n_samples) * 500  # [MHz]
        for i in range(nb_time_bins):
            idx = np.where((i*time_divider <= ts_du) * (ts_du < (i+1)*time_divider))[0]
            time_indices.append(idx)

            traces_np_timei = traces_np[time_indices[i]]
            n_traces = traces_np_timei.shape[0]
            fft_timei = np.fft.rfft(traces_np_timei)
            psd_timei = np.abs(fft_timei**2)
            if n_traces > 0:

                bin_date_min_gmt = datetime.datetime.fromtimestamp(t0+i*time_divider, tz=TZ_GMT())
                bin_date_max_gmt = datetime.datetime.fromtimestamp(t0+(i+1)*time_divider, tz=TZ_GMT())

                bin_date_min_intz = bin_date_min_gmt.astimezone(tz)
                bin_date_max_intz = bin_date_max_gmt.astimezone(tz)

                plt.figure(fig_num, figsize=(15, 15))
                plt.plot(fft_freq, psd_timei.mean(axis=0)[0], label='{} <t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )
                plt.figure(fig_num+1, figsize=(15, 15))
                plt.plot(fft_freq, psd_timei.mean(axis=0)[1], label='{}<t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )

                plt.figure(fig_num+2, figsize=(15, 15))
                plt.plot(fft_freq, psd_timei.mean(axis=0)[2], label='{}<t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )
                plt.figure(fig_num+3, figsize=(15, 15))
                plt.plot(fft_freq, psd_timei.mean(axis=0)[3], label='{}<t< {}'.format(
                    bin_date_min_intz.strftime("%d %b_%Hh%M"),
                    bin_date_max_intz.strftime("%d %b_%Hh%M"))
                )

        plt.figure(fig_num)

        plt.title('du {} ch. X'.format(du_number))
        plt.yscale('log')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Mean PSD [A. U.]')
        plt.ylim(5e3, 2e7)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chX.png".format(du_number)))
        plt.close()

        plt.figure(fig_num+1)
        
        plt.title('du {} ch. Y'.format(du_number))
        plt.yscale('log')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Mean PSD [A. U.]')
        plt.legend(loc=0)
        plt.ylim(5e3, 2e7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chY.png".format(du_number)))
        plt.close()

        plt.figure(fig_num+2)
        
        plt.title('du {} ch. Z'.format(du_number))
        plt.yscale('log')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Mean PSD [A. U.]')
        plt.legend(loc=0)
        plt.ylim(5e3, 2e7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chZ.png".format( du_number)))
        plt.close()

        plt.figure(fig_num+3)
        
        plt.title('du {} ch. W'.format(du_number))
        plt.yscale('log')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Mean PSD [A. U.]')
        plt.legend(loc=0)
        plt.ylim(5e3, 2e7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "psd_std_vs_time_{}_chW.png".format( du_number)))
        plt.close()


def plot_fourier_vs_time(traces_array, date_array, fft_freq, longitude, plot_title, plot_filename, tz=TZ_auger(), figsize=(20, 35), plot_lines=False):

    tz_gmt = TZ_GMT()

    figsize_y = np.int32(traces_array.shape[0]/6/60/24*5) + 10

    fft = np.fft.rfft(traces_array)
    fig, axs = plt.subplots(figsize=(20, figsize_y))
    axs.yaxis.set_major_locator(mdates.HourLocator(interval=4, tz=tz))
    axs.yaxis.set_major_formatter(mdates.DateFormatter('%m/%d %Hh%M', tz=tz))

    X, Y = np.meshgrid(fft_freq, date_array)
    c = axs.pcolormesh(X, Y, np.log10(np.abs(fft)**2), vmin=np.log10(5e3), vmax=7)

    axs.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250])

    ax2 = axs.twinx()

    yticks = axs.get_yticks()
    ax2.set_yticks(yticks)

    y_ticks_date = mdates.num2date(yticks, tz=tz)
    y_ticks_date_gmt = [d.astimezone(tz_gmt) for d in y_ticks_date]

    lst_yticks = np.array(pymap3d.datetime2sidereal(y_ticks_date_gmt, longitude))

    lst_yticks[lst_yticks < 0] += 2*np.pi
    time_yticks = lst_yticks * 24/2/np.pi
    hrs = np.floor(time_yticks)
    mins = time_yticks % 1 * 60

    time_yticks_label = ["{:2.0f}h{:2.0f}".format(h, m) for h, m in zip(hrs, mins)]
    ax2.set_ylim(axs.get_ylim())
    ax2.set_yticklabels(time_yticks_label)
    ax2.set_ylabel('LST')

    axs.set_xlabel('Frequency [MHz]')

    for label in axs.get_yticklabels(which='major'):
        label.set(rotation=30, verticalalignment='center_baseline')

    for label in ax2.get_yticklabels(which='major'):
        label.set(rotation=30, verticalalignment='bottom')

    axs.set_ylabel('Time [{}]'.format(tz.tzname()))
    axs.set_title(plot_title)
    if plot_lines:
        TV_transmitter = 67.25  # [MHz]
        TV_audio_carrier = 71.75  # [MHz]

        f1 = 58.887
        f2 = 61.523
        f3 = 68.555
        f4 = 71.191

        axs.axvline(f1, color='k', ls=(0, (5, 10)))
        axs.axvline(f2, color='k', ls=(0, (5, 10)))
        axs.axvline(f3, color='k', ls=(0, (5, 10)))
        axs.axvline(f4, color='k', ls=(0, (5, 10)))
        axs.axvline(TV_transmitter, color='m', ls=(0, (5, 10)))
        axs.axvline(TV_audio_carrier, color='m', ls=(0, (5, 10)))

    plt.tight_layout()
    fig.colorbar(c, orientation='horizontal', label='PSD [units]', pad=0.10)

    plt.savefig(plot_filename, dpi=500)
