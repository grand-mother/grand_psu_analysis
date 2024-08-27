import numpy as np
from scipy.signal import butter, sosfiltfilt
import grand_psu_lib.utils.utils as utils


def get_trace_std_freqband(tadc, trawv, fft_freq, f1, f2, du_number):

    request = 'trace_ch'
    result, date_array = utils.get_column_for_given_du(tadc, request, du_number)
    traces_np = result[:, 0, 0:3].to_numpy()

    request = 'battery_level'
    bl, date_bl = utils.get_column_for_given_du(trawv, request, du_number)
    bl = bl.to_numpy()

    request = 'gps_temp'
    gps_temp, date_gpstemp = utils.get_column_for_given_du(trawv, request, du_number)
    gps_temp = gps_temp.to_numpy()

    ind = np.where((fft_freq < f1) + (fft_freq > f2))[0]

    fft_np = np.fft.rfft(traces_np)
    fft_np[:, :, ind] = 0

    trace2 = np.fft.irfft(fft_np)
    # return trace2

    std_ch0 = trace2.std(axis=2)[:, 0]
    std_ch1 = trace2.std(axis=2)[:, 1]
    std_ch2 = trace2.std(axis=2)[:, 2]

    return date_array, std_ch0, std_ch1, std_ch2, bl, date_bl, gps_temp, date_gpstemp


def get_filtered_traces(trace_array, fft_freq, f1, f2):

    ind = np.where((fft_freq > f1) * (fft_freq < f2))[0]

    fft_np = np.fft.rfft(trace_array)
    fft_np[:, :, ind] = 0

    trace2 = np.fft.irfft(fft_np)
    return trace2


def get_filtered_traces_inband(trace_array, fft_freq, f1, f2):

    ind = np.where((fft_freq < f1) + (fft_freq > f2))[0]

    fft_np = np.fft.rfft(trace_array)
    fft_np[:, :, ind] = 0

    trace2 = np.fft.irfft(fft_np)
    return trace2


def return_psd(trace_array, sampling_frequency):
    # make sure the trace array has the samples vbalues in the last dimension
    # units of the psd are [trace_array]^2/ [sampling_rate]
    fft = np.fft.rfft(trace_array)
    psd = np.abs(fft)**2
    psd[..., 1:-1] *= 2
    N = trace_array.shape[-1]
    print(N)
    psd = psd / N / sampling_frequency
    return psd


def return_psdx(trace_array, trace_array2, sampling_rate):
    # make sure the trace array has the samples vbalues in the last dimension
    # units of the psd are [trace_array]^2/ [sampling_rate]
    fft = np.fft.rfft(trace_array)
    fft2 = np.fft.rfft(trace_array2)
    psd = np.abs(fft * np.conj(fft2))
    psd[..., 1:-1] *= 2
    N = trace_array.shape[-1]
    print(N)
    psd = psd / N / sampling_rate
    return psd


def harmonic_filter(trace_in,  arg_freqs_list):

    rfft = np.fft.rfft(trace_in)
    rfft_copy = rfft.copy() * 0
    rfft_copy[arg_freqs_list] = rfft[arg_freqs_list]

    trace_out = np.fft.irfft(rfft_copy)
    rfft_out = np.fft.rfft(trace_out)
    return trace_out, rfft_out


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filt(data, lowcut, highcut, fs, order):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    y = sosfiltfilt(sos, data)
    return y, sos


def filter1(tr, f1, f2, f3, f4, order):
    out, sos = butter_bandpass_filt(tr, f1-0.5, f4+0.5, 500, order)
    return out


def filter2(tr, f1, f2, f3, f4, df, order):

    tr1_, _ = butter_bandpass_filt(tr, f1-df, f1+df, 500, order)
    tr2_, _ = butter_bandpass_filt(tr, f2-df, f2+df, 500, order)
    tr3_, _ = butter_bandpass_filt(tr, f3-df, f3+df, 500, order)
    tr4_, _ = butter_bandpass_filt(tr, f4-df, f4+df, 500, order)
    out = tr1_ + tr2_ + tr3_ + tr4_
    return out
