import numpy as np
import grand_psu_lib.utils.utils as utils
#import utils as utils


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

    std_ch0 = trace2.std(axis=2)[:, 0]
    std_ch1 = trace2.std(axis=2)[:, 1]
    std_ch2 = trace2.std(axis=2)[:, 2]

    return date_array, std_ch0, std_ch1, std_ch2, bl, date_bl, gps_temp, date_gpstemp


def return_psd(trace_array, sampling_rate):
    # make sure the trace array has the samples vbalues in the last dimension
    # units of the psd are [trace_array]^2/ [sampling_rate]
    fft = np.fft.rfft(trace_array)
    psd = np.abs(fft)**2
    psd[..., 1:-1] *= 2
    N = trace_array.shape[-1]
    print(N)
    psd = psd / N / sampling_rate
    return psd
    
