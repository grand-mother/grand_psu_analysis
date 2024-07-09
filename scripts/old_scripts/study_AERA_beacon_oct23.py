import uproot
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import utils_abl as ua
import argparse

from scipy.signal import butter, sosfilt, sosfiltfilt


f1 = freq1 = 58.887
f2 = freq2 = 61.523
f3 = freq3 = 68.555
f4 = freq4 = 71.191


def sin_wave(t, f):
    w = 2*np.pi * f
    return np.sin(w * t)



sample_freq = 500   # [MHz]
sample_period = 1/sample_freq # [us]
n_samples = 1024
n_samples2 = 1024 * 16



file1 = '../../data/auger/TD/td002021_f0030.root'  # this file containts the beacon data
file2 = '../../data/auger/GRANDfiles/td002015_f0001.root' # this one does not



tadc = uproot.concatenate({file1: "tadc"})
tadc2 = uproot.concatenate({file2: "tadc"})

trawv = uproot.concatenate({file1: "trawvoltage"})
du_list = ua.get_dulist(tadc)



idx = 151

request = 'trace_ch'
result, time_sec_trace, dtime_sec_trace = ua.get_column_for_given_du(tadc, request, idx)
traces_np1 = result[:, 0, 0:3].to_numpy()

result2, time_sec_trace2, dtime_sec_trace2 = ua.get_column_for_given_du(tadc2, request, idx)
traces_np2 = result2[:, 0, 0:3].to_numpy()


fft151_1 = np.abs(np.fft.rfft(traces_np1)**2)
fft151_2 = np.abs(np.fft.rfft(traces_np2)**2)




idx = 84

request = 'trace_ch'
result, time_sec_trace, dtime_sec_trace = ua.get_column_for_given_du(tadc, request, idx)
traces_np1 = result[:, 0, 0:3].to_numpy()

result2, time_sec_trace2, dtime_sec_trace2 = ua.get_column_for_given_du(tadc2, request, idx)
traces_np2 = result2[:, 0, 0:3].to_numpy()




# Plots of average PSD 

fft84_1 = np.abs(np.fft.rfft(traces_np1)**2)
fft84_2 = np.abs(np.fft.rfft(traces_np2)**2)


fft_freq = np.fft.rfftfreq(n_samples) * 500  # [MHz]

fft_freq2 = np.fft.rfftfreq(n_samples2) * 500  # [MHz]


plt.figure(11)
plt.clf()
plt.plot(fft_freq, fft84_1.mean(axis=0)[0], label='DU84, file1')
plt.plot(fft_freq, fft84_2.mean(axis=0)[0], label='DU84, file2')

plt.plot(fft_freq, fft151_1.mean(axis=0)[0], label='DU151, file1')
plt.plot(fft_freq, fft151_2.mean(axis=0)[0], label='DU151, file2')


plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')
plt.ylabel('NS-channel PSD [$ADC^2$]')
plt.xlabel('Frequency [MHz]')
plt.yscale('log')
plt.legend(loc=0)
plt.savefig('psd_beacon_84_151.png')



plt.figure(21)
plt.clf()
plt.plot(fft_freq, fft84_1.mean(axis=0)[1], label='DU84, file1')
plt.plot(fft_freq, fft84_2.mean(axis=0)[1], label='DU84, file2')

plt.plot(fft_freq, fft151_1.mean(axis=0)[1], label='DU151, file1')
plt.plot(fft_freq, fft151_2.mean(axis=0)[1], label='DU151, file2')


plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')
plt.ylabel('NS-channel PSD [$ADC^2$]')
plt.xlabel('Frequency [MHz]')
plt.yscale('log')
plt.xlim(50, 80)
plt.ylim(1e4, 1e8)
plt.legend(loc=0)
plt.savefig('psd_beacon_84_151_ch1_zoom.png')



plt.figure(111)
plt.clf()
plt.plot(fft_freq, fft84_1.mean(axis=0)[0], label='DU84, file1')
plt.plot(fft_freq, fft84_2.mean(axis=0)[0], label='DU84, file2')

plt.plot(fft_freq, fft151_1.mean(axis=0)[0], label='DU151, file1')
plt.plot(fft_freq, fft151_2.mean(axis=0)[0], label='DU151, file2')


plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')

plt.ylabel('NS-channel PSD [$ADC^2$]')
plt.xlabel('Frequency [MHz]')
plt.yscale('log')
plt.legend(loc=0)
plt.xlim(50, 80)
plt.ylim(1e4, 1e8)
plt.savefig('psd_beacon_84_151_zoom.png')





plt.figure(112)
plt.clf()
plt.plot(fft_freq, fft84_1.mean(axis=0)[0], 'C0',label='DU84, file1')
plt.plot(fft_freq, fft84_1.mean(axis=0)[0], 'C0.')

#plt.plot(fft_freq, fft84_2.mean(axis=0)[0], label='DU84, file2')

plt.plot(fft_freq, fft151_1.mean(axis=0)[0], 'C1',label='DU151, file1')

plt.plot(fft_freq, fft151_1.mean(axis=0)[0], 'C1.')

#plt.plot(fft_freq, fft151_2.mean(axis=0)[0], label='DU151, file2')


plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')

plt.ylabel('NS-channel PSD [$ADC^2$]')
plt.xlabel('Frequency [MHz]')
plt.yscale('log')
plt.legend(loc=0)
plt.xlim(50, 80)
plt.ylim(1e4, 1e8)
plt.savefig('psd_beacon_84_151_zoomonly_file1.png')







trace_id = 40
tr1a = traces_np1[trace_id]
tr2a = traces_np2[trace_id]

trace_id = 42
tr1b = traces_np1[trace_id]
tr2b = traces_np2[trace_id]


df1 = np.diff(fft_freq)[0]

f1 = freq1 = 58.887
f2 = freq2 = 61.523
f3 = freq3 = 68.555
f4 = freq4 = 71.191








# let's create some fake traces with the beacon

t = t1 = np.arange(n_samples) * 1/sample_freq
t2 = np.arange(n_samples2) * 1/sample_freq

trace1 = sin_wave(t, f1)
trace2 = sin_wave(t, f2)
trace3 = sin_wave(t, f3)
trace4 = sin_wave(t, f4)

tracea = sin_wave(t2, f1)
traceb = sin_wave(t2, f2)
tracec = sin_wave(t2, f3)
traced = sin_wave(t2, f4)



noise1 = np.random.randn(n_samples)
noise2 = np.random.randn(n_samples2)

tr_sim_nonoise = trace1 + trace2 + trace3 + trace4
tr_sim_noise = tr_sim_nonoise + noise1 * 1


tr_sim_nonoise2 = tracea + traceb + tracec + traced
tr_sim_noise2 = tr_sim_nonoise2 + noise2 * 1




def harmonic_filter(trace_in, n_df1, fft_freq, deltaf):
    ind1 = np.where(
        (fft_freq >= f1 - n_df1 * deltaf *2 ) *( fft_freq <= f1+n_df1 * 2* deltaf) +
        (fft_freq >= f2 - n_df1 * deltaf ) *( fft_freq <= f2+n_df1 * deltaf) + 
        (fft_freq >= f3 - n_df1 * deltaf *2 ) *( fft_freq <= f3+n_df1 * 2*deltaf) +
        (fft_freq >= f4 - n_df1 * deltaf ) *( fft_freq <= f4+n_df1 * deltaf)
    )

    rfft = np.fft.rfft(trace_in)
    rfft_copy = rfft.copy() * 0
    rfft_copy[ind1] = rfft[ind1]

    trace_out = np.fft.irfft(rfft_copy)
    rfft_out = np.fft.rfft(trace_out)
    return trace_out, rfft_out

n_trace = traces_np1.shape[0]




n_df1 = 1
for i in np.random.permutation(n_trace)[0:5]:
    noise1 = np.random.randn(n_samples)

    tr_sim_nonoise = trace1 + trace2 + trace3 + trace4
    tr_sim_noise = tr_sim_nonoise + noise1 * 1

    trace_with = traces_np1[i][0]
    trace_wout = traces_np2[i][0]

    tr_sim_nonoise_out, rfft_tr_sim_nonoise = harmonic_filter(tr_sim_nonoise, n_df1, fft_freq, df1)
    tr_sim_noise_out, rfft_tr_sim_noise = harmonic_filter(tr_sim_noise, n_df1, fft_freq, df1)

    tr_with_out, rfft_tr_with_out = harmonic_filter(trace_with, n_df1, fft_freq, df1)
    tr_wout_out, rfft_tr_wout_out = harmonic_filter(trace_wout, n_df1, fft_freq, df1)

    fig, axs = plt.subplots(4, 2, figsize=(15, 8))
    # Remove vertical space between axes
    fig.subplots_adjust(hspace=0)

    axs[0, 0].plot(tr_sim_nonoise, alpha=0.5)
    axs[0, 0].plot(tr_sim_nonoise_out, label='Sim. No noise')

    axs[1, 0].plot(tr_sim_noise, alpha=0.5)
    axs[1, 0].plot(tr_sim_noise_out, label='Sim, With noise')
    
    axs[2, 0].plot(trace_with, alpha=0.4)
    axs[2, 0].plot(tr_with_out, label='84-chX, file1')
    axs[3, 0].plot(trace_wout, alpha=0.4)
    axs[3, 0].plot(tr_wout_out, label='84-chX file2')

    axs[0, 0].legend(loc=1)
    axs[1, 0].legend(loc=1)
    axs[2, 0].legend(loc=1)
    axs[3, 0].legend(loc=1)

    axs[0, 1].plot(fft_freq, np.abs(rfft_tr_sim_nonoise)**2)
    axs[1, 1].plot(fft_freq, np.abs(rfft_tr_sim_noise)**2)
    axs[2, 1].plot(fft_freq, np.abs(rfft_tr_with_out)**2)
    axs[3, 1].plot(fft_freq, np.abs(rfft_tr_wout_out)**2)

    axs[0, 1].set_yscale('log')
    axs[1, 1].set_yscale('log')
    axs[2, 1].set_yscale('log')
    axs[3, 1].set_yscale('log')

    axs[0, 1].set_ylim(1e4, 1e6)
    axs[1, 1].set_ylim(1e4, 1e6)
    axs[2, 1].set_ylim(1e3, 1e8)
    axs[3, 1].set_ylim(1e3, 1e8)


    axs[0, 1].set_xlim(50, 80)
    axs[1, 1].set_xlim(50, 80)
    axs[2, 1].set_xlim(50, 80)
    axs[3, 1].set_xlim(50, 80)

    #axs[0].set_title('{}, event #{}, DU {} \n cluster {}'.format(base, trawv.event_number[ind_[i]], idx, lab))
    axs[3, 0].set_xlabel('sample id')
    axs[3, 1].set_xlabel('frequency [MHz]')
    

    axs[0, 0].set_title('Trace in time domain')
    axs[0, 1].set_title('Zoom on filtered Fourier domain ')

    axs[1, 0].set_ylabel('Amplitude')
    axs[0, 0].set_ylabel('Amplitude')
    axs[2, 0].set_ylabel('ADC')
    axs[3, 0].set_ylabel('ADC')

    axs[1, 1].set_ylabel('A.U.')
    axs[1, 1].set_ylabel('A.U.')
    axs[2, 1].set_ylabel('A.U.')
    axs[3, 1].set_ylabel('A.U.')
    plt.savefig('beacon_aera_GAA_84X_{}.png'.format(i))




rfft1a = np.fft.rfft(tr1a)[0]
rfft1a_copy = rfft1a.copy() *0
rfft1a_copy[ind1] = rfft1a[ind1]


tr1a_back = np.fft.irfft(rfft1a_copy)


rfft1 = np.fft.rfft(tr_sim_nonoise)

rfft1_copy = rfft1.copy() *0
rfft1_copy[ind1] = rfft1[ind1]

trace_bakc1 = np.fft.irfft(rfft1_copy)




plt.figure(56)
plt.clf()
plt.plot(t, trace_bakc1)
plt.plot(t, tr_sim_nonoise)
plt.plot(t, tr1a_back)


fft1a = np.fft.rfft(tr1a)[0]
fft1ab = np.fft.rfft(tr1a_back)

plt.figure(456)
plt.clf()
plt.plot(fft_freq, np.abs(fft1a)**2)
plt.plot(fft_freq, np.abs(fft1ab)**2)















psd_sim_nn = np.abs(np.fft.rfft(tr_sim_nonoise)**2)
psd_sim_n = np.abs(np.fft.rfft(tr_sim_noise)**2)


psd_sim_nn_2 = np.abs(np.fft.rfft(tr_sim_nonoise2)**2)
psd_sim_n_2 = np.abs(np.fft.rfft(tr_sim_noise2)**2)

plt.figure(1)
plt.clf()
plt.plot(t, tr_sim_nonoise)
plt.plot(t2, tr_sim_nonoise2)

plt.xlabel('time [$\mu$s]')
plt.title('Noiseless simulated beacon signal')
plt.title('simulated ch X traces')
#plt.savefig('sim_no_noise.png')




plt.figure(2)
plt.clf()
plt.plot(t, tr_sim_noise)
plt.xlabel('time [$\mu$s]')
plt.title('Noised simulated beacon signal')
plt.title('simulated ch X traces')
#plt.savefig('sim_noise.png')


#### plot of the PSD ot the sims
plt.figure(3)
plt.clf()

df1 = np.diff(fft_freq)[0]
df2 = np.diff(fft_freq2)[0]


#plt.plot(fft_freq, psd_sim_n, label='Sim noised')
plt.plot(fft_freq, psd_sim_nn/df1/n_samples**2, '-',label ='Sim no noise')
plt.plot(fft_freq, psd_sim_nn/df1/n_samples**2, '.')


#plt.plot(fft_freq2, psd_sim_n_2, label='Sim noised')
plt.plot(fft_freq2, psd_sim_nn_2/df2/n_samples2**2, '-', label ='Sim no noise2')
plt.plot(fft_freq2, psd_sim_nn_2/df2/n_samples2**2, '.')


plt.ylabel('NS-channel PSD [$ADC^2$]')
plt.xlabel('Frequency [MHz]')
#plt.yscale('log')
plt.legend(loc=0)
#plt.savefig('psd_sims.png')










plt.figure(1)
plt.clf()
plt.plot(t, tr1a[0], label='tr data1')
plt.plot(t, tr2a[0], label='tr data 2')
plt.plot(t, tr_sim_noise, label='sim noise')
plt.plot(t, tr_sim_nonoise, label='sim no noise')
plt.xlabel('time [ns]')
plt.title('real and simulated ch X traces')
plt.legend(loc=0)


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



psd1a = np.abs(np.fft.rfft(tr1a[0])**2)
psd2a = np.abs(np.fft.rfft(tr2a[0])**2)

psd1b = np.abs(np.fft.rfft(tr1b[0])**2)
psd2b = np.abs(np.fft.rfft(tr2b[0])**2)



plt.figure(2)
plt.clf()
plt.plot(fft_freq, psd1a)
plt.plot(fft_freq, psd2a)
plt.plot(fft_freq, psd1b)
plt.plot(fft_freq, psd2b)

#plt.plot(fft_freq, psd_sim_nn)
#plt.plot(fft_freq, psd_sim_n)
plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')

plt.yscale('log')



#### Filter the simulations fisrt


def filter1(tr, f1, f2, f3, f4, order):
    out, sos = butter_bandpass_filt(tr, f1-0.5, f4+0.5, 500, order)
    return out


def filter2(tr, f1, f2, f3, f4, order):

    tr1_, _ = butter_bandpass_filt(tr, f1-0.25, f1+0.25, 500, order)
    tr2_, _ = butter_bandpass_filt(tr, f2-0.25, f2+0.25, 500, order)
    tr3_, _ = butter_bandpass_filt(tr, f3-0.25, f3+0.25, 500, order)
    tr4_, _ = butter_bandpass_filt(tr, f4-0.25, f4+0.25, 500, order)
    out = tr1_ + tr2_ + tr3_ + tr4_
    return out




tr_sim_nn_fl1_o1 = filter1(tr_sim_nonoise, f1, f2, f3, f4, 1)
tr_sim_n_fl1_o1 = filter1(tr_sim_noise, f1, f2, f3, f4, 1)


tr_sim_nn_fl1_o3 = filter1(tr_sim_nonoise, f1, f2, f3, f4, 3)
tr_sim_n_fl1_o3 = filter1(tr_sim_noise, f1, f2, f3, f4, 3)

tr_sim_nn_fl1_o5 = filter1(tr_sim_nonoise, f1, f2, f3, f4, 5)
tr_sim_n_fl1_o5 = filter1(tr_sim_noise, f1, f2, f3, f4, 5)


tr_sim_nn_fl2_o1 = filter2(tr_sim_nonoise, f1, f2, f3, f4, 1)
tr_sim_n_fl2_o1 = filter2(tr_sim_noise, f1, f2, f3, f4, 1)


tr_sim_nn_fl2_o3 = filter2(tr_sim_nonoise, f1, f2, f3, f4, 3)
tr_sim_n_fl2_o3 = filter2(tr_sim_noise, f1, f2, f3, f4, 3)


tr_sim_nn_fl2_o5 = filter2(tr_sim_nonoise, f1, f2, f3, f4, 5)
tr_sim_n_fl2_o5 = filter2(tr_sim_noise, f1, f2, f3, f4, 5)





psd_sim_nn_fl1_o1 = np.abs(np.fft.rfft(tr_sim_nn_fl1_o1)**2)
psd_sim_n_fl1_o1 = np.abs(np.fft.rfft(tr_sim_n_fl1_o1)**2)

psd_sim_nn_fl1_o3 = np.abs(np.fft.rfft(tr_sim_nn_fl1_o3)**2)
psd_sim_n_fl1_o3 = np.abs(np.fft.rfft(tr_sim_n_fl1_o3)**2)

psd_sim_nn_fl1_o5 = np.abs(np.fft.rfft(tr_sim_nn_fl1_o5)**2)
psd_sim_n_fl1_o5 = np.abs(np.fft.rfft(tr_sim_n_fl1_o5)**2)




#### plot of the PSD ot the sims
plt.figure(41)
plt.clf()

plt.plot(fft_freq, psd_sim_nn_fl1_o1, '-', label='psd_sim_nn_fl1_o1')
plt.plot(fft_freq, psd_sim_nn_fl1_o3, '-',  label='psd_sim_nn_fl1_o3')
plt.plot(fft_freq, psd_sim_nn_fl1_o5, '-',  label='psd_sim_nn_fl1_o5')

plt.plot(fft_freq, psd_sim_nn, label ='Sim no noise')

plt.ylabel('NS-channel PSD [$ADC^2$]')
plt.xlabel('Frequency [MHz]')#
#plt.yscale('log')
plt.legend(loc=0)



plt.figure(42)
plt.clf()

plt.plot(fft_freq, psd_sim_n_fl1_o1, '-',  label='psd_sim_n_fl1_o1')
plt.plot(fft_freq, psd_sim_n_fl1_o3, '-',  label='psd_sim_n_fl1_o3')
plt.plot(fft_freq, psd_sim_n_fl1_o5, '-',  label='psd_sim_n_fl1_o5')

plt.plot(fft_freq, psd_sim_n, label='Sim noised')

plt.ylabel('NS-channel PSD [$ADC^2$]')
plt.xlabel('Frequency [MHz]')#
#plt.yscale('log')
plt.legend(loc=0)





plt.figure(31)
plt.clf()

plt.plot(t, tr_sim_nonoise)

plt.plot(t, tr_sim_nn_fl1_o5, label='nn fl1 o5')
plt.plot(t, tr_sim_nn_fl1_o3, label='nn fl1 o3')
plt.plot(t, tr_sim_nn_fl1_o1, label='nn fl1 o1')

plt.legend(loc=0)
plt.title('Sim no noise filter1')


plt.figure(32)
plt.clf()

plt.plot(t, tr_sim_nonoise)


plt.plot(t, tr_sim_nn_fl2_o5, label='nn fl2 o5')
plt.plot(t, tr_sim_nn_fl2_o3, label='nn fl2 o3')
plt.plot(t, tr_sim_nn_fl2_o1, label='nn fl2 o1')
plt.legend(loc=0)
plt.title('Sim no noise filter2')





plt.figure(41)
plt.clf()
plt.plot(t, tr_sim_noise)

plt.plot(t, tr_sim_n_fl1_o5, label='n fl1 o5')
plt.plot(t, tr_sim_n_fl1_o3, label='n fl1 o3')
plt.plot(t, tr_sim_n_fl1_o1, label='n fl1 o1')

plt.legend(loc=0)
plt.title('sim noise filter1')


plt.figure(42)
plt.clf()
plt.plot(t, tr_sim_noise)

plt.plot(t, tr_sim_n_fl2_o5, label='n fl2 o5')
plt.plot(t, tr_sim_n_fl2_o3, label='n fl2 o3')
plt.plot(t, tr_sim_n_fl2_o1, label='n fl2 o1')
plt.legend(loc=0)
plt.title('sim noise filter2')




##### filtering the data with filter2


tr1a_fl2_o1 = filter2(tr1a[0], f1, f2, f3, f4, 1)
tr1a_fl2_o3 = filter2(tr1a[0], f1, f2, f3, f4, 3)
tr1a_fl2_o5 = filter2(tr1a[0], f1, f2, f3, f4, 5)


tr2a_fl2_o1 = filter2(tr2a[0], f1, f2, f3, f4, 1)
tr2a_fl2_o3 = filter2(tr2a[0], f1, f2, f3, f4, 3)
tr2a_fl2_o5 = filter2(tr2a[0], f1, f2, f3, f4, 5)


tr1a_fl1_o1 = filter1(tr1a[0], f1, f2, f3, f4, 1)
tr1a_fl1_o3 = filter1(tr1a[0], f1, f2, f3, f4, 3)
tr1a_fl1_o5 = filter1(tr1a[0], f1, f2, f3, f4, 5)


tr2a_fl1_o1 = filter1(tr2a[0], f1, f2, f3, f4, 1)
tr2a_fl1_o3 = filter1(tr2a[0], f1, f2, f3, f4, 3)
tr2a_fl1_o5 = filter1(tr2a[0], f1, f2, f3, f4, 5)




plt.figure(51)
plt.clf()
#plt.plot(t, tr1a[0], label='data 1a')
plt.plot(t, tr1a_fl2_o1, label='data1a, fl2, o1')
plt.plot(t, tr1a_fl2_o3, label='data1a, fl2, o3')
plt.plot(t, tr2a_fl2_o1, label='data2a, fl2, o1')
plt.plot(t, tr2a_fl2_o3, label='data2a, fl2, o3')
plt.legend(loc=0)
plt.title('data filter2')




plt.figure(52)
plt.clf()
plt.plot(t, tr1a[0], label='data 1a')
plt.plot(t, tr1a_fl1_o1, label='data1a, fl1, o1')
#plt.plot(t, tr1a_fl1_o3, label='data1a, fl1, o3')
plt.plot(t, tr2a_fl1_o1, label='data2a, fl1, o1')
#plt.plot(t, tr2a_fl1_o3, label='data2a, fl1, o3')
plt.legend(loc=0)
plt.xlabel('time [ns]')

plt.title('data filter1')

plt.savefig('data_fileter1.png')



y1, sos = butter_bandpass_filt(tr[0], f1-2.5, f4+2.5, 500, 3)

y1_1, _ = butter_bandpass_filt(tr[0], f1-0.5, f1+0.5, 500, 1)
y1_2, _ = butter_bandpass_filt(tr[0], f2-0.5, f2+0.5, 500, 1)
y1_3, _ = butter_bandpass_filt(tr[0], f3-0.5, f3+0.5, 500, 1)
y1_4, _ = butter_bandpass_filt(tr[0], f4-0.5, f4+0.5, 500, 1)

y1_f = y1_1 + y1_2 + y1_3 + y1_4



y2, sos2 = butter_bandpass_filt(tr5, f1-2.5, f4+2.5, 500, 3)

y2, sos2 = butter_bandpass_filt(tr2[0], f1-2.5, f4+2.5, 500, 3)

y2_1, _ = butter_bandpass_filt(tr5, f1-0.5, f1+0.5, 500, 1)
y2_2, _ = butter_bandpass_filt(tr5, f2-0.5, f2+0.5, 500, 1)
y2_3, _ = butter_bandpass_filt(tr5, f3-0.5, f3+0.5, 500, 1)
y2_4, _ = butter_bandpass_filt(tr5, f4-0.5, f4+0.5, 500, 1)

y2_f = y2_1 + y2_2 + y2_3 + y2_4



plt.figure(2)
plt.clf()
plt.plot(tr[0], label='G@A trace')
plt.plot(tr2[0], label='G@A trace')

plt.plot(y1_f, label='data filtered')

plt.plot(tr5, label='sim')
plt.plot(y2, label='sim filtered')
plt.legend(loc=0)


plt.figure(5)
plt.clf()

plt.plot(fft_freq, np.abs(np.fft.rfft(tr5)**2), label='sim')
plt.plot(fft_freq, np.abs(np.fft.rfft(y2)**2), label='sim filt1')
plt.plot(fft_freq, np.abs(np.fft.rfft(y2_f)**2), label='sim filt2')
plt.yscale('log')
plt.legend(loc=0)




plt.figure(6)
plt.clf()

plt.plot(fft_freq, np.abs(np.fft.rfft(tr[0])**2), label='data')
plt.plot(fft_freq, np.abs(np.fft.rfft(y1)**2), label='data filt1')
plt.plot(fft_freq, np.abs(np.fft.rfft(y1_f)**2), label='data filt2')
plt.yscale('log')
plt.legend(loc=0)
plt.axvline(f1, color='k')
plt.axvline(f2, color='k')
plt.axvline(f3, color='k')
plt.axvline(f4, color='k')


#plt.plot(tr5)
#y2 = butter_bandpass_filt(tr[0], f2-.5, f2+0.5, 500, 20)
#y3 = butter_bandpass_filt(tr[0], f3-.5, f3+0.5, 500, 20)
#y4 = butter_bandpass_filt(tr[0], f4-.5, f4+0.5, 500, 20)