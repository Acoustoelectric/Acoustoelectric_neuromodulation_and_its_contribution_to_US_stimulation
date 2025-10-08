'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date: 23.10.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.stats import ttest_ind
from scipy.signal import hilbert
from scipy.signal import find_peaks
import scipy.stats
from scipy.stats import pearsonr
import pandas as pd
from scipy import signal
# 
# 
# Try with a differently generated VEP filter. 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
# 
savepath = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/temperature/temperature_data/'
ae_filename   = savepath + 'ae0.5-4.npz'     # duration 6
# 
duration        = 6
m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
emg_channel     = 2 
# 
brain_gain      = 10
emg_gain        = 500 


band_limit      = 80 
Fs              = 10000
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
cut             = 1000
sos_low_band    = iirfilter(17, [10], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# get the frequencies ready
start_idx       = int(0*Fs) 
end_idx         = int(duration*Fs) 


# 
d            = np.load(ae_filename, mmap_mode='r')
print (list(d.keys()) )
data = d['aedata']
[dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal] = data 
samples_per_second = len(brain_signal)/duration
print ('samples per second:', samples_per_second)
print (brain_signal)
# 
# 
brain_signal            = sosfiltfilt(sos_low_band, brain_signal)
# 
# new_Fs               = 10000
# downsampling_factor  = int(Fs/new_Fs)
# b_subsampled         = brain_signal[::downsampling_factor]

# print (b_subsampled)

ae = [[24.5,    24.6,   24.6,   24.6,   24.6,   24.6],
[24.7,  25.1,   24.6,   24.6,   24.6,   24.8],
[25.5,  25.5,   25.2,   25, 25.5,   25.6],
[25.7,  25.7,   25.7,   25.7,   26, 26],
[25.9,  25.9,   25.9,   25.9,   26.2,   26],
[25.2,  26, 26, 26, 25.9,   26.1]]
# 
#
from scipy import stats
from scipy.interpolate import interp1d 


x = [1,2,3,4,5,6]
y = np.mean(ae,1)
print (x,y)
upsampling_factor = Fs  
x_upsampled = np.linspace(1,6,6*upsampling_factor)
print ('up',len(x_upsampled))
f = interp1d(x,y,kind='linear')
y_upsampled = f(x_upsampled)

print ('len',len(y_upsampled))
# x = np.mean(ae,1)
# y = b_subsampled

x = brain_signal/(np.max(brain_signal))
y = y_upsampled/(np.max(y_upsampled))
res = stats.pearsonr(x,y)
print ('pearson r,p:',res )
# print (x,y)




fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(x,'k')
plt.plot(y,'r')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.set_ylim([-vlim,vlim])
# ax.set_xlim([0,duration])
plt.tight_layout()
# plot_filename = 'ae.png'
# plt.savefig(plot_filename)
plt.show()


# ae_fsignal     = (1e6*data[m_channel]/brain_gain)
# ae_emgsignal   = (1e6*data[emg_channel]/emg_gain)
# ae_rfsignal    = 10*data[rf_channel]
# ae_vsignal     = 10*data[v_channel]
# ae_isignal     = -5*data[i_channel]/49.9 

# ae_low_signal  = sosfiltfilt(sos_low_band, ae_fsignal)


# # ae_emg_signal = ae_emg
# ae_pp_amp      = np.max(ae_low_signal[start_idx:end_idx]) - np.min(ae_low_signal[start_idx:end_idx])
# print ('ae amplitude:',ae_pp_amp)

# data          = np.load(p_filename)
# p_fsignal     = (1e6*data[m_channel]/brain_gain)
# p_emgsignal   = (1e6*data[emg_channel]/emg_gain)
# p_rfsignal    = 10*data[rf_channel]
# p_vsignal     = 10*data[v_channel]
# p_low_signal  = sosfiltfilt(sos_low_band, p_fsignal)
# p_emg  = sosfiltfilt(sos_emg_band, p_emgsignal)
# p_emg_signal  = mains_stop(p_emg)
# h_p_emg       = abs(hilbert(p_emg_signal))
# hp_emg_signal  = sosfiltfilt(sos_emg_hilbert, h_p_emg)

# # p_emg_signal  = sosfiltfilt(sos_emg_stop, p_emg_signal)
# p_pp_amp      = np.max(p_low_signal[start_idx:end_idx]) - np.min(p_low_signal[start_idx:end_idx])
# print ('p amplitude:',p_pp_amp)

# data          = np.load(v_filename)
# v_fsignal     = (1e6*data[m_channel]/brain_gain)
# v_emgsignal   = (1e6*data[emg_channel]/emg_gain)
# v_rfsignal    = 10*data[rf_channel]
# v_vsignal     = 10*data[v_channel]
# v_low_signal  = sosfiltfilt(sos_low_band, v_fsignal)
# v_emg  = sosfiltfilt(sos_emg_band, v_emgsignal)
# v_emg_signal  = mains_stop(v_emg)
# h_v_emg            = abs(hilbert(v_emg_signal))
# hv_emg_signal      = sosfiltfilt(sos_emg_hilbert, h_v_emg)

# v_pp_amp      = np.max(v_low_signal[start_idx:end_idx]) - np.min(v_low_signal[start_idx:end_idx])
# print ('v amplitude:',v_pp_amp)

# start_pause = int(0*Fs)
# end_pause = int(duration*Fs)
# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]
# # 
# def find_nearest(array, value):
#     idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
#     return idx


# df_idx = find_nearest(frequencies,frequency)
# fft_m = fft(ae_fsignal[start_pause:end_pause])
# fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

# fft_p = fft(p_fsignal[start_pause:end_pause])
# fft_p = np.abs(2.0/(end_pause-start_pause) * (fft_p))[1:(end_pause-start_pause)//2]
# fft_v = fft(v_fsignal[start_pause:end_pause])
# fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]

# # fft_v = fft(v_vsignal[start_pause:end_pause])
# # fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]


# fft_ae_emg = fft(hae_emg_signal[start_pause:end_pause])
# fft_ae_emg = np.abs(2.0/(end_pause-start_pause) * (fft_ae_emg))[1:(end_pause-start_pause)//2]

# fft_p_emg = fft(hp_emg_signal[start_pause:end_pause])
# fft_p_emg = np.abs(2.0/(end_pause-start_pause) * (fft_p_emg))[1:(end_pause-start_pause)//2]

# fft_v_emg = fft(hv_emg_signal[start_pause:end_pause])
# fft_v_emg = np.abs(2.0/(end_pause-start_pause) * (fft_v_emg))[1:(end_pause-start_pause)//2]



# print ('ae df:',2*fft_m[df_idx])
# # 
# # fontsize on plots
# f       = 18
# vlim    = 30
# plim    = 40 
# clim    = 0.1 *1000
# siglim  = 1100
# emglim  = 10
# hilblim = 25
# emg_fft_ylim = 5
# brain_fft_ylim = 1000
# # Turn interactive plotting off
# # plt.ioff()

# # f       = 18
# # vlim    = 100
# # plim    = 100 
# # clim    = 0.1 *1000
# # siglim  = 30000
# # emglim  = 100
# # hilblim = 50

# # # # # # AE plots # # # # 
# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t, ae_rfsignal,'grey')
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.set_ylim([-plim,plim])
# ax.set_xlim([0,duration])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_rf.png'
# plt.savefig(plot_filename)
# plt.show()
# # plt.close(fig)


# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t, ae_vsignal,'pink')
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.set_ylim([-vlim,vlim])
# ax.set_xlim([0,duration])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_v.png'
# plt.savefig(plot_filename)
# plt.show()
# # plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# # plt.plot(t,20*ae_emg_signal,'r')
# plt.plot(t,ae_low_signal,'k')
# ax.set_ylim([-siglim,siglim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_signal.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,ae_emg_signal,'k')
# ax.set_ylim([-emglim,emglim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_emg_signal.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_m,'k')
# ax.set_xlim([0,10])
# # ax.set_xlim([0,cut])
# ax.set_ylim([0,brain_fft_ylim])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_fft_signal.png'
# plt.savefig(plot_filename)
# plt.show()
# # plt.close(fig)


# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_ae_emg,'r')
# ax.set_xlim([0,10])
# # ax.set_xlim([0,cut])
# ax.set_ylim([0,emg_fft_ylim])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_fft_emg_signal.png'
# plt.savefig(plot_filename)
# plt.show()
# # plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,hae_emg_signal,'r')
# ax.set_ylim([0,hilblim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'ae_hilbert_emg_signal.png'
# plt.savefig(plot_filename)
# plt.show()

# # # # # # P plots # # # # 
# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t, p_rfsignal,'grey')
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.set_ylim([-plim,plim])
# ax.set_xlim([0,duration])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_rf.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t, p_vsignal,'pink')
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.set_ylim([-vlim,vlim])
# ax.set_xlim([0,duration])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_v.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,p_low_signal,'k')
# ax.set_ylim([-siglim,siglim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_signal.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,p_emg_signal,'k')
# ax.set_ylim([-emglim,emglim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_emg_signal.png'
# plt.savefig(plot_filename)
# plt.close(fig)


# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_p,'k')
# ax.set_xlim([0,10])
# ax.set_ylim([0,brain_fft_ylim])
# # ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# # plt.yticks([])
# # plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_fft_signal.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_p_emg,'r')
# ax.set_xlim([0,10])
# # ax.set_xlim([0,cut])
# # ax.set_xlim([0,duration])
# ax.set_ylim([0,emg_fft_ylim])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_fft_emg_signal.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,hp_emg_signal,'r')
# ax.set_ylim([0,hilblim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_hilbert_emg_signal.png'
# plt.savefig(plot_filename)
# plt.show()

# # # # # # V plots # # # # 
# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t, v_rfsignal,'grey')
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.set_ylim([-plim,plim])
# ax.set_xlim([0,duration])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'v_rf.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t, v_vsignal,'pink')
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.set_ylim([-vlim,vlim])
# ax.set_xlim([0,duration])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'v_v.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,v_low_signal,'k')
# ax.set_ylim([-siglim,siglim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'v_signal.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,v_emg_signal,'k')
# ax.set_ylim([-emglim,emglim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'v_emg_signal.png'
# plt.savefig(plot_filename)
# plt.close(fig)


# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_v,'k')
# ax.set_xlim([0,10])
# ax.set_ylim([0,brain_fft_ylim])
# # ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# # plt.yticks([])
# # plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'v_fft_signal.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)


# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_v_emg,'r')
# ax.set_xlim([0,10])
# ax.set_ylim([0,emg_fft_ylim])
# # ax.set_xlim([0,cut])
# # ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'v_fft_emg_signal.png'
# plt.savefig(plot_filename)
# plt.show()
# # plt.close(fig)

# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(t,hv_emg_signal,'r')
# ax.set_ylim([0,hilblim])
# ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'v_hilbert_emg_signal.png'
# plt.savefig(plot_filename)
# plt.close(fig)

# # # # # # END PLOTS # # # # 