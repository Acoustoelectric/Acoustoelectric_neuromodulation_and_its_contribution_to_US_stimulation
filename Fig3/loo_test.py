'''

Title: aedc single file representative plot. 

Author: Jean Rintoul
Date: 08.07.2025

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
# savepath      = '/Volumes/extras/aedc/e146t7_aedc5/'
# outpath       = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/aedc/images/'
# # 
# ae_filename   = savepath + 'tae_g10_stream.npy'     # duration 6
# p_filename    = savepath + 'tp_p0.25_g10_stream.npy'   # duration 6 
# v_filename    = savepath + 'tv_v0.12_g10_stream.npy'  # duration 6
# 
# 
# savepath      = '/Volumes/extras/aedc/e147t8_aedc/'
# ae_filename   = savepath + 'tae_g500_stream.npy'     # duration 6
# p_filename    = savepath + 'tp_p0.25_g500_stream.npy'   # duration 6 
# v_filename    = savepath + 'tv_v8_g500_stream.npy'  # duration 6


# 
# 
savepath = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/final_paper_support_figs_code/Fig3/single_representative_file_e146t7aedc5/'
outpath  = savepath

ae_filename   = savepath + 'tae_g10_stream.npy'     # duration 6
p_filename    = savepath + 'tp_p0.25_g10_stream.npy'   # duration 6 
v_filename    = savepath + 'tv_v0.12_g10_stream.npy'  # duration 6

duration        = 6
frequency       = 1.0

m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
emg_channel     = 2 
# 
brain_gain      = 10
emg_gain        = 500 


band_limit      = 80 
Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
cut             = 1000
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_band    = iirfilter(17, [100,1000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_hilbert    = iirfilter(17, [10], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# mains = [50,100,150,300]
# sos_emg_stop    = iirfilter(17, [50-2,50+2], rs=60, btype='bandstop',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
mains_fs = np.arange(50,1000,50)
# print('mains:',mains_fs)
def mains_stop(signal):
    mains = mains_fs
 
    for i in range(len(mains)):
        sos_mains_stop    = iirfilter(17, [mains[i]-4,mains[i]+4], rs=60, btype='bandstop',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        signal = sosfiltfilt(sos_mains_stop, signal)
    return signal


# get the frequencies ready
start_idx       = int(0*Fs) 
end_idx         = int(duration*Fs) 
# 
data           = np.load(ae_filename)
ae_fsignal     = (1e6*data[m_channel]/brain_gain)
ae_emgsignal   = (1e6*data[emg_channel]/emg_gain)
ae_rfsignal    = 10*data[rf_channel]
ae_vsignal     = 10*data[v_channel]
ae_isignal     = -5*data[i_channel]/49.9 

ae_low_signal  = sosfiltfilt(sos_low_band, ae_fsignal)
ae_emg  = sosfiltfilt(sos_emg_band, ae_emgsignal)
ae_emg_signal  = mains_stop(ae_emg)
h_ae_emg       = abs(hilbert(ae_emg_signal))
hae_emg_signal  = sosfiltfilt(sos_emg_hilbert, h_ae_emg)

# ae_emg_signal = ae_emg
ae_pp_amp      = np.max(ae_low_signal[start_idx:end_idx]) - np.min(ae_low_signal[start_idx:end_idx])
print ('ae amplitude:',ae_pp_amp)

data          = np.load(p_filename)
p_fsignal     = (1e6*data[m_channel]/brain_gain)
p_emgsignal   = (1e6*data[emg_channel]/emg_gain)
p_rfsignal    = 10*data[rf_channel]
p_vsignal     = 10*data[v_channel]
p_low_signal  = sosfiltfilt(sos_low_band, p_fsignal)
p_emg  = sosfiltfilt(sos_emg_band, p_emgsignal)
p_emg_signal  = mains_stop(p_emg)
h_p_emg       = abs(hilbert(p_emg_signal))
hp_emg_signal  = sosfiltfilt(sos_emg_hilbert, h_p_emg)

# p_emg_signal  = sosfiltfilt(sos_emg_stop, p_emg_signal)
p_pp_amp      = np.max(p_low_signal[start_idx:end_idx]) - np.min(p_low_signal[start_idx:end_idx])
print ('p amplitude:',p_pp_amp)

data          = np.load(v_filename)
v_fsignal     = (1e6*data[m_channel]/brain_gain)
v_emgsignal   = (1e6*data[emg_channel]/emg_gain)
v_rfsignal    = 10*data[rf_channel]
v_vsignal     = 10*data[v_channel]
v_low_signal  = sosfiltfilt(sos_low_band, v_fsignal)
v_emg  = sosfiltfilt(sos_emg_band, v_emgsignal)
v_emg_signal  = mains_stop(v_emg)
h_v_emg            = abs(hilbert(v_emg_signal))
hv_emg_signal      = sosfiltfilt(sos_emg_hilbert, h_v_emg)

v_pp_amp      = np.max(v_low_signal[start_idx:end_idx]) - np.min(v_low_signal[start_idx:end_idx])
print ('v amplitude:',v_pp_amp)

start_pause = int(0*Fs)
end_pause = int(duration*Fs)
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx


df_idx = find_nearest(frequencies,frequency)
fft_m = fft(ae_fsignal[start_pause:end_pause])
fft_m = np.abs(2.0/(end_pause-start_pause) * (fft_m))[1:(end_pause-start_pause)//2]

fft_p = fft(p_fsignal[start_pause:end_pause])
fft_p = np.abs(2.0/(end_pause-start_pause) * (fft_p))[1:(end_pause-start_pause)//2]
fft_v = fft(v_fsignal[start_pause:end_pause])
fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]

# fft_v = fft(v_vsignal[start_pause:end_pause])
# fft_v = np.abs(2.0/(end_pause-start_pause) * (fft_v))[1:(end_pause-start_pause)//2]


fft_ae_emg = fft(ae_emgsignal[start_pause:end_pause])
fft_ae_emg = np.abs(2.0/(end_pause-start_pause) * (fft_ae_emg))[1:(end_pause-start_pause)//2]

fft_p_emg = fft(p_emgsignal[start_pause:end_pause])
fft_p_emg = np.abs(2.0/(end_pause-start_pause) * (fft_p_emg))[1:(end_pause-start_pause)//2]

fft_v_emg = fft(v_emgsignal[start_pause:end_pause])
fft_v_emg = np.abs(2.0/(end_pause-start_pause) * (fft_v_emg))[1:(end_pause-start_pause)//2]



print ('ae df:',2*fft_m[df_idx])
# 
# fontsize on plots
f       = 18
vlim    = 100
plim    = 100 
clim    = 0.1 *1000
siglim  = 6000
emglim  = 60
hilblim = 30
# Turn interactive plotting off
# plt.ioff()

f       = 18
vlim    = 100
plim    = 100 
clim    = 0.1 *1000
siglim  = 30000
emglim  = 100
hilblim = 50

# # # # # AE plots # # # # 
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, ae_rfsignal,'grey')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.set_ylim([-plim,plim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_rf.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)


fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, ae_vsignal,'pink')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.set_ylim([-vlim,vlim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_v.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
# plt.plot(t,20*ae_emg_signal,'r')
plt.plot(t,ae_low_signal,'k')
ax.set_ylim([-siglim,siglim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_signal.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,ae_emg_signal,'k')
ax.set_ylim([-emglim,emglim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_emg_signal.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_m,'k')
ax.set_xlim([0,20])
# ax.set_xlim([0,cut])
# ax.set_xlim([0,duration])
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_fft_signal.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,hae_emg_signal,'r')
ax.set_ylim([0,hilblim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'ae_hilbert_emg_signal.png'
plt.savefig(plot_filename)
plt.show()

# # # # # P plots # # # # 
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, p_rfsignal,'grey')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.set_ylim([-plim,plim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_rf.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, p_vsignal,'pink')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.set_ylim([-vlim,vlim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_v.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,p_low_signal,'k')
ax.set_ylim([-siglim,siglim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_signal.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,p_emg_signal,'k')
ax.set_ylim([-emglim,emglim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_emg_signal.png'
plt.savefig(plot_filename)
plt.close(fig)


# fig = plt.figure(figsize=(4,2))
# ax = fig.add_subplot(111)
# plt.plot(frequencies,fft_p,'k')
# ax.set_xlim([0,20])
# # ax.set_xlim([0,duration])
# # plt.yticks(fontsize=f)
# # plt.xticks(fontsize=f)
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'p_fft_signal.png'
# plt.savefig(plot_filename)
# # plt.show()
# plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,hp_emg_signal,'r')
ax.set_ylim([0,hilblim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'p_hilbert_emg_signal.png'
plt.savefig(plot_filename)
plt.show()

# # # # # V plots # # # # 
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, v_rfsignal,'grey')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.set_ylim([-plim,plim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_rf.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t, v_vsignal,'pink')
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.set_ylim([-vlim,vlim])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_v.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,v_low_signal,'k')
ax.set_ylim([-siglim,siglim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_signal.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,v_emg_signal,'k')
ax.set_ylim([-emglim,emglim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_emg_signal.png'
plt.savefig(plot_filename)
plt.close(fig)


fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_v,'k')
ax.set_xlim([0,20])
# ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_fft_signal.png'
plt.savefig(plot_filename)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
plt.plot(t,hv_emg_signal,'r')
ax.set_ylim([0,hilblim])
ax.set_xlim([0,duration])
# plt.yticks(fontsize=f)
# plt.xticks(fontsize=f)
plt.yticks([])
plt.xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'v_hilbert_emg_signal.png'
plt.savefig(plot_filename)
plt.close(fig)

# # # # # END PLOTS # # # # 