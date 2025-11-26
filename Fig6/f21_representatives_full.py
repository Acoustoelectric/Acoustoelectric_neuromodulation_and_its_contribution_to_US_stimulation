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
import os 
from scipy.signal import hilbert
from scipy.stats import ttest_ind
import re 
import matplotlib.pyplot as plt
# 
# Plotting: 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
#  
# 
filepath = 'representative/'
outpath = filepath

f21_filename 	= filepath + 'tp_0.3_F21_rep2_stream.npy'
nof21_filename 	= filepath + 'tp_0.25_NOF21_rep1_stream.npy'
print ('f21_filename',f21_filename)
m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
emg_channel     = 2 
duration        = 6
brain_gain      = 200

emg_gain        = 500 
emg_gain        = 500 
#  
Fs              = 5e6
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
cut             = 1000
sos_low_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_band    = iirfilter(17, [100, 1000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_hilbert = iirfilter(17, [15], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
#  
mains_fs = np.arange(50,1000,50)
def mains_stop(signal):
    mains = mains_fs
    for i in range(len(mains)):
        sos_mains_stop    = iirfilter(17, [mains[i]-4,mains[i]+4], rs=60, btype='bandstop',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        signal = sosfiltfilt(sos_mains_stop, signal)
    return signal


data           = np.load(f21_filename)
f21_fsignal     = (1e6*data[m_channel]/brain_gain)
f21_emgsignal   = (1e6*data[emg_channel]/emg_gain)
f21_rfsignal    = 10*data[rf_channel]
f21_vsignal     = 10*data[v_channel]
f21_isignal     = -5*data[i_channel]/49.9 
f21_low_signal  = sosfiltfilt(sos_low_band, f21_fsignal)


f21_emg  	     = sosfiltfilt(sos_emg_band, f21_emgsignal)
f21_emgsignal           = mains_stop(f21_emg)
f21_h_ae_emg            = abs(hilbert(f21_emgsignal))
f21_hae_emg_signal      = sosfiltfilt(sos_emg_hilbert, f21_h_ae_emg)

data                = np.load(nof21_filename)
nof21_fsignal       = (1e6*data[m_channel]/brain_gain)
nof21_emgsignal     = (1e6*data[emg_channel]/emg_gain)
nof21_rfsignal      = 10*data[rf_channel]
nof21_vsignal       = 10*data[v_channel]
nof21_isignal       = -5*data[i_channel]/49.9 
nof21_low_signal    = sosfiltfilt(sos_low_band, nof21_fsignal)
nof21_emg  	        = sosfiltfilt(sos_emg_band, nof21_emgsignal)


nof21_emgsignal           = mains_stop(nof21_emg)
nof21_h_ae_emg            = abs(hilbert(nof21_emgsignal))
nof21_hae_emg_signal      = sosfiltfilt(sos_emg_hilbert, nof21_h_ae_emg)


fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
plt.plot(t, f21_rfsignal,'r')
plt.plot(t, nof21_rfsignal,'k')
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plot_filename = outpath+'f21_rf.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
plt.plot(t, f21_low_signal,'r')
plt.plot(t, nof21_low_signal,'k')
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plot_filename = outpath+'f21_brain.png'
plt.savefig(plot_filename)
plt.show()

# 
# 
fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
plt.plot(t, nof21_emgsignal,'k')
plt.plot(t, f21_emgsignal,'r')
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.set_ylim([-10,10])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'f21_emg.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
# plt.plot(t, nof21_hae_emg_signal-np.mean(nof21_hae_emg_signal),'k')
# plt.plot(t, f21_hae_emg_signal-np.mean(f21_hae_emg_signal),'r')

plt.plot(t, nof21_hae_emg_signal,'k')
plt.plot(t, f21_hae_emg_signal,'r')
plt.yticks([],fontsize=fonts)
plt.xticks([],fontsize=fonts)
ax.set_ylim([0,5])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'f21_hemg.png'
plt.savefig(plot_filename)
plt.show()

# Now plot the carrier frequency decrease representative signal. 
N = len(nof21_fsignal)
fft_m = fft(nof21_fsignal)
fft_m = np.abs(2.0/(N) * (fft_m))[1:(N)//2]
f21_fft_m = fft(f21_fsignal)
f21_fft_m = np.abs(2.0/(N) * (f21_fft_m))[1:(N)//2]
xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies = xf[1:(N)//2]

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
plt.plot(frequencies/1000,fft_m,'k')
plt.plot(frequencies/1000,f21_fft_m,'r')
plt.yticks(fontsize=fonts)
plt.xticks([500],fontsize=fonts)
ax.set_ylim([0,1000])

ax.set_xlim([0,1000])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'fft_brain.png'
plt.savefig(plot_filename)
plt.show()
