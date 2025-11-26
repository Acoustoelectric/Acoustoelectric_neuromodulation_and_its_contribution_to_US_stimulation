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
outpath  = filepath


ACDC_filename   = filepath + 'tp_0.12_ACDC_rep1_stream.npy'
AC_filename  = filepath + 'tp_0.1_AC_rep3_stream.npy' # black


print ('acdc_filename',ACDC_filename)
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


data           = np.load(ACDC_filename)
ACDC_fsignal     = (1e6*data[m_channel]/brain_gain)
ACDC_emgsignal   = (1e6*data[emg_channel]/emg_gain)
ACDC_rfsignal    = 10*data[rf_channel]
ACDC_vsignal     = 10*data[v_channel]
ACDC_isignal     = -5*data[i_channel]/49.9 
ACDC_low_signal  = sosfiltfilt(sos_low_band, ACDC_fsignal)
ACDC_low_signal            = mains_stop(ACDC_low_signal )

ACDC_emg  	     = sosfiltfilt(sos_emg_band, ACDC_emgsignal)
ACDC_emgsignal           = mains_stop(ACDC_emg)
ACDC_h_ae_emg            = abs(hilbert(ACDC_emgsignal))
ACDC_hae_emg_signal      = sosfiltfilt(sos_emg_hilbert, ACDC_h_ae_emg)

data                = np.load(AC_filename)
AC_fsignal       = (1e6*data[m_channel]/brain_gain)
AC_emgsignal     = (1e6*data[emg_channel]/emg_gain)
AC_rfsignal      = 10*data[rf_channel]
AC_vsignal       = 10*data[v_channel]
AC_isignal       = -5*data[i_channel]/49.9 
AC_low_signal    = sosfiltfilt(sos_low_band, AC_fsignal)
AC_low_signal            = mains_stop(AC_low_signal )

AC_emg  	        = sosfiltfilt(sos_emg_band, AC_emgsignal)


AC_emgsignal           = mains_stop(AC_emg)
AC_h_ae_emg            = abs(hilbert(AC_emgsignal))
AC_hae_emg_signal      = sosfiltfilt(sos_emg_hilbert, AC_h_ae_emg)


fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
plt.plot(t, ACDC_rfsignal,'r')
plt.plot(t, AC_rfsignal,'k')


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
plot_filename = outpath+'acdc_rf.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
plt.plot(t, AC_low_signal,'k')
plt.plot(t, ACDC_low_signal,'r')

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
plot_filename = outpath+'acdc_brain.png'
plt.savefig(plot_filename)
plt.show()

# 
# 
fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
plt.plot(t, AC_emgsignal,'k')
plt.plot(t, ACDC_emgsignal,'r')


plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
# ax.set_ylim([-10,10])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plot_filename = outpath+'acdc_emg.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
fig = plt.figure(figsize=(6,2))
ax = fig.add_subplot(111)
# plt.plot(t, AC_hae_emg_signal-np.mean(AC_hae_emg_signal),'k')
# plt.plot(t, ACDC_hae_emg_signal-np.mean(ACDC_hae_emg_signal),'r')
plt.plot(t, ACDC_hae_emg_signal,'r')
plt.plot(t, AC_hae_emg_signal,'k')

plt.yticks([],fontsize=fonts)
plt.xticks([],fontsize=fonts)
# ax.set_ylim([0,5])
ax.set_xlim([0,duration])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'acdc_hemg.png'
plt.savefig(plot_filename)
plt.show()

# Now plot the carrier frequency decrease representative signal. 
N = len(AC_fsignal)
fft_m = fft(AC_fsignal)
fft_m = np.abs(2.0/(N) * (fft_m))[1:(N)//2]
ACDC_fft_m = fft(ACDC_fsignal)
ACDC_fft_m = np.abs(2.0/(N) * (ACDC_fft_m))[1:(N)//2]
xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies = xf[1:(N)//2]

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
plt.plot(frequencies/1000,fft_m,'k')
plt.plot(frequencies/1000,ACDC_fft_m,'r')

plt.yticks(fontsize=fonts)
plt.xticks([500],fontsize=fonts)
ax.set_ylim([0,600])

ax.set_xlim([0,1000])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = outpath+'fft_brain.png'
plt.savefig(plot_filename)
plt.show()
