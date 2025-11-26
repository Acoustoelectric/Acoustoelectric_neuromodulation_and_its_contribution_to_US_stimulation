'''

Title: compare the data going into generator with the data coming out of generator. 

Author: Jean Rintoul
Date:   26.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18

# PRF Wave. 
savepath            = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/PRF_mixing/e151_prf_cont_phantom/'
prf_filename        = savepath +'t10_prf_stream1.npy'


prf_data        = np.load(prf_filename)
a,b             = prf_data.shape
print ('shape',a,b)
# 
# 
prf_gain            = 5000
c_gain              = 5000
duration            = 12
Fs                  = 5e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
t                   = np.linspace(0, duration, N, endpoint=False)
m_channel           = 0 
rf_channel          = 4
marker_channel      = 7 
# 
# 
NN              = len(t)
rbeta           = 0
raw_window      = np.kaiser( (NN), rbeta )


prf_signal = 1e6*prf_data[m_channel]/prf_gain
prf_fft_m = fft(prf_signal*raw_window)
prf_fft_m = np.abs(2.0/(N) * (prf_fft_m))[1:(N)//2]

xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies = xf[1:(N)//2]

carrier = 500000
# 
# 
sos_band   = iirfilter(17, [450000,550000], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
prf_filtsignal            = sosfiltfilt(sos_band, prf_signal)
# 
# 
start_point = int(0*Fs)
end_point   = int(12*Fs) 

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
# plt.plot(t,prf_signal,'k')
plt.plot(t,prf_filtsignal,'k')
ax.set_xlim([5,5.01])
# ax.set_ylim([0,15])
ax.set_xticks([5,5.01])
# plt.xticks(fontsize=fonts)
# plt.yticks(fontsize=fonts)
plt.xticks([])
plt.yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plot_filename = 'time_series.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()
# 
# 
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
# plt.plot(frequencies,4.3*prf_fft_m,'k')
plt.plot(frequencies,prf_fft_m,'k')
ax.set_xlim([0,6000])
# ax.set_ylim([0,45])
ax.set_ylim([0,10])
ax.set_xticks([0,1000,3000,5000])
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'lowpart_fft.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()
# # 
# # 
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
# plt.plot(frequencies,c_fft_m,'k')
plt.plot(frequencies,prf_fft_m,'k')
ax.set_xlim([carrier-6000,carrier+6000])
ax.set_ylim([0,300])
ax.set_xticks([carrier-6000,500000,carrier+6000])
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'raw_carrier_fft.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
# plt.plot(frequencies,c_fft_m,'k')
plt.plot(frequencies,prf_fft_m,'k')
# ax.set_xlim([carrier-100000,carrier+100000])
ax.set_xlim([0,1000000+50])
ax.set_ylim([0,300])
ax.set_xticks([carrier])
# ax.set_xticks([carrier-100000,500000,carrier+100000])
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'carrier_distant_fft.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()
# # 
# # 
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
# plt.plot(frequencies,c_fft_m,'k')
plt.plot(frequencies/1000,prf_fft_m,'k')
# ax.set_xlim([carrier-100000,carrier+100000])
ax.set_xlim([1000-6,1000+6])
ax.set_ylim([0,7.5])
# ax.set_xticks([carrier])
ax.set_xticks([1000-6,1000,1000+6])
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'sum_fft.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()
# # 
# # 

