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
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
f                          = 20
# 
savepath = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/amplitude_w_frequency/e141_t7_mouse_1/'
savepath = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/amplitude_w_frequency/e141_t5_mouse_2/'


# 
outpath              = savepath
# 
outname = 'plots'
# 
# 
freqs           = [0.5,1,2,4,8,20,40]
n_repeats       = 4 
m_channel       = 0 
gain            = 500 
gain            = 100 
duration        = 6 
band_limit      = 40 



Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
# 
# 
# start_idx       = int(2*Fs) 
# end_idx         = int(5.7*Fs) 

start_idx       = int(0*Fs) 
end_idx         = int(6*Fs) 
newN            = int(end_idx-start_idx)
xf              = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies     = xf[1:(newN)//2]
# 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

sos_low_band    = iirfilter(17, [band_limit], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


frequency   = 4
filename    = savepath + str(frequency) + 'hz.npy'
filename    = savepath + str(frequency) + 'hz_ae_mep_g100.npy'        
data        = np.load(filename)
fsignal     = (1e6*data[m_channel]/gain)
fft_raw     = fft(fsignal[start_idx:end_idx])
fft_raw     = np.abs(2.0/(newN) * (fft_raw))[1:(newN)//2]
df_idx      = find_nearest(frequencies,frequency)
lp_brain    = sosfiltfilt(sos_low_band, fsignal)



frequency   = 1

filename    = savepath + str(frequency) + 'hz.npy'
filename    = savepath + str(frequency) + 'hz_ae_mep_g100.npy'      
data        = np.load(filename)
fsignal     = (1e6*data[m_channel]/gain)
fft_raw2     = fft(fsignal[start_idx:end_idx])
fft_raw2     = np.abs(2.0/(newN) * (fft_raw2))[1:(newN)//2]
df_idx      = find_nearest(frequencies,frequency)
lp_brain2    = sosfiltfilt(sos_low_band, fsignal)



fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
plt.plot(t,lp_brain,'r')
plt.plot(t,lp_brain2,'k')
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
plot_filename = 'saline_timeseries.png'
plt.savefig(plot_filename)
# plt.tight_layout()
plt.show()
# 
# 
# 
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_raw,'r')
plt.plot(frequencies,fft_raw2,'k')
ax.set_xlim([0,6])
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.tight_layout()
plot_filename = 'saline_fft.png'
plt.savefig(plot_filename)

plt.show()

