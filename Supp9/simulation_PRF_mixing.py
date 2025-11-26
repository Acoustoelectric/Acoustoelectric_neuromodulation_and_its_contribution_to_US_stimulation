# Pulse Repetition Frequency Experiment. 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.fft import fft, fftfreq,fftshift,ifft,ifftshift
from scipy.signal import butter, lfilter
from scipy.signal import freqs
from scipy.signal import hilbert, chirp
from scipy.signal import hilbert
from scipy.signal import iirfilter,sosfiltfilt
import colorednoise as cn

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2


fig_shape_1 = 6 
fig_shape_2 = 2 

fonts = 16


duration    = 0.1      # second
Fs          = 5e6    # Sample rate same as patch clamp system. 
N           = int(Fs*duration)
timestep    = 1.0/Fs
PRF         = 1000
carrier     = 500000

filter_cutoff           = 300
hp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='highpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

xf          = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]
t           = np.linspace(0, duration, N, endpoint=False)



amplitude           = 1 

counter             = 0 
carrier_signal      = [0]*t
sinusoid_signal     = [0]*t
PRF_signal          = [0]*t
for i in range(len(t)):
    carrier_signal[i] = amplitude*np.cos( 2*np.pi*(carrier)*i*timestep)  
    sinusoid_signal[i] = amplitude*np.cos( 2*np.pi*(PRF)*i*timestep)
    # transient impulse spike implementation.  
    if sinusoid_signal[i] > 0: 
        PRF_signal[i] = carrier_signal[i]
    else:
        PRF_signal[i] = 0 


mixed_PRF      = PRF_signal * PRF_signal


# sine wave fft and its modulated signal. 
fft_PRF = fft(PRF_signal)
fft_PRF = np.abs(2.0/N * (fft_PRF))[1:N//2]

fft_mixed_PRF = fft(mixed_PRF)
fft_mixed_PRF = np.abs(2.0/N * (fft_mixed_PRF))[1:N//2]
# 

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111)
plt.plot(t,PRF_signal,'k')
ax.set_xlim([0,0.01])
# ax.set_ylim([-1,1])
# plt.xlim([0,duration])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.title('500kHz sine wave pulsed at 1020Hz',fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("timeseries_prf.png", bbox_inches="tight")
plt.show()

# 
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_mixed_PRF,'r')
plt.plot(frequencies,fft_PRF,'k')
ax.set_xlim([0,10000])
# ax.set_ylim([-1,1])
# plt.xlim([0,duration])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.title('500kHz sine wave pulsed at 1020Hz',fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("fft_comparison.png", bbox_inches="tight")
plt.show()
# 
# 
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_mixed_PRF,'r')
plt.plot(frequencies,fft_PRF,'k')
ax.set_xlim([0,6000])
ax.set_ylim([0, 0.001])
# plt.xlim([0,duration])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.title('500kHz sine wave pulsed at 1020Hz',fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("fft_comparison_zoom.png", bbox_inches="tight")
plt.show()
# 

# 
# 
fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(111)
plt.plot(frequencies,fft_mixed_PRF,'r')
plt.plot(frequencies,fft_PRF,'k')
ax.set_xlim([carrier-10000,carrier+10000])
# ax.set_ylim([-1,1])
# plt.xlim([0,duration])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.title('500kHz sine wave pulsed at 1020Hz',fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("fft_comparison_carrier.png", bbox_inches="tight")
plt.show()
# 
# 
# 