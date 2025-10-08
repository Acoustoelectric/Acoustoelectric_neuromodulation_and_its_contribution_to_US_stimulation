'''

Title: mouse AE amplitude trends. 

Author: Jean Rintoul
Date: 20.07.2025

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
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
# 
filename          = 'mouse_t7'
filename2         = 'mouse_t5'
# # 
# # 
print ('filename: ', filename)
data                 = np.load(filename+'.npz', mmap_mode='r')
fs                   = data['fs']
amp                  = data['amps']
dcs                  = data['dcs']
flist                = data['flist']
frequencies          = data['frequencies']
print ('fs',fs,amp)


# print ('filename: ', filename)
data2                 = np.load(filename2+'.npz', mmap_mode='r')
fs2                   = data2['fs']
amp2                  = data2['amps']
dcs2                  = data2['dcs']
flist2                = data2['flist']
frequencies2          = data2['frequencies']
print ('fs',fs,amp2)


# 
freqs   = [0.5,1,2,4,8,20,40]
f 			= 18 


print ('first repeat: ',fs,amp)
print ('second repeat: ',fs2,amp2)
amp = [411.93423,  267.18768,   192.146427, 115.49336,  104.325325,  89.77293,
  65.6494]

amp2 = [456.47168 ,  236.14621 , 119.17522,  102.0752,    88.5036,    65.613846,
  52.29043]

# from e141 exp T7 doc record. 
amp3 = [610.45,480.3, 265, 231, 163, 137, 91]
f3   =  [0.5,1,2,4,8,20,40]
print (amp3)

plot_data = np.array([amp,amp2,amp3])
print ([plot_data.shape])
plot_mean = np.mean(plot_data,0)
plot_std = np.std(plot_data,0)


print (plot_mean)

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
plt.plot(fs,amp,'.r')
plt.plot(fs2,amp2,'.r')
plt.plot(f3,amp3,'.r')
plt.plot(fs,plot_mean,'r')
plt.fill_between(fs, plot_mean-plot_std, plot_mean+plot_std, color='r', alpha=0.2)
# plt.legend([filename,filename2],loc='upper right',fontsize=f,framealpha=0 )
#
ax.set_xlim([0,41])
ax.set_ylim([0,650])
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.title('DF amplitudes')
plt.tight_layout()
plot_filename = 'mouse_ae_comparison_vs_freq.png'
plt.savefig(plot_filename)
plt.show()

