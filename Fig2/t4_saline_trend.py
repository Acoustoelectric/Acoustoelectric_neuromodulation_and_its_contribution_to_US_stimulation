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
fonts                          = 16
# 
filename = 'saline_t4.npz'
print ('filename: ', filename)
data2                 = np.load(filename, mmap_mode='r')
fs2                   = data2['fs']
amp2                  = data2['amps']
flist2                = data2['flist']
frequencies2          = data2['frequencies']
dcs2                  = data2['dcs']
# print ('fs',fs,amp)
# 
# 
freqs           = [0.5,1,2,4,8,20,40]
f 				= 18 
# print ('amp2 shape: ',amp2.shape)
# midpoints = np.mean(amp2,1)
print (fs2)
e1 = np.mean(amp2[0:3])
e2 = np.mean(amp2[4:7])
e3 = np.mean(amp2[8:11])
e4 = np.mean(amp2[12:15])
e5 = np.mean(amp2[16:19])
e6 = np.mean(amp2[20:23])
e7 = np.mean(amp2[24:27])

freq_line = [0.5,1,2,4,8,20,40]
amp_line = [e1,e2,e3,e4,e5,e6,e7]

print ('amp2: ',amp2)
stuff = amp2.reshape(7,4)
fs = fs2.reshape(7,4)
ff = fs[:,0]
print ('stuff:',stuff)
# print ('f:',f)
amean = np.mean(stuff,1)
astd  = np.std(stuff,1)
print ('amean:',amean)
print ('astd:',astd)
print ('f',ff)

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
# plt.plot(fs,amp,'.')
plt.plot(fs2,amp2,'.r')
plt.plot(freq_line,amp_line,'r')
plt.fill_between(ff, amean - astd, amean + astd, color='r', alpha=0.2)
# plt.legend([filename,filename2],loc='upper right',fontsize=f,framealpha=0 )
#

plt.xticks([0,10,20,30,40],fontsize=f)
plt.yticks(fontsize=f)
ax.set_xlim([0,41])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.title('DF amplitudes')
plt.tight_layout()
plot_filename = 'saline_ae_comparison_vs_freq.png'
plt.savefig(plot_filename)
plt.show()
