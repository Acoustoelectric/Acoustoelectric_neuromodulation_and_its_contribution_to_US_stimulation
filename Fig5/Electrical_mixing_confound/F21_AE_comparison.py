'''

Title: F21 comparison of AE and carrier amplitudes. 
Author: Jean Rintoul
Date: 03.06.2025

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
fonts                          = 20
# 
# 
filename           = 'F21'
filename2          = 'noF21'
# 
# 
print ('filename: ', filename) # F21
data                 = np.load(filename+'.npz', mmap_mode='r')
# print ('data: ',list(data.keys()) )

fs                   = data['fs']
amp                  = data['amps']
dcs                  = data['dcs']
flist                = data['flist']
frequencies          = data['frequencies']
carriers             = data['carriers']
# 
# 
print ('filename: ', filename2) # no F21
data2                 = np.load(filename2+'.npz', mmap_mode='r')
# print ('data 2: ',list(data2.keys()) )
fs2                   = data2['fs']
amp2                  = data2['amps']
flist2                = data2['flist']
frequencies2          = data2['frequencies']
dcs2                  = data2['dcs']
carriers2             = data2['carriers']
# print ('fs',fs,amp)
# 
# 
freqs           = [0.5,1,2,4,8,20,40]
f 				= 18 

e1 = np.mean(amp[0:3])
e2 = np.mean(amp[4:7])
e3 = np.mean(amp[8:11])
e4 = np.mean(amp[12:15])
e5 = np.mean(amp2[16:19])
e6 = np.mean(amp[20:23])
e7 = np.mean(amp[24:27])
amp_line1 = [e1,e2,e3,e4,e5,e6,e7]

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
# 
freq_line = [0.5,1,2,4,8,20,40]
amp_line  = [e1,e2,e3,e4,e5,e6,e7]
# 
# 
amp_array = np.array([amp[0:3],amp[4:7],amp[8:11],amp[12:15],amp[16:19],amp[20:23],amp[24:27]])
amp1_array = np.array([amp2[0:3],amp2[4:7],amp2[8:11],amp2[12:15],amp2[16:19],amp2[20:23],amp2[24:27]])
# 
print(amp_array.shape)
amp_std   = np.std(amp_array,1)
amp1_std  = np.std(amp1_array,1)
amp_mean  = np.mean(amp_array,1)
amp1_mean = np.mean(amp1_array,1)
# 
# print ('amp std:',amp_std,amp_mean,amp2.shape)
# 
fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
# plt.plot(fs,amp,'.')
plt.plot(fs2,amp,'.k')
plt.plot(fs2,amp2,'.r')
plt.plot(freq_line,amp_line,'r')
plt.plot(freq_line,amp_line1,'k')
ax.fill_between(freq_line, amp_mean - amp_std , amp_mean + amp_std , color='r', alpha=0.2)
ax.fill_between(freq_line, amp1_mean - amp1_std , amp1_mean + amp1_std , color='k', alpha=0.2)
# plt.legend(['F21','no F21'],loc='upper right',fontsize=fonts,framealpha=0 )
#
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.title('DF amplitudes')
plt.tight_layout()
plot_filename = filename+'.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
# Now plot the carrier amplitudes for comparison. 
carrier_array  = np.array([carriers[0:3],carriers[4:7],amp[8:11],amp[12:15],amp[16:19],amp[20:23],amp[24:27]])
carrier2_array = np.array([carriers2[0:3],carriers2[4:7],carriers2[8:11],carriers2[12:15],carriers2[16:19],carriers2[20:23],carriers2[24:27]])
# 
print(amp_array.shape)
carrier_std   = np.std(carrier_array.flatten())
carrier2_std  = np.std(carrier2_array.flatten())
carrier_mean  = np.mean(carrier_array.flatten())
carrier2_mean = np.mean(carrier2_array.flatten())
# 
print ('carriers mean: no F21/F21', carrier2_mean, carrier_mean)
print ('carriers std: no F21/F21', carrier2_std, carrier_std)
# 
# Do a bar plot, that also has points on it. 
# 
# T-test. 
sample1 = carrier_array.flatten()
sample2 = carrier2_array.flatten()
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic: ', t_stat) 
print('P-Value: ', p_value)
print('Total variables: ', len(sample1))

w = 0.8    # bar width
scolors = ['grey','grey']
colors = ['lightgrey','lightgrey']
x = [1,2]
y = np.array([sample1,sample2])

fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
ax.bar(x,
       height=[np.mean(yi) for yi in y],
       yerr=[np.std(yi) for yi in y],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=["no F21", "F21"],
       # color = colors,
       # color=(0,0,0,0),  # face color transparent
       color = colors,
       edgecolor=colors,
       # ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
       )

# scatter plot width. 
w = 0.4 
for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],color=scolors[i])
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
plt.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'carrier_bar_plot.png'
plt.savefig(plot_filename)
plt.show()



# 
# Do a statistical test on the two sets of points. 

# T-test. 
sample1 = amp_array.flatten()
sample2 = amp1_array.flatten()
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic: ', t_stat) 
print('P-Value: ', p_value)
print('Total variables: ', len(sample1))
# 
w = 0.8    # bar width
scolors = ['grey','grey']
colors = ['lightgrey','lightgrey']
x = [1,2]
y = np.array([sample1,sample2])
# 
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
ax.bar(x,
       height=[np.mean(yi) for yi in y],
       yerr=[np.std(yi) for yi in y],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=["no F21", "F21"],
       # color = colors,
       # color=(0,0,0,0),  # face color transparent
       color = colors,
       edgecolor=colors,
       # ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
       )
# 
# scatter plot width. 
w = 0.4 
for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],color=scolors[i])
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
plt.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'AE_amplitudes_bar_plot.png'
plt.savefig(plot_filename)
plt.show()




