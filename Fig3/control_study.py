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
from os import listdir
from os.path import isfile, join
import re 
import heapq
from scipy.stats import tukey_hsd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# 
# 
#  
# 
# 
filepath  = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/new_analysis_with_control/aedc_data/'
control   = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/new_analysis_with_control/DC_subsampled/'
outpath   = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/new_analysis_with_control/'
# 
# 
# 
control_list = [f for f in listdir(control) if isfile(join(control, f))]
control_list = sorted(control_list)
print ('file list:', control_list, len(control_list))

file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
file_list = sorted(file_list)
print ('file list:', file_list, len(file_list))
#  
i           = 0 
ae_results  = []
p_results   = [] 
v_results   = []
c_results   = []

frequencies = []
ae_ffts     = []
ae_emg_ffts = []

p_ffts     = []
p_emg_ffts = []

v_ffts     = []
v_emg_ffts = []

c_ffts     = []
c_emg_ffts = []
#  
#  
for i in range(len(control_list)):

    if control_list[i] != '.DS_Store':

        # data = np.load(os.path.join(control, control_list[i]), allow_pickle=True)
        data           = np.load(control+control_list[i], allow_pickle=True)
        raw            = data['data']
        results        = data['results']        
        [dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal] = raw
        [emg1, b1, p1, snr1, emg2, b2, p2, snr2, emg3, b3, p3, snr3] = results
        # dt = ae_d[0]
        Fs = 10000 
        duration = 6 
        timestep = 1/Fs 
        #  Need brain and emg FFT info 
        start_pause = int(0.5*Fs)
        end_pause   = int(duration*Fs)
        xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
        frequencies = xf[1:(end_pause-start_pause)//2]

        cbrain_signal  = brain_signal
        cemg_signal    = d_emg
        #  
        cfft_brain   = fft(cbrain_signal[start_pause:end_pause])
        cfft_brain   = np.abs(2.0/(end_pause-start_pause) * (cfft_brain))[1:(end_pause-start_pause)//2]
            
        cfft_emg     = fft(cemg_signal[start_pause:end_pause])
        cfft_emg     = np.abs(2.0/(end_pause-start_pause) * (cfft_emg))[1:(end_pause-start_pause)//2]

        c_ffts.append(cfft_brain)
        c_emg_ffts.append(cfft_emg)

        c_results.append(results)


#  
#  
# 
for i in range(len(file_list)):
# for i in range(19):
    # 
    if file_list[i] != '.DS_Store':
        data    = np.load(filepath+file_list[i], mmap_mode='r')
        ae      = data['ae_results'].tolist()
        p       = data['p_results'].tolist()
        v       = data['v_results'].tolist()
        # 
        ae_d    = data['aedata'].tolist()
        p_d     = data['pdata'].tolist()
        v_d     = data['vdata'].tolist()
        # 
        # ae_fft  = data['aefft_data'].tolist()
        # p_fft   = data['pfft_data'].tolist()
        # v_fft   = data['vfft_data'].tolist()
        # 
        dt = ae_d[0]
        Fs = 10000 
        duration = 6 
        timestep = 1/Fs 
        aebrain_signal    = ae_d[6]
        aeemg_signal      = ae_d[5]

        pbrain_signal    = p_d[6]
        pemg_signal      = p_d[5]
        vbrain_signal    = v_d[6]
        vemg_signal      = v_d[5]

        start_pause = int(0.5*Fs)
        end_pause   = int(duration*Fs)
        xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
        frequencies = xf[1:(end_pause-start_pause)//2]

        aefft_brain   = fft(aebrain_signal[start_pause:end_pause])
        aefft_brain   = np.abs(2.0/(end_pause-start_pause) * (aefft_brain))[1:(end_pause-start_pause)//2]
        pfft_brain   = fft(pbrain_signal[start_pause:end_pause])
        pfft_brain   = np.abs(2.0/(end_pause-start_pause) * (pfft_brain))[1:(end_pause-start_pause)//2]
        vfft_brain   = fft(vbrain_signal[start_pause:end_pause])
        vfft_brain   = np.abs(2.0/(end_pause-start_pause) * (vfft_brain))[1:(end_pause-start_pause)//2]


        aefft_emg     = fft(aeemg_signal[start_pause:end_pause])
        aefft_emg     = np.abs(2.0/(end_pause-start_pause) * (aefft_emg))[1:(end_pause-start_pause)//2]
        pfft_emg     = fft(pemg_signal[start_pause:end_pause])
        pfft_emg     = np.abs(2.0/(end_pause-start_pause) * (pfft_emg))[1:(end_pause-start_pause)//2]
        vfft_emg     = fft(vemg_signal[start_pause:end_pause])
        vfft_emg     = np.abs(2.0/(end_pause-start_pause) * (vfft_emg))[1:(end_pause-start_pause)//2]        

        ae_ffts.append(aefft_brain)
        ae_emg_ffts.append(aefft_emg)
        p_ffts.append(pfft_brain)
        p_emg_ffts.append(pfft_emg)
        v_ffts.append(vfft_brain)
        v_emg_ffts.append(vfft_emg)

        # fig = plt.figure(figsize=(3,3))
        # ax  = fig.add_subplot(211)
        # plt.plot(freqs,fft_ae,'k')
        # ax2  = fig.add_subplot(212)        
        # plt.plot(freqs,fft_ae_hemg,'k')
        # plt.show()
        # 
        ae_results.append(ae)
        p_results.append(p)
        v_results.append(v)
ae_array = np.array(ae_results)
p_array  = np.array(p_results)
v_array  = np.array(v_results)
c_array  = np.array(c_results)
print (ae_array.shape)         # 20,7 
# 
ae_ffts     = np.array(ae_ffts).T
ae_emg_ffts = np.array(ae_emg_ffts).T
p_ffts      = np.array(p_ffts).T
p_emg_ffts  = np.array(p_emg_ffts).T
v_ffts      = np.array(v_ffts).T
v_emg_ffts  = np.array(v_emg_ffts).T

c_ffts      = np.array(c_ffts).T
c_emg_ffts  = np.array(c_emg_ffts).T

# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
f_end_idx  = find_nearest(frequencies, 10)
f_df_idx   = find_nearest(frequencies, 0)
# 
# data = [dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal ]
# fft_data = [frequencies[0:f_end_idx],fft_brain[0:f_end_idx],fft_emg[0:f_end_idx]]
# 
# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(211)
# plt.plot(frequencies,ae_ffts,'k')
# ax.set_xlim([0,10])
# ax2  = fig.add_subplot(212)        
# plt.plot(frequencies,ae_emg_ffts,'k')
# ax2.set_xlim([0,10])
# plt.show()
# 
# we can see a robust different in the height of the brain signal. 
# snr, emg area, brain area, ratio, rfpp, vpp, ipp, emg height, brain height. 
# snr, emg area, brain area, ratio, rfpp, vpp, ipp, emg height, brain height, emg height, h emg amp 1hz, brain amp 1hz
# 
# I have the gains wrong. 
# there should be a correlation between the emg snr and the brain height? 
# 
# Plot regression line
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
# 
print ('ae ffts: ',ae_ffts[f_df_idx] )
# 
sample1 = 0.001*ae_ffts[f_df_idx] 
sample2 = 0.001*p_ffts[f_df_idx]
sample3 = 0.001*v_ffts[f_df_idx]
sample4 = 0.001*c_ffts[f_df_idx]

index_max = np.argmax(sample1)
print ('sample 1: ', np.max(sample1),index_max )
sample1[index_max] = 0

index_max = np.argmax(sample2)
print ('sample 2: ', np.max(sample2),index_max )
sample2[index_max] = 0

index_max = np.argmax(sample3)
print ('sample 3: ', np.max(sample3),index_max )
sample3[index_max] = 0

index_max = np.argmax(sample4)
print ('sample 4: ', np.max(sample4),index_max )
sample4[index_max] = 0


# sample1 = sample1.sort()
# sample2 = sample2.sort()
# sample3 = sample3.sort()
# print ('sort: ',sample1)
# sample1= sample1[0]
# sample2.remove(max(sample2))
# sample3.remove(max(sample3))


scolors = ['Black', 'Red', 'Grey','Pink']
x = [1,2,3,4]
y = [sample1,sample2,sample3,sample4]
# 
# T-test. 
# t_stat, p_value = ttest_ind(sample1, sample2) 
# # print('T-statistic value: ', t_stat) 
# print('P-Value: ', p_value)
# print('Number of measurements in each grousp: ', len(ac),len(dc))
# tukey's test for multiple comparisons. 
res = tukey_hsd(sample1,sample2,sample3,sample4)
print (res)
print (f_oneway(sample1,sample2,sample3,sample4))
print ('Tukey p-vals: ',res.pvalue)

print ('DOF: ',len(sample1))
# 
materials = ['ae','p','v','c']



data = [sample1,sample2,sample3,sample4]

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)
# violin_parts = ax.violinplot(data,showmeans = True)
colors = ['Black', 'Red', 'Grey','Pink']
# Set the color of the violin patches
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
# Set the color of the median lines
violin_parts['cmeans'].set_colors(colors)
violin_parts['cbars'].set_colors(colors)
violin_parts['cmins'].set_colors(colors)
violin_parts['cmaxes'].set_colors(colors)

# scatter plot width. 
w = 0.4 
for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],color=scolors[i])
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(materials)
ax.set_ylim([0,7])
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('aedc_brain_violin_plot.png')
plt.show()
#

sample1 = ae_emg_ffts[f_df_idx] 
sample2 = p_emg_ffts[f_df_idx]
sample3 = v_emg_ffts[f_df_idx]
sample4_orig = c_emg_ffts[f_df_idx]
sample4 = v_emg_ffts[f_df_idx]

scolors = ['Black', 'Red', 'Grey','Pink']
x = [1,2,3,4]
y = [sample1,sample2,sample3,sample4]

print ('ae ffts :',sample1 )
print ('pressure ffts :',sample2 )
print ('v ffts :',sample3 )
print ('c ffts :',sample4 )
totals = sample1 + sample2 + sample3 + 0.01
# 
const = 1 
j = 0 
for i in range(len(sample2)): 
    #   
    sample1[i] = np.log(const+sample1[i]/totals[i])
    sample3[i] = np.log(const+sample3[i]/totals[i])
    sample2[i] = np.log(const+sample2[i]/totals[i])
    if i > len(sample4_orig):
        print ('here',j)
        sample4[i] = np.log(const+sample4_orig[j]/totals[i])
        j = j + 1
        if j >= len(sample4_orig):
            j = 0 
    else: 
        sample4[i] = np.log(const+sample4_orig[j]/totals[i])


print ('2 ae ffts :',sample1)
print ('2 pressure ffts :',sample2 )
print ('2 v ffts :',sample3 )
print ('2 c ffts :',sample4 )

# tukey's test for multiple comparisons. 
res = tukey_hsd(sample1,sample2,sample3,sample4)
print (res)
print (f_oneway(sample1,sample2,sample3,sample4))
print ('Tukey p-vals: ',res.pvalue)

materials = ['ae','p','v','c']
data = [sample1,sample2,sample3,sample4]
fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True)
colors = ['Black', 'Red', 'Grey','Pink']
# Set the color of the violin patches
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
# Set the color of the median lines
violin_parts['cmeans'].set_colors(colors)
violin_parts['cbars'].set_colors(colors)
violin_parts['cmins'].set_colors(colors)
violin_parts['cmaxes'].set_colors(colors)

# scatter plot width. 
w = 0.4 
for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],color=scolors[i])
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(materials)
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('aedc_emg_violin_plot.png')
plt.show()



