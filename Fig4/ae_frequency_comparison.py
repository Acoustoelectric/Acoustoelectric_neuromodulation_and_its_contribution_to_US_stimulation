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
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

# Plot regression line
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
#  
filepath              = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/final_paper_support_figs_code/Fig4/1hz_subsampled/'
outpath               = filepath


file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
file_list = sorted(file_list)
print ('ae1hz file list:', file_list, len(file_list))


dc_filepath  = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/final_paper_support_figs_code/Fig3/aedc_data/'
dc_file_list = [f for f in listdir(dc_filepath) if isfile(join(dc_filepath, f))]
dc_file_list = sorted(dc_file_list)
print ('aedc file list:', dc_file_list, len(dc_file_list))
#  
#  
Fs = 10000 
timestep = 1/Fs 
#  
# i           = 0 
ae_ffts     = []
ae_emg_ffts = []
p_ffts      = []
p_emg_ffts  = []
v_ffts      = []
v_emg_ffts  = []
#  
dc_ae_ffts     = []
dc_ae_emg_ffts = []
dc_p_ffts      = []
dc_p_emg_ffts  = []
dc_v_ffts      = []
dc_v_emg_ffts  = []
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
        ae_d    = data['aedata'].tolist()
        p_d     = data['pdata'].tolist()
        v_d     = data['vdata'].tolist()
        dt                = ae_d[0]
        aebrain_signal    = ae_d[6]
        aeemg_signal      = ae_d[5]
        pbrain_signal     = p_d[6]
        pemg_signal       = p_d[5]
        vbrain_signal     = v_d[6]
        vemg_signal       = v_d[5]
        duration = int(len(dt)/Fs) 
        # print ('ae1hz duration:',duration)
        start_pause = int(0.5*Fs)
        end_pause   = int(duration*Fs)
        xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
        freqs = xf[1:(end_pause-start_pause)//2]
        # 1Hz data.  
        aefft_brain   = fft(aebrain_signal[start_pause:end_pause])
        aefft_brain   = np.abs(2.0/(end_pause-start_pause) * (aefft_brain))[1:(end_pause-start_pause)//2]
        pfft_brain   = fft(pbrain_signal[start_pause:end_pause])
        pfft_brain   = np.abs(2.0/(end_pause-start_pause) * (pfft_brain))[1:(end_pause-start_pause)//2]
        vfft_brain   = fft(vbrain_signal[start_pause:end_pause])
        vfft_brain   = np.abs(2.0/(end_pause-start_pause) * (vfft_brain))[1:(end_pause-start_pause)//2]
        # 
        aefft_emg     = fft(aeemg_signal[start_pause:end_pause])
        aefft_emg     = np.abs(2.0/(end_pause-start_pause) * (aefft_emg))[1:(end_pause-start_pause)//2]
        pfft_emg     = fft(pemg_signal[start_pause:end_pause])
        pfft_emg     = np.abs(2.0/(end_pause-start_pause) * (pfft_emg))[1:(end_pause-start_pause)//2]
        vfft_emg     = fft(vemg_signal[start_pause:end_pause])
        vfft_emg     = np.abs(2.0/(end_pause-start_pause) * (vfft_emg))[1:(end_pause-start_pause)//2]        
        # Ratio to determine  
        f_df_idx        = find_nearest(freqs, 1)
        f_noise_idx     = find_nearest(freqs, 0.5)
        f_dc_idx        = find_nearest(freqs, 0)
        # 
        # print ('1hz fft brain: ',aefft_brain[f_df_idx])
        # print ('1hz fft 0.5: ',aefft_brain[f_noise_idx])
        # print ('1hz fft DC: ',aefft_brain[f_dc_idx])
        # print ('ratio:',aefft_brain[f_df_idx]/aefft_brain[f_noise_idx])
        #  
        dc      = aefft_brain[f_dc_idx]
        noise   = aefft_brain[f_noise_idx]
        df      = aefft_brain[f_df_idx]
        ratio   = aefft_brain[f_df_idx]/aefft_brain[f_noise_idx]
        #  
        if ratio > 1.0 and dc < 2000: 
            ae_ffts.append(aefft_brain)
            ae_emg_ffts.append(aefft_emg)
            p_ffts.append(pfft_brain)
            p_emg_ffts.append(pfft_emg)
            v_ffts.append(vfft_brain)
            v_emg_ffts.append(vfft_emg)
            #  
#             fig = plt.figure(figsize=(5,5))
#             ax  = fig.add_subplot(311)
#             plt.plot(freqs,aefft_brain,'k')
#             plt.axvline(x=1)
#             ax.set_xlim([0,10])
#             ax2  = fig.add_subplot(312)        
#             plt.plot(freqs,aefft_emg,'k')
#             plt.axvline(x=1)
#             ax2.set_xlim([0,10])
#             ax3  = fig.add_subplot(313)        
#             plt.plot(dt,aebrain_signal,'k')
#             ax3.set_xlim([0,duration])
#             plt.show()
# # 
# AEDC FILES 
for i in range(len(dc_file_list)):
# for i in range(19):
    # 
    if dc_file_list[i] != '.DS_Store':
        data    = np.load(dc_filepath+dc_file_list[i], mmap_mode='r')
        ae      = data['ae_results'].tolist()
        p       = data['p_results'].tolist()
        v       = data['v_results'].tolist()
        ae_d    = data['aedata'].tolist()
        p_d     = data['pdata'].tolist()
        v_d     = data['vdata'].tolist()
        dt                = ae_d[0]
        aebrain_signal    = ae_d[6]
        aeemg_signal      = ae_d[5]
        pbrain_signal     = p_d[6]
        pemg_signal       = p_d[5]
        vbrain_signal     = v_d[6]
        vemg_signal       = v_d[5]
        duration = int(len(dt)/Fs) 
        if duration == 6:
            print ('aedc duration: ',duration)
            start_pause = int(0.5*Fs)
            end_pause   = int(duration*Fs)
            xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
            freqs = xf[1:(end_pause-start_pause)//2]
            # 1Hz data.  
            aefft_brain   = fft(aebrain_signal[start_pause:end_pause])
            aefft_brain   = np.abs(2.0/(end_pause-start_pause) * (aefft_brain))[1:(end_pause-start_pause)//2]
            pfft_brain   = fft(pbrain_signal[start_pause:end_pause])
            pfft_brain   = np.abs(2.0/(end_pause-start_pause) * (pfft_brain))[1:(end_pause-start_pause)//2]
            vfft_brain   = fft(vbrain_signal[start_pause:end_pause])
            vfft_brain   = np.abs(2.0/(end_pause-start_pause) * (vfft_brain))[1:(end_pause-start_pause)//2]
            # 
            aefft_emg     = fft(aeemg_signal[start_pause:end_pause])
            aefft_emg     = np.abs(2.0/(end_pause-start_pause) * (aefft_emg))[1:(end_pause-start_pause)//2]
            pfft_emg     = fft(pemg_signal[start_pause:end_pause])
            pfft_emg     = np.abs(2.0/(end_pause-start_pause) * (pfft_emg))[1:(end_pause-start_pause)//2]
            vfft_emg     = fft(vemg_signal[start_pause:end_pause])
            vfft_emg     = np.abs(2.0/(end_pause-start_pause) * (vfft_emg))[1:(end_pause-start_pause)//2]        
            # Ratio to determine  
            f_df_idx        = find_nearest(freqs, 1)
            f_noise_idx     = find_nearest(freqs, 0.5)
            f_dc_idx        = find_nearest(freqs, 0)
            # 
            # print ('1hz fft brain: ',aefft_brain[f_df_idx])
            # print ('1hz fft 0.5: ',aefft_brain[f_noise_idx])
            # print ('1hz fft DC: ',aefft_brain[f_dc_idx])
            # print ('ratio:',aefft_brain[f_df_idx]/aefft_brain[f_noise_idx])
            #  
            dc      = aefft_brain[f_dc_idx]
            noise   = aefft_brain[f_noise_idx]
            df      = aefft_brain[f_df_idx]
            ratio   = aefft_brain[f_df_idx]/aefft_brain[f_noise_idx]
            #  
            if dc < 2000: 
                dc_ae_ffts.append(aefft_brain)
                dc_ae_emg_ffts.append(aefft_emg)
                dc_p_ffts.append(pfft_brain)
                dc_p_emg_ffts.append(pfft_emg)
                dc_v_ffts.append(vfft_brain)
                dc_v_emg_ffts.append(vfft_emg)
                #  
    #             fig = plt.figure(figsize=(5,5))
    #             ax  = fig.add_subplot(311)
    #             plt.plot(freqs,aefft_brain,'k')
    #             plt.axvline(x=1)
    #             ax.set_xlim([0,10])
    #             ax2  = fig.add_subplot(312)        
    #             plt.plot(freqs,aefft_emg,'k')
    #             plt.axvline(x=1)
    #             ax2.set_xlim([0,10])
    #             ax3  = fig.add_subplot(313)        
    #             plt.plot(dt,aebrain_signal,'k')
    #             ax3.set_xlim([0,duration])
    #             plt.show()
    # # 
# 
# 
# EMG is bursty - not a precise single spike at the right moment exactly. There is a threshold and when it goes above it - a burst occurs. 
# 
ae_ffts     = np.array(ae_ffts).T
ae_emg_ffts = np.array(ae_emg_ffts).T
p_ffts      = np.array(p_ffts).T
p_emg_ffts  = np.array(p_emg_ffts).T
v_ffts      = np.array(v_ffts).T
v_emg_ffts  = np.array(v_emg_ffts).T
# 
# 
dc_ae_ffts     = np.array(dc_ae_ffts).T
dc_ae_emg_ffts = np.array(dc_ae_emg_ffts).T
dc_p_ffts      = np.array(dc_p_ffts).T
dc_p_emg_ffts  = np.array(dc_p_emg_ffts).T
dc_v_ffts      = np.array(dc_v_ffts).T
dc_v_emg_ffts  = np.array(dc_v_emg_ffts).T
# 
# 
print ('ae1hz ffts ',ae_ffts.shape)
mean_ae      = np.mean(ae_ffts,1)
mean_ae_emg  = np.mean(ae_emg_ffts,1)
std_ae       = np.std(ae_ffts,1)
std_ae_emg   = np.std(ae_emg_ffts,1)
print ('sum_ae1hz',mean_ae.shape)
# 
print ('1hz mean and std brain: ', mean_ae[f_df_idx],std_ae[f_df_idx])
print ('1hz mean and std emg: ', mean_ae_emg[f_df_idx],std_ae_emg[f_df_idx])



print ('ae dcffts ',dc_ae_ffts.shape)
dc_mean_ae      = np.mean(dc_ae_ffts,1)
dc_mean_ae_emg  = np.mean(dc_ae_emg_ffts,1)
dc_std_ae       = np.std(dc_ae_ffts,1)
dc_std_ae_emg   = np.std(dc_ae_emg_ffts,1)
print ('sum_aedc',dc_mean_ae.shape)
# 
print ('dc mean and std brain: ', dc_mean_ae[f_df_idx],dc_std_ae[f_df_idx])
print ('dc mean and std emg: ', dc_mean_ae_emg[f_df_idx],dc_std_ae_emg[f_df_idx])


fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
plt.fill_between(freqs, mean_ae - std_ae, mean_ae + std_ae, color='k', alpha=0.5)
plt.fill_between(freqs, dc_mean_ae - dc_std_ae, dc_mean_ae + dc_std_ae, color='r', alpha=0.5)
plt.plot(freqs,mean_ae,'k')
plt.plot(freqs,dc_mean_ae,'r')
ax.set_ylim([0,1500])
# plt.axvline(x=1)
# plt.legend(['brain 1hz','brain DC'],loc='upper right')
# plt.set_xticklabels([0,1,2,3,4,5])
plt.xticks([0,1,2,3,4,5])
ax.set_xlim([0,5])
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('FFT_brain_comparison.png')
plt.show()

fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)   
plt.fill_between(freqs, mean_ae_emg + std_ae_emg, mean_ae_emg - std_ae_emg, color='k', alpha=0.5)
plt.fill_between(freqs, dc_mean_ae_emg - dc_std_ae_emg, dc_mean_ae_emg + dc_std_ae_emg, color='r', alpha=0.5)
plt.plot(freqs,mean_ae_emg,'k')
plt.plot(freqs,dc_mean_ae_emg,'r')
ax.set_ylim([0,25])
# plt.axvline(x=1)
# plt.legend(['emg 1hz','emg DC'],loc='upper right')
ax.set_xlim([0,5])
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
# plt.suptitle('1Hz and DC data')
plt.savefig('FFT_EMG_comparison.png')
plt.show()


# Do stat sig test between 1hz and DC at FFT(1Hz) 
sample1 = ae_ffts[f_df_idx]
sample2 = dc_ae_ffts[f_df_idx]
scolors = ['Black', 'Red']
x = [1,2]
y = [sample1,sample2]
# T-test. 
t_stat, p_value = ttest_ind(sample1, sample2) 
print('Brain T-statistic value: ', t_stat) 
print('Brain P-Value: ', p_value)


materials = ['1hz','DC']
data = [sample1,sample2]
fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)
colors = ['Black', 'Red']
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
ax.set_xticks([1,2])
ax.set_xticklabels(materials)
ax.set_ylim([0,1500])
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('1hz_fcomp_brain.png')
plt.show()
# # 

# # 
sample1 = ae_emg_ffts[f_df_idx]
sample2 = dc_ae_emg_ffts[f_df_idx]
data = [sample1,sample2]

# T-test. 
t_stat, p_value = ttest_ind(sample1, sample2) 
print('EMG T-statistic value: ', t_stat) 
print('EMG P-Value: ', p_value)


x = [1,2]
y = [sample1,sample2]
fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
# violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True)
colors = ['Black', 'Red']
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

# ax.violinplot(data, showmeans=False,showmedians=True)
# ax.violinplot(d, inner="points")
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
ax.set_xticks([1,2])
ax.set_xticklabels(materials)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.yaxis.grid(True)
ax.set_ylim([0,9])
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('ae1hz_emg_fcomp.png')
plt.show()


