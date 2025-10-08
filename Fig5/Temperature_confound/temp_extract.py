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
from os import listdir
from os.path import isfile, join
import re 
import heapq
from scipy.integrate import simpson
from numpy import trapz
import gc
# 
# 
# Plotting: 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
# 
# File organization for key pictures: 
filepath      = '/Volumes/extras/temperature_test/temp_study/'
outpath       = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/temperature/temperature_data/'
# 
# 
dirlist = [ item for item in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, item)) ]
dirlist = sorted(dirlist)

print ('num files in directory: ',len(dirlist))
# print (dirlist)
brain_gains = 10*np.ones(len(dirlist))
# print ('brain gains',brain_gains)
emg_gains = 500*np.ones(len(dirlist))
# 
# brain_gains = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
# emg_gains   = [500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500]
# 
# It would be better if I had the gains listed in each directory somehow. 
# 
# print (len(brain_gains),len(emg_gains))
# 
Fs              = 5e6
timestep        = 1/Fs
m_channel       = 0 
rf_channel      = 4
v_channel       = 6
i_channel       = 5 
emg_channel     = 2 
cut             = 1000
sos_low_band    = iirfilter(17, [10], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
sos_emg_band    = iirfilter(17, [100,cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
# sos_emg_band    = iirfilter(17, [cut], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# 
sos_emg_hilbert    = iirfilter(17, [10], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
mains_fs = np.arange(50,1000,50)
# print('mains:',mains_fs)
def mains_stop(signal):
    mains = mains_fs
    # mains = [50]    
    for i in range(len(mains)):
        sos_mains_stop    = iirfilter(17, [mains[i]-4,mains[i]+4], rs=60, btype='bandstop',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        signal = sosfiltfilt(sos_mains_stop, signal)
    return signal
# 
#  
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

def metrics(filetoload,filename,typefile):
    ae_data        = np.load(filetoload)
    brain_gain     = brain_gains[i]
    emg_gain       = emg_gains[i]
    ae_fsignal     = (1e6*ae_data[m_channel]/brain_gain)
    duration       = int(len(ae_fsignal)/Fs)
    N              = int(duration*Fs)
    t              = np.linspace(0, duration, N, endpoint=False)    
    ae_emgsignal   = (1e6*ae_data[emg_channel]/emg_gain)
    ae_rfsignal    = 10*ae_data[rf_channel]
    ae_vsignal     = 10*ae_data[v_channel]
    ae_isignal     = -5 *ae_data[i_channel]/50
    rf_pp   = np.max(ae_rfsignal) - np.min(ae_rfsignal)
    v_pp    = np.max(ae_vsignal) - np.min(ae_vsignal)
    i_pp    = np.max(ae_isignal) - np.min(ae_isignal)
    # print ('rf pp, v pp, i pp:',rf_pp,v_pp,i_pp)
    ae_low_signal       = sosfiltfilt(sos_low_band, ae_fsignal)
    ae_emg_signal       = sosfiltfilt(sos_emg_band, ae_emgsignal)
    ae_emg_signal       = mains_stop(ae_emg_signal)
    h_ae_emg            = abs(hilbert(ae_emg_signal))
    hae_emg_signal      = sosfiltfilt(sos_emg_hilbert, h_ae_emg)
    #  
    # FFT CODE. 
    # start_pause = int(0*Fs)
    # end_pause   = int(duration*Fs)
    # xf          = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
    # frequencies = xf[1:(end_pause-start_pause)//2]
    # fft_ae      = fft(ae_fsignal[start_pause:end_pause])
    # fft_ae      = np.abs(2.0/(end_pause-start_pause) * (fft_ae))[1:(end_pause-start_pause)//2]
    # # 
    # fft_ae_hemg = fft(hae_emg_signal[start_pause:end_pause])
    # fft_ae_hemg = np.abs(2.0/(end_pause-start_pause) * (fft_ae_hemg))[1:(end_pause-start_pause)//2]
    # # 
    # f_end_idx       = find_nearest(frequencies, 10)
    # f_1hz_idx       = find_nearest(frequencies, 1)
    #     
    # sub_fft_ae      = fft_ae[0:f_end_idx]    
    # sub_fft_ae_hemg = fft_ae_hemg[0:f_end_idx]
    # sub_freqs       = frequencies[0:f_end_idx]
    # h_amp_1hz       = fft_ae_hemg[f_1hz_idx]
    # b_amp_1hz       = fft_ae[f_1hz_idx]
    # 
    front_gap = 0.5
    # SNR metric based on hilbert transformed EMG. 
    null_times  = [0.1]
    null_start_time = int ((null_times[0])*Fs)
    null_end_time   = int((null_times[0]+front_gap)*Fs)
    h_null_amplitude = np.mean(hae_emg_signal[null_start_time:null_end_time])
    # do it this way to skip the filter artefact at the start.  
    h_amplitude     = np.mean(hae_emg_signal[int(front_gap*Fs):int((duration-0.5)*Fs) ])
    print ('h amplitude',h_amplitude,h_null_amplitude)

    h_snr = 20*np.log(h_amplitude/h_null_amplitude)
    if h_snr < 0: 
        h_snr = 0 
    # print ('h snr:',h_snr)
    # Calculate the area under the hilbert transformed, low pass filtered EMG curve. 
    emg_area = np.round(trapz(hae_emg_signal)/Fs)
    emg_height = abs(np.max(hae_emg_signal) - np.min(hae_emg_signal))
    # print ('emg area =', emg_area )
    # calculate the area under the brain signal curve 
    brain_height = abs(np.max(ae_low_signal) - np.min(ae_low_signal))
    brain_area = np.abs(np.round(trapz(ae_low_signal)/Fs))
    # print ('brain area =',brain_area)
    ratio = brain_area/emg_area
    # 
    # print ('ratio:',ratio)
    #  
    # emg_max_ind= np.argmax(hae_emg_signal[int(0.5*Fs):])
    # emg_max_ind = emg_max_ind + int(0.5*Fs)
    # 
    # print ('max ind:', max_ind)
    # emg_height   =   np.round(hae_emg_signal[emg_max_ind],2 )
    # emg_max_time =   np.round(t[emg_max_ind],2) 
    # print ('emg height, time: ', emg_height,emg_max_time )
    # # 
    # brain_max_ind  = np.argmax(abs(ae_low_signal[int(0.5*Fs):] ) )
    # brain_max_ind  = brain_max_ind + int(0.5*Fs)
    # brain_height   = np.round(ae_low_signal[brain_max_ind],2 )
    # brain_max_time = np.round(t[brain_max_ind],2) 
    # print ('max brain height, time: ', brain_height,brain_max_time )
    # So the brain max metric is messed up...  
    # the emg max metric is good. 
    # for pressure only, it occurs at the end of the ramp, which is correlated to the max in the brain signal. 
    # 
    # Find the peaks  
    x = hae_emg_signal[int(front_gap*Fs):]
    peaks, _ = find_peaks(x, distance=int(front_gap*Fs),prominence=1)
    peaks = peaks +int(front_gap*Fs)
    print ('emg peaks',peaks)
    emg_peaks = peaks 
    # 
    x = ae_low_signal[int(front_gap*Fs):]
    #
    start_height = np.mean(ae_low_signal[0:int(front_gap*Fs)])
    mid_height   = np.mean(ae_low_signal[int(front_gap*Fs):])
    print ('start and mid height:',start_height,mid_height)
    # if mid_height < start_height:      # invert the signal, so they are all up the same way. 
    #     print ('inverted the brain signal for find peaks')
    #     x = -x 
    #     ae_low_signal = -ae_low_signal
    # attempt to center around zero. 
    ae_low_signal = ae_low_signal - np.min(ae_low_signal)    
    x = x - np.min(x)
    # 
    brain_peaks, _ = find_peaks(x, distance=int(front_gap*Fs),prominence=100)
    brain_peaks = brain_peaks +int(front_gap*Fs)
    print ('brain peaks',brain_peaks)
    # values = ae_low_signal[brain_peaks]
    # sorted_vals = np.sort(values)
    # 
    metric_results = [h_snr,emg_area,brain_area,ratio,rf_pp,v_pp,i_pp,emg_height,brain_height] 
    emg_arrays  = emg_peaks
    brain_arrays = brain_peaks
    # 
    # Sub-sample, and save out the brain signal, the hilbert transformed EMG signal, and the original EMG signal. 
    # Downsampling. 
    new_Fs               = 10000
    downsampling_factor  = int(Fs/new_Fs)
    print('downsampling factor: ',downsampling_factor)
    downsampling_filter = iirfilter(17, [new_Fs], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
    #     
    # downsample for easier plotting. 
    dae_emg_signal            = sosfiltfilt(downsampling_filter, ae_emg_signal)
    # Now downsample the data. 
    brain_signal           = ae_low_signal[::downsampling_factor]
    dt                     = t[::downsampling_factor]
    d_emg                  = dae_emg_signal[::downsampling_factor]
    d_hemg                 = hae_emg_signal[::downsampling_factor]  
    h_rf                   = abs(hilbert(ae_rfsignal))  
    h_rf                   = sosfiltfilt(downsampling_filter, h_rf)
    d_rf                   = h_rf[::downsampling_factor]  
    h_v                    = abs(hilbert(ae_vsignal))  
    h_v                    = sosfiltfilt(downsampling_filter, h_v)    
    d_v                    = h_v[::downsampling_factor] 
    h_i                    = abs(hilbert(ae_isignal))  
    h_i                    = sosfiltfilt(downsampling_filter, h_i)    
    d_i                    = h_i[::downsampling_factor] 
    #     
    # 
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(321)
    plt.plot(dt,d_emg,'k')      
    plt.legend(['emg peaks'],loc='upper right')
    ax2 = fig.add_subplot(322)
    plt.plot(dt,d_rf,'k')
    plt.plot(dt,d_v,'r')
    ax3 = fig.add_subplot(323)
    plt.plot(dt,d_hemg,'b')
    ax4 = fig.add_subplot(324)
    plt.plot(dt,d_i,'g')
    ax5 = fig.add_subplot(325)
    plt.plot(dt,d_hemg/np.max(d_hemg),'r')
    plt.plot(dt,brain_signal/np.max(brain_signal),'k' )
    plt.plot(dt,d_rf/np.max(d_rf),'b' )
    ax6 = fig.add_subplot(326)
    plt.plot(dt,brain_signal,'k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    plt.tight_layout()
    plot_filename = outpath+'images/'+filename+ '_'+typefile+'.png'
    plt.savefig(plot_filename)
    plt.close()
    # plt.closefig(plot_filename)    
    # plt.show()

    data = [dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal]
    # 
    # large variable garbage collection 
    del ae_data, ae_fsignal, ae_emgsignal, ae_rfsignal, ae_vsignal, ae_isignal,ae_low_signal,ae_emg_signal
    gc.collect()
    # 
    return metric_results,data
# 
#     
for i in range(len(dirlist)):
# for i in range(1):
#     i = 2
    if i >= 0:
        print ('i',i)
        path = filepath + dirlist[i] 
        print ('dir path:',dirlist[i])
        print ('snr, emg area, brain area, ratio, rfpp, vpp, ipp, emg height, brain height, emg height, h emg amp 1hz, brain amp 1hz')
        file_list = [f for f in listdir(path) if isfile(join(path, f))]
        # print ('file list:', file_list, len(file_list))
        # 
        outfile           = dirlist[i]+ '.npz'
        # 
        # deal with AE file. 
        sub = 'tae_'
        filename = next((s for s in file_list if sub in s), None)
        # print ('filename',filename)
        filetoload = path + '/'+filename
        # print ('file',filetoload)
        ae_results,aedata = metrics(filetoload,dirlist[i],'ae')
        print ('ae results:',ae_results)

        # deal with P file. 
        sub = 'tp_'
        filename = next((s for s in file_list if sub in s), None)
        # print ('filename',filename)
        filetoload = path + '/'+filename
        # print ('file',filetoload)
        p_results,pdata = metrics(filetoload,dirlist[i],'p')
        print ('p_results:',p_results)   
        # 
        # voltage only file. 
        sub = 'tv_'
        filename = next((s for s in file_list if sub in s), None)
        # print ('filename',filename)
        filetoload = path + '/'+filename
        # print ('file',filetoload)
        v_results,vdata = metrics(filetoload,dirlist[i],'v')
        print ('v_results:',v_results)   

        # Save out all the data.  
        np.savez(outpath+outfile,ae_results=ae_results,p_results=p_results,v_results=v_results,aedata=aedata,pdata=pdata,vdata=vdata)
        print ('saved out a data file!')

# 
# 
# 
# What are the group statistics we want here? 
# 
# for i in range(len(file_list)):
#     # 
#     # if file_list[i]
#     r = re.compile("\s+|_")
#     stuff = r.split(file_list[i])

# c1 = ac_filename_1
# c2 = acdc_filename_1
# # 
# # 
# 
# # load the acoustically connected file.
# start_idx      = int(0*Fs) 
# end_idx        = int(duration*Fs) 
# # 
# ac_data        = np.load(c1)
# ac_fsignal     = (1e6*ac_data[m_channel]/brain_gain)
# ac_emgsignal   = (1e6*ac_data[emg_channel]/emg_gain)
# ac_rfsignal    = 10*ac_data[rf_channel]
# ac_low_signal  = sosfiltfilt(sos_low_band, ac_fsignal)
# ac_emg_signal  = sosfiltfilt(sos_emg_band, ac_emgsignal)
# # ac_emg_signal  = mains_stop(ac_emg_signal)
# # 
# # load the acoustically disconnected file. 
# acdc_data        = np.load(c2)
# acdc_fsignal     = (1e6*acdc_data[m_channel]/brain_gain)
# acdc_emgsignal   = (1e6*acdc_data[emg_channel]/emg_gain)
# acdc_rfsignal    = 10*acdc_data[rf_channel]
# acdc_low_signal  = sosfiltfilt(sos_low_band, acdc_fsignal)
# acdc_emg_signal  = sosfiltfilt(sos_emg_band, acdc_emgsignal)
# # acdc_emg_signal  = mains_stop(acdc_emg_signal)
# # 
# # 
# # 
# start_pause = int(0.85*Fs)
# end_pause = int(1.3*Fs)

# xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
# frequencies = xf[1:(end_pause-start_pause)//2]
# # 
# # carrier frequency. 
# carrier_frequency = 500000
# df_idx = find_nearest(frequencies,carrier_frequency)
# # 


# # 
# fft_ac   = fft(ac_fsignal[start_pause:end_pause])
# fft_ac   = np.abs(2.0/(end_pause-start_pause) * (fft_ac))[1:(end_pause-start_pause)//2]
# fft_acdc = fft(acdc_fsignal[start_pause:end_pause])
# fft_acdc = np.abs(2.0/(end_pause-start_pause) * (fft_acdc))[1:(end_pause-start_pause)//2]
# # 
# # 
# fft_ac_emg   = fft(ac_emgsignal[start_pause:end_pause])
# fft_ac_emg   = np.abs(2.0/(end_pause-start_pause) * (fft_ac_emg))[1:(end_pause-start_pause)//2]
# fft_acdc_emg = fft(acdc_emgsignal[start_pause:end_pause])
# fft_acdc_emg = np.abs(2.0/(end_pause-start_pause) * (fft_acdc_emg))[1:(end_pause-start_pause)//2]
# #  
# print('500khz amplitudes ac/dc:',fft_ac[df_idx],fft_acdc[df_idx])
# #  
# #  
# # Hilbert transform of EMG signals. 
# h_emg_acdc          = abs(hilbert(acdc_emg_signal))
# h_emg_ac            = abs(hilbert(ac_emg_signal))
# hac_emg_signal      = sosfiltfilt(sos_emg_hilbert, h_emg_ac)
# hacdc_emg_signal    = sosfiltfilt(sos_emg_hilbert, h_emg_acdc)
# #  
# # Then I want to save the hilbert amplitudes. 
# pulse_times = [1,3,5]
# null_times  = [0.1]
# # 
# # 
# null_start_time = int ((null_times[0])*Fs)
# null_end_time   = int((null_times[0]+0.5)*Fs)
# hac_null_amplitude = np.max(hac_emg_signal[null_start_time:null_end_time])
# hacdc_null_amplitude = np.max(hacdc_emg_signal[null_start_time:null_end_time])   


# # 
# #  
# hamps      = []
# hacdc_amps = []
# for i in range(len(pulse_times)):
#     start_time = int ((pulse_times[i]-0.5 )*Fs)
#     end_time   = int((pulse_times[i]+1)*Fs)
#     # 
#     # 
#     hac_amplitude = np.max(hac_emg_signal[start_time:end_time])
#     hacdc_amplitude = np.max(hacdc_emg_signal[start_time:end_time])   
#     # 
#     # 
#     hac_snr           = 20*np.log(hac_amplitude/hac_null_amplitude)
#     hacdc_snr         = 20*np.log(hacdc_amplitude/hacdc_null_amplitude)

#     hamps.append(hac_snr)
#     hacdc_amps.append(hacdc_snr)
# # 
# # 
# print ('h ac snr:',hamps)
# print ('h acdc snr:',hacdc_amps)
# # 
# # 
# # 
# acamp   = fft_ac[df_idx]
# acdcamp = fft_acdc[df_idx]
# # 
# # Save out all the data. 
# # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
# np.savez(outpath+outfile,acamp=acamp,acdcamp=acdcamp,hamps=hamps,hacdc_amps=hacdc_amps)
# print ('saved out a data file!')
# # 
# #  
# fig = plt.figure(figsize=(3,3))
# ax = fig.add_subplot(111)
# plt.plot(frequencies/1000,fft_acdc,'r')
# plt.plot(frequencies/1000,fft_ac,'k')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlim([500-1,500+1])
# plt.xticks([carrier_frequency/1000],fontsize=fonts)
# plt.yticks(fontsize=fonts)
# plt.tight_layout()
# plot_filename = outpath+'fft_comparison.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# # 
# fig = plt.figure(figsize=(6,2))
# ax = fig.add_subplot(111)
# plt.plot(t,acdc_rfsignal,'r')
# plt.plot(t,ac_rfsignal,'k')
# ax.set_xlim([0,duration])
# # plt.yticks([])
# # plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'rfsignal.png'
# plt.savefig(plot_filename)
# plt.show()
# # 
# # 
# # 
# fig = plt.figure(figsize=(6,2))
# ax = fig.add_subplot(111)
# plt.plot(t,ac_emg_signal ,'k')
# plt.plot(t,acdc_emg_signal ,'r')
# # plt.plot(t,hac_emg_signal ,'-k')
# # plt.plot(t,hacdc_emg_signal,'-r')
# ax.set_ylim([-130,130])
# ax.set_xlim([0,duration])
# # plt.yticks([])
# # plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'emg_signal.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(6,2))
# ax = fig.add_subplot(111)
# # plt.plot(t,ac_emg_signal ,'k')
# # plt.plot(t,acdc_emg_signal ,'r')
# plt.plot(t,hac_emg_signal ,'-k')
# plt.plot(t,hacdc_emg_signal,'-r')
# ax.set_ylim([0,60])
# ax.set_xlim([0,duration])
# # plt.yticks([])
# # plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'emg_hilbert_signal.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(6,2))
# ax = fig.add_subplot(111)
# # plt.plot(t,ac_emg_signal ,'k')
# # plt.plot(t,acdc_emg_signal ,'r')
# plt.plot(t,ac_low_signal ,'-k')
# plt.plot(t,acdc_low_signal,'-r')
# # ax.set_ylim([0,60])
# ax.set_xlim([0,duration])
# plt.yticks([])
# plt.xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.tight_layout()
# plot_filename = outpath+'brain_signal.png'
# plt.savefig(plot_filename)
# plt.show()


# This is the good general everything plot, if I want something informational and not pretty. 
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(511)
# plt.plot(frequencies,fft_acdc,'r')
# plt.plot(frequencies,fft_ac,'k')
# ax.set_xlim([0,600000])
# ax2 = fig.add_subplot(512)
# plt.plot(t,hac_emg_signal ,'k')
# plt.plot(t,hacdc_emg_signal,'r')
# ax3 = fig.add_subplot(513)
# plt.plot(t,ac_emg_signal ,'k')
# ax3.set_xlim([0,duration])
# ax3.set_ylim([-150,150])
# ax4 = fig.add_subplot(514)
# plt.plot(t,acdc_emg_signal ,'r')
# ax4.set_xlim([0,duration])
# ax4.set_ylim([-150,150])
# ax5 = fig.add_subplot(515)
# plt.plot(t,ac_rfsignal,'k')
# ax5.set_xlim([0,duration])
# plt.tight_layout()
# # # plot_filename = outpath+'v_fft_signal.png'
# # # plt.savefig(plot_filename)
# plt.show()

