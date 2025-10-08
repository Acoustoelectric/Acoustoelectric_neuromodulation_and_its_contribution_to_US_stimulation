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
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests
# 
# 
# Plot regression line
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
#  
filepath              = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/aedc_take2/aedc_data/'
outpath               = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_neuromodulation/aedc_take2/'
# 
file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
file_list = sorted(file_list)
print ('file list:', file_list, len(file_list))
# 
i           = 0 
totals      = []
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
        dt               = ae_d[0]
        Fs               = 10000 
        duration         = 6 
        timestep         = 1/Fs 
        ae_rfsignal      = ae_d[1]
        aebrain_signal   = ae_d[6]
        ae_rawemg_signal = ae_d[4]
        aeemg_signal     = ae_d[5]
        pbrain_signal    = p_d[6]
        pemg_signal      = p_d[5]
        vbrain_signal    = v_d[6]
        vemg_signal      = v_d[5]
        # 
        # 
        start_range     = int(0.5*Fs)
        ae_emg_range    = np.max(ae_rawemg_signal[start_range:]) - np.min(ae_rawemg_signal[start_range:])
        ae_brain_range  = np.max(aebrain_signal[start_range:]) - np.min(pbrain_signal[start_range:]) 
        # 
        p_emg_range     = np.max(pemg_signal[start_range:])  - np.min(pemg_signal[start_range:]) 
        p_brain_range   = np.max(pbrain_signal[start_range:]) - np.min(pbrain_signal[start_range:]) 
        # 
        v_emg_range     = np.max(vemg_signal[start_range:])  - np.min(vemg_signal[start_range:]) 
        v_brain_range   = np.max(vbrain_signal[start_range:]) - np.min(vbrain_signal[start_range:]) 
        # 
        # print ('emg_range: ',  ae_emg_range )
        # print ('brain_range: ', ae_brain_range )
        data = [ae_emg_range,ae_brain_range,p_emg_range,p_brain_range,v_emg_range,v_brain_range]
        totals.append(data)
        #  
        # Record this for each ae, v and p. 
        # Then I can do a 0,0.5,1Hz EMG amplitude comparison. 
        # And a proportionality of EMG to AE signal amplitude. 
        # 
        # Make a pretty picture showing what I am doing.     
        # fig = plt.figure(figsize=(4,4))
        # ax  = fig.add_subplot(211)
        # plt.plot(dt,aebrain_signal,'k')
        # ax2  = fig.add_subplot(212)
        # plt.plot(dt,ae_rawemg_signal,'k')         
        # plt.plot(dt,aeemg_signal,'r')    
        # # plt.yticks(fontsize=fonts)
        # # plt.xticks(fontsize=fonts)
        # # plt.yticks([])
        # # plt.xticks([])
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['top'].set_visible(False)  
        # plt.title(file_list[i])      
        # # ax.spines['left'].set_visible(False)
        # # ax.spines['bottom'].set_visible(False)
        # # Save the figure and show. 
        # plt.tight_layout()
        # plt.savefig('correlation_brain.png')
        # plt.show()
        # 
        # 
# 
totals = np.array(totals)
# print ('totals: ',totals.shape)

orders = [['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['ae','p','v'],
['p','ae','v'],
['v','ae','p'],
['ae','p','v'],
['ae','v','p'],
['v','ae','p'],
['ae','p','v'],
['ae','v','p'],
['p','ae','v'],
['v','ae','p'],
['ae','v','p'],
['v','p','ae'],
['p','v','ae']]
# 
print ('len orders:', len(orders))
oo = np.array(orders)

# print ('orderinfo',oo.shape)
thresh          = 5
ae_emg_range    = totals[:,0]
ae_brain_range  = totals[:,1]
p_emg_range     = totals[:,2]
p_brain_range   = totals[:,3]
v_emg_range     = totals[:,4]
v_brain_range   = totals[:,5]

frequency_array = np.zeros((3,3))
print ('fre',frequency_array.shape)
#   
for i in range(len(file_list)-1):
    if i > (len(file_list)-13):

        if oo[i,0] == 'ae': 
            print ('ae first')
            if ae_emg_range[i] > thresh:
                frequency_array[0,0] =frequency_array[0,0]+1; 
        if oo[i,0] == 'v': 
            print ('v first')
            if v_emg_range[i] > thresh:
                frequency_array[1,0] = frequency_array[1,0]+1; 
            else: 
                print ('emg false')
        if oo[i,0] == 'p': 
            print ('vpfirst')
            if p_emg_range[i] > thresh:
                frequency_array[2,0] = frequency_array[2,0]+1; 

        if oo[i,1] == 'ae': 
            print ('ae second')
            if ae_emg_range[i] > thresh:
                frequency_array[0,1] =frequency_array[0,1]+1; 
        if oo[i,1] == 'v': 
            print ('v second')
            if v_emg_range[i] > thresh:
                frequency_array[1,1] = frequency_array[1,1]+1; 
        if oo[i,1] == 'p': 
            print ('p second')
            if p_emg_range[i] > thresh:
                frequency_array[2,1] = frequency_array[2,1]+1; 

        if oo[i,2] == 'ae': 
            print ('ae third')
            if ae_emg_range[i] > thresh:
                frequency_array[0,2] =frequency_array[0,2]+1; 
        if oo[i,2] == 'v': 
            print ('v third')
            if v_emg_range[i] > thresh:
                frequency_array[1,2] = frequency_array[1,2]+1; 
        if oo[i,2] == 'p': 
            print ('p third')
            if p_emg_range[i] > thresh:
                frequency_array[2,2] = frequency_array[2,2]+1; 



# To do a Chi-squared test first you need to have categorical variables. 
# 
# 
from itertools import combinations


# frequency_array = [[5,5,3],
# [0,0,3],
# [3,6,6]]

# res = chi2_contingency(frequency_array)
chi2, p, dof, ex = chi2_contingency(frequency_array, correction=True)

print ('frequency_array',frequency_array)
print ('chi2,p,dof,ex: ',chi2,p,dof,ex)

df = pd.DataFrame(frequency_array)  

# gathering all combinations for post-hoc chi2
all_combinations = list(combinations(df.index, 2))
print("Significance results:")
for comb in all_combinations:
    # subset df into a dataframe containing only the pair "comb"
    new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
    # running chi2 test
    chi2, p, dof, ex = chi2_contingency(new_df, correction=False)
    print(f"Chi2 result for pair {comb}: {chi2}, p-value: {p}")


# gathering all combinations for post-hoc chi2
all_combinations = list(combinations(df.index, 2))
p_vals = []
for comb in all_combinations:
    # subset df into a dataframe containing only the pair "comb"
    new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
    # running chi2 test
    chi2, p, dof, ex = chi2_contingency(new_df, correction=True)
    p_vals.append(p)
#  
#  
reject_list, corrected_p_vals = multipletests(p_vals, method='fdr_bh')[:2]
#  
#  
print("original p-value\tcorrected p-value\treject?")
for p_val, corr_p_val, reject in zip(p_vals, corrected_p_vals, reject_list):
    print(p_val, "\t", corr_p_val, "\t", reject)
#  
# https://neuhofmo.github.io/chi-square-and-post-hoc-in-python/
# To Bonferroni correct: 
#  
# print ('f array: ',frequency_array)
# print ('statistic: ',chi2)
# print ('pvalue: ',p)
# print ('dof: ',dof)
#  
# 
# If P > 0.05, there is no relationship between the categorical variables. 
# 
# Null hypothesis: There are no relationships between the categorical variables. 
# If you know the value of one variable, it does not help you predict the value of another variable.
# 
# When one of the cells is less than 5, use the fisher's exact test. 
# 
# import scipy.stats as stats
# (st, p) = stats.fisher_exact(frequency_array)
# print(st, p)
# 
# 
# 
# 
# stat = 102, pvalue = 3.1e-21
# 
# print ('file_list:', file_list,len(file_list))
# print ('ae_emg_range: ',ae_emg_range)
# print ('p_emg_range: ',p_emg_range)
# print ('v_emg_range: ',v_emg_range)
# 
# 
#   Chi-Squared test
#   NULL Hypothesis
#   It doesn't matter what order the tests were done in. 
#   The table should include observed frequencies. 
#   
#   THRESH can determined via clustering. 
#   
#   ORDER                  1  2  3 
#   AE_EMG_RANGE>THRESH    24  3 2
#   P_EMG_RANGE>THRESH
#   V_EMG_RANGE>THRESH
#   
#   
# from scipy.stats import chisquare,chi2_contingency
#   
# chisquare(f_obs=f_obs, f_exp=f_exp)
# #  
#  Alternatively use a logistic regression? 
#  
#  
#  