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
from itertools import combinations
# 
# 
# Plot regression line
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
#  
#  
# #  
filepath  = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_us_mechanism/ACDC/new_paired/'
outpath   = '/Users/jeanrintoul/Desktop/PhD/analysis/ae_us_mechanism/ACDC/'

filepath = 'ACDC_subsampled/'
outpath = filepath
# 
file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
file_list = sorted(file_list)
print ('file list:', file_list, len(file_list))
# 
i                   = 0 
nof21_emg_range     = []
nof21_brain_range   = []
f21_emg_range       = []
f21_brain_range     = []
# 
for i in range(len(file_list)):
    if file_list[i] != '.DS_Store':
        # put selection criteria here. 
        data           = np.load(filepath+file_list[i], mmap_mode='r')
        raw            = data['data']
        results        = data['results']        
        [dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal] = raw
        # [emg1, b1, p1, emg2, b2, p2, emg3, b3, p3]        = results
        [emg1, b1, p1, hsnr1, emg2, b2, p2,hsnr2, emg3, b3, p3,hsnr3]        = results
        # data = [dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal]
        #  
        if '_ACDC_' in file_list[i]:
            nof21_emg_range.append(emg1)
            nof21_emg_range.append(emg2)
            nof21_emg_range.append(emg3)
            #  
            nof21_brain_range .append(b1)
            nof21_brain_range .append(b2)
            nof21_brain_range .append(b3)

        else:
            f21_emg_range.append(emg1)
            f21_emg_range.append(emg2)
            f21_emg_range.append(emg3)
            #  
            f21_brain_range .append(b1)
            f21_brain_range .append(b2)
            f21_brain_range .append(b3)
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
# 
orders = [['f21','nof21'],
['f21','nof21'],
['f21','nof21'],

['nof21','f21'],
['nof21','f21'],
['nof21','f21'],

['nof21','f21'],
['nof21','f21'],
['nof21','f21'],

['f21','nof21'],
['f21','nof21'],
['f21','nof21'],
['f21','nof21'],
['f21','nof21'],
['f21','nof21'],

['nof21','f21'],
['nof21','f21'],
['nof21','f21'],

['f21','nof21'],
['f21','nof21'],
['f21','nof21'],
['f21','nof21'],
['f21','nof21'],
['f21','nof21'],

['nof21','f21'],
['nof21','f21'],
['nof21','f21'],

['nof21','f21'],
['nof21','f21'],
['nof21','f21'],

['f21','nof21'],
['f21','nof21'],
['f21','nof21'],

['nof21','f21'],
['nof21','f21'],
['nof21','f21']]

oo       = np.array(orders)
(ll, bb) = oo.shape
print ('orderinfo',oo.shape)
thresh               = 1.5

frequency_array = np.zeros((2,2))
print ('fre',frequency_array.shape)
#  
# this has to be done for each pulse. 
for i in range(len(f21_emg_range)-1):
        if oo[i,0] == 'f21': 
            # print ('f21 first')
            print ('emg:',f21_emg_range[i])
            if f21_emg_range[i] > thresh:
                frequency_array[0,0] = frequency_array[0,0]+1; 
        if oo[i,0] == 'nof21': 
            # print ('nof21 first')
            print ('noemg:',nof21_emg_range[i])
            if nof21_emg_range[i] > thresh:
                frequency_array[1,0] = frequency_array[1,0]+1; 
        if oo[i,1] == 'f21': 
            # print ('f21 second')
            if f21_emg_range[i] > thresh:
                frequency_array[0,1] = frequency_array[0,1]+1; 
        if oo[i,1] == 'nof21': 
            # print ('nof21 second')
            if nof21_emg_range[i] > thresh:
                frequency_array[1,1] = frequency_array[1,1]+1; 
# 
# To do a Chi-squared test first you need to have categorical variables. 
print ('f array',frequency_array)

# res = chi2_contingency(frequency_array)
chi2, p, dof, ex = chi2_contingency(frequency_array, correction=True)
print ('chi2,p,dof,ex', chi2, p, dof, ex)

df = pd.DataFrame(frequency_array)  
# 
# gathering all combinations for post-hoc chi2
all_combinations = list(combinations(df.index, 2))
print("Significance results:")
for comb in all_combinations:
    # subset df into a dataframe containing only the pair "comb"
    new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
    # running chi2 test
    chi2, p, dof, ex = chi2_contingency(new_df, correction=False)
    print(f"Chi2 result for pair {comb}: {chi2}, p-value: {p}")
# 
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
print ('f array: ',frequency_array)
print ('statistic: ',chi2)
print ('pvalue: ',p)
print ('dof: ',dof)
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