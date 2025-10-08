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
from itertools import cycle, islice
# 
# 
# Plotting: 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
# 
# 
# ORDER: first, second, third 
e146_t7_ae4 = ['ae','p','v']
e146_t7_ae5 = ['ae','p','v']
e146_t7_ae6 = ['ae','p','v']
e147_t1_ae1 = ['ae','p','v']
e147_t1_ae2 = ['ae','p','v']
e147_t1_ae3 = ['ae','p','v']
e147_t1_ae4 = ['ae','p','v']
e147_t2_ae1 = ['ae','p','v']
e147_t2_ae3 = ['ae','p','v']
# 
e147_t4_ae  = ['ae','p','v']
e147_t4_ae2 = ['ae','p','v']
e147_t4_ae3 = ['ae','p','v']
e147_t4_ae4 = ['ae','p','v']
# 
e147_t5_ae1 = ['ae','p','v']
e147_t5_ae2 = ['ae','p','v']
e147_t5_ae3 = ['ae','p','v']
# 
e147_t6_ae1 = ['ae','p','v']
e147_t6_ae2 = ['p','ae','v']
e147_t6_ae3 = ['v','ae','p']
e147_t6_ae4 = ['ae','p','v']
e147_t7_ae1 = ['ae','v','p']
e147_t7_ae2 = ['v','ae','p']
e147_t7_ae3 = ['ae','p','v']
e147_t8_ae1 = ['ae','v','p']
e147_t8_ae2 = ['p','ae','v']
e147_t8_ae3 = ['v','ae','p']
e147_t9_ae1 = ['ae','v','p']
e147_t9_ae2 = ['v','p','ae']
e147_t9_ae1 = ['p','v','ae']
# 
# 
# Number in each category: 
# ae,v,ps, total 14. 
pos_1 = [6,4,3] 	# these are the real ones. 
pos_2 = [5,4,4] 	# 
pos_3 = [2,5,6] 	# 
# # 
# # 
# # 
pos_1 = [12,10,7] 	# 
pos_2 = [10,8,9] 	# 
pos_3 = [6,10,12] 	# 
# # 
# 
# pos_1 = [6,4,3] 	# 
# pos_2 = [5,4,4] 	# 
# pos_3 = [2,5,6] 	# 
# 
#  
# 
# order = ["1","2","3"]
# names = ["ae","v","p"]
# df = pd.DataFrame({'First': pos_1, 'Second': pos_2, 'Third': pos_3 }, index=names)
# # 
# my_colors = ['r','k','grey']
# print (my_colors)
# # create the dataframe
# fig = plt.figure(figsize=(3,3))
# ax = fig.add_subplot(111)
# ax = df.plot(kind='bar', figsize=(3,3), rot=0,color=my_colors,legend=False,alpha=0.4)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.yticks(fontsize=fonts)
# plt.xticks(fontsize=fonts)
# plt.tight_layout()
# plot_filename = 'order_comparison.png'
# plt.savefig(plot_filename)
# plt.show()

print ('pos_1',pos_1)
pos_1 = [pos_1[0],pos_1[2],pos_1[1]]
pos_2 = [pos_2[0],pos_2[2],pos_2[1]]
pos_3 = [pos_3[0],pos_3[2],pos_3[1]]

order = ["1","2","3"]
names = ["ae","p","v"]
df = pd.DataFrame({'First': pos_1, 'Second': pos_2, 'Third': pos_3 }, index=names)
# 
my_colors = ['k','r','grey']
print (my_colors)
# create the dataframe
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
ax = df.plot(kind='bar', figsize=(3,3), rot=0,color=my_colors,legend=False,alpha=0.4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
plt.tight_layout()
plot_filename = 'order_comparison.png'
plt.savefig(plot_filename)
plt.show()


