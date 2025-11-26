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
# If no f21 is first, then it is not a fake effect. 
# 
# F21           no F21
# 
# 
# Number in each category: 10 
# first is f21, second is no f21. 
pos_1 = [6,7] 	# 
pos_2 = [7,6] 	# 
# 

pos_1 = [18,21] 	# 
pos_2 = [21,18] 	# 
# 
# 
# 
order = ["1","2"]

names = ["No F21","F21"]
df = pd.DataFrame({'First': pos_2, 'Second': pos_1 }, index=names)

# 
my_colors = ['r','k']
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




