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
import re 
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
# 
# 
filepath = 'subsampled_data/'
outpath = filepath
# 
filenames   = sorted(os.listdir(filepath))
print ('filenames',filenames)
# 
Fs                  = 10000 
pressure_cutoff     = 0.5
# 
f21_emgs            = []
f21_brain_amps      = []
nof21_emgs          = []
nof21_brain_amps    = []
# 
f21_rfs             = [] 
nof21_rfs           = []

# pressures
nof21_ps            = []
f21_ps              = []
# 
ratios = []
# 
for i in range(len(filenames)):
    # 
    f21_rat1 = 0   
    nof21_rat1 = 0 
    if '_F21_' in filenames[i] and '.png' not in filenames[i]: 
        r = re.compile("\s+|_")
        stuff = r.split(filenames[i])
        if len(stuff) > 2:
            pressure = float(stuff[2])
            # print ('pressure',pressure)
            f21_ps.append(pressure)
        # print ('stuff',stuff,len(stuff))
        if pressure < pressure_cutoff:
            # put selection criteria here. 
            data           = np.load(filepath+filenames[i], mmap_mode='r')
            raw            = data['data']
            results        = data['results']        
            [dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal] = raw
            [emg1, b1, p1, emg2, b2, p2, emg3, b3, p3]        = results
            # 
            # collect the emg and brain amplitudes from each file. 
            f21_emg = [emg1,emg2,emg3]
            f21_b   = [b1,b2,b3]
            f21_p   = [p1,p2,p3]
            # 
            f21_emgs.append(f21_emg)
            f21_brain_amps.append(f21_b)
            f21_rfs.append(f21_p)
            f21_rat1 = f21_p[0]

            # f21_carriers.append(carrier_amplitude)
    if '_noF21_' in filenames[i] and '.png' not in filenames[i]: 
        r = re.compile("\s+|_")
        stuff = r.split(filenames[i])
        if len(stuff) > 2:
            pressure = float(stuff[2])
            # print ('pressure',pressure)
            nof21_ps.append(pressure)
        # print ('stuff',stuff,len(stuff))
        if pressure < pressure_cutoff:
        # put selection criteria here. 
            data           = np.load(filepath+filenames[i], mmap_mode='r')
            raw            = data['data']
            results        = data['results']        
            [dt,d_rf,d_v,d_i,d_emg,d_hemg,brain_signal] = raw
            [emg1, b1, p1 ,emg2, b2, p2, emg3, b3, p3] = results
            # 
            # if carrier_amplitude > 0:
            # collect the emg and brain amplitudes from each file. 
            nof21_emg = [emg1,emg2,emg3]
            nof21_b   = [b1,b2,b3]
            nof21_p   = [p1,p2,p3]

            nof21_emgs.append(nof21_emg)
            nof21_brain_amps.append(nof21_b)
            nof21_rfs.append(nof21_p)
            nof21_rat1 = nof21_p[0]
            
        ratio = nof21_rat1/f21_rat1
        print ('ratio',nof21_p,f21_p )
# 
# 
nof21_ps      = np.array(nof21_ps)
f21_ps        = np.array(f21_ps)
mean_nof21_p  = np.mean(nof21_ps)
mean_f21_p    = np.mean(f21_ps)
std_nof21_p   = np.std(nof21_ps)
std_f21_p     = np.std(f21_ps)
print ('ratio for pressure:',mean_f21_p/mean_nof21_p)
print ('means and stds:',mean_nof21_p,mean_f21_p,std_nof21_p,std_f21_p)

# 
# 
f21_carriers     = np.array(f21_rfs)
nof21_carriers   = np.array(nof21_rfs)
# 
group1_c = nof21_carriers.flatten()
group2_c = f21_carriers.flatten()
#  
# 
mean_carrier_1  = np.mean(group1_c)
mean_carrier_2  = np.mean(group2_c)
std_carrier_1   = np.std(group1_c)
std_carrier_2   = np.std(group2_c)
print ('ratio for rf relationship:',mean_carrier_2/mean_carrier_1)
print ('means and stds:',mean_carrier_1,mean_carrier_2,std_carrier_1,std_carrier_2)
# 
# Plotting: 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18

# T-test. 
sample1 = group1_c
sample2 = group2_c
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)
print('Number of measurements in each group: ', len(group1_c),len(group2_c))
# Create lists for the plot
materials = ['No F21', 'F21']
x_pos     = np.arange(len(materials))
CTEs      = [mean_carrier_2,mean_carrier_1]
error     = [std_carrier_2,std_carrier_1]
# 
data    = [sample1, sample2]
w       = 0.8    # bar width
scolors = ['grey','grey']
colors  = ['lightgrey','lightgrey']
scolors = ['grey','red']
x       = [1,2]
y       = data

# Build the plot
fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)
colors = ['Black', 'Red']
# colors = ['lightgrey','red']
# Set the color of the violin patches
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
# Set the color of the median lines
violin_parts['cmeans'].set_colors(colors)
violin_parts['cbars'].set_colors(colors)
violin_parts['cmins'].set_colors(colors)
violin_parts['cmaxes'].set_colors(colors)
ax.set_xticks([1,2])
# ax.set_ylim([0,8000])
# scatter plot width. 
w = 0.4 
for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],color=scolors[i])
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
# ax.yaxis.grid(True)
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.title('carrier amplitudes')
# Save the figure and show
plt.tight_layout()
plt.savefig('f21_carrier_amplitudes.png')
plt.show()


# 
nof21_emgs       = np.array(nof21_emgs)
f21_emgs         = np.array(f21_emgs)
nof21_emgs       = nof21_emgs.flatten()
f21_emgs         = f21_emgs.flatten()
# 
# Normalization: 
# 
const = 1
for i in range(len(nof21_emgs)): 
    totals = nof21_emgs[i] + f21_emgs[i] + 0.01
    nof21_emgs[i] = np.log(const+nof21_emgs[i]/totals)
    f21_emgs[i] = np.log(const+f21_emgs[i]/totals)
# need to scale it between 0-1 better. 
group1_emg = nof21_emgs/0.7
group2_emg = f21_emgs/0.7

mean_emg_1  = np.mean(group1_emg)
mean_emg_2  = np.mean(group2_emg)
std_emg_1   = np.std(group1_emg)
std_emg_2   = np.std(group2_emg)
print ('ratio for rf relationship:',mean_emg_2/mean_emg_1)
print ('means and stds:',mean_emg_1,mean_emg_2,std_emg_1,std_emg_2)

# T-test. 
sample1 = group1_emg
sample2 = group2_emg
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)
print('Number of measurements in each group: ', len(group1_emg),len(group2_emg))
# Create lists for the plot
materials = ['No F21', 'F21']
x_pos     = np.arange(len(materials))
CTEs      = [mean_emg_2,mean_emg_1]
error     = [std_emg_2,std_emg_1]

data      = [sample1, sample2]

w = 0.8    # bar width
scolors = ['grey','grey']
colors = ['lightgrey','lightgrey']
scolors = ['grey','red']
x = [1,2]
y = data

# Build the plot
fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)

violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)

colors = ['Black', 'Red']
# colors = ['lightgrey','red']
# Set the color of the violin patches
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
# Set the color of the median lines
violin_parts['cmeans'].set_colors(colors)
violin_parts['cbars'].set_colors(colors)
violin_parts['cmins'].set_colors(colors)
violin_parts['cmaxes'].set_colors(colors)
ax.set_xticks([1,2])

# scatter plot width. 
w = 0.4 
for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],color=scolors[i])

# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
# ax.yaxis.grid(True)
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('f21_emgs.png')
plt.show()
# 
# 


# I'm not doing the normalization right? 
nof21_b = np.array(nof21_brain_amps)
f21_b   = np.array(f21_brain_amps)
nof21_b = nof21_b.flatten()
f21_b = f21_b.flatten()

print ('brain shape:', nof21_b.shape) # 12,3 

const = 1
for i in range(len(nof21_b)): 
    totals = nof21_b[i]  + f21_b[i] + 0.01
    nof21_b[i] = np.log(const+nof21_b[i]/totals)
    f21_b[i] = np.log(const+f21_b[i]/totals)
# 
group1_b = nof21_b/0.7
group2_b = f21_b/0.7

print ('brain length', len(group1_b))
mean_b_1  = np.mean(group1_b)
mean_b_2  = np.mean(group2_b)
std_b_1   = np.std(group1_b)
std_b_2   = np.std(group2_b)
print ('brain ratio for rf relationship:',mean_b_2/mean_b_1)
print ('brain means and stds:',mean_b_1,mean_b_2,std_b_1,std_b_2)

# T-test. 
sample1 = group1_b
sample2 = group2_b
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)
print('Number of measurements in each group: ', len(group1_b),len(group2_b))
# Create lists for the plot
materials = ['No F21', 'F21']
x_pos     = np.arange(len(materials))
CTEs      = [mean_b_2,mean_b_1]
error     = [std_b_2,std_b_1]

data      = [sample1, sample2]

w = 0.8    # bar width
scolors = ['grey','grey']
colors = ['lightgrey','lightgrey']
scolors = ['grey','red']
x = [1,2]
y = data

# Build the plot
fig = plt.figure(figsize=(3,3))
ax  = fig.add_subplot(111)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)
colors = ['Black', 'Red']
# colors = ['lightgrey','red']
# Set the color of the violin patches
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
# Set the color of the median lines
violin_parts['cmeans'].set_colors(colors)
violin_parts['cbars'].set_colors(colors)
violin_parts['cmins'].set_colors(colors)
violin_parts['cmaxes'].set_colors(colors)
ax.set_xticks([1,2])
# scatter plot width. 
w = 0.4 
for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],color=scolors[i])
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
# ax.yaxis.grid(True)
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('f21_brains.png')
plt.show()



# Now do the k means cluster analysis. 
# group1_b
# group1_emg
# 0, 1,
true_labels = np.full(len(group1_b), 1).tolist() + np.zeros(len(group2_b)).tolist() 
true_labels = np.array(true_labels).astype(int).tolist()
print (true_labels)
# 
features = [group1_b.tolist() + group2_b.tolist(), group1_emg.tolist() + group2_emg.tolist()]
features = np.array(features).T
# print (features.shape)
# print (len(true_labels))
# 
# First I think I should perform my own normalization per set.
# 
# standardization. scales everything between 0-1. 
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# 
kmeans = KMeans(n_clusters=2,random_state=42)
kmeans.fit(scaled_features)
print (kmeans.n_iter_)
print (kmeans.labels_)
# print (true_labels)


true = true_labels
pred = kmeans.labels_
print (true)


fig = plt.figure(figsize=(4,3))
ax  = fig.add_subplot(111)
plt.plot(group1_b,group1_emg,'ok')
plt.plot(group2_b,group2_emg,'or')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show. 
plt.tight_layout()
plt.savefig('cluster_brain.png')
plt.show()
# 
# Confusion Matrix. 
# 
labels = ['No F21','F21']
cm = confusion_matrix(true, pred)
cm = np.flip(cm)

print ('cm',cm)
sns.set(font_scale=1.8)
df_cm = pd.DataFrame(cm, index = labels,columns = labels)


fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
sns.heatmap(df_cm,annot=True,cmap="OrRd",fmt='g')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print(cm)