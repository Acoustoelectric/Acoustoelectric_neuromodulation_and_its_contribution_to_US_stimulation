'''


'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import tukey_hsd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16

time = [0,1,2,3,4,5]

voltage = [[24.5,	24.6,	24.6,	24.7,	24.7,	24.6],
[24.5,	24.6,	24.6,	24.7,	24.7,	24.6],
[24.5,	24.6,	24.6,	24.7,	24.7,	24.6],
[24.6,	24.6,	24.6,	24.7,	24.7,	24.6],
[24.6,	24.6,	24.6,	24.7,	24.7,	24.6],
[24.6,	24.6,	24.6,	24.6,	24.7,	24.6]]

pressure = [[24.6,	24.6,	24.5,	24.6,	24.6,	24.6],
[24.8,	24.6,	24.9,	25,	25.3,	25],
[25.9,	25.2,	25.5,	25.7,	25.6,	25.5],
[25.7,	25.7,	25.7,	25.9,	25.8,	25.9],
[25.9,	25.9,	25.9,	26.1,	26,	26.1],
[25.5,	25.9,	25.9,	25.9,	25.6,	25.8]]

ae = [[24.5,	24.6,	24.6,	24.6,	24.6,	24.6],
[24.7,	25.1,	24.6,	24.6,	24.6,	24.8],
[25.5,	25.5,	25.2,	25,	25.5,	25.6],
[25.7,	25.7,	25.7,	25.7,	26,	26],
[25.9,	25.9,	25.9,	25.9,	26.2,	26],
[25.2,	26,	26,	26,	25.9,	26.1]]



ae = np.array(ae)
p  = np.array(pressure)
v  = np.array(voltage)


x = np.mean(ae,1) 
y = np.mean(p,1) 
res = stats.pearsonr(x, y)
print ('pearson r,p:',res )
print (ae.shape)
mean_ae = np.mean(ae,1)
std_ae 	= np.std(ae,1)

mean_p = np.mean(p,1)
std_p 	= np.std(p,1)
mean_v = np.mean(v,1)
std_v 	= np.std(v,1)
print ('mean ae:',mean_ae)
print ('std ae:',std_ae)
print ('time:',time)
#
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
# 
# 
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
plt.fill_between(time, mean_ae-std_ae, mean_ae+std_ae,alpha=0.5, edgecolor='grey', facecolor='grey')
plt.fill_between(time, mean_p-std_p, mean_p+std_p,alpha=0.5, edgecolor='blue', facecolor='blue')
plt.fill_between(time, mean_v-std_v, mean_v+std_v,alpha=0.5, edgecolor='red', facecolor='red')
plt.plot(time,mean_ae,'k')
plt.plot(time,mean_p,'b')
plt.plot(time,mean_v,'r')
plt.legend(['ae','p','v'],loc='upper left',fontsize=fonts, framealpha=0.0)
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.set_xlim([0,np.max(time)+0.1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plot_filename = 'images/ae_temp.png'
plt.savefig(plot_filename)
# plt.close()
# plt.closefig(plot_filename)    
plt.show()




# T-test. 
sample1 = np.array(ae).flatten()
sample2 = np.array(pressure).flatten()
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)
print('Number of measurements in each group: ', len(sample1),len(sample2))
# Create lists for the plot
materials = ['AE', 'P']
x_pos     = np.arange(len(materials))
CTEs      = [np.mean(sample1),np.mean(sample2)]
error     = [np.std(sample1),np.std(sample2)]

data      = [sample1, sample2]

w       = 0.8    # bar width
scolors = ['grey','grey']
colors  = ['lightgrey','lightgrey']
scolors = ['grey','red']
scolors = ['Grey', 'Blue']
x       = [1,2]
y       = data

# Build the plot
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)

colors = ['Grey', 'Blue']
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
plt.savefig('aevsp.png')
plt.show()


sample1 = np.array(ae).flatten()
sample2 = np.array(p).flatten()
sample3 = np.array(v).flatten()


scolors = ['Grey', 'Blue', 'Red']
x = [1,2,3]
y = [sample1,sample2,sample3]
# 
# T-test. 
# t_stat, p_value = ttest_ind(sample1, sample2) 
# # print('T-statistic value: ', t_stat) 
# print('P-Value: ', p_value)
# print('Number of measurements in each grousp: ', len(ac),len(dc))
# tukey's test for multiple comparisons. 
res = tukey_hsd(sample1,sample2,sample3)
print (res)
print (f_oneway(sample1,sample2,sample3))
print ('Tukey p-vals: ',res.pvalue)

print ('DOF: ',len(sample1))
# 
materials = ['ae','p','v']
data = [sample1,sample2,sample3]
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
violin_parts = ax.violinplot(data, widths = 0.9, showmeans = True, showextrema = True)
# violin_parts = ax.violinplot(data,showmeans = True)
colors = ['Grey', 'Blue', 'Red']
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
ax.set_xticks([1,2,3])
ax.set_xticklabels(materials)
ax.set_ylim([24.5,26.2])
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.yaxis.grid(True)
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('temp_violin_plot.png')
plt.show()

