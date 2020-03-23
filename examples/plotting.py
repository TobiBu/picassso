import picasso as p
from picasso.plot.maps import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# make some nice axis adjustments
#plt.switch_backend('agg') 
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 10)

sns.set_style('ticks')
#sns.set_style('darkgrid')
sns.set_context("talk",font_scale=2,rc={"lines.linewidth": 4,"axes.linewidth": 5})

plt.rc('axes', linewidth=3)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['axes.edgecolor'] = 'k'#'gray'
#plt.rcParams['axes.grid'] = True
#plt.rcParams['grid.color'] = 'lightgray'
#plt.rcParams['grid.linestyle'] = 'dashed' #dashes=(5, 1)
plt.rcParams['lines.dashed_pattern'] = 10, 3
plt.rcParams['grid.linewidth'] = 1.5
#plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.shadow'] = False
plt.rcParams['legend.edgecolor'] = 'lightgray'
plt.rcParams['patch.linewidth'] = 3

# set up the axis grid of 5 x 3 panels
fig = plt.figure(figsize=(50,30), constrained_layout=False)
# gridspec inside gridspec
outer_grid = fig.add_gridspec(1, 2, wspace=0.01, hspace=0.0, width_ratios=[2./5.,3./5.])

image_grid = outer_grid[0].subgridspec(3, 2, wspace=0.0, hspace=0.0)
prediction_grid = outer_grid[1].subgridspec(3, 3, wspace=0.0, hspace=0.0)

image_axes = []
true_phys_axes = []
pred_phys_axes = []
diff_axes = []

img_label = ['RGB', 'u', 'g', 'r', 'i', 'z']
true_label = [r'$M_{\star}^{\rm true}$',r'$Z_{\star}^{\rm true}$',r'SFR$^{\rm true}$']
pred_label = [r'$M_{\star}^{\rm pred}$',r'$Z_{\star}^{\rm pred}$',r'SFR$^{\rm pred}$']
diff_label = [r'$\Delta M_{\star}$',r'$\Delta Z_{\star}$',r'$\Delta$SFR']

for i, cell in enumerate(image_grid):
	# these are the axes for the mock images
	image_axes.append(plt.subplot(cell))
	image_axes[-1].set_xticklabels([])
	image_axes[-1].set_yticklabels([])
	# add little panel label
	image_axes[-1].text(0.5,0.9,img_label[i],fontsize=15,color='black',horizontalalignment='center',verticalalignment='center',
		bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=image_axes[-1].transAxes)
		

for i, cell in enumerate(prediction_grid):
	k = 0
	l = 0
	m = 0

	if i in [0,1,2]:
		# now we are dealing with the true physical maps
		true_phys_axes.append(plt.subplot(cell))
		true_phys_axes[-1].set_xticklabels([])
		true_phys_axes[-1].set_yticklabels([])
		true_phys_axes[-1].text(0.5,0.9,true_label[k],fontsize=15,color='k',horizontalalignment='center',verticalalignment='center',
			bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=true_phys_axes[-1].transAxes)
		k += 1
	elif i in [3,4,5]:
		# predicted physical maps
		pred_phys_axes.append(plt.subplot(cell))
		pred_phys_axes[-1].set_xticklabels([])
		pred_phys_axes[-1].set_yticklabels([])
		pred_phys_axes[-1].text(0.5,0.9,pred_label[l],fontsize=15,color='black',horizontalalignment='center',verticalalignment='center',
			bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=pred_phys_axes[-1].transAxes)
		l += 1
	else:
		# the differences between prediction and truth
		diff_axes.append(plt.subplot(cell))
		diff_axes[-1].set_xticklabels([])
		diff_axes[-1].set_yticklabels([])
		diff_axes[-1].text(0.5,0.9,diff_label[m],fontsize=15,color='black',horizontalalignment='center',verticalalignment='center',
			bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=diff_axes[-1].transAxes)
		m += 1

all_axes = fig.get_axes()
# show only the outside spines
for ax in all_axes:
    if ax.is_first_row():
        ax.spines['top'].set_linewidth(5) #set_visible(True)
    if ax.is_last_row():
        ax.spines['bottom'].set_linewidth(5) #set_visible(True)
    if ax.is_first_col():
        ax.spines['left'].set_linewidth(5) #set_visible(True)
    if ax.is_last_col():
        ax.spines['right'].set_linewidth(5) #set_visible(True)
    #if ax.is_second_col():
    #    ax.spines['right'].set_linewidth(8) #set_visible(True)

	# maybe we need to set axis off
	#axes[-1].set_axis('off')

plt.show()