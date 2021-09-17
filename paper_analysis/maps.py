import picasso as p
from picasso.plot.maps import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# make some nice axis adjustments
plt.switch_backend('agg') 
#import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 10)

#sns.set_style('ticks')
#sns.set_style('darkgrid')
#sns.set_context("talk",font_scale=2,rc={"lines.linewidth": 4,"axes.linewidth": 5})

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


survey = p.load('./',  0, 'ugriz') #('./data/prediction_expl_quantiles_19_pp_ds4/') #('./data/prediction_expl_quantiles_18_pp_ds2/') #('./data/prediction_expl_quantiles_17_pp/')
#galaxy = survey['galaxy'][1454] #[839] #[1827] #[1454] #[237] #[4151]

img_label = ['RGB', 'u', 'g', 'r', 'i', 'z']
true_label = [r'$M_{\rm star}^{\rm true}$',r'$Z_{\rm star}^{\rm true}$',r'$\tau_{\rm star}^{\rm true}$']
pred_label = [r'$M_{\rm star}^{\rm pred}$',r'$Z_{\rm star}^{\rm pred}$',r'$\tau_{\rm star}^{\rm pred}$']
diff_label = [r'$\Delta M_{\rm star}$',r'$\Delta Z_{\rm star}$',r'$\Delta\tau_{\rm star}$'] 

star_props = ['stars_Masses','stars_GFM_Metallicity','stars_GFM_StellarFormationTime']
cmaps = ['cividis','inferno','YlOrRd'] #'Spectral_r']
cmap_label = [r'$\log(M_{\rm star}/M_{\odot})$',r'$\log(M_Z^{\rm star}/M_{\odot})$',r'$\tau_{\rm star}$ [Gyr]']
delta_label = [r'$\log(M_{\rm star}^{\rm true}/M_{\rm star}^{\rm pred})$',r'$\log(Z_{\rm star}^{\rm true}/Z_{\rm star}^{\rm pred})$',r'$\log(\tau_{\rm star}^{\rm true}/\tau_{\rm star}^{\rm pred})$']
    

for galaxy in survey['galaxy']: 

    # set up the axis grid of 5 x 3 panels
    # add labels and define style
    fig = plt.figure(figsize=(30,18), constrained_layout=False)
    # gridspec inside gridspec
    outer_grid = fig.add_gridspec(1, 2, wspace=0.005, hspace=0.0, width_ratios=[2./5.,3./5.])   

    image_grid = outer_grid[0].subgridspec(3, 2, wspace=0.0, hspace=0.0)
    prediction_grid = outer_grid[1].subgridspec(3, 3, wspace=0.0, hspace=0.0)   

    image_axes = []
    true_phys_axes = []
    pred_phys_axes = []
    diff_axes = []  

    for i, cell in enumerate(image_grid):
        # these are the axes for the mock images
        image_axes.append(plt.subplot(cell))
        image_axes[-1].set_xticklabels([])
        image_axes[-1].set_yticklabels([])
        # add little panel label
        image_axes[-1].text(0.5,0.9,img_label[i],fontsize=20,color='black',horizontalalignment='center',verticalalignment='center',
            bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=image_axes[-1].transAxes)  

    k = 0
    l = 0
    m = 0
    for i, cell in enumerate(prediction_grid):  

        if i in [0,1,2]:
            # now we are dealing with the true physical maps
            true_phys_axes.append(plt.subplot(cell))
            true_phys_axes[-1].set_xticklabels([])
            true_phys_axes[-1].set_yticklabels([])
            true_phys_axes[-1].text(0.5,0.9,true_label[k],fontsize=20,color='k',horizontalalignment='center',verticalalignment='center',
                bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=true_phys_axes[-1].transAxes)
            k += 1
        elif i in [3,4,5]:
            # predicted physical maps
            pred_phys_axes.append(plt.subplot(cell))
            pred_phys_axes[-1].set_xticklabels([])
            pred_phys_axes[-1].set_yticklabels([])
            pred_phys_axes[-1].text(0.5,0.9,pred_label[l],fontsize=20,color='black',horizontalalignment='center',verticalalignment='center',
                bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=pred_phys_axes[-1].transAxes)
            l += 1
        else:
            # the differences between prediction and truth
            diff_axes.append(plt.subplot(cell))
            diff_axes[-1].set_xticklabels([])
            diff_axes[-1].set_yticklabels([])
            diff_axes[-1].text(0.5,0.9,diff_label[m],fontsize=20,color='black',horizontalalignment='center',verticalalignment='center',
                bbox=dict(facecolor='w', edgecolor='k', boxstyle='round', linewidth=2),transform=diff_axes[-1].transAxes)
            m += 1  

    all_axes = fig.get_axes()
    # show only the outside spines
    for ax in all_axes:
        if ax.is_first_row():
            ax.spines['top'].set_linewidth(6) #set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_linewidth(6) #set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_linewidth(6) #set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_linewidth(6) #set_visible(True)  
    

    img, [i, r, g] = rgb_image(galaxy, ret_img=True, plot=False, ret_indiv_bands=True,lupton_alpha=1e-25, lupton_Q=25, scale_min=5e24, r_scale=.2, g_scale=.3, b_scale=.3) #maybe bone colormap for single channels?
    image_axes[0].imshow(img, origin='lower', interpolation='nearest')  

    image_axes[2].imshow(g, cmap='Greys', origin='lower', interpolation='nearest')
    image_axes[3].imshow(r, cmap='Greys', origin='lower', interpolation='nearest')
    image_axes[4].imshow(i, cmap='Greys', origin='lower', interpolation='nearest')  

    psize = galaxy.properties['psize'] * 0.704 #bug in creation of physical size, one h correction too much
    (xsize, ysize) = galaxy.properties['stars_Masses'].shape
    image_axes[4].plot(np.array([0.6*xsize,0.9*xsize]),np.array([0.04*ysize,0.04*ysize]),linestyle='-',lw=4.0,color='k')
    image_axes[4].text(0.65,0.075,"%.0f kpc"%(0.3*xsize*psize),fontsize=20,color='k',horizontalalignment='left',verticalalignment='center',transform=image_axes[4].transAxes)   

    _, [_, _, u] = rgb_image(galaxy,  r_band='i', g_band='r', b_band='u', ret_img=True, plot=False, ret_indiv_bands=True, lupton_alpha=1e-25, lupton_Q=25, scale_min=5e24, r_scale=.2, g_scale=.3, b_scale=.3)
    image_axes[1].imshow(u, cmap='Greys', origin='lower', interpolation='nearest')  

    _, [z, _, _] = rgb_image(galaxy,  r_band='z', g_band='r', b_band='g', ret_img=True, plot=False, ret_indiv_bands=True,lupton_alpha=1e-25, lupton_Q=25, scale_min=5e24, r_scale=.2, g_scale=.3, b_scale=.3)
    image_axes[5].imshow(z, cmap='Greys', origin='lower', interpolation='nearest')  

    for i, ax in enumerate(true_phys_axes):
        # true maps
        if star_props[i] == 'stars_GFM_StellarFormationTime':
            # mass weigted mean age
            true_map = galaxy.properties[star_props[i]]/(galaxy.properties['stars_Masses']/1.26e6)
            min_val = 0.
            max_val = 13.8
        else:
            true_map = np.log10(galaxy.properties[star_props[i]])
            min_val = 1.05 * np.nanmin(true_map[np.isfinite(true_map)])
            max_val = 0.95 * np.nanmax(true_map[np.isfinite(true_map)])

            if star_props[i] == 'stars_GFM_Metallicity':
                min_val = 4.

        
        c = ax.imshow(true_map, cmap=cmaps[i], origin='lower', interpolation='nearest', vmin=min_val, vmax=max_val) 

        cbbox = inset_axes(ax, width="55%", height="22%", loc=3, bbox_to_anchor=(-0.01,-0.025,1.,1.), bbox_transform=ax.transAxes)
        [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
        cbbox.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off', length=0)
        plt.setp(cbbox.get_xticklabels(), visible=False)
        plt.setp(cbbox.get_yticklabels(), visible=False)
        cbbox.set_facecolor([1,1,1,0.7])    

        cbaxes = inset_axes(cbbox, width="95%", height="20%", loc=8, bbox_to_anchor=(0.,-0.1,1.,1.), bbox_transform=cbbox.transAxes)
        cb = plt.colorbar(c, cax=cbaxes, orientation='horizontal') #, label=r'$\log(M_{\rm star}/M_{\odot})$', size=20)
        cb.set_label(label=cmap_label[i], size=20, labelpad=10) #, bbox=dict(facecolor='w', edgecolor='w', alpha=.5))
        cb.ax.tick_params(labelsize=20, pad=1)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')

        # pred maps
        if star_props[i] == 'stars_GFM_StellarFormationTime':
            pred_map = galaxy.properties[star_props[i]+'_masked_pred']/(galaxy.properties['stars_Masses_masked_pred']/1.26e6)
        else:
            pred_map = np.log10(galaxy.properties[star_props[i]+'_masked_pred'])
        
        pred_phys_axes[i].imshow(pred_map, cmap=cmaps[i], origin='lower', interpolation='nearest', vmin=min_val, vmax=max_val)

        # delta maps
        if star_props[i] == 'stars_GFM_StellarFormationTime':
            c = diff_axes[i].imshow(np.log10(true_map)-np.log10(pred_map), cmap='coolwarm', origin='lower', interpolation='nearest', vmin=-1.15, vmax=1.15) 
        else:
            c = diff_axes[i].imshow(true_map-pred_map, cmap='coolwarm', origin='lower', interpolation='nearest', vmin=-1.15, vmax=1.15) 

        cbbox = inset_axes(diff_axes[i], width="55%", height="22%", loc=3, bbox_to_anchor=(-0.01,-0.025,1.,1.), bbox_transform=diff_axes[i].transAxes)
        [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
        cbbox.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off', length=0)
        plt.setp(cbbox.get_xticklabels(), visible=False)
        plt.setp(cbbox.get_yticklabels(), visible=False)
        cbbox.set_facecolor([1,1,1,0.7])    

        #cbaxes = inset_axes(ax, width="50%", height="5%", loc=3)
        cbaxes = inset_axes(cbbox, width="95%", height="20%", loc=8, bbox_to_anchor=(0.,-0.1,1.,1.), bbox_transform=cbbox.transAxes)
        cb = plt.colorbar(c, cax=cbaxes, orientation='horizontal') #, label=r'$\log(M_{\rm star}/M_{\odot})$', size=20)
        cb.set_label(label=delta_label[i], size=20, labelpad=10) #, bbox=dict(facecolor='w', edgecolor='w', alpha=.5))
        cb.ax.tick_params(labelsize=20, pad=1)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')   

    plt.savefig('./maps/'+str(galaxy.galaxy_id)+'_masked.pdf', bbox_inches='tight')

