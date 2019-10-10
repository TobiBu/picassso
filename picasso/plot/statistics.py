"""

statistics
==========

A set of functions to plot Suvrey statistics and analysis results.

"""



## move some of the stuff to a util.py file???

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns

from tqdm import tqdm


def make_scatterplot(x, y, axis, **kwargs):

    from scipy.stats import binned_statistic_2d

    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    rasterized = kwargs.pop('rasterized', True)
    logscale = kwargs.pop('logscale', True)
    bins = kwargs.pop('bins', 50)
    cmin = kwargs.pop('cmin', 5)
    contours = kwargs.pop('contours', False)
    x_range = kwargs.pop('x_range', None)
    y_range = kwargs.pop('y_range', None)
    colorbar = kwargs.pop('colorbar', False)
    color = kwargs.pop('color', 'k')
    scatter = kwargs.pop('scatter', True)
    density = kwargs.pop('density', False)
    scatter_density = kwargs.pop('scatter_density', True)
    y_label = kwargs.pop('y_label', None)
    msize = kwargs.pop('msize', 5)
    cmap = kwargs.pop('cmap', 'Spectral_r')
    histtype = kwargs.pop('histtype', 'bar')
    add_mean = kwargs.pop('add_mean', True)
    mean_color = kwargs.pop('mean_color', 'c')
    lw = kwargs.pop('lw',2)

    if scatter == True:
        # first do the scatter plot to incorporate outliers
        if scatter_density == True:
            # taken from: 
            # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
            from scipy.stats import gaussian_kde
            # Calculate the point density
            if logscale == True:
                mask = (x != 0) & (y != 0)
                x = x[mask]
                y = y[mask]
                #check also for nan
                #mask = np.isfinite(x) & np.isfinite(y)
                #x = x[mask]
                #y = y[mask]
                xy = np.vstack([np.log10(x),np.log10(y)])
            else:
                #check also for nan
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]

                xy = np.vstack([x,y])
            
            z = gaussian_kde(xy)(xy)

            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

            axis.scatter(x, y, c=z, s=msize, edgecolor='', rasterized=rasterized, cmap=cmap, **kwargs)
        else:
            axis.scatter(x, y, color=color, s=msize, rasterized=rasterized, cmap=cmap, **kwargs)

    if logscale == True:
        x_to_bin = np.log10(x)
        y_to_bin = np.log10(y)
        x_range_bin = np.log10(x_range)
        y_range_bin = np.log10(y_range)
    else:
        x_to_bin = x
        y_to_bin = y
        x_range_bin = x_range
        y_range_bin = y_range

    if contours == True:
        # then make the density or contour plot
        hist, xe, ye, num = binned_statistic_2d(x_to_bin, y_to_bin, np.ones_like(x_to_bin), statistic='sum', bins=bins,
                                            range=[[x_range_bin[0], x_range_bin[1]], [y_range_bin[0], y_range_bin[1]]])
        x_width = (xe[1] - xe[0])
        x_cen = xe[1:] - x_width / 2.
        y_width = (ye[1] - ye[0])
        y_cen = ye[1:] - y_width / 2.

        # how do we get the values for particles in bins with lower than cmin parts per bin?

        num_cont = kwargs.get('num_cont', 3)
        f_size = kwargs.get('fontsize', 10)
        cont_colors = kwargs.get('cont_colors', 'gray')

        X, Y = np.meshgrid(x_cen, y_cen)
        if logscale == True:
            cont = axis.contour(10**X, 10**Y, np.log10(hist).T, num_cont, colors=cont_colors)
        else:
            cont = axis.contour(X, Y, hist, num_cont, colors=cont_colors)

        axis.clabel(cont, inline=True, fontsize=f_size)

    elif density == True:
        # then make the density or contour plot
        hist, xe, ye, num = binned_statistic_2d(x_to_bin, y_to_bin, np.ones_like(x_to_bin), statistic='sum', bins=bins,
                                            range=[[x_range_bin[0], x_range_bin[1]], [y_range_bin[0], y_range_bin[1]]])
        x_width = (xe[1] - xe[0])
        x_cen = xe[1:] - x_width / 2.
        y_width = (ye[1] - ye[0])
        y_cen = ye[1:] - y_width / 2.

        # how do we get the values for particles in bins with lower than cmin parts per bin?

        # cmap = kwargs.get('cmap','inferno')
        # cmap = sns.palplot(sns.color_palette("Blues"))
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        cmap = sns.light_palette('blue', as_cmap=True)

        X, Y = np.meshgrid(xe, ye)

        mask = np.where(hist < cmin)
        hist[mask] = np.nan

        im = axis.pcolormesh(10**X, 10**Y, np.log10(hist).T, cmap=cmap, rasterized=rasterized, **kwargs)
        # im = sns.kdeplot(10**X,10**Y, cmap=cmap, shade=True);

        # im = axis.imshow(np.log10(hist).T, origin='lower', extent=[x_range[0],x_range[1],y_range[0],y_range[1]])#, origin='lower')
        # im = axis.imshow(np.log10(hist).T, extent=[x_range[0],x_range[1],y_range[0],y_range[1]], origin='lower',cmap=cmap,
        #   rasterized=rasterized, **kwargs)
        # h, xe, ye, im = axis.hist2d(x_to_bin, y_to_bin, bins = bins, cmin=cmin, zorder=2, norm=LogNorm(),
        # rasterized=rasterized, **kwargs)
        # else:
        # h, xe, ye, im = axis.hist2d(x, y, bins = bins, cmin=cmin, zorder=2, norm=LogNorm(),
        # rasterized=rasterized, **kwargs)
        if colorbar == True:
            plt.colorbar(im, ax=axis, orientation='vertical', label=r'$\log\left(N_{\rm gal}\right)$')

    if add_mean == True:
        from scipy.stats import binned_statistic

        if logscale == True:
            hist, xe, num = binned_statistic(x_to_bin, y_to_bin, statistic=np.median, range=x_range_bin)
            bin_width = (xe[1] - xe[0])
            bin_center = xe[1:] - bin_width / 2.
            axis.plot(10**bin_center, 10**hist, color=mean_color, lw=lw, zorder=10)

            hist, xe, num = binned_statistic(x_to_bin, y_to_bin, statistic=perc_low, range=x_range_bin)
            axis.plot(10**bin_center, 10**hist, color=mean_color, lw=lw, ls='dashed', zorder=10)
            hist, xe, num = binned_statistic(x_to_bin, y_to_bin, statistic=perc_high, range=x_range_bin)
            axis.plot(10**bin_center, 10**hist, color=mean_color, lw=lw, ls='dashed', zorder=10)
        else:
            hist, xe, num = binned_statistic(x_to_bin, y_to_bin, statistic=np.median, range=x_range)
            bin_width = (xe[1] - xe[0])
            bin_center = xe[1:] - bin_width / 2.
            axis.plot(bin_center, hist, color=mean_color, lw=lw, zorder=10)
            hist, xe, num = binned_statistic(x_to_bin, y_to_bin, statistic=perc_low, range=x_range_bin)
            axis.plot(10**bin_center, 10**hist, color=mean_color, lw=lw, ls='dashed', zorder=10)
            hist, xe, num = binned_statistic(x_to_bin, y_to_bin, statistic=perc_high, range=x_range_bin)
            axis.plot(10**bin_center, 10**hist, color=mean_color, lw=lw, ls='dashed', zorder=10)

    if x_range != None:
        axis.set_xlim(x_range)
        axis.set_ylim(y_range)

    return axis


def scatter_plot(x, y, filename='mstar_Z.pdf', axis=None, add_mean=False, mean_color=None,
                 bins=100, scatter_alpha=None, **kwargs):
    '''Plot a scatter plot of two properties x, y where for high density regions a 2d histogram is plotted and 
    outliers are plotted as a scatter plot.

    x, y: x and y data to be plotted

    filename: name to save the plot. If not given, the axis object is returned.

    axis: if provided, the data is plotted into this object, otherwise a new axis object is created.

    add_mean: if True, a line for the mean relation in the color mean_color is added.

    contours: if True instead of density plot only contours are plotted.

    bins: number of bins to use

    cmin: minimum number of counts in bin

    optional arguments:

    num_cont: number of contour levels

    cont_colors: color of contours

    fontsize: fontsize for contours

    cmap: colormap for the density plot

    xlabel: x axis label

    ylabel: y axis label

    logscale: default True, if not linear scale mostly used for the binning of data for the density plots.
    '''

    from scipy.stats import binned_statistic

    x_label = kwargs.pop('xlabel', r'$M_{\rm star}$ [$M_{\odot}$]')
    y_label = kwargs.pop('ylabel', r'$Z$')
    x_range = kwargs.get('x_range', (0, 15))
    y_range = kwargs.get('y_range', (0, 15))
    add_mean = kwargs.pop('add_mean', True)
    y_label = kwargs.pop('y_label', None)

    if axis == None:
        fig = plt.figure()
        axis = plt.subplot(111)

    rasterized = kwargs.get('rasterized', True)
    logscale = kwargs.get('logscale', True)

    print('Plotting all galaxies!')

    _ = make_scatterplot(x, y, axis, bins=bins, alpha=scatter_alpha, mean_color=mean_color, **kwargs)

#    if add_mean == True:
#        if logscale == True:
#            hist, xe, num = binned_statistic(np.log10(x), y, statistic=np.mean, range=np.log10(x_range))
#            bin_width = (xe[1] - xe[0])
#            bin_center = xe[1:] - bin_width / 2.
#            axis.plot(10**bin_center, hist, color=mean_color, lw=2, zorder=10)
#        else:
#            hist, xe, num = binned_statistic(np.asarray(x), np.asarray(y), statistic=np.mean, range=x_range)
#            bin_width = (xe[1] - xe[0])
#            bin_center = xe[1:] - bin_width / 2.
#            axis.plot(bin_center, hist, color=mean_color, lw=2, zorder=10)

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlim(x_range)
    axis.set_ylim(y_range)

    if filename:
        if 'rasterized':
            dpi = kwargs.get('dpi', 600)
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(filename, bbox_inches='tight')

    return axis


def plot_predicted_vs_true(true_prop_arr, pred_prop_arr, filename='predicted_vs_true.pdf', axis=None,
                           add_unity=False, contours=False, cmin=2, **kwargs):
    '''Plot the predicted vs true properties for all properties.

    true_prop_arr: array containing the true data, can be for a single property or for several properties.

    path_predicted: predicted data. Assumption is that for prediction the same data format as for true data is used.

    contours: if True instead of density plot only contours are plotted.

    cmin: minimum number of counts in bin

    optional arguments:

    bins: number of bins to use

    num_cont: number of contour levels

    cont_colors: color of contours

    fontsize: fontsize for contours

    cmap: colormap for the density plot

    xlabel: x axis label

    ylabel: y axis label

    logscale: default True, if not linear scale 
    '''

    from scipy.stats import binned_statistic

    rasterized = kwargs.get('rasterized', True)
    plot_keys = kwargs.pop('plot_keys', ['gas_GFM_Metallicity', 'gas_Masses', 'gas_NeutralHydrogenAbundance',
                                         'gas_StarFormationRate', 'stars_GFM_Metallicity', 'stars_GFM_StellarFormationTime', 'stars_Masses'])

    #keys = list(pred_prop_arr.keys())

    x_range = kwargs.pop('x_range', [(0, 15)] * len(plot_keys))
    y_range = kwargs.pop('y_range', [(0, 15)] * len(plot_keys))
    y_label = kwargs.pop('y_label', None)
    figsize = kwargs.pop('figsize', (3, 21))
    xin_range = kwargs.pop('xin_range', [None] * len(plot_keys))
    msize = kwargs.pop('msize', 5)
    cmap = kwargs.pop('cmap', 'Spectral_r')
    histtype = kwargs.pop('histtype', 'bar')
    add_mean = kwargs.pop('add_mean', True)
    mean_color = kwargs.pop('mean_color', 'c')
    logscale = kwargs.get('logscale', True)
    inset = kwargs.pop('inset', True)
    inset_pos = kwargs.pop('inset_pos', 'low')
    #yin_range = kwargs.pop('yin_range', [None] * len(keys))

    if axis == None:
        
        fig = plt.figure(figsize=figsize)
        axis = []
        gs = gridspec.GridSpec(len(plot_keys), 1)
        gs.update(wspace=.0, hspace=0.)  # set the spacing between axes.

        for i, key in enumerate(plot_keys):
            
            axis.append(plt.subplot(gs[i]))
            if y_label != None:
                axis[i].set_ylabel(y_label[i])
            else:
                axis[i].set_ylabel(key)
            
            if logscale:
                axis[i].set_xscale('log')
                axis[i].set_yscale('log')

    for i, key in enumerate(plot_keys):
        print('Plotting predicted ' + key + ' vs. true ' + key)
        _ = make_scatterplot(true_prop_arr[key], pred_prop_arr[key], axis[i], x_range=x_range[i], y_range=y_range[i], msize=msize, cmap=cmap, 
            histtype=histtype, add_mean=add_mean, mean_color=mean_color, **kwargs)

        if i != len(plot_keys) - 1:
            axis[i].set_xticklabels([])

        if inset:
            # inset plot which shows the scatter around unity in logscale
            mask = (true_prop_arr[key] != 0) & (pred_prop_arr[key] != 0)
            x_arr = true_prop_arr[key][mask]
            y_arr = pred_prop_arr[key][mask]
            #scatter = np.sqrt(np.sum((np.log10(x_arr) - np.log10(y_arr))**2)/len(x_arr))
            if logscale:
                resid = np.log10(x_arr) - np.log10(y_arr)
            else:
                resid = x_arr - y_arr

            # check also for nan
            resid = resid[np.isfinite(resid)]
            scatter2 = np.std(resid)

            if inset_pos == 'low':
                axin = inset_axes(axis[i], width="30%",  # width = 30% of parent_bbox
                   height="30%",  # height : 1 inch
                   loc='lower right')
                #axin = axis[i].inset_axes([0.61,0.15,0.34,0.34])
                axis[i].text(0.25, 0.9, r'$\sigma=%.3f$'%scatter2, fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axis[i].transAxes)
            else:
                axin = inset_axes(axis[i], width="30%",  # width = 30% of parent_bbox
                   height="30%",  # height : 1 inch
                   loc='upper left')
                #axin = axis[i].inset_axes([0.15,0.65,0.34,0.34])
                axis[i].text(0.75, 0.25, r'$\sigma=%.3f$'%scatter2, fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axis[i].transAxes)
            
            axin.hist(resid, bins='auto', histtype=histtype)#, range=(np.percentile(resid,0.1),np.percentile(resid,0.9)))
            if xin_range[i] != None:
                axin.set_xlim(xin_range[i])
            # Set the tick labels font
            for label in (axin.get_xticklabels() + axin.get_yticklabels()):
                label.set_fontsize(15)
            for sp in ['top','bottom','left','right']:
                axin.spines[sp].set_linewidth(1.5)
            #axis[i].text(0.1, 0.9, r'$\sigma=$ '+str(scatter), horizontalalignment='center', verticalalignment='center', transform=axis[i].transAxes)
            if logscale:
                axin.set_xlabel(r'$\Delta \log$'+y_label[i],fontsize=17, labelpad=0.5)
            else:
                axin.set_xlabel(r'$\Delta$'+y_label[i],fontsize=17, labelpad=0.5)
            axin.set_ylabel(r'$N_{\rm gal}$',fontsize=17)#, labelpad=0)
            #axin.yaxis.set_label_coords(-0.2,.5)

    if add_unity == True:
        for i, key in enumerate(plot_keys):
            ax = axis[i]
            x_lim = ax.get_xbound()
            y_lim = ax.get_ybound()
            ax.plot(x_lim, y_lim, color='k', lw=2, zorder=-10)

    axis[-1].set_xlabel('True property')

    if filename:
        if 'rasterized':
            dpi = kwargs.get('dpi', 600)
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(filename, bbox_inches='tight')

    return axis


def plot_accuracy_vs_radius(true_prop_arr, pred_prop_arr, axis=None, x_edges=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5], score_func='log_diff', 
    filename='accuracy_vs_radius.pdf', bins=25, h_range=(-2,2), **kwargs):
    """
    Plot the accuracy of the prediction as measured by a score function as a function of radius.
    This plots a 2d-hist.

    Input:
        true_prop_arr: 2d array of true properties. For each galaxy this array should contain an array of properties measured in radial bins

        pred_prop_arr: same as true_prop_arr

        axis: axis object to plot on. If None, create new figure.

        x_edges: edges of radial bins. defualt values are in effective or half mass radii

        score_func: string specifying the score function to use. defined in analysis.py

        filename: filename to save the plot

        bins: bins for the histogram

        h_range: range values for the hist

        logscale: True, the scaling of the colorbar

        add_stats: True, adds the median and the percentile ranges to the plot

    """
    from .analysis import score

    rasterized = kwargs.get('rasterized', True)
    cmap = kwargs.pop('cmap', 'inferno')
    figsize = kwargs.pop('figsize', (10,10))
    y_label = kwargs.pop('y_label', None)
    colorbar = kwargs.pop('colorbar', True)
    logscale = kwargs.pop('logscale', True)
    add_stats = kwargs.pop('add_stats', True)

    if axis == None:
        
        fig = plt.figure(figsize=figsize)
            
        axis = plt.subplot(111)
        axis.set_xlabel(r'$R/R_{\rm half}$')
        if y_label:
            axis.set_ylabel(y_label)
        else:
            axis.set_ylabel(key)

    # for loops are slow in python but I am too lazy to think about a smart version of this...
    #probably map would work or numpy.apply_along_axis
    img = []

    if add_stats:
        median = []
        percentile_low = []
        percentile_high = []

    for i in range(len(true_prop_arr[0])-1):
        # do we need a check if xbins is the same length as the arrays in true_prop_arr or pred_prop_arr?
        sc = []
        for true_gal, pred_gal in zip(true_prop_arr,pred_prop_arr):
            if i == 0:
                sc.append(score(true_gal[0],pred_gal[0],score_func))
            else:
                sc.append(score(true_gal[i+1]-true_gal[i],pred_gal[i+1]-pred_gal[i],score_func))

        hist, ye = np.histogram(sc,bins=bins,range=h_range)
        
        if add_stats:
            mask = np.isfinite(sc)
            sc = np.asarray(sc)
            median.append(np.median(sc[mask]))
            percentile_low.append(np.percentile(sc[mask],16))
            percentile_high.append(np.percentile(sc[mask],84))

        if logscale:
            img.append(np.log10(hist))
        else:
            img.append(hist)            

    c = axis.pcolor(x_edges[:-1],ye,np.asarray(img).T,**kwargs)

    cmp = plt.cm.get_cmap(cmap)
    rgba = cmp(0.0)
    axis.set_facecolor(rgba)
    
    axis.plot(axis.get_xlim(),[0,0],c='k', lw=2)
    
    if add_stats:
        delta = 0.5*(x_edges[1]-x_edges[0])
        axis.plot(np.asarray(x_edges[:-2])+delta,median,c='c')
        axis.plot(np.asarray(x_edges[:-2])+delta,percentile_low,c='c')
        axis.plot(np.asarray(x_edges[:-2])+delta,percentile_high,c='c')

    if colorbar:
        plt.colorbar(c, ax=axis, label=r'$\log(N)$')

    if filename:
        plt.savefig(filename,bbox_inches='tight')

    return axis


default_plot_keys = ('gas_GFM_Metallicity', 'gas_Masses', 'gas_NeutralHydrogenAbundance',
                     'gas_StarFormationRate', 'stars_GFM_Metallicity', 'stars_GFM_StellarFormationTime', 'stars_Masses')

def plot_MSE_per_channel(path_true, path_predicted, filename='MSE_per_property.pdf', axis=None,
                            rasterized=True, plot_keys=default_plot_keys, x_range=(0, 15), y_range=(0, 15)):
    """
    Plot the MSE of all predicted properties.
    
    Args:
        path_true: path to folder with true image data, assume for every galaxy and every camera there is one file.
        path_predicted: path to folder with predicted image data. Assumption is that also for prediction the same data format is used.
        
    Returns:
        matplotlib axis object of visualization

    """

    from scipy.stats import binned_statistic

    # rasterized = kwargs.get('rasterized', True)
    # plot_keys = kwargs.pop('plot_keys', ['gas_GFM_Metallicity', 'gas_Masses', 'gas_NeutralHydrogenAbundance',
    #                                      'gas_StarFormationRate', 'stars_GFM_Metallicity', 'stars_GFM_StellarFormationTime', 'stars_Masses'])

    keys = list(pred_prop_arr.keys())

    # x_range = kwargs.pop('x_range', [(0, 15)] * len(keys))
    # y_range = kwargs.pop('y_range', [(0, 15)] * len(keys))

    if axis == None:
        fig = plt.figure(figsize=(3, 21))
        axis = []
        gs = gridspec.GridSpec(len(plot_keys), 1)
        gs.update(wspace=.0, hspace=0.)  # set the spacing between axes.

        for i, key in enumerate(plot_keys):
            axis.append(plt.subplot(gs[i]))
            axis[i].set_ylabel(key)
            # axis[i].set_xscale('log')
            # axis[i].set_yscale('log')

    for i, key in enumerate(plot_keys):
        print('Plotting predicted ' + key + ' vs. true ' + key)
        # _ = make_scatterplot(true_prop_arr[key], pred_prop_arr[key], axis[i],
        #                      x_range=x_range[i], y_range=y_range[i], **kwargs)


    if add_unity == True:
        for i, key in enumerate(plot_keys):
            ax = axis[i]
            x_lim = ax.get_xbound()
            y_lim = ax.get_ybound()
            ax.plot(x_lim, y_lim, color='k', lw=2, zorder=10)

    axis[-1].set_xlabel('True property')

    if filename:
        if 'rasterized':
            dpi = kwargs.get('dpi', 600)
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(filename, bbox_inches='tight')

    return axis


def plot_property_vs_mhalo(prop_arr, dm_arr, filename='prop_vs_mhalo.pdf', axis=None, add_mean=False, mean_color=None, **kwargs):
    '''Plot the properties vs the true halo mass for all properties.
    '''

    label = {'stars_Masses': r'$M_{\rm star}$ [$M_{\odot}$]', 'gas_Masses': r'$M_{\rm gas}$ [$M_{\odot}$]', 'stars_GFM_Metallicity': r'$Z_{\rm star}$',
             'gas_GFM_Metallicity': r'$Z_{\rm gas}$', 'gas_GFM_StellarFormationTime': r'stellar age [Gyr]'}  # dictionary for axis labels
    prop_to_scale = {'stars_Masses': 'Masses', 'gas_Masses': 'Masses', 'stars_GFM_Metallicity': 'stars_Masses',
                     'gas_GFM_Metallicity': 'gas_Masses', 'gas_GFM_StellarFormationTime': 'GFM_StellarFormationTime'}  # dictionary for scaling

    from scipy.stats import binned_statistic

    keys = list(prop_arr.keys())

    rasterized = kwargs.get('rasterized', True)
    x_range = kwargs.pop('x_range', [(0, 15)])
    y_range = kwargs.pop('y_range', [(0, 15)] * len(keys))

    if axis == None:

        fig = plt.figure(figsize=(3, 21))
        axis = []
        gs = gridspec.GridSpec(len(keys), 1)
        gs.update(wspace=.0, hspace=0.1)  # set the spacing between axes.

        for i, key in enumerate(keys):
            axis.append(plt.subplot(gs[i]))
            axis[i].set_ylabel(key)
            axis[i].set_xscale('log')
            axis[i].set_yscale('log')

    for i, key in enumerate(keys):
        if key in label.keys():
            # need to scale the values to physical units... maybe create a separate file?
            _ = make_scatterplot(dm_arr, np.asarray(prop_arr[key]), axis[
                                 i], x_range=x_range, y_range=y_range[i], **kwargs)
            axis[i].set_ylabel(label[key])

    if add_mean == True:
        # keys should exist from the last loop
        for i, key in enumerate(keys):

            hist, xe, num = binned_statistic(np.asarray(dm_arr), np.asarray(prop_arr[key]), statistic=np.mean)
            bin_width = (xe[1] - xe[0])
            bin_center = xe[1:] - bin_width / 2.
            axis[i].plot(bin_center, hist, color=mean_color, lw=2, zorder=10)

    axis[-1].set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')

    if filename:
        if 'rasterized':
            dpi = kwargs.get('dpi', 600)
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(filename, bbox_inches='tight')

    return axis