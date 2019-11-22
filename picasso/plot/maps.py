"""

maps
==========

A set of functions to plot maps of the galaxies.

"""

import numpy as np 
import matplotlib.pylab as plt
import gc 

def make_maps(galaxy, key, save=False, **kwargs):
    '''Make maps of quantities key.

        Input:

            galaxy: galaxy object
    '''

    filename = kwargs.pop('filename', galaxy._base_path + galaxy._Galaxy_id + "_" + str(key) + '.pdf')

    x = galaxy.properties[key]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    if 'GFM_Metallicity' in key:
        cmap = kwargs.pop('cmap', 'plasma')
    elif 'NeutralHydrogenAbundance' in key:
        cmap = kwargs.pop('cmap', 'Blues')
    elif 'StarFormationRate' in key:
        cmap = kwargs.pop('cmap', 'inferno')
    elif 'StellarFormationTime' in key:
        cmap = kwargs.pop('cmap', 'magma_r')
    elif 'stars_Masses':
        cmap = kwargs.pop('cmap', 'viridis')
    else:
        cmap = kwargs.pop('cmap', 'Greys')

    img = ax.imshow(np.log10(x), cmap=cmap, origin='lower')
    
    if save:
        plt.savefig(filename)

    return x

def rgb_image(galaxy, filename=None, r_band='i', g_band='r', b_band='g',
    r_scale=1.3, g_scale=1.0, b_scale=0.7, 
    lupton_alpha=0.5, lupton_Q=0.5, scale_min=1e-4, 
    ret_img=False, axes=None, plot=True, save_indiv_bands=False, **kwargs):

    '''
    Make a 3-color image of the galaxy based on the galaxy maps in
    different wave legnth bands.

    Input:

        galaxy: galaxy object

    **Optional keyword arguments:**

       *filename*: string (default: None)
         Filename to be written to (if a filename is specified)

       *r_band*: string (default: 'i')
         Determines which Johnston filter will go into the image red channel

       *g_band*: string (default: 'v')
         Determines which Johnston filter will go into the image green channel

       *b_band*: string (default: 'b')
         Determines which Johnston filter will go into the image blue channel

       *r_scale*: float (default: 1.3)
         The scaling of the red channel before channels are combined

       *g_scale*: float (default: 1.0)
         The scaling of the green channel before channels are combined

       *b_scale*: float (default: 0.7)
         The scaling of the blue channel before channels are combined

       *lupton_alpha: float (default 0.5)
         luptin alpha parameter

       *lupton_Q: float (default 0.5)
         luptin Q parameter

       *scale_min: float (default 1e-4)
         minimum scale

       *ret_img*: bool (default: False)
         if True, the NxNx3 image array is returned

       *axes*: matplotlib axes object (deault: None)
         if not None, the axes object to plot to

       *save_indiv_bands: bool (default False)
         if True, all three individual bands representing the r,g and b
         channel are saved as well. 

    Returns:

        If ret_im=True, an NxNx3 array representing an RGB image

    '''

    filename = kwargs.pop('filename', galaxy._base_path + galaxy._Galaxy_id + "_rgb")

    # use np.copy to avoid changing the initial images when scaling by r_,g_,b_scale later
    r_image = np.copy(galaxy.properties[r_band+'_band'])
    g_image = np.copy(galaxy.properties[g_band+'_band'])
    b_image = np.copy(galaxy.properties[b_band+'_band'])

    n_pixels = r_image.shape[0]
    img = np.zeros((n_pixels, n_pixels, 3), dtype=float)

    b_image *= b_scale
    g_image *= g_scale
    r_image *= r_scale

    I = (r_image + g_image + b_image) / 3
    val = np.arcsinh( lupton_alpha * lupton_Q * (I - scale_min)) / lupton_Q
    I[ I < 1e-6 ] = 1e100        # from below, this effectively sets the pixel to 0

    img[:,:,0] = r_image * val / I
    img[:,:,1] = g_image * val / I
    img[:,:,2] = b_image * val / I

    maxrgbval = np.amax(img, axis=2)

    changeind = maxrgbval > 1.0
    img[changeind,0] = img[changeind,0]/maxrgbval[changeind]
    img[changeind,1] = img[changeind,1]/maxrgbval[changeind]
    img[changeind,2] = img[changeind,2]/maxrgbval[changeind]

    minrgbval = np.amin(img, axis=2)
    changeind = minrgbval < 0.0
    img[changeind,0] = 0
    img[changeind,1] = 0
    img[changeind,2] = 0

    changind = I < 0
    img[changind,0] = 0
    img[changind,1] = 0
    img[changind,2] = 0
    img[img<0] = 0

    img[img<0] = 0

    if plot:
        if axes is None:
            fig = plt.figure()
            axes = plt.subplot(111)

        if axes:
            axes.imshow(img[::-1, :], origin='lower', interpolation='nearest')
            plt.axis('off')

        plt.savefig(filename+'.pdf', bbox_inches='tight')

    if save_indiv_bands:
        for iii in np.arange(3):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            imgplot = ax.imshow(img[:,:,iii],origin='lower', interpolation='nearest', cmap = 'Greys', vmin=0, vmax=1)
            plt.axis('off')
            fig.savefig(filename[:-4]+"_band_"+str(iii)+'.pdf', bbox_inches='tight')

            del imgplot

    if ret_img:
        return img

    del b_image, g_image, r_image, I, val, img
    gc.collect()

