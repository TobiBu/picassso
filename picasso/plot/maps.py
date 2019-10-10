"""

maps
==========

A set of functions to plot maps of the galaxies.

"""

def make_maps(galaxy, key, **kwargs):
    '''Make maps of quantities key. 

        Input:

            galaxy: galaxy object
    '''

    filename = galaxy._base_path + galaxy._Galaxy_id + "_" + str(key) + '.png'

    x = galaxy[key]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    if 'Masses' in key: 
        ax.imshow(np.log10(x), cmap='viridis', origin='lower')
    elif 'GFM_Metallicity' in key:        
        ax.imshow(x, cmap='plasma', origin='lower')
    elif 'NeutralHydrogenAbundance' in key:
        ax.imshow(x, cmap='Blues', origin='lower')
    elif 'StarFormationRate' in key:
        ax.imshow(x, cmap='inferno', origin='lower')
    elif 'StellarFormationTime' in key:
        ax.imshow(x, cmap='magma_r', origin='lower')
    else:
        ax.imshow(np.log10(x), cmap='viridis', origin='lower')

    plt.savefig(filename)

    return