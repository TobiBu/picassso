"""

image_tools
===========

A set of classes and functions for analysing galaxy images.

"""

import numpy as np
import os, pickle

from photutils.isophote import *

from tqdm import tqdm

from .. import survey, galaxy
from ..survey import Survey
from ..galaxy import Galaxy

# modify the code below to comply with the Survey and galaxy objects derived arrays once the key error is removed...
# get the RGB image functions from pynbody to combine u,g,r,i.z images to nice RGB images...


def circular_geometry(x0=128, y0=128, sma=25):
    '''Initialize circular geometry object from ellipse geometry.
        For further information see documentation of the photutils package "https://photutils.readthedocs.io/en/stable/"
        
        Input:

            x0, y0 : center in pixel 

            sma : semi-major axis of initial ellipse in pixels.
            
            eps: ellipticity of initial ellipse

        Return:

            Function returns a geometry object as specified in the photutils package.
    '''

    from photutils.isophote import EllipseGeometry

    x0 = kwargs.pop('x0', 0.5*image.shape[0])
    y0 = kwargs.pop('y0', 0.5*image.shape[1])
    sma = kwargs.pop('sma', 25/256*image.shape[0]) # this corresponds to the half mass radius

    geometry = photutils.isophote.EllipseGeometry(x0, y0, sma, 1., 0., 0.1, False)

    return geometry


def elliptical_fit(image, thresh=0.1, x0=128, y0=128, sma=25, pa=45, eps=0.25, astep=0.1, linear_growth=False, maxsma=125, minsma=12.5, **kwargs):
    '''
            Fits an elliptical aperture geometry to an image.

            input:

                    image: array containing the image to fit.

                    thresh: threshold value for the ellipse fitting procedure,
                    for more details see photutils package.

                    x0, y0: pixel values of ellipse center 

                    sma: semi-major axis of initial ellipse in pixels. A fiducial value of 50 would correspond to 
                            2 Re in fiducial images of 256 pixel per 10 Re.

                    eps: ellipticity of initial ellipse

                    pa: position angle of initial ellipse

                    astep: The step value for growing/shrinking the semimajor axis. It can be expressed 
                            either in pixels (when linear_growth=True) or as a relative value 
                            (when linear_growth=False). The default is 0.1.

                    linear_growth: The semimajor axis growing/shrinking mode. The default is False.

                    maxsma: Maximum value for semi-major axis in pixel. If not set, the algorithm will determine
                             when to stop growing the ellipse itself. For more details see 
                             https://photutils.readthedocs.io/en/stable/api/photutils.isophote.Ellipse.html#photutils.isophote.Ellipse.fit_image

            Return:

                    Function returns the isophote as specified by photutils. This contains semi-major axis, isophote values plus errors
                     as well as flags specifying goodness of fit. 
    '''

    from photutils.isophote import EllipseGeometry
    from photutils.isophote import Ellipse

    #scale everything to actual image resolution/ pixel size
    
    x0 = kwargs.pop('x0', 0.5*image.shape[0])
    y0 = kwargs.pop('y0', 0.5*image.shape[1])
    sma = kwargs.pop('sma', 25/256*image.shape[0]) # this corresponds to the half mass radius
    maxsma = kwargs.pop('maxsma', 5*sma)
    minsma = kwargs.pop('minsma', 0.5*sma)

    geometry = EllipseGeometry(x0=x0, y0=y0, sma=sma, eps=eps, pa=pa, astep=astep, linear_growth=linear_growth)

    ellipse = Ellipse(image, geometry, threshold=thresh)
    iso = ellipse.fit_isophote(sma,fflag=0.25)
    count = 0
    while iso.stop_code > 2 and count < 10:
        sma *= 1.1
        eps *= 0.9
        pa *= 1.1
        geometry = EllipseGeometry(x0=x0, y0=y0, sma=sma, eps=eps, pa=pa, astep=astep, linear_growth=linear_growth)
        iso = ellipse.fit_isophote(sma,fflag=0.25)
        count += 1
    
    if iso.stop_code > 2:
        print('Bad fit! Please check the fit and perhaps retry fitting manually!')

    return iso

def _fit_image(galaxy, key='stars_Masses', thresh=0.1, plot=False, save=True, **kwargs):

    # make this a lazy load function: 
        # - upon call, fit the ellipse to the mass map and create the geometry and isophotes, return a isophote object, this might as well be saved as a pickle file...
        # - then, use keys to do the actual fitting of the property requested. Similarly, the result might be saved in some way...
        # look at the profile class of pb


    '''Preprocess images and fit an ellipse to the data. The elliptical apertures can then be used
        to calculate total stellar masses or other properties within a given elliptical aperture or to 
        derive gradient values.
        If a data file containing elliptical apertures for each galaxy is present, this function loads 
        and returns the geometries.
        If not, it will create the data file and returns the calculated data. 

            galaxy: galaxy object to perform the fit on.

            key: property from which elliptical geometries should be calculated.

            thresh: threshold for ellipse fittin method

            plot: if True, a plot of property map with the result of the ellipse fit is done.

            save: if True, the resulting geometry is saved on disk for later re-use. 
    '''

    filename = galaxy._base_path + galaxy._Galaxy_id + '_ellipse_geometry.dat'
    if not os.path.isfile(filename):
        # no pre-calculated geometry found, lets do the fit
        try:
            # I think we need to pop the specific kwargs from dict...
            fit = elliptical_fit(galaxy.properties[key], thresh=thresh, **kwargs)
            geometry = fit.sample.geometry
        except:
            print('No elliptical fit possible for ' + galaxy._descriptor + '!')
            print('Rolling back to circular aperture.')
            geometry = circular_geometry() #check how to pass the right kwargs in a smart way
        if save:
            f = open(filename, 'wb')
            pickle.dump({'geometry': geometry}, f)
            f.close()
    else:
        data_ = pickle.load(open(filename, 'rb'))
        geometry = data_['geometry']

    if plot:
        import matplotlib.pylab as plt
        plt.imshow(image)
        x, y, = fit.sampled_coordinates()
        plt.plot(x, y, color='w')
        plt.savefig(filename[:-20] + '.pdf')
        plt.close()

    return geometry

def fit_image(galaxy, key='stars_Masses', thresh=0.1, plot=False, save=True, **kwargs):

    # make this a lazy load function: 
        # - upon call, fit the ellipse to the mass map and create the geometry and isophotes, return a isophote object, this might as well be saved as a pickle file...
        # - then, use keys to do the actual fitting of the property requested. Similarly, the result might be saved in some way...
        # look at the profile class of pb


    '''Preprocess images and fit an ellipse to the data. The elliptical apertures can then be used
        to calculate total stellar masses or other properties within a given elliptical aperture or to 
        derive gradient values.
        If a data file containing elliptical apertures for each galaxy is present, this function loads 
        and returns the geometries.
        If not, it will create the data file and returns the calculated data. 

            galaxy: galaxy or survey object to perform the fit on.

            key: property from which elliptical geometries should be calculated.

            thresh: threshold for ellipse fittin method

            plot: if True, a plot of property map with the result of the ellipse fit is done.

            save: if True, the resulting geometry is saved on disk for later re-use. 
    '''

    if isinstance(galaxy, Galaxy):
        # we are dealing with a single galaxy object.
        geometry = _fit_image(galaxy, key=key, thresh=thresh, plot=plot, save=save, **kwargs)

    elif isinstance(galaxy, Survey):
        # we are dealing with a whole survey of galaxies, lets iterate over all galaxies in the survey.
        # should we parallelize it???
        geometry = np.asarray([_fit_image(x, key=key, thresh=thresh, plot=plot, save=save, **kwargs) for x in tqdm(galaxy['galaxy'])])

    else:
        raise ValueError("Unknown Object type: %s" %galaxy)

    return geometry

def build_isolist(image, geometry, sma_min=12.5, sma_max=125, sma_steps=10, **kwargs):
    '''
    This function loops through a number of semi-major axis values and creates 
    a list of isophotes of given geometry from the input image.

    for further information see:
            https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    Input:

            image: input image

            geometry: instance of EllipseGeometry in order to specify the ellipse geometry to use.

            sma_min, sma_max: minimum and maximum of semi-major axis values in pixel

            sma_steps: number of bins to take in semi-major axis
    '''

    # Temporary list to store instances of Isophote
    isolist_ = []

    sma_min = kwargs.pop('sma_min', 12.5/256*image.shape[0]) # this corresponds to 0.5 half mass radii
    sma_max = kwargs.pop('sma_max', 125/256*image.shape[0])

    # loop over semi-major axis values
    smas = np.linspace(sma_min, sma_max, sma_steps)
    for sma in smas:
        # create ellipse sample
        sample = EllipseSample(image, sma, geometry=geometry)
        sample.update()
        # Create an Isophote instance with the sample, and store it in
        # temporary list. Here we are using '0' as the number of iterations,
        # 'True' for the validity status, and '0' for the stop code. These
        # are in fact arbitrary in this context; you could use anything you
        # like.
        iso_ = Isophote(sample, 0, True, 0)
        isolist_.append(iso_)

    # Build the IsophoteList instance with the result.
    isolist = IsophoteList(isolist_)

    return isolist

def _get_isolists(galaxy, geometry, key=None, plot=False, **kwargs):
    '''
    Get the isocontour lists for the properties as a photutils.isophote.IsophoteList object.

    Inoput:
            galaxy : galaxy object

            geometry: geometry object defining the orientation of the galaxy

            key: if not None, only the isolist for the specified property is returned. 

            plot: parameter to decide if the ellipse should be plotted ontop of the galaxy image.
    '''

    filename = galaxy._base_path + galaxy._Galaxy_id + 'isolists.dat'

    if not os.path.isfile(filename):
        # no precalculated file with isolists found...
        # build from scratch

        # use the geometry in order to define a list of ellipses from which properties
        # inside different ellipses can be calculated (e.g. mass inside ellipse of semi-major axis 2 Re or 1 Re)
        # and gradients can be defined

        # for every property build isolist separately
        iso_dict = {}

        if key:
            isolist = build_isolist(galaxy.properties[key], geometry)

            iso_dict[key] = isolist
        
        else:
            for key in galaxy.properties.keys():

                if isinstance(galaxy.properties[key], np.ndarray):
                    # this check might fail in future, when we have other galaxy properties 
                    # which are of instance np.ndarray but are not an imaage...
                    isolist = build_isolist(galaxy.properties[key], geometry)

                    iso_dict[key] = isolist

        f = open(filename, 'wb')
        pickle.dump(iso_dict, f)
        f.close()

    else:
        iso_dict = pickle.load(open(filename, 'rb'))

    return iso_dict


def get_isolists(galaxy, geometry, key=None, plot=False, **kwargs):
    '''
    Get the isocontour lists for the properties as a photutils.isophote.IsophoteList object.

    Input:
            galaxy : galaxy or Survey object

            geometry: geometry object or in case of Survey object input, the array of geometries corresponding to the galaxies in the survey.

            key: if not None, only the isolist for the specified property is returned. 

            plot: parameter to decide if the ellipse should be plotted ontop of the galaxy image.

    Return:
            dictionary of isocontours or array of dictionaries.
    '''

    if isinstance(galaxy, Galaxy):
        # we are dealing with a single galaxy object.
        iso_dict = _get_isolists(galaxy, geometry, key=key, plot=plot, **kwargs)

    elif isinstance(galaxy, Survey):
        # we are dealing with a whole survey of galaxies, lets iterate over all galaxies in the survey.
        # should we parallelize it???
        iso_dict = np.asarray([_get_isolists(x, g, key=key, plot=plot, **kwargs) for x, g in tqdm(zip(galaxy['galaxy'],geometry))])

    return iso_dict


def _get_pixel_sum(iso_dict, prop, sma, sma_low=None):
    '''
    Takes an isolist dictionary as defined by the function get_isolists and returns the sum of pixel values inside a given aperture
    specified by semi-major axis length sma.
    If sma_low is set, only pixel in the elliptical annulus between sma and sma_low will be used.
    '''

    isolist = iso_dict[prop]

    iso = isolist.get_closest(sma)
    value = iso.tflux_e  # sum of all pixels inside the ellipse
    npix = iso.npix_e  # number of pixels inside the ellipse

    if sma_low:
        iso_low = isolist.get_closest(sma_low)
        value_low = iso_low.tflux_e  # sum of all pixels inside the ellipse
        npix_low = iso_low.npix_e  # number of pixels inside the ellipse
        value = value - value_low
        npix = npix - npix_low

    return value, npix


def get_pixel_sum(iso_dict, prop, sma, sma_low=None):
    '''
    Takes an isolist dictionary as defined by the function get_isolists and returns the sum of pixel values inside a given aperture
    specified by semi-major axis length sma.
    If sma_low is set, only pixel in the elliptical annulus between sma and sma_low will be used.
    '''

    if isinstance(iso_dict, dict):
        value, npix = _get_pixel_sum(iso_dict, prop, sma, sma_low)  

    elif isinstance(iso_dict, np.ndarray):

        tmp = np.asarray([_get_pixel_sum(x, prop, sma, sma_low) for x in tqdm(iso_dict)])
        value = tmp[:,0]
        npix = tmp[:,1]

    return value, npix
    

def _get_image_sum(galaxy, key):
    # This can be moved easily to a decorator function of the Galaxy objects / Survey objects....
    '''
    Get a sum over all pixel.

    Inoput:
        galaxy: galaxy or Survey object

        key: property key
    '''

    if isinstance(galaxy, Galaxy):
        sum = np.sum(galaxy.properties[key])

    elif isinstance(galaxy, Survey):
        sum = np.asarray([np.sum(g.properties[key]) for g in tqdm(galaxy['galaxy'])])

    return sum




##################################################################################################################
# some unused mask arrays

# mimick IFU footprints on images
# in general this is a collection of polygonial masks for numpy arrays
# MANGA footprint on images

# some of the functions defined below are redundant as they are already implemented in the photutils package...
# However, having them coded here we know what they are doing and how they work.

def createHexagonalMask(nx, ny, size=128):
    '''
        Creates a hexagonal shaped mask array of shape (nx, ny). With the diameter parallel to the x-axis.

            input:
                nx, ny: dimensions

                size: of the hexagon in pixels, maximum: half the size of array in x-direction

    '''

    from matplotlib.path import Path

    # define the vertices of the hexagon centered on the image center
    poly_verts = np.asarray([(0, 1), (0.5, 1 - np.sqrt(3) / 2.), (1.5, 1 - np.sqrt(3) / 2.),
                            (2, 1), (1.5, 1 + np.sqrt(3) / 2.), (0.5, 1 + np.sqrt(3) / 2.)]) * size

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid


def createAnnularMask(nx=256, ny=256, center=[128, 128], small_radius=32, big_radius=64):
    '''
        Creates an annulus shaped mask.

        input:
            nx, ny: dimensions
            center: array of center position of annulus in pixel

            big_radius: upper value for annulus big_radius
            small_radius: lower value of annulus radius

    '''
    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = np.logical_and(small_radius <= distance_from_center, distance_from_center <= big_radius)

    return mask


def createEllipticalAnnularMask(nx=256, ny=256, center=[128,128], small_radius=32, big_radius=64, b=32, theta=0.):
    '''
        Creates an elliptically shaped annulus mask based on the photutils package.
        For more details on photutils see https://photutils.readthedocs.io

        input:
            nx, ny: dimensions
            center: array of center position of annulus in pixel

            big_radius: upper value for annulus semi-major axis
            small_radius: lower value of annulus semi-major axis
            b: value for outer semi-minor axis 
            theta: rotation angle from positive x-axis, measured counter-clockwise
    '''

    from photutils import EllipticalAnnulus
    annulus_apertures = EllipticalAnnulus(center,small_radius,big_radius,b)
    annulus_masks = annulus_apertures.to_mask(method='center')

    image = annulus_masks[0].to_image(shape=((nx, ny)))

    mask = image < 1

    return mask

def createEllipticalMask(nx=256, ny=256, center=[128,128], radius=64, b=32, theta=0.):
    '''
        Creates an elliptically shaped mask based on the photutils package.
        For more details on photutils see https://photutils.readthedocs.io

        input:
            nx, ny: dimensions
            center: array of center position of annulus in pixel

            radius: value for annulus semi-major axis
            b: value for semi-minor axis 
            theta: rotation angle from positive x-axis, measured counter-clockwise
    '''

    return createEllipticalAnnularMask(nx, ny, center, 1e-20, radius, b)


def manganize(img, size=128):
    '''Function returns the regualr hexagon of a SDSS MANGA-like observation.
    Define a regular hexagon to mask out pixels outside the hexagon.

        input:

            img: input 2d numpy array containing the image to be masked

            size: half diamter of the hexagon, default: half the image width 

        returns:

            2d masked array of the input image
    '''

    nx = len(im[:,0])
    ny = len(im[0])

    grid = createHexagonalMask(nx, ny, size)
    
    new_arr = np.ma.array(img, mask = ~grid)

    return new_arr


def circ_annulus(img, small_radius, big_radius, center=[128,128]):
    '''
        Returns image values inside annulus of radius (small_radius, big_radius).

        input:

            img: input 2d numpy array containing the image to be masked
            
            big_radius: upper value for annulus big_radius
            small_radius: lower value of annulus radius
            center: array of center position of annulus in pixel
    '''

    nx = len(im[:,0])
    ny = len(im[0])

    grid = createAnnuluslMask(nx, ny, center, small_radius, big_radius)
    
    new_arr = np.ma.array(img, mask = ~grid)

    return new_arr


def elliptical_annulus(img, small_radius, big_radius, b, center=[128,128], theta=0.):
    '''
        Returns image values inside elliptical annulus of radius (small_radius, big_radius) with semi-minor axis b.

        input:

            img: input 2d numpy array containing the image to be masked
            
            big_radius: upper value for semi-major annulus
            small_radius: lower value of semi-major annulus radius
            b: upper value for semi-minor axis radius
            center: array of center position of annulus in pixel
    '''

    nx = len(im[:,0])
    ny = len(im[0])

    grid = createEllipticalAnnularMask(nx, ny, center, small_radius, big_radius, b, theta)
    
    new_arr = np.ma.array(img, mask = ~grid)

    return new_arr

def ellipse_aperture(img, radius, b, center=[128,128], theta=0.):
    '''
        Returns image values inside elliptical aperture of radius radius with semi-minor axis b and angle theta.

        input:

            img: input 2d numpy array containing the image to be masked
            
            radius: value for semi-major annulus
            b: value for semi-minor axis radius
            center: array of center position of annulus in pixel
            theta: rotation angle from positive x-axis, measured counter-clockwise
    '''

    nx = len(img[:,0])
    ny = len(img[0])

    grid = createEllipticalMask(nx, ny, center, radius, b, theta)
    
    new_arr = np.ma.array(img, mask = grid)

    return new_arr