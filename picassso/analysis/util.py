"""

utility functions
=================

A set of classes and functions for analysing galaxy images.

"""

import numpy as np
from . import image_tools
from .. import util

from tqdm import tqdm
import os

def fit_gradient(iso_dict, prop, log=True):
	"""
	fit a straight line to the intensity values which are actually only the mean value along an elliptical path.

	Input:
			iso_dict: isolist dictionary as defined by the function get_isolists.

			prop: key to specify the property

			log: bool to decide if gradient should be calculated from log scaled prop

	Return:
			Radial gradient.

	"""

	from scipy.stats import linregress

	if log:
		y = np.log10(iso_dict[prop].intens)
	else:
		y = iso_dict[prop].intens

	slope, intercept, r_value, p_value, std_err = linregress(iso_dict[prop].sma, y)

	return slope

def fit_gradient_manually(sma_arr, sum_arr, pix_arr, log=True):
	"""
	fit a straight line to the property values.

	Input:
		sma_arr: array containing the semi major axis values of the annuli.

		sum_arr: array containing the sum of pixel values in elliptical annulus.

		pix_arr: array containing the number of pixels in elliptical annulus.

		log: bool to decide if gradient should be calculated from log scaled prop

	Return:
		Radial gradient.

	"""

	from scipy.stats import linregress

	mean = np.asarray(sum_arr)/np.asarray(pix_arr)
	if log:
		y = np.log10( mean )
	else:
		y = mean

	slope, intercept, r_value, p_value, std_err = linregress(sma_arr,y)

	return slope


def _get_property_value(iso_dict, prop, sma, sma_low=None):
    '''
    Takes an isolist dictionary as defined by the function get_isolists and returns the sum of pixel values inside a given aperture
    specified by semi-major axis length sma.
    If sma_low is not set to None only pixel in the elliptical annulus between sma and sma_low will be used.
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


def _get_predicted_vs_true(filename):
    '''
    Load the pre-calculated dataset of predicted and true galaxy properties.
    This is a utility function in cases where the file already exists and you don't want to load the whole survey.

    Input:
        filename: name of the hdf5 file containing the data

    Return:
        dictionaries of true and predicted data as well as the dark matter mass from the halo finder.
    '''
    with util.open_hdf5(filename, "r") as h5file:
        keys = list(h5file.keys())
        keys = [k for k in keys if k not in ['dm_mass']]

        true_keys = [k for k in keys if 'pred' not in k]
        pred_keys = [k for k in keys if 'pred' in k]

        assert(len(true_keys) == len(pred_keys))

        true_prop_arr = {} #{key: [] for key in true_keys}  # fiducial sum at 2Rhalf
        pred_prop_arr = {} #{key: [] for key in true_keys}

        for true_key, pred_key in zip(true_keys, pred_keys):
            pred_prop_arr[true_key] = h5file[pred_key][()]
            true_prop_arr[true_key] = h5file[true_key][()]

        dm_arr = h5file['dm_mass'][()]

    return true_prop_arr, pred_prop_arr, dm_arr

def get_predicted_vs_true_data(survey, filename='predicted_vs_true.h5', plot=False, **kwargs):
    '''
    Preprocess the survey data to extract values of the properties within a given radii. 
    Thus we have all the data needed to make plots of predicted vs true properties for all 
    available galaxy properties.

    Input:

        survey: survey object

        filename: if given, the resulting arrays are stored in this file for future usage.

        plot: bool to decide if intermediate plots of the image fitting should be created.

    Return:

        dictionary of numpy ndarrays of the data ready to plot.
    
    '''

    if not os.path.isfile(filename):
        print('No pre-calculated data file for the plot found.')
        print('Creating the data file.')

        # get dummy keys
        keys = survey['galaxy'][0].keys #survey['galaxy'][0].properties.keys()
        keys = [k for k in keys if k not in ['Galaxy_id','psize','dm_mass']]

        true_keys = [k for k in keys if 'pred' not in k]
        pred_keys = [k for k in keys if 'pred' in k]

        true_prop_arr = {key: [] for key in true_keys}  # fiducial sum at 2Rhalf
        pred_prop_arr = {key: [] for key in pred_keys}

        true_prop_arr_all = {key: [] for key in true_keys}  # fiducial sum over all at pixel
        pred_prop_arr_all = {key: [] for key in pred_keys}

        true_prop_arr_rad = {key: [] for key in true_keys}  # sum of pixel values at different radii
        pred_prop_arr_rad = {key: [] for key in pred_keys}
        
        true_prop_arr_rad_npix = {key: [] for key in true_keys}  # number of pixels for the sums above
        pred_prop_arr_rad_npix = {key: [] for key in pred_keys}
        
        true_prop_grad = {key: [] for key in true_keys}  # gradient values
        pred_prop_grad = {key: [] for key in pred_keys}
        
        dm_arr = []

        # get the geometry of the galaxies
        # not necessarily needed, since its automatically called in the next step
        geom = survey['geom'] # equivalently one could call the underlying function: image_tools.fit_image(survey)

        # calculate the isophotes (where isophotes are actually just ellipses of given geometry and semi-major axis)
        iso_dict_list  = survey['iso_dict'] # again can be obtained by directly calling the underlying function image_tools.get_isolists(survey,geom)

        img_res = survey.properties['image_res']
        r_half_frac = 50./256.

        for idx, gal in tqdm(enumerate(survey['galaxy']), ascii=True, dynamic_ncols=True):

            dm_arr.append(gal.properties['dm_mass'])

            # loop over all properties

            for i, key in enumerate(true_keys):
                
                # get the pixel sum for stellar masses in an ellipse of semi-major axis of 50 pixels == 2 Rhalf for fiducial resolution of 256 pixels
                prop_true, _ = image_tools.get_pixel_sum(iso_dict_list[idx],key,r_half_frac*img_res)
                true_prop_arr[key].append(prop_true)

                prop_true = image_tools._get_image_sum(gal, key) #this can be accessed also by survey['total_star_mass']
                true_prop_arr_all[key].append(prop_true)

                # calculate the sum of all pixels within different apertures
                sum_arr = []
                pix_arr = []
                # loop over all apertures
                for j, sma in enumerate(iso_dict_list[idx]['stars_Masses'].sma):
                        
                    prop_true, npix = _get_property_value(iso_dict_list[idx], key, sma)
                    sum_arr.append(prop_true)
                    pix_arr.append(npix)

                true_prop_arr_rad[key].append(sum_arr)
                true_prop_arr_rad_npix[key].append(pix_arr)

                # calculate the radial gradient corresponding to this property
                # grad = fit_gradient(iso_dict_true, key)
                # true_prop_grad[key].append(grad)
                grad = fit_gradient_manually(iso_dict_list[idx][key].sma ,sum_arr, pix_arr)
                true_prop_grad[key].append(grad)

            #the whole block above should probably be a function so we can call it just twice, once for truth once for prediction...
            for i, key in enumerate(pred_keys):

                prop_pred, _ = image_tools.get_pixel_sum(iso_dict_list[idx],key,r_half_frac*img_res)
                pred_prop_arr[key].append(prop_pred)

                prop_pred = image_tools._get_image_sum(gal, key)
                pred_prop_arr_all[key].append(prop_pred)

                # calculate the sum of all pixels within different apertures
                sum_arr = []
                pix_arr = []
                for j, sma in enumerate(iso_dict_list[idx]['stars_Masses_pred'].sma):
                    #if j == 0:
                    #    prop_pred, npix = _get_property_value(iso_dict_pred, key, sma)
                    #else:
                    #    prop_pred, npix = _get_property_value(iso_dict_pred, key, sma, iso_dict_pred['stars_Masses'].sma[j-1])
                    prop_pred, npix = _get_property_value(iso_dict_list[idx], key, sma)
                    sum_arr.append(prop_pred)
                    pix_arr.append(npix)

                pred_prop_arr_rad[key].append(sum_arr)
                pred_prop_arr_rad_npix[key].append(pix_arr)

                # calculate the radial gradient corresponding to this property
                # grad_pred = fit_gradient(iso_dict_pred, key)
                # pred_prop_grad[key].append(grad_pred)
                grad_pred = fit_gradient_manually(iso_dict_list[idx][key].sma, sum_arr, pix_arr)
                pred_prop_grad[key].append(grad_pred)

        # save the data
        with util.open_hdf5(filename, "w") as h5file:
            for i, key in enumerate(true_keys):
                h5file.create_dataset(key, data=np.asarray(true_prop_arr[key]))
                h5file.create_dataset(key + '_all', data=np.asarray(true_prop_arr_all[key]))
                h5file.create_dataset(key + '_rad', data=np.asarray(true_prop_arr_rad[key]))
                h5file.create_dataset(key + '_rad_npix', data=np.asarray(true_prop_arr_rad_npix[key]))
                h5file.create_dataset(key + '_grad', data=np.asarray(true_prop_grad[key]))
            
            for i, key in enumerate(pred_keys):
                h5file.create_dataset(key, data=np.asarray(pred_prop_arr[key]))
                h5file.create_dataset(key + '_all', data=np.asarray(pred_prop_arr_all[key]))
                h5file.create_dataset(key + '_rad', data=np.asarray(pred_prop_arr_rad[key]))
                h5file.create_dataset(key + '_rad_npix', data=np.asarray(pred_prop_arr_rad_npix[key]))
                h5file.create_dataset(key + '_grad', data=np.asarray(pred_prop_grad[key]))

            h5file.create_dataset('dm_mass', data=np.asarray(dm_arr))

    else:
        print('Pre-calculated data file for the plot found.')

        true_prop_arr, pred_prop_arr, dm_arr = _get_predicted_vs_true(filename)

    return true_prop_arr, pred_prop_arr, dm_arr
