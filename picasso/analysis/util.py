"""

utility functions
=================

A set of classes and functions for analysing galaxy images.

"""

import numpy as np


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
