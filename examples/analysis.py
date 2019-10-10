"""

example analysis
================

An example of analysis flow using the picasso framework.
Here we examplify how to read in a survey, fit the images and the use
the resulting geometry of the galaxies to instantiate isophote objects.
From there we can calculate property totals in different radii bins and
perform gradient fits...

"""

import picasso as p
from picasso.analysis import image_tools
from picasso.analysis import util


s = p.load('./')

geom = image_tools.fit_image(survey)

iso_dict_list  = image_tools.get_isolists(survey,geom)

tmp = image_tools.get_pixel_sum(iso_dict_list,'stars_Masses',75)



def get_predicted_vs_true_data(survey, filename='predicted_vs_true.h5', plot=False, **kwargs):
    '''
    Preprocess the survey data to extract property totals within given radii. 
    Thus we have all the data needed to make plots of predicted vs true properties for all available properties.

    Input:

        survey: survey object

        filename: if given, the resulting arrays are stored in this file for future usage.

        plot: bool to decide if intermediate plots of the image fitting should be created.

    Return:

        numpy ndarray of the data ready to plot.
    
    '''

    # get dummy keys
    keys = survey['galaxy'][0].properties.keys()

    true_prop_arr = {key: [] for key in keys}  # fiducial sum at 2Rhalf
    pred_prop_arr = {key: [] for key in keys}

    true_prop_arr_all = {key: [] for key in keys}  # fiducial sum over all at pixel
    pred_prop_arr_all = {key: [] for key in keys}

    true_prop_arr_rad = {key: [] for key in keys}  # sum of pixel values at different radii
    pred_prop_arr_rad = {key: [] for key in keys}
    true_prop_arr_rad_npix = {key: [] for key in keys}  # number of pixels for the sums above
    pred_prop_arr_rad_npix = {key: [] for key in keys}
    true_prop_grad = {key: [] for key in keys}  # gradient values
    pred_prop_grad = {key: [] for key in keys}
    dm_arr = []

    if not os.path.isfile(filename):
        print('No pre-calculated data file for the plot found.')
        print('Creating data file.')

        for predicted_file in tqdm(files_predicted[:150], ascii=True, dynamic_ncols=True):

            iso_dict_true = get_isolists(path_true, predicted_file.split('/')[-1], plot=plot)
            iso_dict_pred = get_isolists(path_predicted, predicted_file.split('/')[-1], plot=plot)

            dm_arr.append(load_dm_mass(galnr, halo_file))

            # why did we do this weird if statement?
            for i, key in enumerate(keys):
                if key in list(true_prop_arr.keys()):

                    # calculate the sum of all pixels within 3 Rhalf
                    prop_true, _ = _get_property_value(iso_dict_true, key, 75)  # 50 corresponds to 2 Rhalf_3D
                    if prop_true == 0:
                        print('zero entry in file: ' + path_true + predicted_file.split('/')[-1])
                        print('retry calculating ' + key)
                        prop_true, _ = _get_property_value(iso_dict_true, key, 75)
                        if prop_true == 0:
                            print('Still 0, you need to retry manually!')

                    true_prop_arr[key].append(prop_true)
                    
                    prop_true = _get_pixel_sum(path_true, predicted_file.split('/')[-1], key)
                    true_prop_arr_all[key].append(prop_true)

                    # calculate the sum of all pixels within different apertures
                    sum_arr = []
                    pix_arr = []
                    for j, sma in enumerate(iso_dict_true['stars_Masses'].sma):
                        #if j == 0:
                        #    prop_true, npix = _get_property_value(iso_dict_true, key, sma)
                        #else:
                        #    prop_true, npix = _get_property_value(iso_dict_true, key, sma, iso_dict_true['stars_Masses'].sma[j-1])
                        prop_true, npix = _get_property_value(iso_dict_true, key, sma)

                        sum_arr.append(prop_true)
                        pix_arr.append(npix)

                    true_prop_arr_rad[key].append(sum_arr)
                    true_prop_arr_rad_npix[key].append(pix_arr)

                    # calculate the radial gradient corresponding to this property
                    #grad = fit_gradient(iso_dict_true, key)
                    #true_prop_grad[key].append(grad)
                    grad = fit_gradient_manually(iso_dict_true[key].sma ,sum_arr, pix_arr)
                    true_prop_grad[key].append(grad)

            #the whole block above should probably be a function so we can call it just twice, once for truth once for prediction...
            for i, key in enumerate(keys):
                if key in list(pred_prop_arr.keys()):

                    prop_pred, _ = _get_property_value(iso_dict_pred, key, 75)  # 50 corresponds to 2 Rhalf_3D

                    pred_prop_arr[key].append(prop_pred)

                    prop_pred = _get_pixel_sum(path_predicted, predicted_file.split('/')[-1], key)
                    pred_prop_arr_all[key].append(prop_pred)

                    # calculate the sum of all pixels within different apertures
                    sum_arr = []
                    pix_arr = []
                    for j, sma in enumerate(iso_dict_pred['stars_Masses'].sma):
                        #if j == 0:
                        #    prop_pred, npix = _get_property_value(iso_dict_pred, key, sma)
                        #else:
                        #    prop_pred, npix = _get_property_value(iso_dict_pred, key, sma, iso_dict_pred['stars_Masses'].sma[j-1])
                        prop_pred, npix = _get_property_value(iso_dict_pred, key, sma)

                        sum_arr.append(prop_pred)
                        pix_arr.append(npix)

                    pred_prop_arr_rad[key].append(sum_arr)
                    pred_prop_arr_rad_npix[key].append(pix_arr)

                    # calculate the radial gradient corresponding to this property
                    #grad_pred = fit_gradient(iso_dict_pred, key)
                    #pred_prop_grad[key].append(grad_pred)
                    grad_pred = fit_gradient_manually(iso_dict_pred[key].sma, sum_arr, pix_arr)
                    pred_prop_grad[key].append(grad_pred)

        # save the data
        with open_hdf5(filename, "w") as h5file:
            for i, key in enumerate(keys):
                h5file.create_dataset(key + '_true', data=np.asarray(true_prop_arr[key]))
                h5file.create_dataset(key + '_pred', data=np.asarray(pred_prop_arr[key]))
                h5file.create_dataset(key + '_all_true', data=np.asarray(true_prop_arr_all[key]))
                h5file.create_dataset(key + '_all_pred', data=np.asarray(pred_prop_arr_all[key]))
                h5file.create_dataset(key + '_rad_true', data=np.asarray(true_prop_arr_rad[key]))
                h5file.create_dataset(key + '_rad_pred', data=np.asarray(pred_prop_arr_rad[key]))
                h5file.create_dataset(key + '_rad_npix_true', data=np.asarray(true_prop_arr_rad_npix[key]))
                h5file.create_dataset(key + '_rad_npix_pred', data=np.asarray(pred_prop_arr_rad_npix[key]))
                h5file.create_dataset(key + '_grad_true', data=np.asarray(true_prop_grad[key]))
                h5file.create_dataset(key + '_grad_pred', data=np.asarray(pred_prop_grad[key]))

            h5file.create_dataset('dm_mass', data=np.asarray(dm_arr))

    else:
        print('Pre-calculated data file for the plot found.')

        with open_hdf5(filename, "r") as h5file:
            keys = list(h5file.keys())

            for i, key in enumerate(keys):
                if key[-5:] == '_pred':
                    true_prop_arr[key[:-5]] = h5file[key[:-5] + '_true'][()]
                    pred_prop_arr[key[:-5]] = h5file[key[:-5] + '_pred'][()]

                    true_prop_arr[key[:-5]] = h5file[key[:-5] + '_true'][()]
                    pred_prop_arr[key[:-5]] = h5file[key[:-5] + '_pred'][()]
                    true_prop_arr[key[:-5]] = h5file[key[:-5] + '_true'][()]
                    pred_prop_arr[key[:-5]] = h5file[key[:-5] + '_pred'][()]
                    true_prop_arr[key[:-5]] = h5file[key[:-5] + '_true'][()]
                    pred_prop_arr[key[:-5]] = h5file[key[:-5] + '_pred'][()]

            dm_arr = h5file['dm_mass'][()]

    return true_prop_arr, pred_prop_arr, dm_arr