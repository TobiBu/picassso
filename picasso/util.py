from time import sleep
import os
import h5py
import numpy as np

# bad move but simplest solution...
from astropy.cosmology import WMAP7 as cosmo

# this file contains utility functions used to load data

def congrid(a, newdims, centre=False, minusone=False):
    ''' Slimmed down version of congrid as originally obtained from:
        http://wiki.scipy.org/Cookbook/Rebinning
    '''

    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []



    for i in range( ndims ):
        base = np.arange( newdims[i] )
        dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
    # specify old dims
    olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

    # first interpolation - for ndims = any
    mint = scipy.interpolate.interp1d( olddims[-1], a, kind='linear', bounds_error=False, fill_value=0.0 )
    newa = mint( dimlist[-1] )

    trorder = [ndims - 1] + list(range( ndims - 1))
    for i in range( ndims - 2, -1, -1 ):
        newa = newa.transpose( trorder )

        mint = scipy.interpolate.interp1d( olddims[i], newa, kind='linear', bounds_error=False, fill_value=0.0 )
        newa = mint( dimlist[i] )

    if ndims > 1:
        # need one more transpose to return to original dimensions
        newa = newa.transpose( trorder )

    return newa

def open_hdf5(filename, *args, **kwargs):
    # Function for delayed hdf5 file opening, when opend by another process
    waits = 0
    while waits < 10:
        try:
            hdf5_file = h5py.File(filename, *args, **kwargs)
            break  # Success!
        except OSError:
            if not os.path.exists(filename):
                raise KeyError(f"file {filename} does not exist")
            sleep(5)  # Wait a bit
            waits += 1

    else:
        raise NameError("Wait limit exceeded for", filename)
    return hdf5_file


def to_logspace(data):
    data[data <= 0] = 0
    data += 1e-14
    np.log10(data, out=data)
    return data


def from_logspace(data):
    np.power(10, data, out=data)
    data -= 1e-14
    data[data <= 0] = 0
    return data

def scale_to_physical_units(x, psize):
    '''Account for spacial scales.'''
    #ensure that we do not modify x itself.
    xx = x * psize * psize

    return xx

def load_halo_properties(halo_idx, camera, scale=None, smoothing=0., path='./data/halodata/', log=False):
    target_file_name = path + f'halo_{halo_idx}_camera_{camera}_physical_data.h5'

    # offer alternative camera
    with open_hdf5(target_file_name, "r") as h5file:
        keys = list(h5file.keys())
            
        if "metadata" in keys:
            keys.remove("metadata")

        if scale is None:
            gt_stack = [h5file[k].value for k in h5file.keys()]
        else:
            # rebin image to the same scale as data
            gt_stack = [congrid(h5file[k].value, scale) for k in h5file.keys()]

        data = np.stack(gt_stack)

        if smoothing > 0.:
            gt_stack = [gaussian_filter(x, sigma=smoothing, mode='constant') for x in gt_stack]

        if log:
            data = to_logspace(data)

        psize = None
        if "psize" in h5file[keys[0]].attrs.keys():
            psize = h5file[keys[0]].attrs["psize"]

        return data.transpose(0, 2, 1), keys, psize


def load_dm_mass(galnr, filename='illustris_fof_props.h5'):

    with open_hdf5(filename, "r") as h5file:

        key = galnr + '_dm_mass'
        dm = h5file[key].value

    dm *= 1e10 / (cosmo.H(0) / 100).value

    return dm

def load_mass(galnr, filename='illustris_fof_props.h5'):

    with open_hdf5(filename, "r") as h5file:

        key = galnr + '_mass'
        dm = h5file[key].value

    dm *= 1e10 / (cosmo.H(0) / 100).value

    return dm
