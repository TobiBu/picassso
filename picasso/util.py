from time import sleep
import os
import h5py
import numpy as np

# bad move but simplest solution...
from astropy.cosmology import WMAP7 as cosmo

# this file contains utility functions used to load data


def score(true_prop_arr, pred_prop_arr, score_type):
    """
    defines a wrapper around a scoring function. This can be used to calculate the score of the predicition by passing an array containing
    either pixel values of a single galaxy for which the score will be evaluated or an array containing ensemble values, 
    like e.g. the total stellar mass of the galaxy. score_type the specifies what kind of scoring funtion should be used, e.g. simple standard deviation,
    or standard error estimates or self-defined measures.
    """

    if score_type == 'log_diff':
        score = np.log10(true_prop_arr) - np.log10(pred_prop_arr)
    elif score_type == 'diff':
        score = true_prop_arr - pred_prop_arr
        

    return score

    
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

def _load_halo_properties(h5file, scale=None, smoothing=0., log=False):

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

def gcf(a, b):
    while b > 0:
        a, b = b, a % b
    return a


def lcm(a, b):
    return (a * b) // gcf(a, b)


def intersect_slices(s1, s2, array_length=None):
    """Given two python slices s1 and s2, return a new slice which
    will extract the data of an array d which is in both d[s1] and
    d[s2].

    Note that it may not be possible to do this without information on
    the length of the array referred to, hence all slices with
    end-relative indexes are first converted into begin-relative
    indexes. This means that the slice returned may be specific to
    the length specified."""

    assert array_length is not None or \
        (s1.start >= 0 and s2.start >= 0 and s1.stop >= 0 and s2.start >= 0)

    s1_start = s1.start
    s2_start = s2.start
    s1_stop = s1.stop
    s2_stop = s2.stop
    s1_step = s1.step
    s2_step = s2.step

    if s1_step == None:
        s1_step = 1
    if s2_step == None:
        s2_step = 1

    assert s1_step > 0 and s2_step > 0

    if s1_start < 0:
        s1_start = array_length + s1_start
    if s1_start < 0:
        return slice(0, 0)

    if s2_start < 0:
        s2_start = array_length + s2_start
    if s2_start < 0:
        return slice(0, 0)

    if s1_stop < 0:
        s1_stop = array_length + s1_stop
    if s1_stop < 0:
        return slice(0, 0)

    if s2_stop < 0:
        s2_stop = array_length + s2_stop
    if s2_stop < 0:
        return slice(0, 0)

    step = lcm(s1_step, s2_step)

    start = max(s1_start, s2_start)
    stop = min(s1_stop, s2_stop)

    if stop <= start:
        return slice(0, 0)

    s1_offset = start - s1_start
    s2_offset = start - s2_start
    s1_offset_x = int(s1_offset)
    s2_offset_x = int(s2_offset)

    if s1_step == s2_step and s1_offset % s1_step != s2_offset % s1_step:
        # slices are mutually exclusive
        return slice(0, 0)

    # There is surely a more efficient way to do the following, but
    # it eludes me for the moment
    while s1_offset % s1_step != 0 or s2_offset % s2_step != 0:
        start += 1
        s1_offset += 1
        s2_offset += 1
        if s1_offset % s1_step == s1_offset_x % s1_step and s2_offset % s2_step == s2_offset_x % s2_step:
            # slices are mutually exclusive
            return slice(0, 0)

    if step == 1:
        step = None

    return slice(start, stop, step)


def relative_slice(s_relative_to, s):
    """Given a slice s, return a slice s_prime with the property that
    array[s_relative_to][s_prime] == array[s]. Clearly this will
    not be possible for arbitrarily chosen s_relative_to and s, but
    it should be possible for s=intersect_slices(s_relative_to, s_any)
    which is the use case envisioned here (and used by SubSim).
    This code currently does not work with end-relative (i.e. negative)
    start or stop positions."""

    assert (s_relative_to.start >= 0 and s.start >= 0 and s.stop >= 0)

    if s.start == s.stop:
        return slice(0, 0, None)

    s_relative_to_step = s_relative_to.step if s_relative_to.step is not None else 1
    s_step = s.step if s.step is not None else 1

    if (s.start - s_relative_to.start) % s_relative_to_step != 0:
        raise ValueError("Incompatible slices")
    if s_step % s_relative_to_step != 0:
        raise ValueError("Incompatible slices")

    start = (s.start - s_relative_to.start) // s_relative_to_step
    step = s_step // s_relative_to_step
    stop = start + \
        (s_relative_to_step - 1 + s.stop - s.start) // s_relative_to_step

    if step == 1:
        step = None

    return slice(start, stop, step)


def chained_slice(s1, s2):
    """Return a slice s3 with the property that
    ar[s1][s2] == ar[s3] """

    assert (s1.start >= 0 and s2.start >= 0 and s1.stop >= 0 and s2.stop >= 0)
    s1_start = s1.start or 0
    s2_start = s2.start or 0
    s1_step = s1.step or 1
    s2_step = s2.step or 1

    start = s1_start + s2_start * s1_step
    step = s1_step * s2_step
    if s1.stop is None and s2.stop is None:
        stop = None
    elif s1.stop is None:
        stop = start + step * (s2.stop - s2_start) // s2_step
    elif s2.stop is None:
        stop = s1.stop
    else:
        stop_s2 = start + step * (s2.stop - s2_start) // s2_step
        stop_s1 = s1.stop
        stop = stop_s2 if stop_s2 < stop_s1 else stop_s1
    return slice(start, stop, step)


def index_before_slice(s, index):
    """Return an index array new_index with the property that, for a
    slice s (start, stop and step all positive), ar[s][index] ==
    ar[new_index]."""

    start = s.start or 0
    step = s.step or 1

    assert start >= 0
    assert step >= 0
    assert s.stop is None or s.stop >= 0

    new_index = start + index * step
    if s.stop is not None:
        new_index = new_index[np.where(new_index < s.stop)]

    return new_index


def concatenate_indexing(i1, i2):
    """Given either a numpy array or slice for both i1 and i2,
    return either a numpy array or slice i3 with the property that

    ar[i3] == ar[i1][i2].

    As a convenience, if i2 is None, i1 is returned
    """
    if isinstance(i1, tuple) and len(i1) == 1:
        i1 = i1[0]
    if isinstance(i2, tuple) and len(i2) == 1:
        i2 = i2[0]

    if i2 is None:
        return i1
    if isinstance(i1, slice) and isinstance(i2, slice):
        return chained_slice(i1, i2)
    elif isinstance(i1, slice) and isinstance(i2, (np.ndarray, list)):
        return index_before_slice(i1, i2)
    elif isinstance(i1, (np.ndarray, list)) and isinstance(i2, (slice, np.ndarray, slice)):
        return np.asarray(i1)[i2]
    else:
        raise TypeError("Don't know how to chain these index types")


def indexing_length(sl_or_ar):
    """Given either an array or slice, return len(ar[sl_or_ar]) for any
    array ar which is large enough that the slice does not overrun it."""

    if isinstance(sl_or_ar, slice):
        step = sl_or_ar.step or 1
        diff = (sl_or_ar.stop - sl_or_ar.start)
        return diff // step + (diff % step > 0)
    else:
        return len(sl_or_ar)


def arrays_are_same(a1, a2):
    """Returns True if a1 and a2 are numpy views pointing to the exact
    same underlying data; False otherwise."""
    try:
        return a1.__array_interface__['data'] == a2.__array_interface__['data'] \
            and a1.strides == a2.strides
    except AttributeError:
        return False


def set_array_if_not_same(a_store, a_in, index=None):
    """This routine checks whether a_store and a_in ultimately point to the
    same buffer; if not, the contents of a_in are copied into a_store."""
    if index is None:
        index = slice(None)
    if not arrays_are_same(a_store[index], a_in):
        a_store[index] = a_in


def index_of_first(array, find):
    """Returns the index to the first element in array
    which satisfies array[index]>=find. The array must
    be sorted in ascending order."""

    if len(array) == 0:
        return 0

    left = 0
    right = len(array) - 1

    if array[left] >= find:
        return 0

    if array[right] < find:
        return len(array)

    while right - left > 1:
        mid = (left + right) // 2
        if array[mid] >= find:
            right = mid
        else:
            left = mid

    return right


def equipartition(ar, nbins, min=None, max=None):
    """

    Given an array ar, return nbins+1 monotonically increasing bin
    edges such that the number of items in each bin is approximately
    equal.

    """

    a_s = np.sort(ar)

    if max is not None:
        a_s = a_s[a_s <= max]
    if min is not None:
        a_s = a_s[a_s > min]

    return a_s[np.array(np.linspace(0, len(a_s) - 1, nbins + 1), dtype='int')]


def bisect(left, right, f, epsilon=None, eta=0, verbose=False, niter_max=200):
    """

    Finds the value x such that f(x)=0 for a monotonically increasing
    function f, using a binary search.

    The search stops when either the bounding domain is smaller than
    epsilon (by default 10^-7 times the original region) OR a value
    f(x) is found such that |f(x)|<eta (by default eta=0, so this
    criterion is never satisfied).

    """

    if epsilon is None:
        epsilon = (right - left) * 1.e-7

    logger.info("Entering bisection search algorithm")
    for i in xrange(niter_max):

        if (right - left) < epsilon:
            return (right + left) / 2

        mid = (left + right) / 2
        z = f(mid)

        logger.info("%f %f %f %f" % (left, mid, right, z))

        if (abs(z) < eta):
            return mid
        elif(z < 0):
            left = mid
        else:
            right = mid

    raise ValueError("Bisection algorithm did not converge")


class ExecutionControl(object):

    def __init__(self):
        self.count = 0
        self.on_exit = None

    def __enter__(self):
        self.count += 1

    def __exit__(self, *excp):
        self.count -= 1
        assert self.count >= 0
        if self.count == 0 and self.on_exit is not None:
            self.on_exit()

    def __nonzero__(self):
        return self.count > 0

    def __repr__(self):
        return "<ExecutionControl: %s>" % ('True' if self.count > 0 else 'False')

