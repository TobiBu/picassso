
"""
survey
========

This module implements the  :class:`~picasso.survey.Survey` class which manages and stores Survey data.
It also implements the :class:`~picasso.survey.SubSurvey` class (and relatives) which
represent different views of an existing :class:`~picasso.survey.Survey`.

"""

from .. import array
from .. import family
from .. import util
from .. import filt
from .. import configuration
from ..configuration import config
from .. import surveydict

import numpy as np
import weakref
import threading
import gc # garbage collector
import re

class Survey(object):

    """The class for managing Survey data.

    For most purposes, Surveys should be initialized through
    :func:`~picasso.load` or :func:`~picasso.new`.

    For a basic tutorial explaining how to load a file as a Survey
    see :doc:`tutorials/data_access`.

    *Getting arrays or suburveys*

    Once a :class:`Survey` object ``f`` is instantiated, it can
    be used in various ways. The most common operation is to
    access something with the code ``f[x]``.  Depending on the
    type of ``x``, various behaviours result:

    - If ``x`` is a string, the array named by ``x`` is returned. If
      no such array exists, the framework attempts to load or
      derive an array of that name (in that order). If this is
      unsuccessful, a `KeyError` is raised.

    - If ``x`` is a python `slice` (e.g. ``f[5:100:3]``) or an array of
      integers (e.g. ``f[[1,5,100,200]]``) a subsurvey containing only the
      mentioned galaxies is returned.

      See :doc:`tutorials/data_access` for more information.

    - If ``x`` is a numpy array of booleans, it is interpreted as a mask and
      a subsurvey containing only those galaxies for which x[i] is True.
      This means that f[condition] is a shortcut for f[np.where(condition)].

    - If ``x`` is a :class:`picasso.filt.Filter` object, a subsurvey
      containing only the galaxies which pass the filter condition
      is returned.

      See :doc:`tutorials/data_access` for more information.

    - If ``x`` is a :class:`picasso.family.Family` object, a subsurvey
      containing only the galaxies in that family is returned. In practice
      for most code it is more convenient to write e.g. ``f.true`` in place of
      the equivalent syntax f[picasso.family.true].

    *Getting metadata*

    The property `path` gives the path to a survey.

    There is also a `properties` dictionary which
    contains further metadata about the snapshot. See :ref:`subsurveys`.
    """

    _derived_quantity_registry = {}

    _decorator_registry = {}

    _loadable_keys_registry = {}

     # The following will be objects common to a SimSnap and all its SubSnaps
    _inherited = ["properties", "immediate_mode"]
    #["lazy_off", "lazy_derive_off", "lazy_load_off", "_derived_array_names", "_family_derived_array_names"]

    _split_arrays = {'mass': ('mass_s', 'mass_g', 'mass_d'),
                     'metals': ('metals_s', 'metals_g')}

    @classmethod
    def _array_name_1D_to_ND(self, name):
        """Map a 1D array name to a corresponding 3D array name, or return None
        if no such mapping is possible.

        e.g. 'vy' -> 'vel'; 'acc_z' -> 'acc'; 'mass' -> None"""
        for k, v in self._split_arrays.iteritems():
            if name in v:
                return k

        generic_match = re.findall("^(.+)_[sgd]$", name)
        if len(generic_match) is 1 and generic_match[0] not in self._split_arrays:
            return generic_match[0]

        return None

    @classmethod
    def _array_name_ND_to_1D(self, array_name):
        """Give the 3D array names derived from a 3D array.

        This routine makes no attempt to establish whether the array
        name passed in should indeed be a 3D array. It just returns
        the 1D slice names on the assumption that it is. This is an
        important distinction between this procedure and the reverse
        mapping as implemented by _array_name_1D_to_ND."""

        if array_name in self._split_arrays:
            array_name_1D = self._split_arrays[array_name]
        else:
            array_name_1D = [array_name + "_" + i for i in ['s', 'g', 'd']]

        return array_name_1D

    def _array_name_implies_ND_slice(self, array_name):
        """Returns True if, at best guess, the array name corresponds to a 1D slice
        of a ND array, on the basis of names alone.

        This routine first looks at special cases (pos -> x,y,z for example),
        then looks for generic names such as acc_x - however this would only be
        considered a "match" for a ND subslice if 'acc' is in loadable_keys().
        """
        for v in self._split_arrays.itervalues():
            if array_name in v:
              return True

        generic_match = re.findall("^(.+)_[sgd]$", array_name)
        loadable_keys = self.loadable_keys()
        keys = self.keys()
        if len(generic_match) is 1 and generic_match[0] not in self._split_arrays:
            return generic_match[0] in loadable_keys or generic_match[0] in keys
        return False


    def __init__(self):
        """Initialize an empty, zero-length Survey.

        For most purposes Surveys should instead be initialized through
       :func:`~picasso.load` or :func:`~picasso.new`.
       """

        self._arrays = {}
        self._num_galaxies = 0
        self._family_slice = {}
        self._family_arrays = {}
        self._derived_array_names = []
        self._family_derived_array_names = {}
        for i in family._registry:
            self._family_derived_array_names[i] = []

        self._immediate_cache_lock = threading.RLock()

        self.immediate_mode = util.ExecutionControl()
        # use 'with immediate_mode: ' to always return actual numpy arrays, rather
        # than IndexedSubArrays which point to sub-parts of numpy arrays
        self.immediate_mode.on_exit = lambda: self._clear_immediate_mode()
        
        self._unifamily = None
        
        # If True, when new arrays are created they are in shared memory by
        # default
        self._shared_arrays = False

        self.properties = surveydict.SurveyDict({})

    ############################################
    # THE BASICS: SIMPLE INFORMATION
    ############################################

    @property
    def path(self):
        return self._path

    def __len__(self):
        return self._num_galaxies

    def __repr__(self):
        if self._path != "":
            return "<Survey \"" + self._path + "\" len=" + str(len(self)) + ">"
        else:
            return "<Survey len=" + str(len(self)) + ">"

    def families(self):
        """Return the survey families which have representitives in this Survey.
        This might simply be all the galaxies in the survey or something like the
        subsamples for training, prediction or validation.
        """
        out = []
        start = {}
        for fam in family._registry:
            sl = self._get_family_slice(fam)
            if sl.start != sl.stop:
                out.append(fam)
                start[fam] = (sl.start)
        out.sort(key=start.__getitem__)
        return out

    ############################################
    # THE BASICS: GETTING AND SETTING
    ############################################

    def __getitem__(self, i):
        """Return either a specific array or a subview of this simulation. See
        the class documentation (:class:`Survey`) for more information."""

        if isinstance(i, str):
            return self._get_array_with_lazy_actions(i)
        elif isinstance(i, slice):
            return SubSurvey(self, i)
        elif isinstance(i, family.Family):
            return FamilySubSurvey(self, i)
        elif isinstance(i, np.ndarray) and np.issubdtype(np.bool, i.dtype):
            return self._get_subsurvey_from_mask_array(i)
        elif isinstance(i, (list, tuple, np.ndarray, filt.Filter)):
            return IndexedSubSurvey(self, i)
        elif isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):
            return IndexedSubSurvey(self, (i,))

        raise TypeError


    def __setitem__(self, name, item):
        """Set the contents of an array in this survey"""
        if self.is_derived_array(name):
            raise RuntimeError("Derived array is not writable")

        if isinstance(name, tuple) or isinstance(name, list):
            index = name[1]
            name = name[0]
        else:
            index = None

        self._assert_not_family_array(name)

        if isinstance(item, array.SurveyArray):
            ax = item
        else:
            ax = np.asanyarray(item).view(array.SurveyArray)

        if name not in self.keys():
            # Array needs to be created. We do this through the
            # private _create_array method, so that if we are operating
            # within a particle-specific subview we automatically create
            # a particle-specific array
            try:
                ndim = len(ax[0])
            except TypeError:
                ndim = 1
            except IndexError:
                ndim = ax.shape[-1] if len(ax.shape) > 1 else 1

            # The dtype will be the same as an existing family array if
            # one exists, or the dtype of the source array we are copying
            dtype = self._get_preferred_dtype(name)
            if dtype is None:
                dtype = getattr(item, 'dtype', None)

            self._create_array(name, ndim, dtype=dtype)

        # Copy in contents if the contents isn't actually pointing to
        # the same data (which will be the case following operations like
        # += etc, since these call __setitem__).
        self._set_array(name, ax, index)

    def __delitem__(self, name):
        if name in self._family_arrays:
            # mustn't have simulation-level array of this name
            assert name not in self._arrays
            del self._family_arrays[name]

            for v in self._family_derived_array_names.itervalues():
                if name in v:
                    del v[v.index(name)]

        else:
            del self._arrays[name]
            if name in self._derived_array_names:
                del self._derived_array_names[
                    self._derived_array_names.index(name)]


    def _get_subsurvey_from_mask_array(self,mask_array):
        if len(mask_array.shape) > 1 or mask_array.shape[0] > len(self):
            raise ValueError("Incorrect shape for masking array")
        else:
            return self[np.where(mask_array)]

    def _get_array_with_lazy_actions(self, name):
        """This allows to load arrays from disk if pre-calculated or to
        derive them from data products if present. E.g. if an array of
        total stellar mass is requested and galaxy images are present,
        a sum over all pixels will easily create this array. On the other hand,
        this operation might have been performed before and the result was stored
        on disk, then reading the array from disk might be the action of choice."""

        if name in self.keys():
            # array was calculated before
            return self._get_array(name)
        
        self.__resolve_obscuring_family_array(name)

        
        self.__load_if_required(name)
        
        self.__derive_if_required(name)

        return self._get_array(name)


    def __load_if_required(self, name):
        if name not in self.keys():
            try:
                self.__load_array_and_perform_postprocessing(name)
            except IOError:
                pass

    def __derive_if_required(self, name):
        if name not in self.keys():
            self._derive_array(name)

    def __resolve_obscuring_family_array(self, name):
        if name in self.family_keys():
            self.__remove_family_array_if_derived(name)

        if name in self.family_keys():
            self.__load_remaining_families_if_loadable(name)

        if name in self.family_keys():
            in_fam, out_fam = self.__get_included_and_excluded_families_for_array(name)
            raise KeyError("""%r is a family-level array for %s. To use it over the whole simulation you need either to delete it first, or create it separately for %s.""" % (
                name, in_fam, out_fam))

    def __get_included_and_excluded_families_for_array(self,name):
        in_fam = []
        out_fam = []
        for x in self.families():
            if name in self[x]:
                in_fam.append(x)
            else:
                out_fam.append(x)

        return in_fam, out_fam

    def __remove_family_array_if_derived(self, name):
        if self.is_derived_array(name):
            del self.ancestor[name]


    def __load_remaining_families_if_loadable(self, name):
        in_fam, out_fam = self.__get_included_and_excluded_families_for_array(name)
        try:
            for fam in out_fam:
                self.__load_array_and_perform_postprocessing(name, fam=fam)
        except IOError:
            pass



    def __getattr__(self, name):
        """This function overrides the behaviour of f.X where f is a Survey object.

        It serves two purposes; first, it provides the family-handling behaviour
        which makes f.truth equivalent to f[picasso.family.truth]. """

        try:
            return self[family.get_family(name)]
        except ValueError:
            pass

        raise AttributeError("%r object has no attribute %r" % (
            type(self).__name__, name))

    def __setattr__(self, name, val):
        """This function overrides the behaviour of setting f.X where f is a Survey object.

        It serves two purposes; first it prevents overwriting of family names (so you can't
        write to, for instance, f.truth)."""

        if name in family.family_names():
            raise AttributeError("Cannot assign family name " + name)
        else:
            return object.__setattr__(self, name, val)

    ############################################
    # DICTIONARY EMULATION FUNCTIONS
    ############################################
    def keys(self):
        """Return the directly accessible array names (in memory)"""
        return self._arrays.keys()

    def has_key(self, name):
        """Returns True if the array name is accessible (in memory)"""
        return name in self.keys()

    def values(self):
        """Returns a list of the actual arrays in memory"""
        x = []
        for k in self.keys():
            x.append(self[k])
        return x

    def items(self):
        """Returns a list of tuples describing the array
        names and their contents in memory"""
        x = []
        for k in self.keys():
            x.append((k, self[k]))
        return x

    def get(self, key, alternative=None):
        """Standard python get method, returns self[key] if
        key in self else alternative"""
        try:
            return self[key]
        except KeyError:
            return alternative

    def iterkeys(self):
        for k in self.keys():
            yield k

    __iter__ = iterkeys

    def itervalues(self):
        for k in self:
            yield self[k]

    def iteritems(self):
        for k in self:
            yield (k, self[k])

    ############################################
    # DICTIONARY-LIKE FUNCTIONS
    # (not in the normal interface for dictionaries,
    # but serving similar purposes)
    ############################################

    def has_family_key(self, name):
        """Returns True if the array name is accessible (in memory) for at least one family"""
        return name in self.family_keys()

    def loadable_keys(self, fam=None):
        """Returns a list of arrays which can be lazy-loaded from
        an auxiliary file."""
        return []

    def derivable_keys(self):
        """Returns a list of arrays which can be lazy-evaluated."""
        res = []
        for cl in type(self).__mro__:
            if cl in self._derived_quantity_registry:
                res += self._derived_quantity_registry[cl].keys()
        return res

    def all_keys(self):
        """Returns a list of all arrays that can be either lazy-evaluated
        or lazy loaded from an auxiliary file."""
        return self.derivable_keys() + self.loadable_keys()

    def family_keys(self, fam=None):
        """Return list of arrays which are not accessible from this
        view, but can be accessed from family-specific sub-views.

        If *fam* is not None, only those keys applying to the specific
        family will be returned (equivalent to self.fam.keys())."""
        if fam is not None:
            return [x for x in self._family_arrays if fam in self._family_arrays[x]]
        else:
            return self._family_arrays.keys()

    ############################################
    # ANCESTRY FUNCTIONS
    ############################################

    def is_ancestor(self, other):
        """Returns true if other is a subview of self"""

        if other is self:
            return True
        elif hasattr(other, 'base'):
            return self.is_ancestor(other.base)
        else:
            return False

    def is_descendant(self, other):
        """Returns true if self is a subview of other"""
        return other.is_ancestor(self)

    @property
    def ancestor(self):
        """The original Survey from which this view is derived (potentially self)"""
        if hasattr(self, 'base'):
            return self.base.ancestor
        else:
            return self

    def get_index_list(self, relative_to, of_galaxies=None):
        """Get a list specifying the index of the galaxies in this view relative
        to the ancestor *relative_to*, such that relative_to[get_index_list(relative_to)]==self."""

        # Implementation for base snapshot

        if self is not relative_to:
            raise RuntimeError("Not a descendant of the specified survey")
        if of_galaxies is None:
            of_galaxies = np.arange(len(self))

        return of_galaxies

    ############################################
    # SET-LIKE OPERATIONS FOR SUBSurveyS
    ############################################
    def intersect(self, other, op=np.intersect1d):
        """Returns the set intersection of this survey view with another view
        of the same survey"""

        anc = self.ancestor
        if not anc.is_ancestor(other):
            raise RuntimeError("Parentage is not suitable")

        a = self.get_index_list(anc)
        b = other.get_index_list(anc)
        return anc[op(a, b)]

    def union(self, other):
        """Returns the set union of this survey view with another view
        of the same survey"""

        return self.intersect(other, op=np.union1d)

    def setdiff(self, other):
        """Returns the set difference of this survey view with another view
        of the same survey"""

        return self.intersect(other, op=np.setdiff1d)


    def galaxies(self, *args, **kwargs):
        """Tries to instantiate a galaxy catalogue object for the given
        snapshot, using the first available method (as defined in the
        configuration files)."""

        from .. import galaxy

        for c in config['survey-class-priority']:
        # we keep this weird structure from pynbody here since in future we will deal with different surveys 
        # which might have different data specifics. E.g. the illustris survey has simple hdf5 files for each 
        # galaxy while SDSS MANGA has other data formats.

            try:
                if c._can_load(self, *args, **kwargs):
                    return c(self, *args, **kwargs)
            except TypeError:
                pass

        for c in config['survey-class-priority']:
            # this stays here because we might want to use it in order to download or create survey data from a list of galaxies
            # e.g. see Nik Arora's script
            try:
                if c._can_run(self, *args, **kwargs):
                    return c(self, *args, **kwargs)
            except TypeError:
                pass

        raise RuntimeError("No galaxy catalogue found for %r" % str(self))


    ############################################
    # HELPER FUNCTIONS FOR LAZY LOADING
    ############################################
    def _load_array(self, array_name, fam=None):
        """This function is called by the framework to load an array
        from disk and should be overloaded by child classes.

        If *fam* is not None, the array should be loaded only for the
        specified family.
        """
        raise IOError("No lazy-loading implemented")

    def __load_array_and_perform_postprocessing(self, array_name, fam=None):
        """Calls _load_array for the appropriate subclass, but also attempts to convert
        units of anything that gets loaded and automatically loads the whole ND array
        if this is a subview of an ND array"""
        array_name = self._array_name_1D_to_ND(array_name) or array_name

        # keep a record of every array in existence before load (in case it
        # triggers loading more than we expected, e.g. coupled pos/vel fields
        # etc)
        anc = self.ancestor

        pre_keys = set(anc.keys())

        # the following function builds a dictionary mapping families to a set of the
        # named arrays defined for them.
        fk = lambda: dict([(fami, set([k for k in anc._family_arrays.keys() if fami in anc._family_arrays[k]]))
                           for fami in family._registry])
        pre_fam_keys = fk()

        if fam is not None:
            self._load_array(array_name, fam)
        else:
            try:
                self._load_array(array_name, fam)
            except IOError:
                for fam_x in self.families():
                    self._load_array(array_name, fam_x)

        # Find out what was loaded
        new_keys = set(anc.keys()) - pre_keys
        new_fam_keys = fk()
        for fami in new_fam_keys:
            new_fam_keys[fami] = new_fam_keys[fami] - pre_fam_keys[fami]


    ############################################
    # WRITING FUNCTIONS
    ############################################
    def write(self, fmt=None, filename=None, **kwargs):
        if filename is None and "<" in self.filename:
            raise RuntimeError(
                'Cannot infer a filename; please provide one (use obj.write(filename="filename"))')

        if fmt is None:
            if not hasattr(self, "_write"):
                raise RuntimeError(
                    'Cannot infer a file format; please provide one (e.g. use obj.write(filename="filename", fmt=hdf5)')

            self._write(self, filename, **kwargs)
        else:
            fmt._write(self, filename, **kwargs)

    def write_array(self, array_name, fam=None, overwrite=False, **kwargs):
        """
        Write out the array with the specified name.

        Some of the functionality is available via the
        :func:`picasso.array.SurveyArray.write` method, which calls the
        present function with appropriate arguments.

        **Input**

        *array_name* - the name of the array to write

        **Optional Keywords**

        *fam* (None) - Write out only one family; or provide a list to
         write out a set of families.
         """

        # Determine whether this is a write or an update
        if fam is None:
            fam = self.families()

        # It's an update if we're not fully replacing the file on
        # disk, i.e. there exists a family f in self.families() but
        # not in fam for which array_name is loadable
        is_update = any([array_name in self[
                        f].loadable_keys() and f not in fam for f in self.families()])

        if not hasattr(self, "_write_array"):
            raise IOError(
                "The underlying file format class does not support writing individual arrays back to disk")

        if is_update and not hasattr(self, "_update_array"):
            raise IOError(
                "The underlying file format class does not support updating arrays on disk")

        # It's an overwrite if we're writing over something loadable
        is_overwriting = any([array_name in self[
                             f].loadable_keys() for f in fam])

        if is_overwriting and not overwrite:
            # User didn't specifically say overwriting is OK
            raise IOError(
                "This operation would overwrite existing data on disk. Call again setting overwrite=True if you want to enable this behaviour.")

        if is_update:
            self._update_array(array_name, fam=fam, **kwargs)
        else:
            self._write_array(self, array_name, fam=fam, **kwargs)

    ############################################
    # LOW-LEVEL ARRAY MANIPULATION
    ############################################
    def _get_preferred_dtype(self, array_name):
        """Return the 'preferred' numpy datatype for a named array.

        This is mainly useful when creating family arrays for new families, to be
        sure the datatype chosen matches"""

        if hasattr(self, 'base'):
            return self.base._get_preferred_dtype(array_name)
        elif array_name in self.keys():
            return self[array_name].dtype
        elif array_name in self.family_keys():
            return self._family_arrays[array_name][self._family_arrays[array_name].keys()[0]].dtype
        else:
            return None

    def _create_array(self, array_name, ndim=1, dtype=None, zeros=True, derived=False, shared=None):
        """Create a single survey-level array of dimension len(self) x ndim, with
        a given numpy dtype.

        *kwargs*:

          - *ndim*: the number of dimensions for each galaxy
          - *dtype*: a numpy datatype for the new array
          - *zeros*: if True, zeros the array (which takes a bit of time); otherwise
            the array is uninitialized
          - *derived*: if True, this new array will be flagged as a derived array
            which makes it read-only
          - *shared*: if True, the array will be built on top of a shared-memory array
            to make it possible to access from another process
        """

        # Does this actually correspond to a slice into a 3D array?
        NDname = self._array_name_1D_to_ND(array_name)
        if NDname:
            self._create_array(
                NDname, ndim=3, dtype=dtype, zeros=zeros, derived=derived)
            return

        if ndim == 1:
            dims = self._num_galaxies
        elif isinstance(ndim,tuple):
            dims = (self._num_galaxies,) + ndim
        else:
            dims = (self._num_galaxies, ndim)
            print(dims)

        if shared is None:
            shared = self._shared_arrays

        new_array = array._array_factory(dims, dtype, zeros, shared)
        new_array._survey = weakref.ref(self)
        new_array._name = array_name
        new_array.family = None
        # new_array.set_default_units(quiet=True)
        self._arrays[array_name] = new_array

        if derived:
            if array_name not in self._derived_array_names:
                self._derived_array_names.append(array_name)

        if ndim == 3:
            array_name_1D = self._array_name_ND_to_1D(array_name)

            for i, a in enumerate(array_name_1D):
                self._arrays[a] = new_array[:, i]
                self._arrays[a]._name = a

    def _create_family_array(self, array_name, family, ndim=1, dtype=None, derived=False, shared=None):
        """Create a single array of dimension len(self.<family.name>) x ndim,
        with a given numpy dtype, belonging to the specified family. For arguments
        other than *family*, see the documentation for :func:`~picasso.survey.Survey._create_array`.

        Warning: Do not assume that the family array will be available after
        calling this funciton, because it might be a 'completion' of existing
        family arrays, at which point the routine will actually be creating
        a survey-level array, e.g.

        survey._create_family_array('bla', truth)
        'bla' in survey.family_keys() # -> True
        'bla' in survey.keys() # -> False
        survey._create_family_array('bla', pred)
        'bla' in survey.keys() # -> True
        'bla' in survey.family_keys() # -> False

        survey[pred]['bla'] *is* guaranteed to exist, however, it just might
        be a view on a survey-length array.

        """

        NDname = self._array_name_1D_to_ND(array_name)
        if NDname:
            self._create_family_array(
                NDname, family, ndim=3, dtype=dtype, derived=derived)
            return

        self_families = self.families()

        if len(self_families) == 1 and family in self_families:
            # If the file has only one family, just go ahead and create
            # a normal array
            self._create_array(
                array_name, ndim=ndim, dtype=dtype, derived=derived)
            return

        if ndim == 1:
            dims = self[family]._num_galaxies
        else:
            dims = (self[family]._num_galaxies, ndim)

        # Determine what families already have an array of this name
        fams = []
        dtx = None
        try:
            fams = self._family_arrays[array_name].keys()
            dtx = self._family_arrays[array_name][fams[0]].dtype
        except KeyError:
            pass

        fams.append(family)

        if dtype is not None and dtx is not None and dtype != dtx:

            # We insist on the data types being the same for, e.g. survey.truth['my_prop'] and surbey.pred['my_prop']
            # This makes promotion to survey-level arrays possible.
            raise ValueError("Requested data type %r is not consistent with existing data type %r for family array %r" % (
                str(dtype), str(dtx), array_name))

        if all([x in fams for x in self_families]):
            # If, once we created this array, *all* families would have
            # this array, just create a survey-level array
            if self._promote_family_array(array_name, ndim=ndim, derived=derived, shared=shared) is not None:
                return None

        # if we get here, either the array cannot be promoted to survey level, or that would
        # not be appropriate, so actually go ahead and create the family array

        if shared is None:
            shared = self._shared_arrays
        new_ar = array._array_factory(dims, dtype, False, shared)
        new_ar._sim = weakref.ref(self)
        new_ar._name = array_name
        new_ar.family = family

        def sfa(n, v):
            try:
                self._family_arrays[n][family] = v
            except KeyError:
                self._family_arrays[n] = dict({family: v})

        sfa(array_name, new_ar)
        if derived:
            if array_name not in self._family_derived_array_names[family]:
                self._family_derived_array_names[family].append(array_name)

        if ndim is 3:
            array_name_1D = self._array_name_ND_to_1D(array_name)
            for i, a in enumerate(array_name_1D):
                sfa(a, new_ar[:, i])
                self._family_arrays[a][family]._name = a

    def _del_family_array(self, array_name, family):
        """Delete the array with the specified name for the specified family"""
        del self._family_arrays[array_name][family]
        if len(self._family_arrays[array_name]) == 0:
            del self._family_arrays[array_name]

        derive_track = self._family_derived_array_names[family]
        if array_name in derive_track:
            del derive_track[derive_track.index(array_name)]

    def _get_from_immediate_cache(self, name, fn):
        """Retrieves the named numpy array from the immediate cache associated
        with this survey. If the array does not exist in the immediate
        cache, function fn is called with no arguments and must generate
        it."""

        with self._immediate_cache_lock:
            if not hasattr(self, '_immediate_cache'):
                self._immediate_cache = [{}]
            cache = self._immediate_cache[0]
            hx = hash(name)
            if hx not in cache:
                cache[hx] = fn()

        return cache[hx]

    def _get_array(self, name, index=None, always_writable=False):
        """Get the array of the specified *name*, optionally
        for only the galxies specified by *index*.

        If *always_writable* is True, the returned array is
        writable. Otherwise, it is still normally writable, but
        not if the array is flagged as derived by the framework."""

        x = self._arrays[name]
        if x.derived and not always_writable:
            x = x.view()
            x.flags['WRITEABLE'] = False

        if index is not None:
            if type(index) is slice:
                ret = x[index]
            else:
                ret = array.IndexedSurveyArray(x, index)

            ret.family = None
            return ret

        else:
            return x

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        """Get the family-level array with specified *name* for the family *fam*,
        optionally for only the galaxies specified by *index* (relative to the
        family slice).

        If *always_writable* is True, the returned array is writable. Otherwise
        it is still normally writable, but not if the array is flagged as derived
        by the framework.
        """

        try:
            x = self._family_arrays[name][fam]
        except KeyError:
            raise KeyError("No array " + name + " for family " + fam.name)

        if x.derived and not always_writable:
            x = x.view()
            x.flags['WRITEABLE'] = False

        if index is not None:
            if type(index) is slice:
                x = x[index]
            else:
                if _subarray_immediate_mode or self.immediate_mode:
                    x = self._get_from_immediate_cache(name,
                                                       lambda: x[index])
                else:
                    x = array.IndexedSurveyArray(x, index)
        return x

    def _set_array(self, name, value, index=None):
        """Update the contents of the survey-level array to that
        specified by *value*. If *index* is not None, update only that
        subarray specified."""
        util.set_array_if_not_same(self._arrays[name], value, index)

    def _set_family_array(self, name, family, value, index=None):
        """Update the contents of the family-level array to that
        specified by *value*. If *index* is not None, update only that
        subarray specified."""
        util.set_array_if_not_same(self._family_arrays[name][family],
                                   value, index)

    def _create_arrays(self, array_list, ndim=1, dtype=None, zeros=True):
        """Create a set of arrays *array_list* of dimension len(self) x ndim, with
        a given numpy dtype."""
        for array in array_list:
            self._create_array(array, ndim, dtype, zeros)

    def _get_family_slice(self, fam):
        """Turn a specified Family object into a concrete slice which describes
        which galaxies in this SurveySnap belong to that family."""
        try:
            return self._family_slice[fam]
        except KeyError:
            return slice(0, 0)

    def _family_index(self):
        """Return an array giving the family number of each galaxy in this survey,
        something like 0,0,0,0,1,1,2,2,2, ... where 0 means self.families()[0] etc"""

        if hasattr(self, "_family_index_cached"):
            return self._family_index_cached

        ind = np.empty((len(self),), dtype='int8')
        for i, f in enumerate(self.ancestor.families()):
            ind[self._get_family_slice(f)] = i

        self._family_index_cached = ind

        return ind

    def _assert_not_family_array(self, name):
        """Raises a ValueError if the specified array name is connected to
        a family-specific array"""
        if name in self.family_keys():
            raise KeyError("Array " + name + " is a family-level property")


    def _promote_family_array(self, name, ndim=1, dtype=None, derived=False, shared=None):
        """Create a survey-level array (if it does not exist) with
        the specified name. Copy in any data from family-level arrays
        of the same name."""

        if ndim == 1 and self._array_name_1D_to_ND(name):
            return self._promote_family_array(self._array_name_1D_to_ND(name), 3, dtype)

        if dtype is None:
            try:
                x = self._family_arrays[name].keys()[0]
                dtype = self._family_arrays[name][x].dtype
                for x in self._family_arrays[name].values():
                    if x.dtype != dtype:
                        warnings.warn("Data types of family arrays do not match; assuming " + str(
                            dtype),  RuntimeWarning)

            except IndexError:
                pass

        dmap = [name in self._family_derived_array_names[
            i] for i in self._family_arrays[name]]
        some_derived = any(dmap)
        all_derived = all(dmap)

        if derived:
            some_derived = True
        if not derived:
            all_derived = False

        if name not in self._arrays:
            self._create_array(name, ndim=ndim, dtype=dtype, derived=all_derived, shared=shared)
        try:
            for fam in self._family_arrays[name]:
                self._arrays[name][self._get_family_slice(fam)] = self._family_arrays[name][fam]
            del self._family_arrays[name]
            if ndim == 3:
                for v in self._array_name_ND_to_1D(name):
                    del self._family_arrays[v]
            gc.collect()

        except KeyError:
            pass

        if some_derived:
            if all_derived:
                self._derived_array_names.append(name)
            else:
                warnings.warn(
                    "Conjoining derived and non-derived arrays. Assuming result is non-derived, so no further updates will be made.", RuntimeWarning)
            for v in self._family_derived_array_names.itervalues():
                if name in v:
                    del v[v.index(name)]

        return self._arrays[name]

    ############################################
    # DERIVED ARRAY SYSTEM
    ############################################
    @classmethod
    def derived_quantity(cl, fn):
        if cl not in Survey._derived_quantity_registry:
            Survey._derived_quantity_registry[cl] = {}
        Survey._derived_quantity_registry[cl][fn.__name__] = fn
        fn.__stable__ = False
        return fn

    @classmethod
    def stable_derived_quantity(cl, fn):
        if cl not in Survey._derived_quantity_registry:
            Survey._derived_quantity_registry[cl] = {}
        Survey._derived_quantity_registry[cl][fn.__name__] = fn
        fn.__stable__ = True
        return fn

    def _find_deriving_function(self, name):
        for cl in type(self).__mro__:
            if cl in self._derived_quantity_registry \
                    and name in self._derived_quantity_registry[cl]:
                return self._derived_quantity_registry[cl][name]
        else:
            return None

    def _derive_array(self, name, fam=None):
        """Calculate and store, for this Survey, the derivable array 'name'.
        If *fam* is not None, derive only for the specified family.

        This searches the registry of @X.derived_quantity functions
        for all X in the inheritance path of the current class.
        """
        global config

        calculated = False
        fn = self._find_deriving_function(name)
        if fn:
            if fam is None:
                result = fn(self)
                ndim = result.shape[-1] if len(
                    result.shape) > 1 else 1
                self._create_array(
                    name, ndim, dtype=result.dtype, derived=not fn.__stable__)
                write_array = self._get_array(
                    name, always_writable=True)
            else:
                result = fn(self[fam])
                ndim = result.shape[-1] if len(
                    result.shape) > 1 else 1

                # check if a family array already exists with a different dtype
                # if so, cast the result to the existing dtype
                # numpy version < 1.7 does not support doing this in-place

                if self._get_preferred_dtype(name) != result.dtype \
                   and self._get_preferred_dtype(name) is not None:
                    if int(np.version.version.split('.')[1]) > 6 :
                        result = result.astype(self._get_preferred_dtype(name),copy=False)
                    else :
                        result = result.astype(self._get_preferred_dtype(name))

                self[fam]._create_array(
                    name, ndim, dtype=result.dtype, derived=not fn.__stable__)
                write_array = self[fam]._get_array(
                    name, always_writable=True)

            write_array[:] = result

#    def _dirty(self, name):
#        """Declare a given array as changed, so deleting any derived
#        quantities which depend on it"""#

#        name = self._array_name_1D_to_ND(name) or name#

#        if not self.auto_propagate_off:
#            for d_ar in self._dependency_tracker.get_dependents(name):
#                if d_ar in self or self.has_family_key(d_ar):
#                    if self.is_derived_array(d_ar):
#                        del self[d_ar]
#                        self._dirty(d_ar)


    def is_derived_array(self, name, fam=None):
        """Returns True if the array or family array of given name is
        auto-derived (and therefore read-only)."""
        fam = fam or self._unifamily
        if fam:
            return (name in self._family_derived_array_names[fam]) or name in self._derived_array_names
        elif name in self.keys():
            return name in self._derived_array_names
        elif name in self.family_keys():
            return all([name in self._family_derived_array_names[i] for i in self._family_arrays[name]])
        else:
            return False


    ############################################
    # CONVENIENCE FUNCTIONS
    ############################################
    def mean_by_mass(self, name):
        """Calculate the mean by mass of the specified array."""
        m = np.asanyarray(self["mass"])
        ret = array.SurveyArray(
            (self[name].transpose() * m).transpose().mean(axis=0) / m.mean(), self[name].units)

        return ret

    ############################################
    # SURVEY DECORATION
    ############################################

    @classmethod
    def decorator(cl, fn):
        if cl not in Survey._decorator_registry:
            Survey._decorator_registry[cl] = []
        Survey._decorator_registry[cl].append(fn)
        return fn

    def _decorate(self):
        for cl in type(self).__mro__:
            if cl in self._decorator_registry:
                for fn in self._decorator_registry[cl]:
                    fn(self)

    ############################################
    # HASHING AND EQUALITY TESTING
    ############################################

    @property
    def _inclusion_hash(self):
        try:
            rval = self.__inclusion_hash
        except AttributeError:
            try:
                index_list = self.get_index_list(self.ancestor)
                hash = hashlib.md5(index_list.data)
                self.__inclusion_hash = hash.digest()
            except:
                print("Encountered a problem while calculating your inclusion hash. %s" % traceback.format_exc())
            rval = self.__inclusion_hash
        return rval

    def __hash__(self):
        return hash((object.__hash__(self.ancestor), self._inclusion_hash))

    def __eq__(self, other):
        """Equality test for Surveys. Returns true if both sides of the
        == operator point to the same data."""

        if self is other:
            return True
        return hash(self) == hash(other)


_subarray_immediate_mode = False
# Set this to True to always get copies of data when indexing is
# necessary. This is mainly a bug testing/efficiency checking mode --
# shouldn't be necessary

class SubSurvey(Survey):

    """Represent a sub-view of a Survey, initialized by specifying a
    slice.  Arrays accessed through __getitem__ are automatically
    sub-viewed using the given slice."""

    def __init__(self, base, _slice):
        self.base = base
        self._file_units_system = base._file_units_system
        self._unifamily = base._unifamily

        self._inherit()

        if isinstance(_slice, slice):
            # Various slice logic later (in particular taking
            # subsurveys-of-subsurveys) requires having positive
            # (i.e. start-relative) slices, so if we have been passed a
            # negative (end-relative) index, fix that now.
            if _slice.start is None:
                _slice = slice(0, _slice.stop, _slice.step)
            if _slice.start < 0:
                _slice = slice(len(
                    base) + _slice.start, _slice.stop, _slice.step)
            if _slice.stop is None or _slice.stop > len(base):
                _slice = slice(_slice.start, len(base), _slice.step)
            if _slice.stop < 0:
                _slice = slice(_slice.start, len(
                    base) + _slice.stop, _slice.step)

            self._slice = _slice

            descriptor = "[" + str(_slice.start) + ":" + str(_slice.stop)
            if _slice.step is not None:
                descriptor += ":" + str(_slice.step)
            descriptor += "]"

        else:
            raise TypeError("Unknown SubSurvey slice type")

        self._num_galaxies = util.indexing_length(_slice)

        self._descriptor = descriptor

    def _inherit(self):
        for x in self._inherited:
            setattr(self, x, getattr(self.base, x))

    def _get_array(self, name, index=None, always_writable=False):
        if _subarray_immediate_mode or self.immediate_mode:
            return self._get_from_immediate_cache(name,
                                                  lambda: self.base._get_array(
                                                      name, None, always_writable)[self._slice])

        else:
            ret = self.base._get_array(name, util.concatenate_indexing(
                self._slice, index), always_writable)
            ret.family = self._unifamily
            return ret

    def _set_array(self, name, value, index=None):
        self.base._set_array(
            name, value, util.concatenate_indexing(self._slice, index))

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        base_family_slice = self.base._get_family_slice(fam)
        sl = util.relative_slice(base_family_slice,
                                 util.intersect_slices(self._slice, base_family_slice, len(self.base)))
        sl = util.concatenate_indexing(sl, index)
        if _subarray_immediate_mode or self.immediate_mode:
            return self._get_from_immediate_cache((name, fam),
                                                  lambda: self.base._get_family_array(
                name, fam, None, always_writable)[sl])
        else:
            return self.base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        fslice = self._get_family_slice(family)
        self.base._set_family_array(
            name, family, value, util.concatenate_indexing(fslice, index))

    def _promote_family_array(self, *args, **kwargs):
        self.base._promote_family_array(*args, **kwargs)

    def __delitem__(self, name):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base survey")

    def _del_family_array(self, name, family):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base survey")

    @property
    def _path(self):
        return self.base._path + ":" + self._descriptor

    def keys(self):
        return self.base.keys()

    def loadable_keys(self, fam=None):
        if self._unifamily:
            return self.base.loadable_keys(self._unifamily)
        else:
            return self.base.loadable_keys(fam)

    def derivable_keys(self):
        return self.base.derivable_keys()

    def _get_family_slice(self, fam):
        sl = util.relative_slice(self._slice,
                                 util.intersect_slices(self._slice, self.base._get_family_slice(fam), len(self.base)))
        return sl

    def _load_array(self, array_name, fam=None, **kwargs):
        self.base._load_array(array_name, fam)

    def write_array(self, array_name, fam=None, **kwargs):
        fam = fam or self._unifamily
        if not fam or self._get_family_slice(fam) != slice(0, len(self)):
            raise IOError(
                "Array writing is available for entire survey arrays or family-level arrays, but not for arbitrary subarrays")

        self.base.write_array(array_name, fam=fam, **kwargs)

    def _derive_array(self, array_name, fam=None):
        self.base._derive_array(array_name, fam)

    def family_keys(self, fam=None):
        return self.base.family_keys(fam)

    def _create_array(self, *args, **kwargs):
        self.base._create_array(*args, **kwargs)

    def _create_family_array(self, *args, **kwargs):
        self.base._create_family_array(*args, **kwargs)

    def is_derived_array(self, v, fam=None):
        return self.base.is_derived_array(v)

    def unlink_array(self, name):
        self.base.unlink_array(name)

    def get_index_list(self, relative_to, of_galaxies=None):
        if of_galaxies is None:
            of_galaxies = np.arange(len(self))

        if relative_to is self:
            return of_galaxies

        return self.base.get_index_list(relative_to, util.concatenate_indexing(self._slice, of_galaxies))


class IndexedSubSurvey(SubSurvey):

    """Represents a subset of the survey galaxies according
    to an index array."""

    def __init__(self, base, index_array):

        self._descriptor = "indexed"
        self.base = base
        self._inherit()

        self._unifamily = base._unifamily

        if isinstance(index_array, filt.Filter):
            self._descriptor = index_array._descriptor
            index_array = index_array.where(base)[0]

        elif isinstance(index_array, tuple):
            if isinstance(index_array[0], np.ndarray):
                index_array = index_array[0]
            else:
                index_array = np.array(index_array)
        else:
            index_array = np.asarray(index_array)

        findex = base._family_index()[index_array]
        # Check the family index array is monotonically increasing
        # If not, the family slices cannot be implemented
        if not all(np.diff(findex) >= 0):
            raise ValueError(
                "Families must retain the same ordering in the SubSurvey")

        self._slice = index_array
        self._family_slice = {}
        self._family_indices = {}
        self._num_galaxies = len(index_array)

        # Find the locations of the family slices
        for i, fam in enumerate(self.ancestor.families()):
            ids = np.where(findex == i)[0]
            if len(ids) > 0:
                new_slice = slice(ids.min(), ids.max() + 1)
                self._family_slice[fam] = new_slice
                self._family_indices[fam] = np.asarray(index_array[
                                                       new_slice]) - base._get_family_slice(fam).start


    def _get_family_slice(self, fam):
        # A bit messy: jump out the SubSurvey inheritance chain
        # and call Survey method directly...
        return Survey._get_family_slice(self, fam)

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        sl = self._family_indices.get(fam,slice(0,0))
        sl = util.concatenate_indexing(sl, index)

        return self.base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        self.base._set_family_array(name, family, value,
                                    util.concatenate_indexing(self._family_indices[family], index))

    def _create_array(self, *args, **kwargs):
        self.base._create_array(*args, **kwargs)


class FamilySubSurvey(SubSurvey):

    """Represents a one-family portion of a parent survey object"""

    def __init__(self, base, fam):
        self.base = base

        self._inherit()

        self._slice = base._get_family_slice(fam)
        self._unifamily = fam
        self._descriptor = ":" + fam.name
        # Use the slice attributes to find sub array length
        self._num_galaxies = self._slice.stop - self._slice.start

    def __delitem__(self, name):
        if name in self.base.keys():
            raise ValueError(
                "Cannot delete global simulation property from sub-view")
        elif name in self.base.family_keys(self._unifamily):
            self.base._del_family_array(name, self._unifamily)

    def keys(self):
        global_keys = self.base.keys()
        family_keys = self.base.family_keys(self._unifamily)
        return list(set(global_keys).union(family_keys))

    def family_keys(self, fam=None):
        # We now define there to be no family-specific subproperties,
        # because all properties can be accessed through standard
        # __setitem__, __getitem__ methods
        return []

    def _get_family_slice(self, fam):
        if fam is self._unifamily:
            return slice(0, len(self))
        else:
            return slice(0, 0)

    def _get_array(self, name, index=None, always_writable=False):
        try:
            return SubSurvey._get_array(self, name, index, always_writable)
        except KeyError:
            return self.base._get_family_array(name, self._unifamily, index, always_writable)

    def _create_array(self, array_name, ndim=1, dtype=None, zeros=True, derived=False, shared=None):
        # Array creation now maps into family-array creation in the parent
        self.base._create_family_array(
            array_name, self._unifamily, ndim, dtype, derived, shared)

    def _set_array(self, name, value, index=None):
        if name in self.base.keys():
            self.base._set_array(
                name, value, util.concatenate_indexing(self._slice, index))
        else:
            self.base._set_family_array(name, self._unifamily, value, index)

    def _create_family_array(self, array_name, family, ndim, dtype, derived, shared):
        self.base._create_family_array(
            array_name, family, ndim, dtype, derived, shared)

    def _promote_family_array(self, *args, **kwargs):
        pass

    def _load_array(self, array_name, fam=None, **kwargs):
        if fam is self._unifamily or fam is None:
            self.base._load_array(array_name, self._unifamily)

    def _derive_array(self, array_name, fam=None):
        if fam is self._unifamily or fam is None:
            self.base._derive_array(array_name, self._unifamily)



def load(path, *args, **kwargs):
    """Loads a file using the appropriate class, returning a Survey
    instance."""

    for c in config['survey-class-priority']:
        if c._can_load(path):
            return c(path, *args, **kwargs)

    raise IOError("File %r: format not understood or does not exist" % path)

def new(n_galaxies=0, order=None, **families):
    """Create a blank Survey, with the specified number of galaxies.

    Galaxy, mass and metallicity arrays are created and filled
    with zeros. Do we need more general properties describing a galaxy something like distance/redshift?

    By default all galaxies are taken to be training galaxies.
    To specify otherwise, pass in keyword arguments specifying
    the number of galaxies for each family, e.g.

    f = new(train=50, pred=25, validation=25)

    The order in which the different families appear in the survey
    is unspecified unless you add an 'order' argument:

    f = new(train=50, pred=25, validation=25, order='val,train,pred')

    guarantees the validation, then train, then prediction galaxies appear
    in sequence.
    """

    if len(families) == 0:
        families = {'train': n_galaxies}

    t_fam = []
    tot_galaxies = 0

    if order is None:
        for k, v in families.items():

            assert isinstance(v, int)
            t_fam.append((family.get_family(k), v))
            tot_galaxies += v
    else:
        for k in order.split(","):
            v = families[k]
            assert isinstance(v, int)
            t_fam.append((family.get_family(k), v))
            tot_galaxies += v

    x = Survey()
    x._num_galaxies = tot_galaxies
    x._path = "<created>"

    #x._create_arrays(["image"], 5) #default u,g,r,i,z images
    x._create_arrays(["galaxy", "mass", "metals", "distance"], 1)

    rt = 0
    for k, v in t_fam:
        x._family_slice[k] = slice(rt, rt + v)
        rt += v

    x._decorate()
    return x

def _get_survey_classes():
    from . import illustris#, sdss

    _survey_classes = [illustris.SDSSMockSurvey]#,sdss.MANGASurvey,sdss.ClassicSurvey]

    return _survey_classes
