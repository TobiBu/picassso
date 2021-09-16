"""illustris
=====

Implements classes and functions for handling illustris (mock SDSS) files.  
You rarely need to access this module directly as it will be invoked
automatically via picasso.load.

**Input**:

*path*: path to survey folder string

**Optional Keywords**:

"""

from .. import array, util
from .. import family
from .. import configuration #do we need this line?
from ..configuration import config 
from . import Survey
from ..galaxy import SDSSMockGalaxy

import struct
import os
import glob
import numpy as np
import sys
import warnings
import copy
import types
import math
import h5py
from tqdm import tqdm

from picasso.analysis import image_tools

class MultiFileManager(object) :
    _suffix = '_physical_data.h5'

    def __init__(self, path, mode='r') :
        self._mode = mode
        if h5py.is_hdf5(path):
            tmp_filename = path.split("/")[-1]
            self._path  = path[:-len(tmp_filename)]
            self._filenames = [tmp_filename]
            self._numfiles = 1
        else:
            self._path = path
            self._filenames = glob.glob(path + "halo_*_camera_?" + self._suffix)
            self._numfiles = len(self._filenames)

    def __iter__(self):
        for i in range(self._numfiles):
            self.idx = i
            yield self._filenames[self.idx]

    def __next__(self):
        if self.idx <= self._numfiles:
            file = self._filenames[self.idx]
            self.idx += 1
            return file
        else:
            raise StopIteration

    def __getitem__(self, i) :
        return self._filenames[i]

    def has_file(self, filename):
        if self._path in filename:
            return filename in self._filenames
        else:
            return self.path+filename in self._filenames

    def get_file_idx(self, filename):
        if self._path in filename:
            return self._filenames.index(filename) #np.where(filename == self._filenames)[0]
        else:
            return self._filenames.index(self._path+filename) #np.where(self._path+filename == self._filenames)[0]

    def reopen_in_mode(self, mode):
        if mode!=self._mode:
            self._mode = mode


#should we add a function/method to create the galaxy images from illustris or leave that as a seperate script as is now?

class SDSSMockSurvey(Survey):
    # decide later which are basic loadables...
    _basic_loadable_keys = {family.training: set(['galaxy', 'mass', 'metals','distance']),
                            family.testing: set(['galaxy', 'mass', 'metals','distance']),
                            family.validation: set(['galaxy', 'mass', 'metals','distance']),
                            None: set(['galaxy'])}

    _readable_hdf5_test_key = "stars_Masses" #"galaxy"
    _suffix = '_physical_data.h5'

    _multifile_manager_class = MultiFileManager

    def __init__(self, path, **kwargs):

        global config

        super(SDSSMockSurvey, self).__init__()

        verbose = kwargs.get('verbose', config['verbose'])

        self._path = path
        self._num_galaxies = len(glob.glob(path + "halo_*_camera_?" + self._suffix))

        self.__init_filemanager(path)
        #self.__init_galaxies()  ## fill with content to load all galaxies
        self.__init_file_map()
        self.__init_family_map()
        self.__init_loadable_keys()

        #gal_list = glob.glob(path + "true/halo_*_camera_?" + _suffix)

        self._decorate()

    def __init_filemanager(self, path):
        self._files = self._multifile_manager_class(path)


    def __init_file_map(self, family_map_file='family_map.h5'):

        # modify later: idea here is: all galaxies in one folder and there exists a file splitting the gals into different families
        # if not present, this file can be created with survey.split_for_train() which further has to reshuffle the survey array to
        # comply with the family slicing...

        if h5py.is_hdf5(family_map_file):
            self._file_map = h5py.File(family_map_file, "r") # this is supposed to be a dictionary with keys family and arrays of indices or filenames/galaxy names belonging to that family 
        else:
            self._file_map = {}
            for x in family.family_names():
                fam = family.get_family(x)
                if x == "training":
                    self._file_map[fam] = glob.glob(self.path + "halo_*_camera_?" + self._suffix)
                else:
                    self._file_map[fam] = []

    def __init_family_map(self):
        family_slice_start = 0
        #here we could use some scikit stuff to split the survey into train, test, val and then save this split in the survey folder.

        # assumption below is that true properties exist for all the survey and prediction and validation are disjoint.
        # thus slicing will be possible by first loading the disjoint galaxies into memory and then later all galaxies not loaded...
        
        # do the reading in an ordered way to instantiate the family slicing
        family_length = 0
        for x in family.family_names():
            fam = family.get_family(x)

            if self._file_map:
                family_length = len(self._file_map[fam])
            else:
                family_length = 0

            self._family_slice[fam] = slice(family_slice_start, family_slice_start + family_length)
            family_slice_start += family_length
            #assert(family_slice_start==self._num_galaxies)

        
    def __init_loadable_keys(self):
        self._loadable_family_keys = {}
        all_fams = self.families()

        for fam in all_fams:
            self._loadable_family_keys[fam] = set(["galaxy"])
            tmp_file = self._file_map[fam][0] #first file in family
            
            _file_idx = self._files.get_file_idx(tmp_file)
            with h5py.File(self._files[_file_idx], self._files._mode) as f: 
                for this_key in f.keys():
                    self._loadable_family_keys[fam].add(this_key)
            self._loadable_family_keys[fam] = list(self._loadable_family_keys[fam]) 

        self._loadable_keys = set(self._loadable_family_keys[all_fams[0]])
        for fam_keys in self._loadable_family_keys.itervalues():
            self._loadable_keys.intersection_update(fam_keys)

        self._loadable_keys = list(self._loadable_keys)

    def _family_has_loadable_array(self, fam, name):
        """Returns True if the array can be loaded for the specified family.
        If fam is None, returns True if the array can be loaded for all families."""
        return name in self.loadable_keys(fam)

    def loadable_keys(self, fam=None):
        if fam is None:
            return self._loadable_keys
        else:
            return self._loadable_family_keys[fam]

    @staticmethod
    def _write(self, filename=None):
        raise RuntimeError("Not implemented")

    # we could think about writing auxiliary array files instead of adding fields to the single galaxy files...
    def write_array(self, array_name, fam=None, overwrite=False):
        #how do we want to write arrays? as separate files or as fields in each galaxy file?
        self._files.reopen_in_mode('r+')

        if fam is None:
            target = self
            all_fams_to_write = self.families()
        else:
            target = self[fam]
            all_fams_to_write = [fam]

        for writing_fam in all_fams_to_write:
            target_array = self[writing_fam][array_name]
            for file in self._file_map[writing_fam]:
                _file_idx = self._files.get_file_idx(file)
            
                target_array_this = target_array[_file_idx] #assumption is still we preserve the order in arrays and files on disk

                with h5py.File(self._files[_file_idx], self._files._mode) as f: 
                    f.create_dataset(array_name, data=target_array_this)


    def _load_array(self, array_name, fam=None):
        if not self._family_has_loadable_array(fam, array_name):
            raise IOError("No such array on disk")
        else:

            dtype, dy = self.__get_dtype_dims_and_units(fam, array_name) 

            if fam is None:
                target = self
                all_fams_to_load = self.families()
            else:
                target = self[fam]
                all_fams_to_load = [fam]

            target._create_array(array_name, dy, dtype=dtype) #need to check how to deal with images

            for loading_fam in all_fams_to_load:
                tmp_arr = []
                
                for file in tqdm(self._file_map[loading_fam], ascii=True, dynamic_ncols=True):
                    _file_idx = self._files.get_file_idx(file)
                    if array_name == "galaxy":
                        #instantiate the galaxy object
                        tmp_arr.append(SDSSMockGalaxy(self._files[_file_idx]))
                        #with h5py.File(self._files[_file_idx], self._files._mode) as f: 
                        #    tmp_arr.append(SDSSMockGalaxy(f))
                    else:
                        #if we do not ask for the galaxy itself, load the "postprocessed" array
                        #tmp_arr.append(self._files[file_idx][array_name].value)   
                        tmp_arr.append(self['galaxy'].properties[array_name])  
                        #raise NotImplementedError("Sorry, this feature is not implemented yet.")
                        # Do we actually need to construct any other survey arrays except for the galaxies objects themselves? 
                        # maybe in some cases it migth be useful to have the DM mass or the Galaxy ID directly accessable...
                        # could think oif making this arrays of total stellar mass, etc... or do we need to get an array of galaxy images?

                target_array = self[loading_fam][array_name]
                assert target_array.size == np.asarray(tmp_arr).size
                
                target_array[:] = tmp_arr


    def __get_dtype_dims_and_units(self, fam, translated_name):

        if translated_name == "galaxy":
            # we deal with galaxy objects
            dtype = object
            dy = 1
            #inferred_units = None
        else:
            
            #if fam is None:
            #    fam = self.families()[0]

            representative_dset = None
            representative_hdf = None
            # not all arrays are present in all hdfs so need to loop
            # until we find one
            # needs modification to work with images as array entries...
            for file in self._files:
                try:
                    with h5py.File(file, self._files._mode) as hdf0:
                        representative_dset = hdf0[translated_name]
                        #if hasattr(hdf0, "psize"):
                        #    inferred_units = hdf0[psize] #do something

                        if len(representative_dset)!=0:
                            # suitable for figuring out everything we need to know about this array
                            break
                except KeyError:
                    continue
            if representative_dset is None:
                raise KeyError("Array is not present in HDF file")

            assert len(representative_dset.shape) <= 2 

            if len(representative_dset.shape) > 1:
                dy = representative_dset.shape # here we deal with images...
            else:
                dy = 1

            dtype = representative_dset.dtype

        return dtype, dy#, inferred_units

    @classmethod
    def _test_for_hdf5_key(cls, f):
        with h5py.File(f, "r") as h5test:
            test_key = cls._readable_hdf5_test_key
            #if test_key[-1]=="?":
                # try all particle numbers in turn
                #for p in range(6):
                #    test_key = test_key[:-1]+str(p)
                #    if test_key in h5test:
                #        return True
            #    return False
            #else:
            return test_key in h5test

    @classmethod
    def _can_load(cls, f):

        test = glob.glob(f + "halo_*_camera_?" + cls._suffix)
        if hasattr(h5py, "is_hdf5"):
            if h5py.is_hdf5(test[0]):
                return cls._test_for_hdf5_key(test[0])
            else:
                return False
        else:
            if "hdf5" in test[0] or "h5" in test[0]:
                warnings.warn(
                    "It looks like you're trying to load HDF5 files, but python's HDF support (h5py module) is missing.", RuntimeWarning)
            return False


@SDSSMockSurvey.decorator
def do_properties(survey):
    # do we have any gloabl propertie we want to set?...
    #raise RuntimeError("Not implemented")

    with h5py.File(survey._files[0], "r") as  test_file:
        # for mock images so far resolution is fixed
        try:
            survey.properties['image_res'] = test_file["stars_Masses"].value.shape[0]
        except KeyError:
            survey.properties['image_res'] = np.nan

# when finally dealing with real observartions, we can add further properties here...


#    # not all omegas need to be specified in the attributes
#    try:
#        sim.properties['omegaB0'] = atr['OmegaBaryon']
#    except KeyError:
#        pass#

#    sim.properties['omegaM0'] = atr['Omega0']
#    sim.properties['omegaL0'] = atr['OmegaLambda']
#    sim.properties['boxsize'] = atr['BoxSize'] * sim.infer_original_units('cm')
#    sim.properties['z'] = (1. / sim.properties['a']) - 1
#    sim.properties['h'] = atr['HubbleParam']#

#    # time unit might not be set in the attributes
#    if "Time_GYR" in atr:
#        sim.properties['time'] = units.Gyr * atr['Time_GYR']
#    else:
#        from .. import analysis
#        sim.properties['time'] = analysis.cosmology.age(sim) * units.Gyr#

#    for s,value in sim._get_hdf_header_attrs().iteritems():
#        if s not in ['ExpansionFactor', 'Time_GYR', 'Time', 'Omega0', 'OmegaBaryon', 'OmegaLambda', 'BoxSize', 'HubbleParam']:
#            sim.properties[s] = value



## We have some internally derive quantities...

#add all quantities, such as the stellar mass in half mass radius, the photutils isocontours when we have a working survey object...

#TODO#
# do we want to add the geometry and iso dicts to the galaxy properties dictionary or do we keep this clean?
# might think about a derived property dict, where we can put the geometry and iso_lists once they are pre-calculated.

@SDSSMockSurvey.derived_quantity
def total_star_mass(self) :
    tmp = [np.sum(x.properties['stars_Masses']) for x in self['galaxy']]
    return np.asarray(tmp)

@SDSSMockSurvey.derived_quantity
def geom(self):
    """ ellipse geometry fitted to mass map """
    return image_tools.fit_image(self)

@SDSSMockSurvey.derived_quantity
def iso_dict(self):
    """ dictionary of available iso contours """
    return image_tools.get_isolists(self,self['geom'])

@SDSSMockSurvey.derived_quantity
def M2Rh(self):
    """Mass within 2 R_half """
    value, _ = image_tools.get_pixel_sum(self['iso_dict'],'stars_Masses',50)
    return value


