import numpy as np
import os, h5py

from . import Galaxy
from .. import survey, array
from .. import util

# To Do
# do we want to add a derived_property dict for things like the geometry fits and the isolists?

class PropertyDict(dict):

    def __init__(self, base_path, filename, galnr):
        self._file = filename
        self._base_path = base_path
        self._galnr = galnr

    def __getitem__(self, k):
        if k in self:
            return dict.__getitem__(self, k)
        else:
            try:
                self.__init_properties()
                return dict.__getitem__(self, k)
            except:
                raise KeyError(k)

    def __load_photometric_maps(self, subfolder='photometry/'):
        """ load the photometric maps in all wave length bands. """

        _phot_file = h5py.File(self._base_path+subfolder+self._file.split('/')[-1][:-16]+'synthetic_image.h5', "r")
        
        for band, key in zip(_phot_file['data'],['u','g','r','i','z']):
            # the illustris SDSS mock images created with sunrise are somehow transposed compared to our 
            # physical maps. Thus to prevent confusion when plotting them we transpose them here to
            # comply with our image defintion.
             
            self.__setitem__(key+'_band', band.T[:,::-1])

    def __init_predicted_properties(self, subfolder='prediction/'):
        # try to load properties which are the result of the ML prediction 

        _pred_path = self._base_path+subfolder+self._file.split('/')[-1]
        _file = h5py.File(_pred_path, "r")
        image_stack, keys, psize = util._load_halo_properties(_file)
        
        # self.properties['psize_pred'] = psize #psize should be the same as the true image
        # dm mass should be the same, unless we want to predict it in the future as well.
        #self.properties['dm_mass'] = util.load_dm_mass(self._galnr, self._base_path+'illustris_fof_props.h5')
        
        for key in keys:
            idx = keys.index(key)
            #should we make the following below a separate step? 
            self.__setitem__(key+'_pred', np.asarray(util.scale_to_physical_units(image_stack[idx], psize)))

    def __init_properties(self, subfolder='prediction/'):
        # init true properties
        with h5py.File(self._file, "r") as h5file:
            image_stack, keys, psize = util._load_halo_properties(h5file)
        
        self.__setitem__('psize', psize)
        self.__setitem__('dm_mass', util.load_dm_mass(self._galnr, self._base_path+'illustris_fof_props.h5'))
        
        for key in keys:
            idx = keys.index(key)
            #should we make the following below a separate step? 
            self.__setitem__(key, np.asarray(util.scale_to_physical_units(image_stack[idx], psize)))

        try:
            self.__load_photometric_maps()
        except:
            pass

        # in case we deal with galaxies belonging to the validation or prediction set we have further properties which come as the output of the ML algortith.
        # assumption here is these are seperate files in a subfolder named < prediction >
        # so, let's try to load them as well

        if os.path.exists(self._base_path+subfolder):
            self.__init_predicted_properties(subfolder=subfolder)



class SDSSMockGalaxy(Galaxy):
    """
    Illustris SDSS Mock galaxy class
    """

    def __init__(self, filename, *args) :
        self._file = filename
        self._galnr = self.__get_galnr(filename)
        self._camera = self.__get_camera(filename)
        self._Galaxy_id = self.__build_gal_id(filename)
        self._base_path = filename[:-len(filename.split('/')[-1])]

        super(SDSSMockGalaxy,self).__init__(self._Galaxy_id, *args)

        self._descriptor = "Galaxy_"+self._Galaxy_id

        self.properties = PropertyDict(self._base_path, self._file, self._galnr)
        self._keys = []

        # load properties

        #self.__init_properties()
    
    def __get_galnr(self, filename):
        return filename.split('_')[-5]

    def __get_camera(self, filename):
        return filename.split('_')[-3]

    def __build_gal_id(self, filename):
        return self._galnr + "c" + self._camera

    @property
    def keys(self):
        # return all keys for this galaxy, this involves for now loading all associated files of this galaxy
        #to make sure we are not missing any key

        if self._keys == []:
            _keys = ['Galaxy_id']
            # ground truth keys
            with h5py.File(self._file, "r") as h5file:
                _, keys, _ = util._load_halo_properties(h5file)

            _keys.extend(keys)
            _keys.extend(['psize'])
            # is the dm file present?
            try:
                _ = util.load_dm_mass(self._galnr, self._base_path+'illustris_fof_props.h5')
                _keys.extend(['dm_mass'])
            except:
                pass
            # predicted keys not a clever solution with the explicit path hard coding of the path...
            #try:
            _pred_path = self._base_path+'prediction/'+self._file.split('/')[-1]
            _file = h5py.File(_pred_path, "r")
            _, keys, _ = util._load_halo_properties(_file)
            _keys = _keys + [k+'_pred' for k in keys]
            #except:
            #    pass
            # and now photometric data
            try:
                _phot_file = h5py.File(self._base_path+'photometry/'+self._file.split('/')[-1][:-16]+'synthetic_image.h5', "r")
                keys = ['u_band','g_band','r_band','i_band','z_band']
                _keys.extend(keys)
            except:
                pass   
            
            self._keys = _keys

        return self._keys

