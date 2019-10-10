import numpy as np
import os, h5py

from . import Galaxy
from .. import survey, array
from .. import util

class SDSSMockGalaxy(Galaxy):
    """
    Illustris SDSS Mock galaxy class
    """

    def __init__(self, file, *args) :
        self._file = file
        self._galnr = self.__get_galnr(file.filename)
        self._camera = self.__get_camera(file.filename)
        self._Galaxy_id = self.__build_gal_id(file.filename)
        self._base_path = file.filename[:-len(file.filename.split('/')[-1])]

        super(SDSSMockGalaxy,self).__init__(self._Galaxy_id, *args)

        self._descriptor = "Galaxy_"+self._Galaxy_id

        # load properties

        self.__init_properties()


    def __get_galnr(self, filename):
        return filename.split('_')[-5]

    def __get_camera(self, filename):
        return filename.split('_')[-3]

    def __build_gal_id(self, filename):
        return self._galnr + "c" + self._camera

    def __init_predicted_properties(self, subfolder='prediction/'):
        # try to load properties which are the result of the ML prediction 

        _pred_path = self._base_path+'/'+subfolder
        _file = h5py.File(pred_path, "r")
        image_stack, keys, psize = util._load_halo_properties(_file)
        
        # self.properties['psize_pred'] = psize #psize should be the same as the true image
        # dm mass should be the same, unless we want to predict it in the future as well.
        #self.properties['dm_mass'] = util.load_dm_mass(self._galnr, self._base_path+'illustris_fof_props.h5')
        
        for key in keys:
            idx = keys.index(key)
            #should we make the following below a separate step? 
            self.properties[key+'_pred'] = np.asarray(util.scale_to_physical_units(image_stack[idx], psize)) 

    def __init_properties(self):
        # init true properties
        image_stack, keys, psize = util._load_halo_properties(self._file)
        
        self.properties['psize'] = psize
        self.properties['dm_mass'] = util.load_dm_mass(self._galnr, self._base_path+'illustris_fof_props.h5')
        
        for key in keys:
            idx = keys.index(key)
            #should we make the following below a separate step? 
            self.properties[key] = np.asarray(util.scale_to_physical_units(image_stack[idx], psize)) 
        
        # in case we deal with galaxies belonging to the validation or prediction set we have further properties which come as the output of the ML algortith.
        # assumption here is these are seperate files in a subfolder named < prediction >
        # so, let's try to load them as well

        if os.path.exists(self._base_path+'/'+subfolder):
            __init_predicted_properties(self, subfolder='prediction/')


    #ideas for more utilities:
    # - plot the galaxy image
    # - calculate total mass, metallicity etc? --> will be a decorator to the property (see pynbody)
    # - fitting ellipses etc. or does this go to analysis?  