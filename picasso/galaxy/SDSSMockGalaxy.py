import numpy as np
import os

from . import Galaxy
from .. import survey, array
from .. import util

class SDSSMockGalaxy(Galaxy):
    """
    Illustris SDSS Mock galaxy class
    """

    def __init__(self, filename, *args) :

        self._Galaxy_id = self.__build_gal_id(filename)
        self._galnr = self.__get_galnr(filename)
        self._camera = self.__get_camera(filename)
        self._base_path = os.path.join(filename.split('/')[:-1])

        super(SDSSMockGalaxy,self).__init__(self._Galaxy_id, *args)

        self._descriptor = "Galaxy_%d"%,self._Galaxy_id

        # load properties

        self.__init_properties()


    def __get_galnr(filename):
        return filename.split('_')[-5]

    def __get_camera(filename):
        return filename.split('_')[-3]

    def __build_gal_id(filename):
        return __get_galnr + "c" + __get_camera

    def __init_properties():
        image_stack, keys, psize = util.load_halo_properties(self._galnr, self._camera, path=self._base_path)
        
        self.properties['dm_mass'] = util.load_dm_mass(self.galnr, self._base_path+'illustris_fof_props.h5')
        
        for key in keys:
            idx = keys.index(key)
            self.properties[key] = np.asarray(scale_to_physical_units(image_stack[idx], psize)) 
        