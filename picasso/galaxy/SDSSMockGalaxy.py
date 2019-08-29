import numpy as np

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

    def __init_properties(self):
        image_stack, keys, psize = util._load_halo_properties(self._file)
        
        self.properties['psize'] = psize
        self.properties['dm_mass'] = util.load_dm_mass(self._galnr, self._base_path+'illustris_fof_props.h5')
        
        for key in keys:
            idx = keys.index(key)
            #should we make the following below a separate step? 
            self.properties[key] = np.asarray(util.scale_to_physical_units(image_stack[idx], psize)) 
        