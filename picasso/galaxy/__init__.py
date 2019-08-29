"""

galaxy
====

Implements the :class:`~picasso.Galaxy` class which represents a container class for 
the galaxy (image) data and derived data products.

See the `galaxy tutorial
<http://picasso.github.io/picasso/tutorials/galaxies.html>`_ for some
examples.

"""

import copy

from .. import survey


class Galaxy(object):

    """
    Generic class representing a galaxy.
    """

    def __init__(self, galaxy_id):
        
        self._galaxy_id = galaxy_id
        self._descriptor = "Galaxy_" + str(galaxy_id)
        self.properties = {}
        self.properties['Galaxy_id'] = galaxy_id


    ############################################
    # THE BASICS: SIMPLE INFORMATION
    ############################################

    @property
    def galaxy_id(self):
        return self._galaxy_id

    def __repr__(self):
        return "<Galaxy " + self._galaxy_id + ">"

    #We need the basic getitem, setitem, delitem methods, I think

from picasso.galaxy.SDSSMockGalaxy import SDSSMockGalaxy

def _get_galaxy_classes():

    _galaxy_classes = [SDSSMockGalaxy]

    return _galaxy_classes
