"""
picasso
=======

A light-weight, portable, format-transparent machine learning
and analysis framework for astronomical galaxy images.

For more information, either build the latest documentation included
in our git repository, or view the online version here:
http:balbla

"""

############## change things below ###############

from . import configuration

from .configuration import config

from . import util, filt, array, family, survey
from .survey import *
from . import analysis


#fill these functions with meaning
from . import load, new


configuration.configure_physical_scaling()

__version__ = '0.1'

__all__ = ['load', 'new', 'analysis', 'train']
