"""
surveydict
=======

This submodule defines an augmented dictionary class
(:class:`SurveyDict`) for :class:`~picasso.survey.Survey` properties
where entries need to be managed e.g.  for defining default entries,
or for ensuring consistency between equivalent properties like
redshift and scalefactor.

By default, a :class:`SurveyDict` automatically implements default entries
for image resolution of galaxy images.

Adding further properties
-------------------------

To add further properties use the SurveyDict.getter and SurveyDict.setter decorators.
For instance, to add a property 'X_copy' which just reflects the value of the
property 'X', you would use the following code:

.. code-block:: python

 @SurveyDict.getter
 def X_copy(d) :
     return d['X']

 @SurveyDict.setter
 def X_copy(d, value) :
     d['X'] = value
   
"""

import warnings
from . import configuration

__all__ = ['SurveyDict']


class SurveyDict(dict):
    _getters = {}
    _setters = {}

    @staticmethod
    def getter(f):
        """Define a getter function for all SurveyDicts"""
        SurveyDict._getters[f.__name__] = f

    @staticmethod
    def setter(f):
        """Define a setter function for all SurveyDicts"""
        SurveyDict._setters[f.__name__] = f

    def __getitem__(self, k):
        if k in self:
            return dict.__getitem__(self, k)
        elif k in SurveyDict._getters:
            return SurveyDict._getters[k](self)
        else:
            raise KeyError(k)

    def __setitem__(self, k, v):
        if k in SurveyDict._setters:
            SurveyDict._setters[k](self, v)
        else:
            dict.__setitem__(self, k, v)


@SurveyDict.getter
def z(d):
    if d["a"] is None:
        return None
    try:
        return 1.0 / d["a"] - 1.0
    except ZeroDivisionError:
        return None


@SurveyDict.setter
def z(d, z):
    if z is None:
        d["a"] = None
    else:
        d["a"] = 1.0 / (1.0 + z)


def default_fn(name, value):
    """Return a getter function for the default name/value pair"""
    def f(d):
        warnings.warn("Assuming default value for property '%s'=%.2e" % (
            name, value), RuntimeWarning)
        d[name] = value
        return value
    f.__name__ = name

    return f

#for k in config['default-cosmology']:
#    SurveyDict.getter(default_fn(k, config['default-cosmology'][k]))
