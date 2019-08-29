"""

filt
====

Defines and implements 'filters' which allow abstract subsets
of data to be specified.

See the `filter tutorial
<http://picasso.github.io/picasso/tutorials/filters.html>`_ for some
sample usage.

"""


import numpy as np
#import cPickle
from . import util, _util, family

class Filter(object):

    def __init__(self):
        self._descriptor = "filter"
        pass

    def where(self, survey):
        return np.where(self(survey))

    def __call__(self, survey):
        return np.ones(len(survey), dtype=bool)

    def __and__(self, f2):
        return And(self, f2)

    def __invert__(self):
        return Not(self)

    def __or__(self, f2):
        return Or(self, f2)

    def __repr__(self):
        return "Filter()"

    #def __hash__(self):
    #    return hash(cPickle.dumps(self))

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        for k, v in self.__dict__.iteritems():
            if k not in other.__dict__:
                return False
            else:
                equal = other.__dict__[k]==v
                if isinstance(equal, np.ndarray):
                    equal = equal.all()
                if not equal:
                    return False

        return True


class FamilyFilter(Filter):
    def __init__(self, family_):
        assert isinstance(family_, family.Family)
        self._descriptor = family_.name
        self.family = family_

    def __repr__(self):
        return "FamilyFilter("+self.family.name+")"

    def __call__(self, survey):
        slice_ = survey._get_family_slice(self.family)
        flags = np.zeros(len(survey), dtype=bool)
        flags[slice_] = True
        return flags

class And(Filter):

    def __init__(self, f1, f2):
        self._descriptor = f1._descriptor + "&" + f2._descriptor
        self.f1 = f1
        self.f2 = f2

    def __call__(self, survey):
        return self.f1(survey) * self.f2(survey)

    def __repr__(self):
        return "(" + repr(self.f1) + " & " + repr(self.f2) + ")"


class Or(Filter):

    def __init__(self, f1, f2):
        self._descriptor = f1._descriptor + "|" + f2._descriptor
        self.f1 = f1
        self.f2 = f2

    def __call__(self, survey):
        return self.f1(survey) + self.f2(survey)

    def __repr__(self):
        return "(" + repr(self.f1) + " | " + repr(self.f2) + ")"


class Not(Filter):

    def __init__(self, f):
        self._descriptor = "~" + f._descriptor
        self.f = f

    def __call__(self, survey):
        x = self.f(survey)
        return np.logical_not(x)

    def __repr__(self):
        return "~" + repr(self.f)



class BandPass(Filter):

    """
    Return galaxies whose property `prop` is within `min` and `max`.
    """

    def __init__(self, prop, min, max):
        self._descriptor = "bandpass_" + prop

        self._prop = prop
        self._min = min
        self._max = max

    def __call__(self, survey):
        min_ = self._min
        max_ = self._max
        prop = self._prop

        return ((survey[prop] > min_) * (survey[prop] < max_))

    def __repr__(self):
        min_, max_ = ['%.2e' % x for x in [self._min, self._max]]

        return "BandPass('%s', %s, %s)" % (self._prop, min_, max_)


class HighPass(Filter):

    """
    Return particles whose property `prop` exceeds `min`.
    """

    def __init__(self, prop, min):
        self._descriptor = "highpass_" + prop

        self._prop = prop
        self._min = min

    def __call__(self, survey):
        min_ = self._min

        prop = self._prop

        return (survey[prop] > min_)

    def __repr__(self):
        min = ('%.2e' % self._min)
        return "HighPass('%s', %s)" % (self._prop, min)


class LowPass(Filter):

    """Return particles whose property `prop` is less than `max`.
    """

    def __init__(self, prop, max):
        self._descriptor = "lowpass_" + prop

        self._prop = prop
        self._max = max

    def __call__(self, survey):
        max_ = self._max

        prop = self._prop

        return (survey[prop] < max_)

    def __repr__(self):
        max = ('%.2e' % self._max)
        return "LowPass('%s', %s)" % (self._prop, max)

