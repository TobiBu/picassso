.. data_access tutorial

.. _data-access:

A walk through picassso's features
==================================

The following examples show how to load survey data either represented
by a bunch of files or a single file. We further exemplify how to 
determine various attributes and access individual data.

If you're more interested in making pretty pictures and plots straight
away, you may wish to jump to the end of this section.


First steps
-----------

.. note:: Before you start make sure `picassso` is properly
 installed. See :ref:`picassso-installation`
 for more information. You will also need the standard `picassso` test
 files if you want to follow the tutorial.
 These files are available separately here:
 <https://github.com/picassso/picassso/releases>
 (`testdata.tar.gz`).

After you have extracted the testdata folder (e.g. with ``tar -xzf
testdata.tar.gz``), launch `ipython`. At the prompt, type ``import
picassso``. If everythin is properly installed, this should
succeed, and you are ready to use `picassso` commands. Here's an
example.

.. ipython::

 In [1]: import picassso

 In [2]: s = picassso.load("./testdata/")

This loads a sample of Illustris images. 
Calling the function :func:`picassso.load`, will store a link
to the whole content of the folder in the variable ``s`` which is 
now your calling point for accessing the data.

In fact ``s`` is an object known as a `Survey` (see
:class:`picassso.survey.Survey`).

Behind the scenes, the function inspects the provided path and decides
what file/data format is present. So far only custom formats for 
Illustris images is supported but extension to any other file format
is easily achieved by providing the appropriate backend in a similar
fashion as for the `SDSSMockSurvey` (see :class:`picassso.survey.SDSSMockSurvey`).
This kind of flexibility is the whole point of the `picassso` framework.

.. note:: If you look inside the `testdata` folder, you'll notice that
  the content of this folder contains several single files for each 
  object as well as a subfolder for the corresponding predictions.
  `picassso` knows how to load and connect both the true maps and the
  predictions as well as all auxilliary files present. Under the hood
  `picassso` will concatenate all files belonging to the same object
  into one single representation of that object and all those 
  reprensentations into a single survey object.

The `Survey` object that's returned is currently a fairly empty
holder for the data. The data will only be loaded from disk as you
request it. However, this object knows already about the total number 
of files present in the `Survey` and we can already get some global
information by typing:

.. ipython::

 In [1]: print(s)

 Out[2]: <Survey "./test_data/" len=21>

Similarly the standard python operator ``len`` can be used to query the number
of files in the `Survey`:

.. ipython::

 @doctest
 In [3]: len(s)
 Out[3]: 21

Finding out something more about the `Survey`
---------------------------------------------

Let's start to inspect the `Survey` we've opened.

In machine learning a standard procedure is to split the dataset into
sub-parts used for training, testing and validation. We have implemented
this dataset splitting by separating the entities of each subset into
different `families` in picassso. To find out which families are present
in the survey, use :func:`~picassso.survey.Survey.families`:

.. ipython::

 In [3]: s.families()
 Out[3]: [<Family training>, <Family testing>, <Family validation>]

You can pick out just the objects belonging to a family by using the
syntax ``s.family``. So, for example, we can see how many objects of
each type are present:


.. ipython::

 @doctest
 In [4]: len(s.training)
 Out[4]: 21

 @doctest
 In [5]: len(s.testing)
 Out[5]: 0

 @doctest
 In [6]: len(s.validation)
 Out[6]: 0

Actually in this very first release of `picassso` the familiy slicing is not 
implemented.


Useful information about the survey is stored in a python dictionary
called `properties`:

.. ipython::

 In [7]: s.properties

Like any python dictionary, specific properties can be accessed by
name:

.. ipython::

 In [8]: s.properties['image_res']

These names are standardized across different file formats. Here for example `image_res`
means the number of pixels along one image axis.

.. note:: Actually ``s.properties`` has some behaviour which is
 very slightly different from a normal python dictionary. For further
 information see :class:`~picassso.surveydict.SurveyDict`.


Retrieving data
---------------

Like ``s.properties``, ``s`` itself also behaves like a python
dictionary. The standard python method
``s.``:func:`~picassso.survey.Survey.keys` returns a list of arrays
that are currently in memory.

.. ipython::

  In [9]: s.keys()
  Out[9]: []

Right now it's empty! That's actually correct because data is only
retrieved when you first access it. To find out what `could` be loaded,
use the `picassso`-specific method
``s.``:func:`~picassso.survey.Survey.loadable_keys`.

.. ipython::

  In [10]: s.loadable_keys()
  Out[10]: ['stars_Masses','stars_GFM_Metallicity','galaxy','stars_GFM_StellarFormationTime']

This looks a bit more promising.
To access data, simply use the normal dictionary syntax. For example
``s['galaxy']`` returns an array containing the galaxy objects of the
survey.

.. ipython::

 In [11]: f['galaxy']
 Out[11]:
 SurveyArray([<Galaxy 12c1>, <Galaxy 18c0>, <Galaxy 31c0>, <Galaxy 1c2>,
             <Galaxy 24c0>, <Galaxy 18c1>, <Galaxy 12c0>, <Galaxy 1c3>,
             <Galaxy 31c1>, <Galaxy 24c1>, <Galaxy 18c2>, <Galaxy 12c3>,
             <Galaxy 24c2>, <Galaxy 1c0>, <Galaxy 31c2>, <Galaxy 12c2>,
             <Galaxy 18c3>, <Galaxy 37c0>, <Galaxy 24c3>, <Galaxy 31c3>,
             <Galaxy 1c1>], dtype=object)



.. note::

 So far only the galaxy key is accessable for `Survey` arrays.
 the other loadable keys are only accessable through single galaxy
 objects.

To access data which is only present for single galaxy objects, 
you need to use the following syntax. For example
``s['galaxy'][0].properties['stars_Masses']`` returns an array 
containing the stellar mass map of the galaxy object ``s['galaxy'][0]``.

.. ipython::

 In [12]: s['galaxy'][0].properties['stars_Masses']
 Out[12]:
array([[      0.        ,       0.        ,       0.        , ...,
              0.        ,       0.        ,       0.        ],
       [      0.        ,       0.        ,       0.        , ...,
              0.        , 1906931.37786104,       0.        ],
       [      0.        ,       0.        ,       0.        , ...,
              0.        , 1471050.03777746, 1717792.46115242],
       ...,
       [      0.        ,       0.        , 1279748.00653322, ...,
              0.        , 1367954.93295583,       0.        ],
       [      0.        ,       0.        ,       0.        , ...,
              0.        , 1668024.45279829,       0.        ],
       [      0.        , 2158359.09779766,       0.        , ...,
              0.        ,       0.        ,       0.        ]])


Some arrays can actually be derived for the whole survey and stored separately 
as survey arrays. For example, it is useful to derive an array storing the total
stellar mass of a galaxy object and make this separately accessable from the 
galaxy object. Which derivable arrays are present can be inspected via
``s.``:func:`~picassso.survey.Survey.derivable_keys`:

.. ipython::

 In [13]: s.derivable_keys()
 Out[13]: ['total_star_mass', 'geom', 'iso_dict', 'M2Rh']


 In [14]: s['total_star_mass']
 Out[14]:
 SurveyArray([1.43720477e+11, 1.93832207e+11, 2.64063622e+10,
             3.79995081e+11, 8.49663794e+10, 1.78271390e+11,
             1.48894226e+11, 4.56245418e+11, 2.57942480e+10,
             8.15334328e+10, 1.22176095e+11, 1.51387723e+11,
             8.53532336e+10, 4.28729492e+11, 2.71191629e+10,
             1.42284708e+11, 1.81036331e+11, 5.63070103e+10,
             8.41919264e+10, 2.69313124e+10, 4.10429199e+11])


.. _create_arrays :

Creating your own arrays
------------------------

`Picassso` has the feature of creating your own arrays 
using the obvious assignment syntax:

.. ipython::

  In [15]: s['twicemass'] = s['total_star_mass']*2

You can also define new arrays for one family of particles:

.. ipython::

  In [14]: s.train['myarray'] = s.train['total_star_mass']**2

An array created in this way exists *only* for the training
objects; trying to access it for other objects raises an
exception.

Alternatively, as we have seen befor, there exist *derived arrays*.
You can define those *derived arrays* which are calculated on demand
via thew following syntax:

.. ipython::

  In [15]: @picassso.derived_array
     ...: def thricethemass(s) :
     ...:     return s['total_star_mass']*3
     ...:


At this point, nothing has been calculated. However, when you ask for
the array, the values are calculated and stored

.. ipython::

  In [4]: s['thricethemass']

This has the advantage that your new `thricethemass` array is
automatically updated when you change the `total_star_mass` array:

.. ipython::

  In [4]: s['total_star_mass'][0] = 1

  In [6]: s['thricethemass']

Note, however, that the array is not re-calculated every time you
access it, only if the `total_star_mass` array has changed. 
Therefore you don't waste any time by using derived arrays. 
For more information see the reference documentation 
for :ref:`derived arrays <derived>`.



Where next?
-----------

This concludes the tutorial for basic use of `picassso`. Further
:ref:`tutorials <tutorials>` for specific tasks are available.
