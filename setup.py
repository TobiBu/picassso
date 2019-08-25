import os
from setuptools import setup, Extension

import numpy
import numpy.distutils.misc_util
import glob
import tempfile
import subprocess
import shutil
import sys

ext_modules = []
libraries=[ ]

if sys.version_info[0:2]==(3,4) :
    # this fixes the following bug with the python 3.4 build:
    # http://bugs.python.org/issue21121
    extra_compile_args.append("-Wno-error=declaration-after-statement")

extra_link_args = []

incdir = numpy.distutils.misc_util.get_numpy_include_dirs()


install_requires = [
    'h5py',
    'matplotlib',
    'numpy>=1.9.2',
    'scipy',
]

tests_require = [
    'tests'
]

docs_require = [
    'ipython>=3',
    'Sphinx==1.6.*',
    'sphinx-bootstrap-theme',
],

extras_require = {
    'docs': docs_require,
    'tests': tests_require,
}

extras_require['all'] = []
for name, reqs in extras_require.items():
    extras_require['all'].extend(reqs)

dist = setup(name = 'picasso',
             author = 'Tobias Buck',
             author_email = 'tbuck@aip.de',
             version = '0.1',
             description = 'Light-weight astronomical image and machine learning framework for python',
             url = '',
             package_dir = {'picasso/': ''},
             packages = ['picasso', 'picasso/analysis', 'picasso/plot',
                         'picasso/survey', 'picasso/galaxy', 'picasso/training' ],

             package_data={'picasso': ['default_config.ini'],#check later what we need
                           'picasso/analysis': ['cmdlum.npz', 
                                                'h1.hdf5',
                                                'ionfracs.npz',
                                                'CAMB_WMAP7',
                                                'cambtemplate.ini'],
                           'picasso/plot': ['tollerud2008mw']},

             ext_modules = ext_modules,
             classifiers = ["Development Status :: 5 - Production/Stable",
                            "Intended Audience :: Developers",
                            "Intended Audience :: Science/Research",
                            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                            "Programming Language :: Python :: 2",
                            "Programming Language :: Python :: 3",
                            "Topic :: Scientific/Engineering :: Astronomy",
                            "Topic :: Scientific/Engineering :: Visualization"],
             install_requires=install_requires,
             tests_require=tests_require,
             extras_require=extras_require,
      )

#if dist.have_run.get('install'):
#    install = dist.get_command_obj('install')
