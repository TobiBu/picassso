.. picasso documentation master file, created by
   sphinx-quickstart on Sat Aug 24 23:20:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Picassso Documentation
===================================

Picassso is a python 3 toolkit and machine learning framework to operate on astronomical images.
Picassso is an acronym for **P**\ ainting\ **I**\ ntrinsi\ **C** **A**\ ttributes onto\ **S**\ D\ **SS**\  
**O**\ bjects and was designed for but not limited to **image-to-image** translation tasks.
Correspondingly, Picassso contains modules for image data handling, machine learning and analysis.  

In order to get started follow the :ref:`picassso-installation` guide and try the :ref:`tutorials`.


What's next?
------------

Consult the :doc:`installation` documentation for instructions on how
to use Picassso and try out the :ref:`tutorial <survey_manipulation>` which 
explains some of picassso's analysis features.

The full documentation is organized into three sections:

.. toctree::
   :maxdepth: 1

   Installation <installation>
   Tutorial <tutorial/tutorial>
   Reference <reference/index>


All of the information in the reference guide is also available
through the interactive python help system. In ipython, this is as
easy as putting a `?` at the end of a command:

.. ipython::

   In [1]: import picassso

   In [2]: picassso.load?



.. _getting-help:

Seeking Further Assistance
---------------------------


If the tutorials don't answer your question, and you find yourself stuck, 
don't hesitate to contact the author.

We greatly value feedback from users, especially when things are not
working correctly because this is the best way for us to correct
bugs. This also includes any problems you encounter with this documentation.

Feel free to use the `github issue tracker <https://github.com/picassso/picassso/issues>`_ if you
encounter a problem, and create an issue there.

If you use the code regularly for your projects, please consider contributing
your code back using a `pull request
<https://help.github.com/articles/using-pull-requests>`_.

.. _acknowledging-picassso:

Acknowledging Picassso in Scientific Publications
------------------------------------------------

Picassso is an open-source development. We would appreciate if you cite Picasssso via its
`Astrophysics Source Code Library <http://ascl.net/????.???>`_ entry when using it in preparing 
a scientific publication using the following BibTex::

   @misc{picassso,
     author = {{Buck}, T. and {Wolf}, S.,
     title = "{picassso: Painting Intrinsic Attributes onto SDSS objects}",
     note = {Astrophysics Source Code Library, ascl:????.???},
     year = 2020
   }
