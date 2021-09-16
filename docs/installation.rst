.. summary How to install picassso

.. _picassso-installation:

Picassso Installation
=====================

Great, you decided to give picassso a try! This section shows
you how you install picassso on your machine. If you encounter issues
during the installation process, please let us know right away. 
If you decide you like picassso and you end up using it for your scientific work,
we would appreciate if you contribute your developments for all to use. And we 
are always happy when you acknowledge the code. For this please see
:ref:`acknowledging-picassso`. Enjoy!


In brief
--------

To install the latest release version (which depends on numpy and scipy and 
the `photutils package <https://photutils.readthedocs.io>`_), use

``pip install picassso``

To install from our development branch, use

``pip install git+git://github.com/picassso/picassso.git``

If you have problems or need more help, read on.


.. _install-picassso:

Install picassso
---------------

In your shell try to type:

::

   pip install git+git://github.com/picassso/picassso.git

and everything should happen automatically. This will give you
whatever the latest code from the `git repository <https://github.com/picassso/picassso>`_.

.. note:: If your distutils are not installed properly and you don't have root permissions, this will fail -- see :ref:`distutils`.

If you want to install picassso manually or if you want to develop `picassso` or `pip` is not available on 
your machine proceed as follows.

First, clone the `git repository from Github
<https://github.com/picassso/picassso>`_. picassso uses `git
<http://git-scm.com/>`_ for development:

0. `git` is probably already on your machine -- try typing ``git`` from the shell. If it exists, go to step 2.

1. get the appropriate binary from http://git-scm.com/downloads and install `git`

2. ``$ git clone https://github.com/picassso/picassso.git``

3. to get the newest developments from the repository, run ``git pull``.

4. ``$ cd picassso``

5. ``$ pip install .[all]``

Now the package is installed wherever your python packages reside and should be importable from within python:

6. ``$ cd ~``

7. ``$ python``

8. ``>>> import picassso``

If this yields no errors, you are done!

.. note::
   If you plan on joining the development efforts and you are
   unfamiliar with git, we recommend that you spend some time getting
   familiar with it. The `git documentation <http://git-scm.com/doc>`_
   is quite good and it's worth a read through Chapter 3 on
   branching. You may also choose to `fork the repo
   <https://help.github.com/articles/fork-a-repo>`_ if you already
   have a `github <http://github.com>`_ account.



Upgrading your installation and testing features or bug-fixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use the most recent version from the repository you 
can easily update your installation. We refer you to the upgrading 
your installation section of the 
`picassso package <http://picassso.github.io/picassso/installation.html>`_ which explains 
this procedure quite well.


Load your survey data and start analyzing
-----------------------------------------

Check out the rest of the :ref:`tutorials section <tutorial>` and
especially the :ref:`data-access` to get going.


