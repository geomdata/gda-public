README for Advanced Installations
=================================

If the Quick Start in `README` didn't work, or if you want to work in an
offline/air-gapped environment, see below.


Online Setup and Testing with Anaconda on Linux/Mac 
---------------------------------------------------

Download and install the 3.5 (or newer) version of Anaconda from https://www.continuum.io/downloads

In the following code examples, suppose you have cloned this repository to 

.. code:: 
    
    /path/to/gda-public

Find the GDA code

.. code::
    
    bash$ cd /path/to/gda-public

Download the packages for numerical computation to ``/path/to/gda-public/downloads``

.. code::

    bash$ python ./download.py
    
Download the packages for visualization (optional!)

.. code::

    bash$ python ./download-viz.py

Setup a basic conda environment (on linux or Mac OSX)

.. code::

    bash$ conda create --name gda_env --offline --channel file:///path/to/gda-public/downloads --file requirements.txt python=3

Enter the environment

.. code::

    bash$ source activate gda_env

To see what has been installed

.. code::

    (gda_env) bash$ conda list 

Make sure you can compile the Cython code

.. code::

    (gda_env) bash$ python setup.py build_ext --inplace --force

Run doctests on all code

.. code::

    (gda_env) bash$ py.test 

*NOTE!*  If tests fails, maybe you compiled previously with another version of Python.  Remove the cached bytecode with

.. code::
    
    (gda_env) bash$ python setup.py clean
    (gda_env) bash$ py3clean -v /path/to/gda-public

*NOTE!*  If tests fail, it might be because your PYTHONPATH is wrong.
On Linux or MacOS, try:

.. code::

    (gda_env) bash$ export PYTHONPATH="$PYTHONPATH:/path/to/gda-public"

Now, play around.  Look at the documentation and the ``examples/``.  If you installed the visualization requirements, you can use jupyter to use the code in a graphical/interactive way.

Leave the environment when you are done working.

.. code::

    (gda_env) bash$ source deactivate
    bash$

Online Setup and Testing with Anaconda on Windows
-------------------------------------------------

To compile the software, you first need a compatible version of Microsoft's 
*Visual Studio Community 2015 for Windows Desktop* from 
https://www.visualstudio.com/en-us/downloads#d-express-windows-desktop

Be sure to enable the *Visual C++* Language option, as well as the *Windows
Development Kit*.

With those packages installed, you can proceed with the Anaconda setup, which is similar to the Linux/MacOS setup. 

Download and install the 3.5 (or newer) version of Anaconda from https://www.continuum.io/downloads

In the following code examples, suppose you have cloned this repository to 

.. code::
    
    C:\abs\path\to\gda-public

Find the GDA code

.. code::

    C:\> cd \abs\path\to\gda-public

Download the packages for numerical computation to ``\abs\path\to\gda-public\downloads``

.. code::

    C:\abs\path\to\gda-public> python download.py
    
Download the packages for visualization (optional!)

.. code::

    C:\abs\path\to\gda-public> python download-viz.py

Setup a basic conda environment

.. code::

    C:\abs\path\to\gda-public> conda create --name gda_env --offline --channel file:///path/to/gda-public/downloads --file requirements.txt python=3

Enter the environment

.. code::

    C:\abs\path\to\gda-public> activate gda_env

To see what has been installed

.. code::

    [gda_env] C:\abs\path\to\gda-public> conda list 

Make sure you can compile the Cython code

.. code::

    [gda_env] C:\abs\path\to\gda-public python setup.py build_ext --inplace

**Note!** If you get an error about ``vcvarsall.bat`` or ``cl.exe``, it means the Visual Studio was not installed correctly, or that you have the wrong version of Visual Studio.  Go back and re-install Visual Studio 2015 (v14.0).

Run doctests on all code

.. code::

    [gda_env] C:\abs\path\to\gda-public> py.test

*NOTE!*  If this fails, it might be because your ``PYTHONPATH`` is wrong.
On Windows, try:

.. code::

    [gda_env] C:\abs\path\to\gda-public> set PYTHONPATH=%PYTHONPATH%;C:\abs\path\to\gda-public

Now, play around.  Look at the contents of ``examples/`` and ``scripts/`` and ``tests/``.  If you installed the visualization requirements, you can use jupyter to use the code in a graphical/interactive way.

Leave the environment when you are done working.

.. code::

    [gda_env] C:\abs\path\to\gda-public> deactivate
    C:\abs\path\to\gda-public>


Offline Setup and Testing with Anaconda
---------------------------------------

To install the software on a computer without internet access, you will need the following saved to a disk:

On Windows, you need Visual Studio Community 2015 (14.0).  The install file is usually called ``vs_community__ENU.exe``.  You can pre-download all the packages for an offline install by running

.. code::

     vs_community_ENU.exe /layout

 For more information on alternate installs of Visual Studio, see https://msdn.microsoft.com/en-us/library/e2h7fzkw(v=vs.140).aspx#bkmk_offline
 
 For all platforms, you need:
 
 - The newest Anaconda installer file, from https://www.continuum.io/downloads
 - A clone of this repository, say at ``/path/to/gda-public``
 - The ``/path/to/gda-public/downloads`` folder, copied from an *online* machine after running the above procedure there.

The total is about 600 MB. With all these files copied to the offline machine, follow the procedure for an online machine, but skip the ``python ./download.py`` and ``python ./download-viz.py`` steps.


Installation
------------

If you intend to *use* the code in a larger pipeline (as opposed to developing
and tinkering in the repository itself), you can install the packages to your
environment in the following way.


Enter the environment

.. code::
    
    # Using Anaconda
    bash$ source activate gda_env

Run the commands above in the appropriate section above first to make sure things work!  

Then, install the package to your environment.

.. code::

    (gda_env) bash$ pip install file:///path/to/gda-public

Change directory to somewhere else, just to be sure import works globally

.. code::

    (gda_env) bash$ cd /tmp

Then, run python or ipython/jupyter and import the modules you want:

.. code::
        
    (gda_env) bash$ python
    >>> import homology
    >>> help(homology)

When you are done doing work, you can leave the environment: 

.. code::

    # Using Anaconda
    (gda_env) bash$ source deactivate
    # Using Python.org
    (gda_env) bash$ deactivate
    
    bash$ 

Porting Changes
---------------

Suppose you have your own copy of this repo, and you want to copy in changes
in-bulk, but a merge request is not possible because they are on different
platforms.

.. code::

    repo_old> git format-patch --root  # produce all changes from history

.. code::

    repo_new> go to head, update, and remove *all* working files
    repo_new> ls /path/to/repo_old/\*.patch | sort -n  | while read r; do echo $r; [ $(ls -s $r | cut -d\  -f1) -gt 0 ] && git apply --whitespace=nowarn $r; done
    repo_new> build your code and run your tests
    repo_new> git add -A
    repo_new> git commit -m "imported to version XXXXXXX from internal repo." -a

