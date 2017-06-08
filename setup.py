r"""
Basic setup/install script, customized for Cython

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
from sphinx.setup_command import BuildDoc
import shutil

name = 'gda-public'
version = '1.0.0'
release = 'stable'


class MyBuildDoc(BuildDoc):
    """ 
    A copy of sphinx's BuildDoc command, which knows to build Cython first,
    and to move the documentation afterward
    """
    sub_commands = [('build_ext', lambda self:True), ]

    def run(self):
        self.inplace = True  # need the --inplace option
        for cmd_name in self.get_sub_commands():
            print(cmd_name)
            cmd = self.get_finalized_command(cmd_name)
            cmd.inplace = self.inplace
            self.run_command(cmd_name)
        BuildDoc.run(self)
        shutil.rmtree('docs')
        shutil.copytree('doc_build/html', 'docs', symlinks=True)
        shutil.rmtree('doc_build/html')
        with open("docs/.nojekyll", "w") as f:
            f.write("This file fixes documentation on GitHub.")


cmdclass = {'build_doc_html': MyBuildDoc, 'build_doc_latex': MyBuildDoc}

setup(
    name=name,
    version=version,
    description='Tools for topological data analysis',
    author='Geometric Data Analytics, Inc.',
    author_email='abraham.smith@geomdata.com',
    url='https://www.geomdata.com/',
    packages=['homology', 'timeseries', 'multidim' ],
    ext_modules=cythonize(['multidim/fast_algorithms.pyx', 
                           'homology/dim0.pyx',
                           'homology/dim1.pyx',
                           'timeseries/curve_geometry.pyx', 
                           'timeseries/fast_algorithms.pyx']),
    include_dirs=[get_include()],
    requires=['numpy', 'pandas', 'scipy', 'sphinx', 'cython',
              'docopt', 'numpydoc', 'pytest', 'sklearn'],
    cmdclass=cmdclass,
    command_options={  # also see setup.cfg
        'build_doc_html': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            },
        'build_doc_latex': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            },
        }
)
