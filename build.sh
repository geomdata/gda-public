#!/bin/bash
rm -rfv dist/ gda_public.egg-info/ _skbuild/ 

mamba env create -f gda_env.yml  
source activate gda_env 
pip install $(./ppinfo.py -d build) 

# set up the source, to allow users to build from pip
python -m build --sdist 
# Build the wheel.  NOTE! `manylinux`_x86_64` is necessary.  linux and manylinux are NOT OK.
# https://peps.python.org/pep-0513/#platform-detection-for-installers
python setup.py bdist_wheel --plat-name manylinux1_x86_64 
conda deactivate 

# Gary needs an older version for numba compatibility
mamba env create -f gda_env38.yml
source activate gda_env38 
pip install numpy\<1.22 
pip install $(./ppinfo.py -d build)  
python setup.py bdist_wheel --plat-name manylinux1_x86_64 


echo "type 'upload' to proceed"
read go
if [ x"${go}" == "xupload" ]; then
	pip install twine
	python -m twine upload --user __token__ --password  ${PYPI_PASSWORD}  dist/* --verbose
fi

source deactivate
mamba env remove -n gda_env
mamba env remove -n gda_env38
