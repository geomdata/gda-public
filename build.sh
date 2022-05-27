#!/bin/bash
rm -rfv dist/ gda_public.egg-info/ _skbuild/ 

conda env create -f gda_env.yml || exit 1 
conda activate gda_env || exit 1
pip install $(./ppinfo.py -d build) || exit 1

# set up the source, to allow users to build from pip
python -m build --sdist || exit 1
# Build the wheel.  NOTE! `manylinux`_x86_64` is necessary.  linux and manylinux are NOT OK.
# https://peps.python.org/pep-0513/#platform-detection-for-installers
python setup.py bdist_wheel --plat-name manylinux1_x86_64 || exit 1
conda deactivate || exit 1


# Gary needs an older version for numba compatibility
conda create -n gda_env_py38 python=3.8 numpy\<1.22 pip setuptools || exit 1
conda activate gda_env_py38 || exit 1
pip install $(./ppinfo.py -d build) || exit 1 
python setup.py bdist_wheel --plat-name manylinux1_x86_64 || exit 1


echo "type 'upload' to proceed"
read go
if [ x"${go}" == "xupload" ]; then
	pip install twine
	python -m twine upload --user __token__ --password  ${PYPI_PASSWORD}  dist/* --verbose
fi

conda deactivate
conda env remove -n gda_env
conda env remove -n gda_env_py38
