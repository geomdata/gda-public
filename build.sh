#!/bin/bash
conda env create -f gda_env.yml
source activate gda_env

rm -rfv dist/ gda_public.egg-info/ _skbuild/

pip install $(./ppinfo.py -d build)


# set up the source, to allow users to build from pip
python -m build --sdist
# Build the wheel.  NOTE! `manylinux`_x86_64` is necessary.  linux and manylinux are NOT OK.
# https://peps.python.org/pep-0513/#platform-detection-for-installers
python setup.py bdist_wheel --plat-name manylinux1_x86_64

echo "type 'upload' to proceed"
read go
if [ x"${go}" == "xupload" ]; then
	pip install twine
	python -m twine upload --user __token__ --password  ${PYPI_PASSWORD}  dist/* --verbose
fi

