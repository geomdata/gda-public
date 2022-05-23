# set up the source, to allow users to build from pip
python -m build --sdist
# Build the wheel.  NOTE! `manylinux`_x86_64` is necessary.  linux and manylinux are NOT OK.
# https://peps.python.org/pep-0513/#platform-detection-for-installers
#python setup.py bdist_wheel --plat-name manylinux1_x86_64


# this is how I was testing it
conda create -y -n testgda python=3; conda activate testgda;  pip install -v  -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple  gda-public==1.5;  python -c 'from homology.dim0 import mkforestDBL'; pip uninstall -y gda-public; conda deactivate; conda env remove -y -n testgda;  find .cache/ -name "gda_public-*.whl"  -delete
