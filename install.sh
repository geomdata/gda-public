#!/bin/bash
conda env create -f gda_env.yml
source activate gda_env
# rm -rfv dist/ _skbuild/
# python -m build
# python -m twine upload --user __token__ --password  pypi-XXXXX --repository testpypi  dist/*
pip install -U $(./dependencies.py)
pip install -i . 
