#!/bin/bash
conda env create -f gda_env.yml
source activate gda_env
pip install -e . 
