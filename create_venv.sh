#!/bin/bash

# create venv:
VENVNAME=gromenv
virtualenv --python=python3 $VENVNAME

source $VENVNAME/bin/activate

# requirements:
pip install matplotlib
pip install vtk
pip install scipy
pip install dgl
pip install torch
pip install sigopt

sigopt config
