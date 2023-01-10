#!/bin/bash

# create venv:
VENVNAME=gromenv
virtualenv $VENVNAME

source $VENVNAME/bin/activate

# requirements:
pip install matplotlib
pip install vtk
pip install scipy
pip install dgl
pip install torch
pip install tqdm