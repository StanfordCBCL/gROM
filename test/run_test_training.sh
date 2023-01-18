#!/bin/bash

set -e

source gromenv/bin/activate 
python test/test_training.py --epochs 3