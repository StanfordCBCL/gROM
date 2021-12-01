#!/bin/bash

export OMP_NUM_THREADS=3

if [ "$#" -ne 2 ]; then
    echo "Usage: arg#1 = experiment id, arg#2 = num workers"
fi

export EXPCODE=$1
export NUMWORKERS=$2

for i in {1..$NUMWORKERS}
do
    ( sigopt start-worker $EXPCODE python training_sigopt.py 0111_0001 > $i.txt & )
done
