#!/bin/bash
# jupyter nbconvert --to script $1.ipynb
# python $1.py
# rm $1.py
if [ -d ../source/$1_files ]; then
    rm -rf ../source/$1_files
fi
if [ -f ../source/$1.rst ]; then
    rm -rf ../source/$1.rst
fi

jupyter nbconvert --to rst --execute --ExecutePreprocessor.timeout=600 $1.ipynb
mv $1.rst ../source/
if [ -d $1_files ]; then
    mv $1_files ../source/
fi
