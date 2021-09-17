#!/bin/bash
# jupyter nbconvert --to script $1.ipynb
# python $1.py
# rm $1.py
if [ -d ../$1_files ]; then
    rm -rf ../$1_files
fi
if [ -f ../$1.rst ]; then
    rm -rf ../$1.rst
fi

jupyter nbconvert --to rst --execute --ExecutePreprocessor.timeout=600 $1.ipynb
mv $1.rst ../
if [ -d $1_files ]; then
    mv $1_files ../
fi
# pastille colab:
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/chemin_vers_le_notebook.ipynb]
