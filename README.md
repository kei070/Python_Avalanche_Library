# Past and future changes in avalanche danger
This repository contains the code used to preprocess the data and train the machine-learning models for the article Eiselt & Graversen: "Past and future changes in avalanche problems in northern Norway estimated with machine-learning models" submitted to The Cryosphere.

This repository includes the Python package `ava_functions` in the subdirectory `Ava_Functions`. In that subdirectory execute <br>
`pip install .` <br>
to install the package. This is necessary for the other scripts to be functioning.
Before installing the package, the paths of to the data and scripts must be set in `Ava_Functions/ava_functions/Lists_and_Directories/Paths.py`. The introductory comment in this file gives some information about the general directory structure expected by the other scripts.

Some packages and their versions used: <br>
`scikit-learn 1.5.1` <br>
`imbalanced-learn 0.12.3` <br>
