# Past and future changes in avalanche danger in northern Norway
This repository contains the code used to preprocess the data and train the machine-learning models for the article Eiselt & Graversen: "Past and future changes in avalanche problems in northern Norway estimated with machine-learning models" submitted to The Cryosphere.

This repository includes the Python package `ava_functions` in the subdirectory `Ava_Functions`. In that subdirectory execute <br>
`pip install .` <br>
to install the package. This is necessary for the other scripts to be functioning.
Before installing the package, the paths of to the data and scripts must be set in `Ava_Functions/ava_functions/Lists_and_Directories/Paths.py`. The introductory comment in this file gives some information about the general directory structure expected by the other scripts.

To build the SNOWPACK model in a container the following instructions were used: https://research-software.uit.no/blog/2023-building-snowpack/

The procedures to generate the predictive feature data are described in the pdf-document `Doc_Gen_Avalanche_Predictors.pdf`.

The procedure to download and prepare the avalanche danger/problem data from the Norwegian avalanche bulletin is described in the pdf-document `Doc_Gen_Modified_AvaProb_Files.pdf`.

The procedure to optimise and train the random forest models is presented in the pdf-document `Doc_Train_ML_Model.pdf`.

The procedure to hindcast and future-project the avalanche-day frequency as well as the generation of the corresponding (and some other) plots are dexcribed in `Doc_Predict_and_Plot.pdf`. 

The models and the data used to train them are published on Zenodo (https://doi.org/10.5281/zenodo.17106819). For the data not available in the Zenodo repository, consult the documentation pdfs and see above.

Note that the scripts do not generally create the output directories themselves. This must either be implemented in the scripts or the output directories must already exist.

Some packages and their versions used: <br>
`matplotlib 3.7.2` <br>
`pandas 2.0.3` <br>
`geopandas 0.12.2` <br>
`seaborn 0.13.1` <br>
`xarray 2022.11.0` <br>
`scikit-learn 1.3.0` <br>
`imbalanced-learn 0.12.3` <br>

Using conda version 25.7.0, functionality was partly reproduced with an environment built (on 18 Sep 2025) with the following command:

`conda create -n ava_env matplotlib pandas geopandas seaborn xarray scikit-learn=1.3.0 imbalanced-learn=0.12.3` 

