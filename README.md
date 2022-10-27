# melting_point_prediction_paper

Paper written with streamlit

Only the files and scripts required for publication are included in this repo.

Virtual Environment Information:
    * Make sure you have conda installed on your computer and up to date
    * Following these instructions: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/
    * In the terminal, type: conda create -n mppredictionpapervenv python=3.7 anaconda
        * You can replace mppredictionpapervenv with whatever you want your virtual environment to be called
    * To activate virtual environnment, in the terminal, type: conda activate mppredictionpapervenv
        * You should see (mppredictionpapervenv) in front of your command line input now
    * When in vscode, make sure you select the right interpreter by clicking on Python X.X.X in the bottom left corner, then selecting the (mppredictionpapervenv) interpreter from the top menu. You can also navigate to this by pressing command+shift+p and typing interpreter -> select interpreter -> choose the one in the virtual environment. You might need to restart VSCode to get it to show up.

* Python 3.7.9 64-bit
* Packages to install (with commands):
    * conda install -c rdkit rdkit
    * conda install -c mordred-descriptor mordred

All other required packages are installed as dependencies of the above packages. These packages are all that's needed for both Wunmi's automated physics-based model generation script and Antonio's machine learning model using Reaxys data.

Repository Structure:
* Archive:
    * Old plots, errors, and parameters from previous model iterations. No longer used
* Data Files:
    * Entropy and Volume Data - Hydrocarbons.csv (Thermodynamic Features for Thermodynamic Model for Hydrocarbons)
    * Entropy and Volume Data - Hydroquinones.csv (Thermodynamic Features for Thermodynamic Model for Hydroquinones)
    * Entropy and Volume Data - Quinones.csv (Thermodynamic Features for Thermodynamic Model for Quinones)
    * featurized_bq.csv (Mordred Features for ML Models for Quinones)
    * featurized_hq.csv (Mordred Features for ML Models for Hydroquinones)
    * parsed_p_benzoquinone_216.csv (Reaxys data for Quinones)
    * parsed_p_hydroquinone_204.csv (Reaxys data for Hydroquinones)
* Errors:
    * csv files for errors (AAEs and RMSEs) for all datasets and model structures
* Parameters:
    * csv files for parameters/coefficients for all datasets and model structures
* Plots:
    * png files flots for all datasets and model structures. Plots are from the last run of the model (last random train - test split)
* mp_prediction_models.ipynb: 
    * main notebook for analysis. Model functions are defined in a cell and can be called as desired in subsequent cells
