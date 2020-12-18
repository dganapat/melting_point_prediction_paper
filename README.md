# melting_point_prediction_paper

Paper written with streamlit
Only the files and scripts required for publication included 

I'm making this repository to clean up all the code that we use to write our paper. If we use streamlit we can integrate our paper with our plots and the code for the plots so everything is all in one place and can be updated live. This repository will also be better organized so that we only have the plots and the scripts that we need. We can choose to show snippets of our code in the paper if we want, and also this will upload well as the SI for our paper when we submit it.

Virtual Environment Information:
    Make sure you have conda installed on your computer and up to date
    Following these instructions: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/
    In the terminal, type: conda create -n mppredictionpapervenv python=3.7 anaconda
        You can replace mppredictionpapervenv with whatever you want your virtual environment to be called
    To activate virtual environnment, in the terminal, type: conda activate mppredictionpapervenv
        You should see (mppredictionpapervenv) in front of your command line input now
    When in vscode, make sure you select the right interpreter by clicking on Python X.X.X in the bottom left corner, then selecting the (mppredictionpapervenv) interpreter from the top menu. You can also navigate to this by pressing command+shift+p and typing interpreter -> select interpreter -> choose the one in the virtual environment. You might need to restart VSCode to get it to show up.

Python 3.7.9 64-bit
Packages to install (with commands):
    pip install streamlit
    conda install -c rdkit rdkit
    conda install -c mordred-descriptor mordred

All other required packages are installed as dependencies of the above packages. These packages are all that's needed for both Wunmi's automated physics-based model generation script and Antonio's machine learning model using Reaxys data.

Once you've installed all the packages you need in your virtual environment, in the terminal in vscode streamlit run mp_prediction_paper.py. A link should be outputted which you can open in a browser window. The easiest way to edit and  view your code is by split-screening your coding editor (ex. vscode) and the streamlit browser. If you make changes in your script you can refresh the streamlit window (ex. cmd+r) or clicking rerun in the top right corner. Pressing run in your coding editor won't do anything in streamlit.

Notes:
Some blocks of code are hidden inside "regions". To open them up, just click on the little arrow next to where you see the code #region...