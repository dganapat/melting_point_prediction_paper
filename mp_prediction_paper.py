
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import statistics
from sklearn.linear_model import LinearRegression
import re
import matplotlib.backends.backend_pdf
import os
import time
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from matplotlib import gridspec

## STATIC BLOCK - DON'T MODIFY ANYTHING BELOW HERE ##
#region
# Helper Functions that will be used 
def findstr(test_str,test_sub):
    res = [i for i in range(len(test_str)) if test_str.startswith(test_sub, i)]
    return res
    
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def split_data(dataset):
    dataset=dataset.sample(frac=1)
    length= len(dataset)
    eightypercent= int(length*0.8)
    trainingset= dataset[:eightypercent]
    testset=dataset[eightypercent:]
    return trainingset, testset

def fit_tm_model(predictors,*parameter):
    parameters= parameter
    return eval(model_form)

def fit_tm_model_err(predictors,*parameter):
    parameters= parameter[0]
    return eval(model_form)

def make_plots(dataset_test,dataset_train,letters_in_use, dataset_name,avg_model_err,rmse_err):
    Tbuffer= 25
    lowerT =min ( min(dataset_test['T_m (K)']),min(dataset_train['T_m (K)'])
    ,min(fit_tm_model_err(dataset_test,letters_in_use)),
    min (fit_tm_model_err(dataset_train,letters_in_use)))
    lowerT=lowerT- Tbuffer

    higherT =max( max(dataset_test['T_m (K)']),max(dataset_train['T_m (K)'])
    ,max(fit_tm_model_err(dataset_test,
    letters_in_use)),max(fit_tm_model_err(dataset_train,letters_in_use)))
    higherT=higherT+Tbuffer

    fig, ax=plt.subplots()
    ax.plot(dataset_train['T_m (K)'],fit_tm_model_err(dataset_train,
    letters_in_use),'ko')
    ax.plot(dataset_test['T_m (K)'],fit_tm_model_err(dataset_test,
    letters_in_use),'ro')
    ax.plot([lowerT,higherT],[lowerT,higherT],color=((0.6,0.6,0.6)))
    ax.set_ylabel('Predicted $T_m$ (K)')
    ax.set_xlabel('Experimental $T_m$ (K)')
    #ax.set_title('Calculated vs Experimental $T_m$ for '+ model_form_name + ' - ' + dataset_name +'\n Test Error: ' + str(avg_model_err[0])+ '\n Train Error: ' + str(avg_model_err[1]) + '\n RMSE Test:'+ str(rmse_err[0])+ '\n RMSE Train:'+str(rmse_err[1]))
    ax.legend(('Training Set','Test Set'))
    ax.set_xlim([lowerT,higherT])
    ax.set_ylim([lowerT,higherT])
    return fig

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Import data files for physics-based model into pandas dataframes
quinone_data = pd.read_csv("Entropy and Volume Data - Quinones.csv")
hydroquinone_data= pd.read_csv("Entropy and Volume Data - Hydroquinones.csv")
hydrocarbon_data= pd.read_csv("Entropy and Volume Data - Hydrocarbons.csv")
mega_database=pd.concat([hydroquinone_data,quinone_data])[['sigma','tau','V_m (nm3)','T_m (K)','Eccentricity(Ear)','Eccentricity(Eal)']].reset_index(drop=True)

# Add to this array if new models need additional parameters
all_possible_predictors= ['tau', 'V_m (nm3)', 'sigma','Eccentricity(Eal)', 'Eccentricity(Ear)']

letters_in_use= ['a','b','c','d','f','g','h','k','l','m']
predicted= ['T_m (K)']

#endregion
## STATIC BLOCK - DON'T MODIFY ANYTHING ABOVE HERE ##

MAKING_NEW_PLOTS=True

if MAKING_NEW_PLOTS:
    ## EDIT BELOW HERE
    ### Change datasets used, model form, starting guesses
    #region
    # Change the datasets that you're interested in looking at in this block. Make sure you change the names of the datasets appropriately. Note: All datasets you include here will be tested with the same model form. If you want to test different model forms for different datasets, you will have to test one dataset at a time and change the model form as desired for that single dataset.
    datasets= [hydrocarbon_data]
    dataset_names= ['Hydrocarbon']
    num_datasets= len(datasets)

    # CHANGE MODEL NAME AND FORM HERE:
    model_form_name= '$V_m^{-2}$ Numerator, Full Denominator'
    model_form= '(parameters[0]*predictors["V_m (nm3)"]**(-2)+parameters[1])/(parameters[2]*np.log(predictors["sigma"])+parameters[3]*predictors["tau"]+1+parameters[4]*np.log(predictors["Eccentricity(Ear)"])+parameters[5]*np.log(predictors["Eccentricity(Eal)"]))'

    # CHANGE STARTING GUESSES HERE
    # Note: You must have the correct number of starting guesses to match the number of parameters in the model form, and you must also have the correct number of sets of starting guesses depending how many datasets you're testing at once.
    starting_guesses= [[-4.32603500e+00,2.87801207e+02,-7.93904846e-02,1.22626036e-02,-9.56732927e-02,-4.24685604e-02]]
    # HQ starting guesses: ,[-2.46237655e+01,6.04795762e+02,-6.14060740e-02,4.49395401e-02,3.03647346e-02,3.38981530e-02],[-1.83846686e+01,2.34877509e+02,-1.21714771e-01,-1.30258253e-02,-1.14720566e-01,-1.09430299e-01]

    #endregion
    ## EDIT ABOVE HERE

    ## STATIC BLOCK - DON'T MODIFY ANYTHING BELOW HERE ##
    #region
    num_predictors= list(range(len(findstr(model_form, 'predictors'))))
    used_predictors= []

    for i in range(len(all_possible_predictors)):
        this_predictor= all_possible_predictors[i]
        if findstr(model_form, this_predictor):
            used_predictors.append(this_predictor)
    used_predictors= used_predictors+predicted
    num_parameters= len(findstr(model_form, 'parameter'))

    if num_parameters!=len(starting_guesses[0]):
        raise Exception("Number of starting guesses doesn't match the number of parameters")

    letters_in_use=letters_in_use[0:num_parameters]
    num_parameters= list(range(num_parameters))

    plots=list(range(num_datasets))
    fig = plt.figure()      
    for i in range(num_datasets):
        avg_model_err=[0,0]
        rmse_err=[0,0]
        number_of_runs=5
        count=0
        while count < number_of_runs:
            dataset= datasets[i]
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            dataset_name=dataset_names[i]
            dataset_length= len(dataset)
            predictors= dataset[used_predictors].loc[2:dataset_length]
            predictors= predictors.astype('float64')
            [dataset_train,dataset_test]=split_data(predictors)
            (letters_in_use,_)=opt.curve_fit(fit_tm_model,dataset_train,dataset_train['T_m (K)'],(starting_guesses[i]))
            
            if statistics.mean(np.absolute(fit_tm_model_err(dataset_test,letters_in_use)
            -dataset_test['T_m (K)']))/(number_of_runs)<(70/number_of_runs) and letters_in_use[0]<0:
                #print(letters_in_use)
                avg_model_err[0]=avg_model_err[0]+(statistics.mean(np.absolute(fit_tm_model_err
                (dataset_test,letters_in_use)-dataset_test['T_m (K)']))/(number_of_runs))
                avg_model_err[1]=avg_model_err[1]+(statistics.mean(np.absolute(fit_tm_model_err
                (dataset_train,letters_in_use)-dataset_train['T_m (K)']))/(number_of_runs))
                count=count+1
                rmse_err[0]=rmse_err[0]+rmse((fit_tm_model_err(dataset_test,letters_in_use)),dataset_test['T_m (K)'])/(number_of_runs)
                rmse_err[1]=rmse_err[1]+rmse((fit_tm_model_err(dataset_train,letters_in_use)),dataset_train['T_m (K)'])/(number_of_runs)

        print(rmse_err)
        ax= (make_plots(dataset_test,dataset_train,letters_in_use,dataset_name,avg_model_err,rmse_err))
        
        plots[i]= ax

    
#endregion
    ## STATIC BLOCK - DON'T MODIFY ANYTHING ABOVE HERE ##



r'''
# Melting Point Prediction for Quinones and Hydroquinones

## Devi Ganapathi, Wunmi Akinlemibola, Antonio Baclig, Emily Penn, and William C Chueh

### * Department of Materials Science and Engineering, Stanford University, Stanford, CA, 94305 *

### Email: wchueh@stanford.edu

### Phone: (650) 725-7515

'''
# Abstract
r'''
## Abstract

In this study, two different computational approaches were developed to predict the melting points of quinone and hydroquinone-based molecules. A traditional machine learning approach was used to calculate features and generate a model using a ridge regression. A more fundamental, thermodynamics-based model that utilizes volume-based thermodynamics to describe the enthalpy of fusion and previously published equations to capture the entropy of melting was also developed. Different functional forms of the enthalpy term were also tested and compared to the original physics-based model to determine whether the accuracy of the model was  based on scientific understanding. The machine learning-based model resulted in errors of X for the quinone dataset and Y for the hydroquinone dataset. The physics-based model resulted in test set and training set errors of Y for the quinone set and Z for the hydroquinone set.

'''
# Introduction
'''
## Introduction

Quinone- and hydroquinone-based molecules have gathered attention recently as electrolyte candidate molecules for redox flow batteries, among other applications\cite{Shimizu2017,Kwabi2018,Goulet2019}. Our group has recently proposed using a eutectic mixture of benzoquinone-based molecules as a high energy density positive electrolyte for flow batteries that remains liquid at room temperature. In order to identify promising materials for this application, knowing the melting temperatures of the quinone and hydroquinone molecules is essential. With the model we have developed, the melting points of the pure component quinones and hydroquinones can be used to predict the melting point of a eutectic mixture. However, melting data is not available for all the quinones and hydroquinones of interest. To address this challenge, we developed a computational model that can be used to predict the melting points of quinone and hydroquinone molecules that are not available in literature. 

Many melting point prediction models employ group contribution method (GCM). This is an additive method that works by summing the contributions from all the various groups in a molecule (hence "group contribution method"). However, this method does not account for interactions between groups \cite{JOBACK1987}.

Our model utilizes simple molecular descriptors that can be calculated from the two-dimensional structure of the model from a semi-empirical model previously proposed by Dannenfelser and Yalkowsky\cite{DannenfelserEntropy}. The model also uses molecular volume data, which can be calculated using crystal structure data or density measurements, or predicted computationally\cite{Day2011}. The molecular volume data contains information about the strength of the interactions between molecules in the solid phase. This approach provides an advantage over GCM in that the molecular volume in the solid phase of a species inherently accounts for the interactions between molecules.

'''
# Methods
r'''
## Methods

### Machine Learning Model

### Thermodynamics-Based Model

We begin with the fundamental thermodynamic equation for melting temperature: 
$$
T_m=\frac{\Delta H_m}{\Delta S_m}
$$

From here, we sought to find equations that would describe the enthalpy and entropy of melting using descriptors that could be easily calculated from the structure of the molecule or from experimental data.

#### Entropy of Melting

We use the equation developed by Dannenfelser and Yalkowsky\cite{DannenfelserEntropy} to estimate the entropy of melting:
$$
\Delta S_m= a*\textrm{ln}\sigma + b*\tau + c
$$
where $a,b,$ and $c$ are adjustable parameters to be fit with a non-linear least squares optimization function. 

The first descriptor $\sigma$, which is the molecular symmetry number, is the number of unique rotations that can be performed on the molecule and return an indistinguishable orientation, in addition to the identity. At minimum, $\sigma$ must be 1. As sigma increases, the number of microstates (orientations of the molecule) that produce the same crystal structure increases, thus increasing the overall entropy of the solid (according to the Boltzmann equation, $\Delta S = k_B ln\Omega$. This results in a lower entropy of melting (difference between the entropy of the solid and liquid phases decreases).

The second descriptor, $\tau$, is the number of torsional angles in the molecule and can be calculated using the formula\cite{DannenfelserEntropy}:

$$
\tau = \textrm{SP3} + 0.5(\textrm{SP2}+\textrm{RING}) - 1
$$

Here, SP3 is the number of $\textrm{sp}^3$ chain atoms (not including end carbons), SP2 is the number of $\textrm{sp}^2$ chain atoms (also not including end carbons), and RING is the number of fused-ring systems. As the effect of tau on the entropies of the solid and liquid phases varies depending on the class of molecules, it is difficult to say whether it increases or decreases the entropy of melting.

This model was updated by Lian and Yalkowsky to include two more descriptors for entropy of melting in the Unified Physiochemical Property Estimation Relationships (UPPER) method \cite{LianUPPER2013}. These additional descriptors are 1) aromatic eccentricity - the number of atoms in aromatic rings, and 2) aliphatic eccentricity - the number of atoms in aliphatic (non-aromatic) rings. These terms capture the tendency of flat or elongated molecules to be partially ordered in the liquid (which decreases the change in entropy between the solid and liquid phase, thus decreasing the entropy of melting). Thus, the final form of the equation we used to model the entropy of melting was:

$$
    \Delta S_m= a*\textrm{ln}\sigma + b*\tau + c*\textrm{ln}\epsilon_{ar} + d*\textrm{ln}\epsilon_{al} + f
$$

#### Enthalpy of Melting

For the enthalpy of melting, we use volume-based thermodynamics (VBT), a physics-based approach for simply predicting thermodynamic properties of compounds. Rather than adding contributions from all the side groups on the molecule, a single property, molecular volume, is used. For ionic compounds, interactions in the solid are primarily captured by the Coulomb energy, which scales with $r^{-1}$, which can be be re-written as $V_m^{-\frac{1}{3}}$, where $r$ is the distance between ions and $V_m$ is the molecular volume\cite{Glasser2005,Glasser2011}.

We extend this understanding of interactions in the solid phase from ionic compounds to molecular solids. As Coulomb forces dominate ionic interactions, we expect dipole-dipole interactions to dominate intermolecular interactions for quinone and hydroquinone solids, due to the carbon-oxygen bonds. It can be derived from quantum mechanics that dipole-dipole interactions scale with distance as $r^{-3}$ or $V_m^{-1}$\cite{Berlin1952}. Similarly, Van der Waals (VdW) interactions scale with distance as $r^{-6}$ or $V_m^{-2}$\cite{Holstein2001}. We estimate that the lattice energy of our quinone and hydroquinone solids is well described by the dipole-dipole interaction.

By definition, the lattice energy is the  difference between the energies of the ions in the solid phase and the gas phase. This simple model assumes that the differences between enthalpies of sublimation for various compounds are dominated by differences in enthalpies of melting - i.e. the enthalpy of vaporization is similar for these quinone-based and hydroquinone-based models. This is captured in the constant $h$ in (\ref{enthalpyVm2}). 

We confirmed that a correlation does exist between enthalpy of melting and lattice energy as calculated using VBT by examining a previously collected set of data for hydrocarbon molecules\cite{LianUPPER2013}. In these set of molecules, we expect VdW interactions to dominate. We use this understanding to represent the enthalpy of melting as:

$$
\Delta H_m= g*V_m^{-2}+h
$$

Combining this with our equation for entropy of melting, we get the overall equation for melting point for hydrocarbons:

$$
    T_m=\frac{g*V_m^{-2}+h}{a*\textrm{ln}\sigma + b*\tau + c*\textrm{ln}\epsilon_{ar} + d*\textrm{ln}\epsilon_{al} + f}
$$

'''
# This is where we need to start inserting plots. I'm going to copy and paste the code from Wunmi's T_m code, and modify it to be just for hydrocarbons, Vm^(-2)
# Hydrocarbon plot
MAKING_NEW_PLOTS=True
if MAKING_NEW_PLOTS:
    ## EDIT BELOW HERE
    ### Change datasets used, model form, starting guesses
    #region
    # Change the datasets that you're interested in looking at in this block. Make sure you change the names of the datasets appropriately. Note: All datasets you include here will be tested with the same model form. If you want to test different model forms for different datasets, you will have to test one dataset at a time and change the model form as desired for that single dataset.
    datasets= [hydrocarbon_data]
    dataset_names= ['Hydrocarbon']
    num_datasets= len(datasets)

    # CHANGE MODEL NAME AND FORM HERE:
    model_form_name= '$V_m^{-2}$ Numerator, Full Denominator'
    model_form= '(parameters[0]*predictors["V_m (nm3)"]**(-2)+parameters[1])/(parameters[2]*np.log(predictors["sigma"])+parameters[3]*predictors["tau"]+1+parameters[4]*np.log(predictors["Eccentricity(Ear)"])+parameters[5]*np.log(predictors["Eccentricity(Eal)"]))'

    # CHANGE STARTING GUESSES HERE
    # Note: You must have the correct number of starting guesses to match the number of parameters in the model form, and you must also have the correct number of sets of starting guesses depending how many datasets you're testing at once.
    starting_guesses= [[-4.32603500e+00,2.87801207e+02,-7.93904846e-02,1.22626036e-02,-9.56732927e-02,-4.24685604e-02]]
    # HQ starting guesses: ,[-2.46237655e+01,6.04795762e+02,-6.14060740e-02,4.49395401e-02,3.03647346e-02,3.38981530e-02],[-1.83846686e+01,2.34877509e+02,-1.21714771e-01,-1.30258253e-02,-1.14720566e-01,-1.09430299e-01]

    #endregion
    ## EDIT ABOVE HERE

    ## STATIC BLOCK - DON'T MODIFY ANYTHING BELOW HERE ##
    #region
    num_predictors= list(range(len(findstr(model_form, 'predictors'))))
    used_predictors= []

    for i in range(len(all_possible_predictors)):
        this_predictor= all_possible_predictors[i]
        if findstr(model_form, this_predictor):
            used_predictors.append(this_predictor)
    used_predictors= used_predictors+predicted
    num_parameters= len(findstr(model_form, 'parameter'))

    if num_parameters!=len(starting_guesses[0]):
        raise Exception("Number of starting guesses doesn't match the number of parameters")

    letters_in_use=letters_in_use[0:num_parameters]
    num_parameters= list(range(num_parameters))

    plots=list(range(num_datasets))
    fig = plt.figure()      
    for i in range(num_datasets):
        avg_model_err=[0,0]
        rmse_err=[0,0]
        number_of_runs=5
        count=0
        while count < number_of_runs:
            dataset= datasets[i]
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            dataset_name=dataset_names[i]
            dataset_length= len(dataset)
            predictors= dataset[used_predictors].loc[2:dataset_length]
            predictors= predictors.astype('float64')
            [dataset_train,dataset_test]=split_data(predictors)
            (letters_in_use,_)=opt.curve_fit(fit_tm_model,dataset_train,dataset_train['T_m (K)'],(starting_guesses[i]))
            
            if statistics.mean(np.absolute(fit_tm_model_err(dataset_test,letters_in_use)
            -dataset_test['T_m (K)']))/(number_of_runs)<(70/number_of_runs) and letters_in_use[0]<0:
                #print(letters_in_use)
                avg_model_err[0]=avg_model_err[0]+(statistics.mean(np.absolute(fit_tm_model_err
                (dataset_test,letters_in_use)-dataset_test['T_m (K)']))/(number_of_runs))
                avg_model_err[1]=avg_model_err[1]+(statistics.mean(np.absolute(fit_tm_model_err
                (dataset_train,letters_in_use)-dataset_train['T_m (K)']))/(number_of_runs))
                count=count+1
                rmse_err[0]=rmse_err[0]+rmse((fit_tm_model_err(dataset_test,letters_in_use)),dataset_test['T_m (K)'])/(number_of_runs)
                rmse_err[1]=rmse_err[1]+rmse((fit_tm_model_err(dataset_train,letters_in_use)),dataset_train['T_m (K)'])/(number_of_runs)

        print(rmse_err)
        ax= (make_plots(dataset_test,dataset_train,letters_in_use,dataset_name,avg_model_err,rmse_err))
        
        plots[i]= ax

    st.write(plots[i])
#endregion
    ## STATIC BLOCK - DON'T MODIFY ANYTHING ABOVE HERE ##

st.markdown('''VBT model assuming Van der Waals interaction for hydrocarbon dataset. Training set absolute average error is {:.2f} C and test set average absolute error is {:.2f} C. Training set RMSE is {:.2f} C and test set RMSE is {:.2f} C, based on the average over five runs of the model.'''.format(avg_model_err[1],avg_model_err[0],rmse_err[1],rmse_err[0]) )

r'''
Our initial model for enthalpy of melting for the benzoquinone and hydroquinones was:
$$
\Delta H_m=g*V_m^{-1}+h
$$

Resulting in an overall equation of: 
$$
T_m=\frac{g*V_m^{-1}+h}{a*\textrm{ln}\sigma + b*\tau + c*\textrm{ln}\epsilon_{ar} + d*\textrm{ln}\epsilon_{al} + f}
$$

With a free constant in both the numerator and the denominator there are infinite possible solutions to the optimization problem. This makes the fitted parameters difficult to compare between different datasets (quinones, hydroquinones, and hydrocarbons). To mitigate this issue, we can normalize the equation by dividing numerator and denominator by one of the constants (this necessarily assumes that the parameter we normalize by is nonzero). Our model then becomes:

$$
T_m=\frac{g*V_m^{-1}+h}{a*\textrm{ln}\sigma + b*\tau + c*\textrm{ln}\epsilon_{ar} + d*\textrm{ln}\epsilon_{al} + 1}
$$
where the parameters a, b, c.... will be different values from before.

'''

# How to put code in markdown using python formatting
    # '''
    # ```python
    # for i in range(0, 4):
    #     st.write('Devi')
    # ```
    # '''

# Questions for Antonio:
    # How did you split the datasets into training and test sets?
    # Do you use any of the functions in the helper_functions.py file? I couldn't find them in either parse_reaxys_data or featurization_regression.py