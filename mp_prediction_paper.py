
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
from PIL import Image
import scipy.stats as stt


###################### Thermodynamics-Based Model Code #########################

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
    lowerT =min ( min(dataset_test['T_m (K)']),min(dataset_train['T_m (K)']),min(fit_tm_model_err(dataset_test,letters_in_use)),min (fit_tm_model_err(dataset_train,letters_in_use)))
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

# function to calculate root mean square error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Function for rounding tuples in confidence intervals - copied from online
def re_round(li, _prec=3):
    ''' Rounds argument li to desired number of decimals (_prec) '''
    try:
         return round(li, _prec)
    except TypeError:
         return type(li)(re_round(x, _prec) for x in li)
        
def vbt_model_automated(dataset_dictionary, dataset_name, model_form, model_form_name, starting_guesses, number_of_runs):
    '''Generates a dictionary of VBT model plot, Avg Absolute Error, RMSE Error, and Parameter values'''
    all_possible_predictors= ['tau', 'V_m (nm3)', 'sigma','Eccentricity(Eal)', 'Eccentricity(Ear)']
    letters_in_use= ['a','b','c','d','f','g','h','k','l','m']
    predicted= ['T_m (K)']
    used_predictors= []
    # This for loop finds which parameters are used in the model form and adds them to a list of used predictors
    for i in range(len(all_possible_predictors)):
        this_predictor= all_possible_predictors[i]
        if findstr(model_form, this_predictor):
            used_predictors.append(this_predictor)
    used_predictors= used_predictors+predicted
    num_parameters= len(findstr(model_form, 'parameter'))

    if num_parameters!=len(starting_guesses):
        raise Exception("Number of starting guesses doesn't match the number of parameters")

    letters_in_use=letters_in_use[0:num_parameters]
    num_parameters= list(range(num_parameters))
    avg_model_err=[0,0]
    rmse_err=[0,0]
    count=0
    parameters_from_runs=np.zeros((number_of_runs,len(num_parameters)))
    while count < number_of_runs:
        dataset= dataset_dictionary[dataset_name]
        dataset_length= len(dataset)
        predictors=dataset[used_predictors].astype('float64')
        [dataset_train,dataset_test]=split_data(predictors)
        (letters_in_use,_)=opt.curve_fit(fit_tm_model,dataset_train,dataset_train['T_m (K)'],(starting_guesses))
        
        if statistics.mean(np.absolute(fit_tm_model_err(dataset_test,letters_in_use)
        -dataset_test['T_m (K)']))/(number_of_runs)<(70/number_of_runs) and letters_in_use[0]<0:
            #print(letters_in_use)
            parameters_from_runs[count,:]=letters_in_use
            avg_model_err[0]=avg_model_err[0]+(statistics.mean(np.absolute(fit_tm_model_err
            (dataset_test,letters_in_use)-dataset_test['T_m (K)']))/(number_of_runs))
            avg_model_err[1]=avg_model_err[1]+(statistics.mean(np.absolute(fit_tm_model_err
            (dataset_train,letters_in_use)-dataset_train['T_m (K)']))/(number_of_runs))
            count=count+1
            rmse_err[0]=rmse_err[0]+rmse((fit_tm_model_err(dataset_test,letters_in_use)),dataset_test['T_m (K)'])/(number_of_runs)
            rmse_err[1]=rmse_err[1]+rmse((fit_tm_model_err(dataset_train,letters_in_use)),dataset_train['T_m (K)'])/(number_of_runs)
    fig=make_plots(dataset_test,dataset_train,letters_in_use,dataset_name,avg_model_err,rmse_err)
    #print(rmse_err)
    model_dict = {
        'Plot': fig,
        'Average Absolute Error': avg_model_err,
        'RMSE Error': rmse_err,
        'Parameter Values': parameters_from_runs
    }
    plt.close()
    return model_dict

def ml_model(ml_dataset_dict, dataset_key, alpha=100, do_featurization = False):  
    if do_featurization:

        working_ML_dataset=ml_dataset_dict[dataset_key]
        mols_full=[Chem.MolFromSmiles(m) for m in working_ML_dataset.SMILES.tolist() if Chem.MolFromSmiles(m) != None]
        calc = Calculator(descriptors, ignore_3D=True)
        mordredresults = calc.pandas(mols_full)
        working_ML_dataset = working_ML_dataset.join(mordredresults,how='inner')
        # save to the appropriate file name depending on which dataset you're looking at
        if dataset_key=='Quinones':
            working_ML_dataset.to_csv('featurized_bq.csv',index=False)
        elif dataset_key=='Hydroquinones':
            working_ML_dataset.to_csv('featurized_hq.csv',index=False)
        else:
            print('Not a valid dataset key')            
    else:
        if dataset_key=='Quinones':
            working_ML_dataset=pd.read_csv('featurized_bq.csv')
        elif dataset_key=='Hydroquinones':
            working_ML_dataset=pd.read_csv('featurized_hq.csv')
   
   # Now the training set and test set shuffle every time you run the code. If we want to average the errors over multiple runs we can put this in another loop.
    [trainset,testset] = split_data(working_ML_dataset)
    trainset=trainset.reset_index()
    testset=testset.reset_index()        
    
    ######
    ## Standardization (want each feature to be a gaussian with zero mean and unit variance)
    ######

    # drop non-numeric columns and ones for the melting point, so we only have columns of features
    # I don't know why, but LogP caused a problem during the standardization - dropping for now, but have to figure out
    # Now dropping everything but MW from the Reaxys, since we don't have it for the Na and K salts
    columns_to_drop_from_reaxys = ['InChI Key','SMILES','Type of Substance','mp_mean','mp_std','LogP','H Bond Donors','H Bond Acceptors','Rotatable Bonds','TPSA','Lipinski Number','Veber Number']

    if dataset_key=='Quinones':
        columns_to_drop_thatgavetrouble = ['MAXdO','MINdO']
        # for bq these gave trouble: ['Unnamed: 0','MAXdO','MINdO']
    elif dataset_key=='Hydroquinones':
        columns_to_drop_thatgavetrouble = []

    columns_to_drop = columns_to_drop_from_reaxys + columns_to_drop_thatgavetrouble
    trainset_s = trainset.drop(columns=columns_to_drop)
    testset_s = testset.drop(columns=columns_to_drop)

    # drop columns where there is an error from mordred
    #print('Started with '+str(len(trainset_s.columns))+' features')
    trainset_s = trainset_s.select_dtypes(include=['float64','int'])
    #print('After dropping columns with mordred errors, have '+str(len(trainset_s.columns))+' features')

    # drop the same columns from the test set
    testset_s = testset_s[trainset_s.columns]

    # finally, do the standardization
    X = preprocessing.scale(trainset_s)
    # apply the same standardization to the test set
    scaler = preprocessing.StandardScaler().fit(trainset_s)
    X_test = scaler.transform(testset_s)

    ######
    ## Ridge regression
    ######

    y = trainset.mp_mean.tolist()
    y_test = testset.mp_mean.tolist()

    rr = Ridge(alpha=alpha)
    rr.fit(X, y)
    w = rr.coef_
    intercept = rr.intercept_

    avg_abs_err = np.zeros(2)
    rmse_err = np.zeros(2)
    # The test set is at the 0 index and the training set is at the 1 index to match the convention in the other model 
    avg_abs_err[1] = np.mean(np.abs(y-(np.dot(X,w)+intercept)))
    avg_abs_err[0] = np.mean(np.abs(y_test-(np.dot(X_test,w)+intercept)))
    rmse_err[1] = np.sqrt(((y - (np.dot(X,w)+intercept)) ** 2).mean())
    rmse_err[0] = np.sqrt(((y_test-(np.dot(X_test,w)+intercept))**2).mean())

    #print('Training error: '+str(ml_err_train))
    #print('Test error: '+str(ml_err_test))
    coeffs = pd.DataFrame(data={'label':trainset_s.columns, 'w':w, 'w_abs':np.abs(w)})

    #print(coeffs.sort_values(by='w_abs',ascending=False).head(20))

    # Make plot
    model_plot=plt.figure(figsize=(4,4), dpi=300)
    ax1 = plt.gca()
    ml_plot_train_points=ax1.scatter(y,np.dot(X,w)+intercept,s=5,c='k',alpha=0.7,linewidth=0)
    ax1.plot([-273,2000],[-273,2000],'k--',lw=1)
    ml_plot_test_points=ax1.scatter(y_test,np.dot(X_test,w)+intercept,s=5,c='r',alpha=0.7,linewidth=0)

    lims = [min(y+y_test)-5,max(y+y_test)+5]

    for theaxis in [ax1]:
        
        theaxis.set_aspect(1)
        theaxis.set_xlim(lims)
        theaxis.set_ylim(lims)

        theaxis.set_xlabel(r"mp data ($^{\circ}$C)")
        theaxis.set_ylabel("mp predicted ($^{\circ}$C)")
        
        for item in ([theaxis.xaxis.label, theaxis.yaxis.label, theaxis.yaxis.get_offset_text(), theaxis.xaxis.get_offset_text()]):
            item.set_fontsize(12)
        for item in (theaxis.get_xticklabels() + theaxis.get_yticklabels()):
            item.set_fontsize(10)
    plt.legend((ml_plot_train_points,ml_plot_test_points),('Training Set','Test Set'))    
    plt.gcf().subplots_adjust(left=0.2,top=0.95,bottom=0.15,right=0.95)
   
    plt.close()

    ml_model_dict={
        'Plot': model_plot,
        'Average Absolute Error': avg_abs_err,
        'RMSE Error': rmse_err,
        'Model Coefficients': coeffs
    }
    return ml_model_dict


# Import data files for physics-based model into pandas dataframes
# VBT Data
quinone_data = pd.read_csv("Entropy and Volume Data - Quinones.csv")
hydroquinone_data= pd.read_csv("Entropy and Volume Data - Hydroquinones.csv")
hydrocarbon_data= pd.read_csv("Entropy and Volume Data - Hydrocarbons.csv")
mega_database=pd.concat([hydroquinone_data,quinone_data])[['sigma','tau','V_m (nm3)','T_m (K)','Eccentricity(Ear)','Eccentricity(Eal)']].reset_index(drop=True)
# ML Data
quinone_ML_data=pd.read_csv('parsed_p_benzoquinone_216.csv')
hydroquinone_ML_data=pd.read_csv('parsed_p_hydroquinone_204.csv')
# Make dictionary for VBT data
dataset_dict={'Quinones':quinone_data,'Hydroquinones':hydroquinone_data,'Hydrocarbons':hydrocarbon_data,'Quinones + Hydroquinones':mega_database}
# Make dictionary for ML data
ml_dataset_dict={'Quinones':quinone_ML_data,'Hydroquinones':hydroquinone_ML_data}
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

r'''
## Introduction

Quinone- and hydroquinone-based molecules have gathered attention recently as electrolyte candidate molecules for redox flow batteries, among other applications\cite{Shimizu2017,Kwabi2018,Goulet2019}. Our group has recently proposed using a eutectic mixture of benzoquinone-based molecules as a high energy density positive electrolyte for flow batteries that remains liquid at room temperature. In order to identify promising materials for this application, knowing the melting temperatures of the quinone and hydroquinone molecules is essential. With the model we have developed, the melting points of the pure component quinones and hydroquinones can be used to predict the melting point of a eutectic mixture. However, melting data is not available for all the quinones and hydroquinones of interest. To address this challenge, we developed a computational model that can be used to predict the melting points of quinone and hydroquinone molecules that are not available in literature. 

Many melting point prediction models employ group contribution method (GCM). This is an additive method that works by summing the contributions from all the various groups in a molecule (hence "group contribution method"). However, this method does not account for interactions between groups \cite{JOBACK1987}.

Our model utilizes simple molecular descriptors that can be calculated from the two-dimensional structure of the model from a semi-empirical model previously proposed by Dannenfelser and Yalkowsky\cite{DannenfelserEntropy}. The model also uses molecular volume data, which can be calculated using crystal structure data or density measurements, or predicted computationally\cite{Day2011}. The molecular volume data contains information about the strength of the interactions between molecules in the solid phase. This approach provides an advantage over GCM in that the molecular volume in the solid phase of a species inherently accounts for the interactions between molecules.

'''
# Methods
r'''
## Methods

### Machine Learning Model

For the more traditional machine learning approach, we used the Mordred molecular descriptor calculator (available as a python package) to featurize our quinone and hydroquinone based molecules using just the chemical SMILES string as the input \cite{Moriwaki2018}. This generated approximately 1000 usable features for each molecule. After generating the features, they were then standardized so that each feature was a gaussian with zero mean and unit variance. The standardization was applied independently to the training set and test set. The standardized features were then used in a ridge regression ($\alpha=100$) to generate the melting point prediction model. 

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

With a free constant in both the numerator and the denominator there are infinite possible solutions to the optimization problem. This makes the fitted parameters difficult to compare between different datasets (quinones, hydroquinones, and hydrocarbons). To mitigate this issue, we can normalize the equation by dividing numerator and denominator by one of the constants (this necessarily assumes that the parameter we normalize by is nonzero). Our model then becomes:
$$
    T_m=\frac{g*V_m^{-2}+h}{a*\textrm{ln}\sigma + b*\tau + c*\textrm{ln}\epsilon_{ar} + d*\textrm{ln}\epsilon_{al} + 1}
$$

where the parameters $a, b, c....$ will be different values from before.

'''

# Hydrocarbon plot
MAKING_NEW_PLOTS=True
if MAKING_NEW_PLOTS:

    # CHANGE MODEL DATASET, NAME AND FORM HERE:
    dataset_name='Hydrocarbons'

    model_form_name= '$V_m^{-2}$ Numerator, Full Denominator'

    model_form= '(parameters[0]*predictors["V_m (nm3)"]**(-2)+parameters[1])/(parameters[2]*np.log(predictors["sigma"])+parameters[3]*predictors["tau"]+1+parameters[4]*np.log(predictors["Eccentricity(Ear)"])+parameters[5]*np.log(predictors["Eccentricity(Eal)"]))'

    starting_guesses= [-1.5,1.8e+02,-1e-01,-2e-02,-1.1e-01,-9e-02]
    num_runs=5
    vbt_hc_dict=vbt_model_automated(dataset_dict,dataset_name,model_form,model_form_name,starting_guesses,num_runs)

    vbt_hc_dict['Plot'].savefig('HC_plot.png',dpi=300)
    st.write(vbt_hc_dict['Plot'])
    st.markdown('''VBT model assuming Van der Waals interaction for hydrocarbon dataset. Training set absolute average error is {:.2f} C and test set average absolute error is {:.2f} C. Training set RMSE is {:.2f} C and test set RMSE is {:.2f} C, based on the average over five runs of the model.'''.format(vbt_hc_dict['Average Absolute Error'][1],vbt_hc_dict['Average Absolute Error'][0],vbt_hc_dict['RMSE Error'][1],vbt_hc_dict['RMSE Error'][0]) )
else: 
    hc_plot=Image.open('HC_plot.png')
    # Here we just use the saved plot from the previous run. The caption won't have errors because the model wasn't recalculated
    st.image(hc_plot,caption='VBT model assuming Van der Waals interaction for hydrocarbon dataset.',use_column_width=True)

r'''
Our initial model for enthalpy of melting for the benzoquinone and hydroquinones was:
$$
\Delta H_m=g*V_m^{-1}+h
$$

Resulting in an overall equation of: 

$$
T_m=\frac{g*V_m^{-1}+h}{a*\textrm{ln}\sigma + b*\tau + c*\textrm{ln}\epsilon_{ar} + d*\textrm{ln}\epsilon_{al} + 1}
$$

Where we have normalized the constant in the denominator for reasons discussed above. 

'''
# Results and Discussion

r'''
## Results and Discussion

### Machine Learning Model
'''



# Quinone ML Plot
alpha_bq=st.slider('Quinone Ridge Regression alpha',min_value=1,max_value=200,value=100)
ml_bq_dict=ml_model(ml_dataset_dict,'Quinones',alpha_bq,False)

ml_bq_dict['Plot'].savefig('ML_BQ_plot.png',dpi=300)
st.write(ml_bq_dict['Plot'])
st.markdown('''ML model for quinone dataset. Training set absolute average error is {:.2f} C and test set average absolute error is {:.2f} C. Training set RMSE is {:.2f} C and test set RMSE is {:.2f} C.'''.format(ml_bq_dict['Average Absolute Error'][1],ml_bq_dict['Average Absolute Error'][0],ml_bq_dict['RMSE Error'][1],ml_bq_dict['RMSE Error'][0]) )

# Hydroquinone ML Plot
alpha_hq=st.slider('Hydroquinone Ridge Regression alpha',min_value=1,max_value=200,value=100)
ml_hq_dict=ml_model(ml_dataset_dict,'Hydroquinones',alpha_hq,False)
ml_hq_dict['Plot'].savefig('ML_HQ_plot.png',dpi=300)
st.write(ml_hq_dict['Plot'])
st.markdown('''ML model for hydroquinone dataset. Training set absolute average error is {:.2f} C and test set average absolute error is {:.2f} C. Training set RMSE is {:.2f} C and test set RMSE is {:.2f} C.'''.format(ml_hq_dict['Average Absolute Error'][1],ml_hq_dict['Average Absolute Error'][0],ml_hq_dict['RMSE Error'][1],ml_hq_dict['RMSE Error'][0]) )

#print('Time to Run = ' + str(time.time()-starttime) + ' s')

#endregion
## STATIC BLOCK - DON'T MODIFY ANYTHING ABOVE HERE ##

r'''
### Thermodynamics-Based Model
The quinone and hydroquinone datasets were initially fitted to the model independently (thus generating different values for a, b, c,...) to reflect the assumption that the strength of the dipole-dipole interaction varies between quinones and hydroquinones. This assumption will be evaluated and discussed later in the paper. They were later also combined into one dataset and fitted together to generate one model.

We tested several different functional forms for the numerator, but found that $V_m^{-1}$ consistently yielded the lowest errors. We calculated both the average absolute error (which has been reported in other melting point prediction literature\cite{Preiss2011}) and the root mean square error, which is commonly used in machine learning approaches\cite{Nigsch2006}.
'''
# Quinone Plot
MAKING_NEW_PLOTS=True
if MAKING_NEW_PLOTS:
    ## EDIT BELOW HERE

    # CHANGE MODEL NAME AND FORM HERE:
    model_form_name= '$V_m^{-1}$ Numerator, Full Denominator'
    model_form= '(parameters[0]*predictors["V_m (nm3)"]**(-1)+parameters[1])/(parameters[2]*np.log(predictors["sigma"])+parameters[3]*predictors["tau"]+1+parameters[4]*np.log(predictors["Eccentricity(Ear)"])+parameters[5]*np.log(predictors["Eccentricity(Eal)"]))'

    starting_guesses= [-1e+01,3e+02,-1e-01,1e-02,-8e-02,-3e-02]
    dataset_name='Quinones'
    num_runs=5
    vbt_bq_dict=vbt_model_automated(dataset_dict,dataset_name,model_form,model_form_name,starting_guesses,num_runs)

    vbt_bq_dict['Plot'].savefig('BQ_plot.png',dpi=300)

    st.write(vbt_bq_dict['Plot'])
    # Figure caption incorporates calculated errors
    st.markdown('''VBT model assuming dipole-dipole interaction for quinone dataset. Training set absolute average error is {:.2f} C and test set average absolute error is {:.2f} C. Training set RMSE is {:.2f} C and test set RMSE is {:.2f} C, based on the average over five runs of the model.'''.format(vbt_bq_dict['Average Absolute Error'][1],vbt_bq_dict['Average Absolute Error'][0],vbt_bq_dict['RMSE Error'][1],vbt_bq_dict['RMSE Error'][0]) )
else: 
    bq_plot=Image.open('BQ_plot.png')
    st.image(bq_plot,caption='VBT model assuming dipole-dipole interaction for quinone dataset.',use_column_width=True)

# Hydroquinone Plot
MAKING_NEW_PLOTS=True
if MAKING_NEW_PLOTS:

    # CHANGE MODEL NAME AND FORM HERE:
    model_form_name= '$V_m^{-1}$ Numerator, Full Denominator'
    model_form= '(parameters[0]*predictors["V_m (nm3)"]**(-1)+parameters[1])/(parameters[2]*np.log(predictors["sigma"])+parameters[3]*predictors["tau"]+1+parameters[4]*np.log(predictors["Eccentricity(Ear)"])+parameters[5]*np.log(predictors["Eccentricity(Eal)"]))'

    starting_guesses= [-2e+01,5e+02,-7e-02,3e-02,2e-02,2e-02]
    dataset_name='Hydroquinones'
    num_runs=5
    vbt_hq_dict=vbt_model_automated(dataset_dict,dataset_name,model_form,model_form_name,starting_guesses,num_runs)
    
    vbt_hq_dict['Plot'].savefig('HQ_plot.png',dpi=300)

    st.write(vbt_hq_dict['Plot'])

    st.markdown('''VBT model assuming dipole-dipole interaction for hydroquinone dataset. Training set absolute average error is {:.2f} C and test set average absolute error is {:.2f} C. Training set RMSE is {:.2f} C and test set RMSE is {:.2f} C, based on the average over five runs of the model.'''.format(vbt_hq_dict['Average Absolute Error'][1],vbt_hq_dict['Average Absolute Error'][0],vbt_hq_dict['RMSE Error'][1],vbt_hq_dict['RMSE Error'][0]) )
else: 
    hq_plot=Image.open('HQ_plot.png')
    st.image(hq_plot,caption='VBT model assuming dipole-dipole interaction for hydroquinone dataset.',use_column_width=True)

r'''
We note that there appears to be a systematic underestimation of the melting points of the higher $T_m$ and an overestimate of the melting points of the lower $T_m$ molecules. We attribute this systematic error to our model for enthalpy. By using all of the molecules in the dataset to generate one set of fitted parameters, we are effectively assuming that all molecules have the same types of intermolecular interactions. However, this is unlikely true, as the higher melting molecules most likely have stronger intermolecular interactions (perhaps hydrogen bonding), and thus should have higher enthalpies of melting. By combining different types of quinones molecules in this way, we are essentially taking an intermediate strength of intermolecular interaction and applying it to all the molecules in the dataset, which results in the over- and under- estimation that we see in our data. This was verified by analyzing the types of molecules on both ends of the spectrum to see if there were obvious reasons why the higher melting compounds might have stronger interactions and the lower melting compounds would have weaker interactions.

Upon observing the difficulties in maintaining a consistent sign for the eccentricity parameters, which we suspected was due to the relative unimportance of it and the small dataset, we decided to combine both the quinone and hydroquinone datasets into a single dataset and fit a model to this larger dataset (~200 molecules). 
'''
# Combined Quinone + Hydroquinone Plot
MAKING_NEW_PLOTS=True
if MAKING_NEW_PLOTS:

    # CHANGE MODEL NAME AND FORM HERE:
    model_form_name= '$V_m^{-1}$ Numerator, Full Denominator'
    model_form= '(parameters[0]*predictors["V_m (nm3)"]**(-1)+parameters[1])/(parameters[2]*np.log(predictors["sigma"])+parameters[3]*predictors["tau"]+1+parameters[4]*np.log(predictors["Eccentricity(Ear)"])+parameters[5]*np.log(predictors["Eccentricity(Eal)"]))'

    starting_guesses= [-2e+01,5e+02,-7e-02,3e-02,2e-02,2e-02]
    dataset_name='Quinones + Hydroquinones'
    num_runs=5
    vbt_bqhq_dict=vbt_model_automated(dataset_dict,dataset_name,model_form,model_form_name,starting_guesses,num_runs)
    vbt_bqhq_dict['Plot'].savefig('BQHQ_plot.png',dpi=300)

    st.write(vbt_bqhq_dict['Plot'])

    st.markdown('''VBT model assuming dipole-dipole interaction for combined quinone and hydroquinone dataset. Training set absolute average error is {:.2f} C and test set average absolute error is {:.2f} C. Training set RMSE is {:.2f} C and test set RMSE is {:.2f} C, based on the average over five runs of the model.'''.format(vbt_bqhq_dict['Average Absolute Error'][1],vbt_bqhq_dict['Average Absolute Error'][0],vbt_bqhq_dict['RMSE Error'][1],vbt_bqhq_dict['RMSE Error'][0]) )
else: 
    hq_plot=Image.open('HQ_plot.png')
    st.image(hq_plot,caption='VBT model assuming dipole-dipole interaction for hydroquinone dataset.',use_column_width=True)

# Experimental Section
r'''
## Experimental

### Machine Learning Model

All of the data for this method was downloaded from the Reaxys database online (reaxys.com). For each of the quinone and hydroquinone datasets, a substructure search was performed with a structure editor query (benzoquinone example shown below in Figure \cite{bq_reaxys_search}). We limited our search to compounds witha molecular weight of less than 216 g/mol for the quinone-based molecules and 204 g/mol for the hydroquinone-based molecules. We then filtered the data by compounds which had melting points available from literature, and downloaded them using Reaxys's download feature.

Once all the compounds were downloaded we had to further process the data for molecules that had multiple reported melting points. The Tietjen-Moore outlier test was employed to determine whether there were any outliers in the set of melting point data for each molecule, and remove outliers if they did exist. This test requires an initial hypothesis for how many outliers exist in the dataset, so we incrementally increased the hypothesized number of outliers until the range was less than 15 C, or we had thrown out more than half of the melting points in the set. If we had to eliminate more than half the melting points (outlier test failed), we removed that molecule from our dataset.

### Thermodynamics-Based Model
To build our quinone and hydroquinone datasets, we used crystal structure data acquired from the Cambridge Crystallographic Data Centre (CCDC) to calculate the molecular volume for each compound. Molecular symmetry, torsional angles, and eccentrity values were calculated by visual inspection of the 2D molecular structure. The experimental melting points recorded in the CCDC database were used in our dataset, if they were reported. If the melting points were not reported in the CCDC, we found the reported melting points for the molecules in literature (sources listed in the database in SI). A total of 94 quinones, 94 hydroquinones, and 224 hydrocarbons were used in the analysis. 

We wrote a script that allowed us to specify any functional form of the model that we wanted to test on our datasets. This allowed us to test several different forms of the model to determine what enthalpy relation had the most predictive power. The function ``optimize.curve\textunderscore fit" in the python library ``scipy" was used to calculate values for the parameters in our model, given reasonable starting guesses. All datasets were randomly split into a training and test set, and the model parameters were fitted using only the training set. The resulting equation was then used to calculate the predicted melting points for both the training and test sets. Both the training and test set errors were calculated independently - both the average absolute error and the root mean square errors were calculated.

'''
# Edit here
# number of decimal points to round to in our parameter  tables
dec_points=3
# confidence interval we want to calculate in our tables
ci_int_val=0.95
##### STATIC BLOCK BELOW (you know the drill)
#region
# This code will only work if we use the same model - can't change the number of parameters otherwise the columns will be mis-labeled
hc_parameters_df=pd.DataFrame(data=vbt_hc_dict['Parameter Values'],index=["Run 1","Run 2","Run 3","Run 4","Run 5"],columns=["g","h","a","b","c","d"])
bq_parameters_df=pd.DataFrame(data=vbt_bq_dict['Parameter Values'],index=["Run 1","Run 2","Run 3","Run 4","Run 5"],columns=["g","h","a","b","c","d"])
hq_parameters_df=pd.DataFrame(data=vbt_hq_dict['Parameter Values'],index=["Run 1","Run 2","Run 3","Run 4","Run 5"],columns=["g","h","a","b","c","d"])
bqhq_parameters_df=pd.DataFrame(data=vbt_bqhq_dict['Parameter Values'],index=["Run 1","Run 2","Run 3","Run 4","Run 5"],columns=["g","h","a","b","c","d"])

# datasets = [hc_parameters, bq_parameters, hq_parameters, bqhq_parameters]
# for dataset in datasets:

# Calculate parameter means and confidence intervals for each dataset. We can do this in very few lines using a dictionary and f strings
df_dict={'hc':hc_parameters_df,'bq':bq_parameters_df,'hq':hq_parameters_df,'bqhq':bqhq_parameters_df}

for key in df_dict:
    # iterate through all the dfs in our dictionary, initialize mean and ci lists
    mean_list=[]
    ci_list=[]
    for column in df_dict[key]:
        # calculate the mean and 95% CI for each column in the current df
        mean_list.append(np.mean(df_dict[key][column]))
        ci_list.append(stt.t.interval(ci_int_val,len(df_dict[key][column])-1, np.mean(df_dict[key][column]), stt.sem(df_dict[key][column])))
# Add these statistics to a new dataframe called xx_parameters_df_w_stats
    exec(f'{key}_parameters_df_w_stats=df_dict[key].copy().round(dec_points)')
    exec(f'{key}_parameters_df_w_stats.loc["Mean"]=re_round(mean_list,dec_points)')
    exec(f'{key}_parameters_df_w_stats.loc["95% CI"]=re_round(ci_list,dec_points)')


#### STATIC BLOCK ABOVE

#endregion
st.markdown('''### Hydrocarbon Parameters ''')
st.write(hc_parameters_df_w_stats)
st.markdown('''### Quinone Parameters ''')
st.write(bq_parameters_df_w_stats)
st.markdown('''### Hydroquinone Parameters ''')
st.write(hq_parameters_df_w_stats)
st.markdown('''### Quinone + Hydroquinone Parameters ''')
st.write(bqhq_parameters_df_w_stats)

# How to put code in markdown using python formatting
    # '''
    # ```python
    # for i in range(0, 4):
    #     st.write('Devi')
    # ```
    # '''


# To do:
    # Figure out if g should be + or -