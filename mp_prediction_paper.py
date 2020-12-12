
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

# References Section


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

'''

# How to put code in markdown using python formatting
    # '''
    # ```python
    # for i in range(0, 4):
    #     st.write('Devi')
    # ```
    # '''