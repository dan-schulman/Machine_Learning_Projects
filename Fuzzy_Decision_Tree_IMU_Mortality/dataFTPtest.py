#testing FTP.py
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:42:42 2020

@author: jmbelda
"""

from FuzzyTree import *
import pandas as pd

#%% Reading the data
data = pd.read_csv("./demo/wine.csv", index_col=False)

#%% Vars in the file
variables = list(data.keys())


#%% Names of the vars
# LHS stands for Left Hand Side: Predictors
# RHS stands for Right Hand Side: Predicted

varRHS = variables[1:14]
varLHS = variables[0]
#varRHS.append(variables[0])

#%% Fuzzification

fnVars = {}  # Dictionary of fuzzification functions
fvVars = {}  # Dictionary of fuzzified variables

#levels = ["1. Extreme Low","2. Low", "3. Medium", "4. High","5. Extreme High"]
levels = ["Low", "Med", "Hi"]


for v in  variables:

    fnVars[v], fvVars[v] = percentile_partition(data[v], v, levels)
    
    


#%% Creating a fuzzy set
fs = FuzzySet(*fvVars.values())

#%% Building the FuzzyTree

ft =  FPT_forest("Testing Tree",levels)
ft.fit(fs,varLHS, varRHS)

ft.graphviz_export()