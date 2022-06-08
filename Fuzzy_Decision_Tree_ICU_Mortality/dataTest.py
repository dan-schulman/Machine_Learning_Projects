# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:42:42 2020

@author: jmbelda
"""

from FuzzyTree import *
import pandas as pd

#%% Reading the data
data = pd.read_csv("./demo/data_clean_a.csv")

#%% Vars in the file
variables = list(data.keys())


#%% Names of the vars
# LHS stands for Left Hand Side: Predictors
# RHS stands for Right Hand Side: Predicted

varRHS = variables[3:20]
varLHS = variables[2]

## Balance the data
# This is for binary data
dataR = np.array(data[varRHS])
dataL = np.array(data[varLHS])

alive = sum(dataL==0)
dead = sum(dataL==1)

wa = alive/(alive + dead)
wd = dead/(alive + dead)

print(wa)
print(wd)

#dataR[dataL == 0,:] *= wa
#dataR[dataL == 1,:] *= wd
#%% Fuzzification

fnVars = {}  # Dictionary of fuzzification functions
fvVars = {}  # Dictionary of fuzzified variables

#levels = ["1. Extreme Low","2. Low", "3. Medium", "4. High","5. Extreme High"]
levels = ["1. Low----", "2. Medium-", "3. Hi-----"]


for v in  variables:
    fnVars[v], fvVars[v] = gaussFuzz(data[v], v, levels)
    

    
#%% Drawing the variables
    
from numpy import linspace
from matplotlib.pyplot import *


fig = figure()
fig.tight_layout()




for c, v in enumerate(variables):
    subplot(6,6,c+1)
    
    # Creating an array for the extremes of the variable
    mini = min(data[v])
    maxi = max(data[v])    
    vals = linspace(mini, maxi, 500)
    
    # Fuzzyfiyng the array
    fuzzified = fnVars[v](vals)
        
    for l in levels:
        plot(vals, fuzzified[l], label=l)
        
    legend()
    title(v)

#%% Creating a fuzzy set
fs = FuzzySet(*fvVars.values())

#%% Building the FuzzyTree
Beta = 0.8
Alpha = 0.8
ft = FuzzyTree(fs, Beta, Alpha, varRHS, varLHS)
print(ft)

#%% Confussion matrix
ft.confussion_matrix(fvVars[varLHS], fs)

#%% Save a graphviz file for graphic output
ft.output_to_dot_graphviz('./draw_the_tree.dot')

#%% Showing the fuzzification (matplotlib)
show()

