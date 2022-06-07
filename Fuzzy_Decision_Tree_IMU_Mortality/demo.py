# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:42:42 2020

@author: jmbelda
"""

from FuzzyTree import *
import itertools  
import pandas as pd
import pydot

#%% Reading the data
data = pd.read_csv("./demo/set_a_clean_survival_days_reduced_1.csv", index_col=0)

#%% Vars in the file
variables = list(data.keys())


#%% Names of the vars
# LHS stands for Left Hand Side: Predictors
# RHS stands for Right Hand Side: Predicted

varRHS = variables[1:9]
varLHS = variables[0]
#varRHS.append(variables[0])

#%% Fuzzification

fnVars = {}  # Dictionary of fuzzification functions
fvVars = {}  # Dictionary of fuzzified variables

#levels = ["1. Extreme Low","2. Low", "3. Medium", "4. High","5. Extreme High"]
levels = ["1. Low","2. High"]

for v in  variables:
    fnVars[v], fvVars[v] = percentile_partition(data[v], v, levels)
    
    
#%% Drawing the variables
    
from numpy import linspace
from matplotlib.pyplot import *


fig = figure()
fig.tight_layout()




for c, v in enumerate(variables):
    subplot(4,4,c+1)
    
    # Creating an array for the extremes of the variable
    mini = min(data[v])
    maxi = max(data[v])    
    vals = linspace(mini, maxi, 5000)
    
    # Fuzzyfiyng the array
    fuzzified = fnVars[v](vals)
        
    for l in levels:
        plot(vals, fuzzified[l], label=l)
        
    legend()
    title(v)

#%% Creating a fuzzy set
fs = FuzzySet(*fvVars.values())

#%% Building the FuzzyTree
Beta = 0.9999999
Alpha = 0.1
ft = FuzzyTree(fs, Beta, Alpha, varRHS, varLHS)
print(ft)

#%% Confussion matrix training
ft.confussion_matrix(fvVars[varLHS], fs)
print(fs)

#%% Save a graphviz file for graphic output
ft.output_to_dot_graphviz('./draw_the_tree.dot')
(graph,) = pydot.graph_from_dot_file('draw_the_tree.dot')
graph.write_png('somefile.png')

#%% Showing the fuzzification (matplotlib)
show()

