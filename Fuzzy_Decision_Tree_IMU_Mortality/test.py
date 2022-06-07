#testing FTP.py
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:42:42 2020

@author: jmbelda
Heavily modified by Alec Socha
"""

from FuzzyTree import *
import pandas as pd
from sklearn import model_selection
import numpy as np
from numpy import linspace
from matplotlib.pyplot import *

#%% Reading the data
data = pd.read_csv("./demo/set_a_clean_survival_days_reduced.csv", index_col=0)

#%% Vars in the file
variables = list(data.keys())


#%% Names of the vars
# LHS stands for Left Hand Side: Predictors
# RHS stands for Right Hand Side: Predicted

varRHS = variables[4:36]
varLHS = variables[0]
#varRHS.append(variables[0])
features = data[varRHS].values
labels = data[varLHS].values

#X_train, X_test, y_train, y_test = model_selection.train_test_split(features,labels,test_size=0.05)
#y_train = y_train.reshape((y_train.shape[0],1))
#y_test = y_test.reshape((y_test.shape[0],1))
#train = pd.DataFrame(np.hstack((y_train,X_train)),columns= variables)
#test = pd.DataFrame(np.hstack((y_test,X_test)),columns= variables)

train = data
#%% Fuzzification

fnVarsTRAIN = {}  # Dictionary of fuzzification functions
fvVarsTRAIN = {}  # Dictionary of fuzzified variables
fnVarsTEST = {}  # Dictionary of fuzzification functions
fvVarsTEST = {}  # Dictionary of fuzzified variables

levels = ["Low","Med", "Hi"]
#testIter = [{}]*X_test.shape[0]
for v in  variables:

    fnVarsTRAIN[v], fvVarsTRAIN[v] = percentile_partition(train[v], v, levels)  
    #for i in range(X_test.shape[0]):
        #testIter[i][v] = fnVarsTRAIN[v](test[v]._values[i]) 
    
'''
Plotting the three different fuzzification variables
t = 'Heart Rate'
fnVarsG , temp = gaussFuzz(train[t],t, levels)
fnVarsTr, temp = percentile_partition(train[t],t, levels)
fnVarsTrap,temp = trapFuzz(train[t],t, levels)

fig = figure()
subplot(1,3,1)
mini = min(data[t])
maxi = max(data[t])
vals = linspace(mini, maxi , 500)
fuzzified = fnVarsG(vals)
for l in levels:
    plot(vals, fuzzified[l], label=l)

title('Gaussian')

subplot(1,3,2)
fuzzified = fnVarsTr(vals)
for l in levels:
    plot(vals, fuzzified[l], label=l)

title('Triangle')

subplot(1,3,3)
fuzzified = fnVarsTrap(vals)
for l in levels:
    plot(vals, fuzzified[l], label=l)

title('Trapezoid')
legend()
'''
#%% Drawing the variables


fig = figure()
fig.tight_layout()

for c, v in enumerate(variables):
    subplot(5,3,c+1)
    
    # Creating an array for the extremes of the variable
    mini = min(data[v])
    maxi = max(data[v])    
    vals = linspace(mini, maxi, 500)
    
    # Fuzzyfiyng the array
    fuzzified = fnVarsTRAIN[v](vals)
        
    for l in levels:
        plot(vals, fuzzified[l], label=l)
        
    legend()
    title(v)   
show()

#%% Creating a fuzzy set
fsTRAIN = FuzzySet(*fvVarsTRAIN.values())
#%% Building the FuzzyTree
#y_test = np.array(y_test).flatten()-1
ft =  FPT_forest("Testing Tree",levels,maxdepth=3)
ft.fit(fsTRAIN,varLHS, varRHS)
'''
results = ft.predict(fsTEST,len(y_test))            print(p)
correctLabel = np.argmax([fsTEST[varLHS]._values[i] for i in levels],axis=0)
print(results)
print(y_test)
correct = sum(results ==y_test)/len(y_test)
print(correct)
'''
#for i in testIter:
    #print(ft.predict(FuzzySet(*i.values()),1 ))
guess = ft.predict(fsTRAIN,1473)
#print(np.array(y_train).flatten()-1)
actual = np.zeros(1473)
for i in range(1473):
    current = np.zeros(3)
    current[0] = fsTRAIN['Risk']._values['Low'][i]
    current[1] = fsTRAIN['Risk']._values['Med'][i]
    current[2] = fsTRAIN['Risk']._values['Hi'][i]

    actual[i] = np.argmax(current)

print(sum(guess == actual))
ft.graphviz_export()