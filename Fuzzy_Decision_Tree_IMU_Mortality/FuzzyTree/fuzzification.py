import skfuzzy as fuzz
import numpy as np
from .FuzzyVars import *
from scipy.optimize import fmin_slsqp
from numpy import percentile, array, diff

def lffTrap(x,m1,m2):
    '''Left Fuzzyfication function'''
    if x < m1:
        return 1.
    elif x > m2:
        return 0.
    else:
        return (m2 - x)/(m2 - m1)
        
def rffTrap(x,m1,m2):
    '''Right Fuzzyfication function'''
    if x < m1:
        return 0.
    elif x > m2:
        return 1.
    else:
        return (x - m1)/(m2 - m1)
        
def cffTrap(x,m1,m2,m3,m4):
    '''Center Fuzzyfication function'''
    if x < m1:
        return 0.
    elif (x >= m1) & (x < m2):
        return (x - m1)/(m2 - m1)
    elif (x >= m2) & (x < m3):
        return 1.
    elif (x >= m3) & (x < m4):
        return (m4 - x)/(m4 - m3)
    elif (x >= m4):
        return (0.)
def zeroLine(x):
    return (0.)
def fuzz(Variable,VarName, terms,stepsize=0.01):

    #variables
    lowBound = np.min(Variable)
    highBound = np.max(Variable)
    c = np.percentile(Variable,50)


    dataRange = np.arange(lowBound,highBound+1,stepsize)
    nPoints =len(terms)
    std = np.std(Variable)

    #initialization
    linguistic_functions = np.zeros(dataRange.shape[0])    
    fuzz_func = dict()



    #Creating the functions based on the medians
    for i in range(nPoints):
        if i == 0:
            linguistic_functions = cFF(lff, lowBound, highBound)
        elif i == nPoints-1:
            linguistic_functions = cFF(rff,lowBound, highBound)
        else:
            linguistic_functions = cFF(cff,lowBound,c,highBound)

        fuzz_func[terms[i]] = linguistic_functions

    fzFunc = Fuzzification(VarName,**fuzz_func)
    fzVar = fzFunc(Variable)
    
    return (fzFunc,fzVar)

def gaussFuzz(Variable,VarName, terms,stepsize=0.01):

    #variables
    lowBound = np.min(Variable)
    highBound = np.max(Variable)
    binVar = False

    dataRange = np.arange(lowBound,highBound+1,stepsize)
    nPoints =len(terms)
    std = np.std(Variable)

    #initialization
    linguistic_functions = np.zeros(dataRange.shape[0])    
    fuzz_func = dict()

    #binary check
    if np.percentile(Variable,50) == highBound or np.percentile(Variable,50) == lowBound:
        binVar = True 

    #Creating the functions based on the medians
    for i in range(nPoints):
        if i == 0:
            linguistic_functions = cFF(fuzz.gaussmf, lowBound+std, std)
        elif i == nPoints-1:
            linguistic_functions = cFF(fuzz.gaussmf,highBound-std, std)
        else:
            if binVar:
                linguistic_functions = cFF(zeroLine)
            else:
                p = np.percentile(Variable,100*i/(nPoints-1))
                linguistic_functions = cFF(fuzz.gaussmf,p,std)

        fuzz_func[terms[i]] = linguistic_functions

    fzFunc = Fuzzification(VarName,**fuzz_func)
    fzVar = fzFunc(Variable)
    
    return (fzFunc,fzVar)


#work in progress
def trapFuzz(Variable,VarName, terms,stepsize=0.01):

    #variables
    lowBound = np.min(Variable)
    highBound = np.max(Variable)

    dataRange = np.arange(lowBound,highBound+1,stepsize)
    nPoints =len(terms)
    std = np.std(Variable)

    #initialization
    linguistic_functions = np.zeros(dataRange.shape[0])    
    fuzz_func = dict()

    #Creating the functions based on the medians
    for i in range(nPoints):
        p = np.percentile(Variable,100*i/(nPoints-1))
        plow = np.percentile(Variable,100*i/((nPoints-1)*2))
        print(100*i/((nPoints-1)*2))
        
        phigh = np.percentile(Variable,100*(2*nPoints-3)/((nPoints-1)*2))
        if i == 0:
            linguistic_functions = cFF(lffTrap, plow,np.percentile(Variable,100*(i+1)/(nPoints-1)))
        elif i == nPoints-1:
            linguistic_functions = cFF(rffTrap,np.percentile(Variable,100*(i-1)/(nPoints-1)), phigh)
        else:
            linguistic_functions = cFF(cffTrap,np.percentile(Variable,100*(i-1)/(nPoints-1)),np.percentile(Variable,100*(2*i-1)/((nPoints-1)*2)),np.percentile(Variable,100*(2*i+1)/((nPoints-1)*2)),np.percentile(Variable,100*(i+1)/(nPoints-1)))

        fuzz_func[terms[i]] = linguistic_functions

    fzFunc = Fuzzification(VarName,**fuzz_func)
    fzVar = fzFunc(Variable)
    
    return (fzFunc,fzVar)


## Functions for aggregation
def algebraicAND(a,b):
    return np.multiply(a,b)

def algebraicOR(a,b):
    return np.subtract(np.add(a, b), algebraicAND(a,b))

def lukasiewiczAND(a,b):
    return np.maximum(np.subtract(np.add(a, b), 1),np.zeros(len(a)))

def lukasiewiczOR(a,b):
    return np.minimum(np.add(a, b), np.ones(len(a)))

def einsteinAND(a,b):
    return np.divide(algebraicAND(a,b), np.subtract(2, algebraicOR(a,b)))

def einsteinOR(a,b):
    return np.divide(np.subtract(algebraicOR(a,b), algebraicAND(a,b)) , np.add(1,algebraicAND(a,b)) )

def OWA(a,b,weights):
    sortedArray = np.sort(np.concatenate((a,b),axis=0), axis=0)
    return np.matmul(weights,sortedArray)



