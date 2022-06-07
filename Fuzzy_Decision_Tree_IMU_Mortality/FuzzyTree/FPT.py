from .FuzzyVars import *
from numpy import zeros, isnan, minimum, maximum
from .fuzzification import *
import pandas as pd


class Node:

    def __init__(self,name,fuzzySubset, similarity,var=None,ling=None, isLeaf=False,isClass=False,
    Children=None, Parent=None, level=0):
        self._name = name
        self._var = var
        self._ling = ling
        self._fuzzySubset = fuzzySubset
        self._similarity = similarity
        self._isLeaf = isLeaf
        self._isClass = isClass
        self._Children = Children
        self._Parent = Parent
        self._level = level
        self._agg = None

class FPT_tree:

    def __init__(self, name, levels, maxdepth=5):
        self._name = name
        self._lowestLeaf = []
        self._aggList = []
        self._candidate =[]
        self._levels = levels # Categorical levels
        self._fuzzySet = {} # Includes all the fuzzified variables
        self._labelsSet = []
        self._maxdepth = maxdepth
        #store the lowest leafs of the tree. Track what level each node is on, where class node is level 0

    def fit(self, fuzzySet, output, features, c):
        
        #fuzzify data into fuzzy sets
        for key in features:    
             self._fuzzySet[key] = fuzzySet[key]
        
        self._labelsSet = fuzzySet[output]
        
        #Create a pool of simple trees with class

        #making a tree for only one class
        #self._classSubset = np.divide(self._labelsSet._values[c],sum(self._labelsSet._values[c]))
        self._classSubset = self._labelsSet._values[c]
        Pool = {}
        for feat in self._fuzzySet.keys():
            for lings in fuzzySet[feat]._values.keys():
                

                #subset =np.divide(fuzzySet[feat][lings]._value,sum(fuzzySet[feat][lings]._value) )
                subset = fuzzySet[feat][lings]._value
                name = lings + " " + feat
                similarity = self.similarity(self._classSubset, subset)
                Pool[name] = Node(name,subset, similarity,  var=feat, ling=lings,isLeaf=True, isClass=True)
        sortedKeys = sorted(Pool,key=lambda name:Pool[name]._similarity,reverse=True)

        #Choose the simple tree that has the most similarity to start
        maxName = sortedKeys[0]
        maxVal = Pool[maxName]._similarity
       

        #Remove best one from the pool
        self._candidate = Pool.pop(maxName)
        self._lowestLeaf.append(self._candidate  ) #This is the first SPT to be apart of the tree
        sortedKeys.remove(maxName)
        
        aggFuncs = {'MIN-AND':minimum,'MAX-OR': maximum,
                    #'Algebraic-AND': algebraicAND,'Algebraic-OR':algebraicOR,
                     }
                    #'lukasiewicz-AND': lukasiewiczAND,'lukasiewicz-OR': lukasiewiczOR,
                    # 'MIN-AND':minimum,'MAX-OR': maximum, 'einstein-AND':einsteinAND,'einstein-OR':einsteinOR # min is AND max is OR
        #Search for through trees and aggregations to increase similarity. If similarity can not increase or depth is too large. stop growing the tree
        for i in range(self._maxdepth):
            bestAgg = None
            bestSlave = None
            best = self._candidate._similarity
            for spt in sortedKeys:
                for agg in aggFuncs:
                    #similarity to beat
                    
                    together = self.aggregate_to_similarity(self._candidate,Pool[spt],aggFuncs[agg])
                    if together > best:
                        bestAgg = agg
                        bestSlave = spt
                        best = together


            if bestSlave != None:
                self._candidate = self.aggregate(self._candidate,Pool[bestSlave], aggFuncs[bestAgg],bestAgg)
                self._aggList.append(aggFuncs[bestAgg])
                self._lowestLeaf.append( Pool.pop(bestSlave) ) #This is the first SPT to be apart of the tree
                sortedKeys.remove(bestSlave)    

    #Aggregate two Nodes together
    def aggregate(self,node1, node2, func, funcName):

        #new inputs for new node
        name = "(" +node1._name +" "+ funcName +" "+ node2._name +")"
        subset = func(node1._fuzzySubset, node2._fuzzySubset)
        #subset = np.divide(subset,sum(subset))
        similarity = self.similarity(self._classSubset,subset)
        
        newParent = Node(name ,subset, similarity, isLeaf=False, isClass=True, Children=[node1, node2], level=node1._level+1)

        #Adjusting children nodes
        node1._isClass = False
        node2._isClass = False

        #node1._level += 1
        node2._level = node1._level

        node1._Parent = newParent
        node2._Parent = newParent

        newParent._agg = func

        return newParent
    
    def aggregate_to_similarity(self,node1, node2, func):
        subset = func(node1._fuzzySubset, node2._fuzzySubset)
        similarity = self.similarity(self._classSubset,subset)

        return similarity

    def similarity(self,fuzzC, fuzzN):
        #must be the same size
        #normalize
        #fuzzC = np.divide(fuzzC,sum(fuzzC))
        #fuzzN = np.divide(fuzzN,sum(fuzzN))
        n = 0.
        d = 0.
        for c in range(0,len(fuzzC)-1):
            #print vals[c]
            n += max(fuzzC[c],fuzzN[c])
            d += min(fuzzC[c],fuzzN[c])
        
        d += 1
        return n/d

    def _output_Node_tree(self, node):
        output = '"%s" \n' % (node._name)
        
        if node._isLeaf:
            return output
        
        for child in node._Children:
            cad = '"%s"->"%s" \n' % ( child._name,node._name)
            output += cad
            output += self._output_Node_tree(child)

        return output

    def graphviz_export(self):
        tree = self._output_Node_tree(self._candidate)
        filename = self._name
        f = open(filename + ".dot", "w")
        print("digraph G{",file=f)

        print(tree,file=f)

        print("}",file=f)
        
        f.close()
        return filename
    

    def predict(self, fuzzifiedSet):
        
        val = []
        for leaf in self._lowestLeaf:
            if leaf._isClass:
                return fuzzifiedSet[leaf._var]._values[leaf._ling]
            val.append(fuzzifiedSet[leaf._var]._values[leaf._ling]) 
        val = np.array(val)
        out = val[0,:]
        for i in range(len(self._aggList)-1):
            out = self._aggList[i](out,val[i+1,:])

        return out



class FPT_forest():
    def __init__(self, name, levels, maxdepth=5):
        self._name = name
        self._levels = levels
        self._maxdepth = maxdepth
        self._treeList = {}

    def fit(self, fuzzySet, output, features):
        for c in self._levels:
            self._treeList[c] = FPT_tree(c + " " + output, self._levels, self._maxdepth)
            self._treeList[c].fit(fuzzySet, output, features, c)
    

    def predict(self, fuzzifiedSet,n):
        p = np.zeros((n,3) )
        for tree in range(len(self._treeList)):
            A = list(self._treeList.values())[tree]
            
            p[:,tree] = A.predict(fuzzifiedSet)

        print(p)
        return np.argmax(p, axis=1)

            

    def graphviz_export(self):
        f = open("convertPNG", "w")
        
        for c in self._treeList.keys():
            filename = self._treeList[c].graphviz_export()
            print("dot -Tpng " + "'"+filename +".dot" +"'"+ " -o " + "'"+filename + ".png"+"'" ,file=f)

        f.close()
