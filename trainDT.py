

"""
File: trainDT.py
Description: Training the decision with the given training data samples
Course Name: Foundations of Intellient Systems
Course Code: CSCI630
__author__ = "Abhilash Peddinti, Girish Kumar Reddy Veerepalli"
"""
import math
import sys
import operator
import numpy as np
import matplotlib.pyplot as plot

class Tree:
    """
    Tree Class to represent nodes in a tree
    """

    __slots__=('data','greater','lesser','classCounter','leaf','split')
    
    def __init__(self):
        """
        Initializing the parameters required for the node in a tree
        :return:None
        """
        self.data=None
        self.greater=None
        self.lesser=None
        self.classCounter=[0,0,0,0]
        self.leaf=False
        self.split=None
        
    def setData(self,data):
        """
        Sets the current node to the given data and assigns the node as a leaf node
        :param data: Class of the current node
        :return: None
        """
        self.data=data
        self.leaf=True    
    def setSplit(self,split):
        """
        Sets the split attribute and split value to the current node
        :param split: A tuple containing the split attribute and split value
        :return: None
        """
        self.split=split    
    def setGreater(self,Node):
        """
        Sets the given node to greater value of the current node
        :param Node: A node in the tree
        :return: None
        """
        self.greater=Node
    def setLesser(self,Node):
        """
        Sets the given node to lesser link of the current node
        :param Node: A node in the tree
        :return: None
        """
        self.lesser=Node
    
    
     
def readData(file):
    """
    Reads the files and split into input attributes, output classes
    and combination of both input and output values
    :param file : training data file
    :return     : list of input values, output values, total data
    """
    
    inputValues=list()
    outputValue=list()
    totalData=list()
    
    with open(file) as fp :
        for line in fp:
            if line.strip( ) == ' ':
                continue
            attributeValue = line.strip().split(",")
            inputValue1 = float(attributeValue[0])
            inputValue2 = float(attributeValue[1])
            
            inputValues+=[[inputValue1]+[inputValue2]]
            outputValue+=[int(attributeValue[2])]
            totalData+=[[inputValue1]+[inputValue2]+[int(attributeValue[2])]]
    
   
    return inputValues,outputValue,totalData


def countClasses(dataSamples):
    """
    Calculates the number of values in each class
    :param dataSamples: Samples to be classified
    :return: list of number of values in each class
    """

    classCountList=[0,0,0,0]
    for sample  in dataSamples:
        i = sample[-1]
        classCountList[i-1]+=1
    return classCountList

def probabilityCalculation(dataSamples):
    """
    Calculates the probability of each class of samples in total samples
    :param dataSamples: dataSamples to be classified
    :return: A list of probabilities for each class of samples
    """
    probability = []
    countList = []
    countList = countClasses(dataSamples)
    for i in countList:
        probability.append((i/sum(countList)))
    
    return probability
    

def entropyCalculation(dataSamples):
    """
    Calculates the entropy of the given datasamples.
    :param samples: dataSamples to be classified
    :return: total entropy of the dataSamples
    """
    probabilityList = []
    probabilityList = probabilityCalculation(dataSamples)
    entropy = 0
    for i in probabilityList:
        if i!=0:
            entropy = entropy + i*math.log2(i)
    
    return -entropy

def splitbyMidpoint(attributeValues,attribute,i):
    """
    Splits the attribute values by calculating the given attribute's midpoint
    :param attributeValues: attributes to be classified
    :param attribute: attribute based on which samples are to be split
    :param i: index of the attribute
    :return: midpoint of the attribute value and the split data based on the midpoint
    """
    midpoint = 0
    greaterList = list() 
    lesserList = list()
    midpoint = (attributeValues[i][attribute] + attributeValues[i+1][attribute]) / 2
    
    for value in range(len(attributeValues)):
        if attributeValues[value][attribute] <= midpoint:
            lesserList.append(attributeValues[value])
        else:
            greaterList.append(attributeValues[value])
    return midpoint,lesserList, greaterList

def informationGainCalculation(attributeValues,attributes,totalEntropy):
    """
    Calculates the information gain for the given list of attribute values
    :param attributeValues: values of the attribute to be classified
    :param attribute: attribute of attributeVamples
    :param currentEntropy: entropy of the current node
    :return: information gain calculated based on attributes.
    """
    informationGain=[[] for _ in range(attributes)]
    for attribute in range(attributes):
        attributeValues.sort(key=operator.itemgetter(attribute))
        for i in range(len(attributeValues)-1):
            midpoint, lesserList, greaterList = splitbyMidpoint(attributeValues, attribute, i)
            lowerEntropy = entropyCalculation(lesserList)
            higherEntropy = entropyCalculation(greaterList)

            difference = ((len(lesserList)/len(attributeValues))*lowerEntropy +
                             (len(greaterList)/len(attributeValues))*higherEntropy)

            informationGain[attribute]+=[(midpoint , totalEntropy-difference)]
    
    return informationGain


def chisquareTestCalculation(rootNode, significance):
    """
    Performs the chisquare test on the given root node with a given value of significance
    :param rootNode: node of the tree to perform pruning
    :param significance: significance to be considered on chi-square testing.
    :return: deciding factor for the pruning
    """
    
    leftSum = sum(rootNode.lesser.classCounter)
    rightSum = sum(rootNode.greater.classCounter)
    totalSum = sum(rootNode.classCounter)
    left= leftSum/totalSum
    right= rightSum/totalSum
    
    leftNodeClass=list()
    rightNodeClass=list()
    distribution = 0
    
    for value in rootNode.classCounter:
        leftNodeClass.append(value*left)
        rightNodeClass.append(value*right)
    
    
   
    for index in range(len(rootNode.classCounter)):
        if leftNodeClass[index]!=0:
            distribution +=(rootNode.lesser.classCounter[index] - leftNodeClass[index])**2/leftNodeClass[index]
        if rightNodeClass[index]!=0:
            distribution += (rootNode.greater.classCounter[index]-rightNodeClass[index])**2/rightNodeClass[index]
    
    if significance == 0.01:
        return distribution < 11.345
    else:
        return distribution < 7.185
    

def pruning (rootNode, significance, nodes,nodeLeaves,totalCount=0) :  
    """
    Prunes the decision tree based on chi-square test calculation.
    :param rootNode: root node or current node of the tree
    :param significance: significance to be considered on chi-square testing.
    :param nodes: number of nodes in tree
    :param nodeLeaves: list of leaves in tree
    :param totalCount: level of the current node in the tree
    :return: pruned nodes and pruned leaves of the tree
    """


    if not rootNode.leaf:
        nodeLeaves,nodes = pruning(rootNode.lesser,significance,nodes,nodeLeaves,totalCount+1)
        nodeLeaves,nodes = pruning(rootNode.greater,significance, nodes,nodeLeaves,totalCount+1)
        if rootNode.lesser.leaf and rootNode.greater.leaf:
            if chisquareTestCalculation(rootNode, significance):
                rootNode.data,_=max(enumerate(rootNode.classCounter),key=operator.itemgetter(1))
                
                
                rootNode.data+=1
                rootNode.lesser=rootNode.greater=None
                rootNode.leaf=True
                
                nodeLeaves.remove(totalCount+1)
                nodeLeaves.remove(totalCount+1)
                nodeLeaves.append(totalCount)
                return nodeLeaves,nodes-2
    
    return nodeLeaves, nodes

def decisionTreeLearning(attributeValues,attributes,rootNode,currentNode,nodeLeaves=list(),nodes=1,totalCount=0):
    """
    Decision Tree Learning is performed based on the given attribute values
    :param attributeValues: data samples to be classified
    :param attributes: attributes of samples
    :param root Node: parent values of the values that are in recursion
    :param currentNode: Root or current node of the tree
    :param nodes: number of nodes in the tree
    :param nodeLeaves: list of leaves in the tree
    :param totalCount: level of the current node at a particular recursion
    :return: Root node of the tree along with count of total number of nodes and list of nodeLeaves.
    """
    val = 0
    for value in attributeValues:
        if attributeValues[0][-1] != value[-1]:
            val = 1
    
   
    if val == 0:
        currentNode.data=str(attributeValues[0][-1])
        currentNode.leaf=True
        currentNode.split="BASE CASE"
        nodeLeaves.append(totalCount)
        
        return None,nodeLeaves,nodes
    
    #print("ROWS ",rows)
    totalEntropy = entropyCalculation(attributeValues)
    informationGain= informationGainCalculation(attributeValues,attributes,totalEntropy)
    #for i in range(attributes):
     #   maximumList = [max(informationGain[i], key = operator.itemgetter(1))]
                           
    
    maximumList=[max(informationGain[i],key=operator.itemgetter(1)) for i in range(attributes)]
    if maximumList[0][1] < maximumList[1][1]:
        splitAttribute=1
        splitValue=maximumList[1][0]
    else:
        splitAttribute=0
        splitValue=maximumList[0][0]
    
    #Split by value
    greaterList = list()
    lesserList = list()
    for value in range(len(attributeValues)):
        if attributeValues[value][splitAttribute] <= splitValue:
            lesserList.append(attributeValues[value])
        else:
            greaterList.append(attributeValues[value])
    
    currentNode.split=(splitAttribute+1,splitValue)
    
    for childValue in [lesserList,greaterList]:
        childnodeTree=Tree()
        if childValue == greaterList:
            childnodeTree.classCounter=countClasses(greaterList)
            currentNode.greater=childnodeTree
            
        else:
            childnodeTree.classCounter=countClasses(lesserList)
            currentNode.lesser=childnodeTree
            
        _,leaves,nodes=decisionTreeLearning(childValue,attributes,attributeValues,childnodeTree,nodeLeaves,nodes+1,totalCount+1)
    
    return currentNode,nodeLeaves,nodes

def printTree(root,level=0):
    """
    prints the tree from the given root of the tree.
    :param root: root node or current node in the tree
    :param level: level of current node in the tree
    :return: None
    """
    if root:
        print(" "*level,"Attribute Value:", root.split, "CLASS CLASSIFICATION: ",root.classCounter,  "Class:", root.data)
        printTree(root.lesser, level+3)
        printTree(root.greater, level+3)
        

def decisionBoundary(root,figure,fileName):
    """
    It plots a graph of decision boundary for all the data samples in the classes
    :param root: root node of the decision tree
    :param figure: figure in plot
    :param fileName: training data file
    :return: decision plot
    """
    stepValue = 0.01
    classClassification = [1,2,3,4]
    colorClassification=['b','g','r','m']
    markerClassification=['x','+','*','o']
    classesList = ["Bolts", "Nuts", "Rings", "Scraps"]
    
    decisionPlot=figure.add_subplot(111)
    attributeValues,classes,_=readData(fileName)
    attributeValues=np.array(attributeValues)
    classes=np.array(classes)
    
    #Classifications
     
    attribute1,attribute2=np.meshgrid(np.arange(0,1,stepValue),np.arange(0,1,stepValue))

    predicted_class=[]
    for a in range(attribute1.shape[0]):
        predicted_class.append([])
        for b in range(attribute1.shape[1]):
            result =[attribute1[a][b],attribute2[a][b]]
            value =np.array(result)
            val=root
            while(val.data == None):
                splitAttribute,splitValue= val.split
                if value[int(splitAttribute)-1]>float(splitValue):
                    val=val.greater
                else:
                    val=val.lesser
            
            
            predicted_value=val.data
            predicted_class[a].append(predicted_value)
    decisionPlot.contourf(attribute1,attribute2,np.array(predicted_class))
    
    for i in classClassification:
        attribute1=[]
        attribute2=[]
        for j in range(len(attributeValues[:])):
            if classes[j]==i:
                attribute1 +=[attributeValues[j][0]]
        for k in range(len(attributeValues[:])):
            if classes[k]==i:
                attribute2 +=[attributeValues[k][1]]
        
        
        decisionPlot.scatter(attribute1,attribute2,label=classesList[i-1],marker=markerClassification[i-1],color=colorClassification[i-1],s=100)
        
    decisionPlot.legend(loc='upper right')
    decisionPlot.set_xlabel(" Six Fold Rotational symmetry")
    decisionPlot.set_ylabel("Eccentricity")
    decisionPlot.set_title("Decision Boundary")
    return decisionPlot

def writeData(file,root):
    """
    Writes the decision tree to a file from the root node of the tree
    :param file: file to which the data of the tree is written
    :param root: root node or current node of the decision tree
    :return: None
    """
    
    if root is None:
        file.write("!,")
        return
    
    if not root.leaf:
        writeFile=str(root.split[0])+'-'+str(root.split[1])+'-'+ str(root.data)+','
    else:
        writeFile='$'+'-'+'$'+'-'+str(root.data)+','
    
    
    
    file.write(writeFile)
    writeData(file,root.lesser)
    writeData(file,root.greater)
    
    
def main():
    """
    The main function to call other functions
    :return: None
    """
    
    fileName=sys.argv[1]
    numberofAttributes = 2
    significance = 0.01
    inputData,outputData,totalData=readData(fileName)
    DecisionTree=Tree()
    print('\n ///////////////////DECISION TREE////////////////////////////////// \n')
   
    rootNode,nodeLeaves,nodes=decisionTreeLearning(totalData,numberofAttributes,totalData,DecisionTree)
    printTree(rootNode)


    print("\n THE VALUES GENERATED FOR THE DECISION TREE ARE: \n")
    print("NUMBER OF NODES GENERATED" , nodes )
    print("NUMBER OF LEAF NODES GENERATED :", len(nodeLeaves))
    print("MAXIMUM DEPTH OF THE DECISION TREE ",max(nodeLeaves))
    print("MINIMUM DEPTH OF THE DECISON TREE ",min(nodeLeaves))
    print("AVERAGE DEPTH OF THE DECISION TREE ",sum(nodeLeaves)/len(nodeLeaves))
    
    figure=plot.figure(1)
    decisionBoundary(rootNode,figure,fileName)
    
    file=open('DecisionTree.csv','w')
    writeData(file,rootNode)
    file.close()
    
    print (" \n /////////////////////// PRUNED DECISION TREE ////////////////////////// \n")

    prunedLeaves, prunedNodes = pruning(rootNode, significance, nodes, nodeLeaves)
    printTree(rootNode)
    

    
    print(" \n THE VALUES GENERATED FOR THE PRUNED DECISION TREE ARE: \n ")
    print("NUMBER OF NODES GENERATED:" , prunedNodes )
    print("NUMBER OF LEAF NODES GENERATED :", len(prunedLeaves))
    print("MAXIMUM DEPTH OF THE DECISION TREE: ",max(prunedLeaves))
    print("MINIMUM DEPTH OF THE DECISON TREE: ",min(prunedLeaves))
    print("AVERAGE DEPTH OF THE DECISION TREE: ",sum(prunedLeaves)/len(prunedLeaves))
    
    figure1=plot.figure(2)
    decisionBoundary(rootNode,figure1,fileName)
    plot.show()
    
    file=open('PrunedDecisionTree.csv','w')
    writeData(file,rootNode)
    file.close()
    
    
    
if __name__ == "__main__":   
    main()

