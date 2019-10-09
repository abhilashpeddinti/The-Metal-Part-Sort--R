

"""
File: executeDT.py
Description: Classifies the test data based on the provided decision tree
Course Name: Foundations of Intelligent Systems
Course Code: CSCI630
__author__ = "Abhilash Peddinti, Girish Kumar Reddy Veerepalli"
"""

import sys
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
        self.data=0
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
            if line.strip( ) == '':
                continue
            attributeValue = line.strip().split(",")
            inputValue1 = float(attributeValue[0])
            inputValue2 = float(attributeValue[1])
            
            inputValues+=[[inputValue1]+[inputValue2]]
            outputValue+=[int(attributeValue[2])]
            totalData+=[[inputValue1]+[inputValue2]+[int(attributeValue[2])]]
    
   
    return inputValues,outputValue,totalData



def classify(sample,currentNode):
    """
    classifies the given data Samples
    :param sample: Sample to be classified
    :param currentNode: Root node or current node of the tree
    :return: class of the given sample
    """
    
    while(currentNode.data == 0):
        splitAttribute,splitValue= currentNode.split
        if sample[int(splitAttribute)-1]>float(splitValue):
            currentNode = currentNode.greater
        else:
            currentNode = currentNode.lesser
    return currentNode.data
            


def treeList(tree):
    """
    Converts the given csv file of a tree into list
    :param treeFile: csv file containing the tree in DFS traversal
    :return: list of nodes in the tree in DFS traversal order
    """
    with open(tree) as fp:
        line = fp.read()
        nodesList = line.strip().split(',')
        return nodesList


def extractTree(nodesList, rootNode):
    """
    Given list of nodes it extracts the attribute values and split attribute and split value of the atrribute
    :param nodesList: list of nodes in the tree 
    :param rootn=Node: Root node of the tree
    :return: remaining nodes in list of nodes 
    """
    if len(nodesList) == 0:
        return
    if nodesList[0] == '!':
        return nodesList[1:]

    splitAttribute, splitValue, attributeValue = nodesList[0].strip().split('-')
    nodesList = nodesList[1:]
    
    if splitAttribute != splitValue or splitAttribute != '$' or splitValue != '$':
        rootNode.setSplit((splitAttribute, splitValue))
    else:
        rootNode.setSplit("Base case")
        rootNode.setData(attributeValue)
        return nodesList[2:]
        
   
    leftTree = Tree()
    rightTree = Tree()
    rootNode.setLesser(leftTree)
    rootNode.setGreater(rightTree)
    nodesList = extractTree(nodesList, leftTree)

    
    
    nodesList = extractTree(nodesList, rightTree)

    return nodesList


def printTree(rootNode, level = 0):
    """
    Given the root node of the tree it prints the tree.
    :param level: level of current node in the tree
    :param node: Root node or current node of the tree
    """
    
    if rootNode:
        print("  " * level, rootNode.split, "CLASS:", rootNode.data)
        printTree(rootNode.lesser, level + 3)
        printTree(rootNode.greater, level + 3)


def decisionBoundary(root, figure, fileName):
    """
    It plots a graph of decision boundary for all the data samples in the classes
    :param root: root node of the decision tree
    :param figure: figure in plot
    :param fileName: test data file
    :return: decision plot
    """
    stepValue = 0.001
    classClassification = [1, 2, 3, 4]
    colorClassification = ['b', 'g', 'r', 'm']
    markerClassification = ['x', '+', '*', 'o']
    classesList = ["Bolts", "Nuts", "Rings", "Scraps"]
    decisionPlot = figure.add_subplot(111)
    attributeValues, classes, _ = readData(fileName)
    attributeValues = np.array(attributeValues)
    classes = np.array(classes)
    
    

    attribute1, attribute2 = np.meshgrid(np.arange(0, 1, stepValue), np.arange(0, 1, stepValue))

    predicted_class = []
    for i in range(attribute1.shape[0]):
        predicted_class.append([])
        for j in range(attribute1.shape[1]):
            result = [attribute1[i][j], attribute2[i][j]]
            predicted_value = classify(np.array(result), root)
            predicted_class[i].append(predicted_value)

    decisionPlot.contourf(attribute1, attribute2, np.array(predicted_class))

    for a in classClassification:
        attribute1=[]
        attribute2=[]
       
        for j in range(len(attributeValues[:])):
        
            if classes[j]==a:
                attribute1 +=[attributeValues[j][0]]
        for k in range(len(attributeValues[:])):
            if classes[k]==a:
                attribute2 +=[attributeValues[k][1]]
        
        
        decisionPlot.scatter(attribute1, attribute2, color=colorClassification[a - 1], marker=markerClassification[a - 1]
                              , label=classesList[a - 1], s=100)

    decisionPlot.legend(loc='upper right')
    decisionPlot.set_xlabel("Six fold Rotational Symmetry")
    decisionPlot.set_ylabel("Eccentricity")
    decisionPlot.set_title("Decision boundary")
    return decisionPlot


def confusionMatrixCalculation(node,fileName):
    """
    Calculates the confusion matrix by predicting the values for the given root node of the decision tree
    :param node: root node of the decision tree
    :param fileName: test_data csv file
    :return:confusion matrix for the given decision tree
    """
    attributeValues, classes, _ = readData(fileName)
    attributeValues = np.array(attributeValues)
    numberofClasses = 4

    confusionMatrix = []

    for _ in range(numberofClasses):
        confusionMatrix.append([])
        for _ in range(numberofClasses):
            confusionMatrix[-1].append(0)
    for val in range(attributeValues.shape[0]):
        result = classes[val]
        predicted_value = classify(attributeValues[val], node)
        confusionMatrix[int(predicted_value) - 1][int(result) - 1] += 1

   
    return confusionMatrix


def printData(confusion_matrix):
    """
    Prints the confusion matrix after classification
    :param confusion_matrix: A confusion matrix
    :return: None
    """
    numberofClasses = 4
    confusion_matrix = np.array(confusion_matrix)
    value = 5
    
    print("MATRIX ", "*\t",
          padding("Class 1 ", value), "\t|\t",
          padding("Class 2 ", value), "\t|\t",
          padding("Class 3 ", value), "\t|\t",
          padding("Class 4 ", value), "\t*\tTotal")
    for i in range(numberofClasses):
        print("Class " + str(i + 1), " *\t",
              padding(confusion_matrix[i][0], value), "\t\t|\t",
              padding(confusion_matrix[i][1], value), "\t\t|\t",
              padding(confusion_matrix[i][2], value), "\t\t|\t",
              padding(confusion_matrix[i][3], value), "\t\t*\t",
              padding(np.sum(confusion_matrix[i]), value))
    print("--------------------------------------------------------------------------------------------------------------\n")
    print("Total    -->\t",
          padding(np.sum(confusion_matrix[:, 0]), value), "\t\t+\t",
          padding(np.sum(confusion_matrix[:, 1]), value), "\t\t+\t",
          padding(np.sum(confusion_matrix[:, 2]), value), "\t\t+\t",
          padding(np.sum(confusion_matrix[:, 3]), value), "\t\t=\t",
          padding(np.sum(confusion_matrix[:, 0:4]), value))
    print("--------------------------------------------------------------------------------------------------------------\n")


def padding(input_value, value):
    """
    Helper method for padding the input number so that program can maintain
    symmetry in tables
    :param input_num: a number
    :param digit: number of digits required
    :return: Padded value
    """
    padding_value = str(input_value)
    for i in range(value - len(str(input_value))):
        padding_value += " "
    return padding_value


def classification(fileName, node):
    """
    calculate the number of correctly classified and incorrectly classified data samples in the given test data set
    :param fileName:test data csv file
    :param node: root node or current node of the tree
    :return: Number of incorrectly and correctly classified samples
    """
    wrongPrediction = [0, 0, 0, 0]
    exactPrediction = [0, 0, 0, 0]
    correct_classes = 0
    incorrect_classes = 0
    total_classes = 0
    numberofClasses = 4
    meanofClassAccuracy = 0
    attributeValues, classes, _ = readData(fileName)
    attributeValues = np.array(attributeValues)
    for i in range(attributeValues.shape[0]):
        predicted_value = classify(attributeValues[i], node)
        if int(classes[i]) != int(predicted_value):
            wrongPrediction[classes[i] - 1] += 1
            incorrect_classes += 1
        else:
            exactPrediction[classes[i] - 1] += 1
            correct_classes += 1
    total_classes = correct_classes + incorrect_classes
    accuracy_value= correct_classes / total_classes
    class_sum = numberofClasses * total_classes
    
    for count in range(numberofClasses):
        incorrect_classes = wrongPrediction[count]
        correct_classes = exactPrediction[count]
        class_sum = numberofClasses * (correct_classes + incorrect_classes)
        
        meanofClassAccuracy += correct_classes / class_sum
   
   
    return accuracy_value, meanofClassAccuracy


def profitCalculation(confusion_matrix):
    """
    Calculates the profit from the confusion matrix
    :param confusion_matrix: confusion matrix
    :return: None
    """
    numberofClasses = 4
    profits = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]
    totalProfit = 0
    for count in range(numberofClasses):
        for counter in range(numberofClasses):
            totalProfit += confusion_matrix[count][counter] * profits[count][counter]

    return totalProfit

def main():
    """
    Main method
    return: None
    """
    nodesList = treeList(sys.argv[1])  # DecisionTree.csv
    file = sys.argv[2]    # test_data file

    node = Tree()
    extractTree(nodesList, node)
    
    print("\n///////////////////////////////// DECISION TREE //////////////////////////////////////\n")
    printTree(node)
    
    print("\n ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ CLASSIFICATION OF TEST DATA ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    
    
    print("\n-------------------------------------------CONFUSION MATRIX ----------------------------------------\n")
    
    confusion_matrix = confusionMatrixCalculation(node, file)
    printData(confusion_matrix)
    
    classifier1, classifier2 = classification(file, node)
    print("\n--------------------------------- RECOGNITION RATE --------------------------------------------")
    print("TOTAL ACCURACY OF THE CLASSIFICATION : ", classifier1)
    print(" MEAN OF THE EACH CLASS ACCURACY : ", classifier2)
    print("-------------------------------------------------------------------------------------------------\n")
    
    total_profit = profitCalculation(confusion_matrix)
    print('\n -------------------------------------- PROFIT ---------------------------------------')
    print('TOTAL PROFIT:', total_profit)
    print('----------------------------------------------------------------------------------------\n')
    
    figure = plot.figure()
    decisionBoundary(node, figure, file)
    plot.show()


if __name__ == "__main__":
    main()

