#Authors : Girish Kumar Reddy Veerepalli & Abhilash Peddinti
#File Name : executeMLP.py
#The ecexuteMLP program takes a trained weights file and a test dataset file as inputs and
#implements the network on the dataset.
#Outputs the prediction accuracy rate.
#Prints the confusion matrix.
#Calculates the profit.
#A class region image.

import numpy as np
import matplotlib.pyplot as plt
from trainMLP import *
import sys

NUM_LAYER=2

def retrieve_weights(network):
    """
    Function to retrieve the network weights
    :param network: neural network input
    :return: weights of the neural network
    """
    tempWeights = []
    count = 0
    for each_layer in self.neural_network:
        tempWeights.append([])
        for neuron in each_layer.neurons:
            tempWeights[count].append(neuron.weights)
        count += 1
    return tempWeights

def Classification(attributes,labels,network):
    """
    This function calculates the no. of correct and incorrect classifications.
    Helps in estimating the prediction accuracy of our network.
    Prints the prediction accuracy as well.
    :param attributes: attributes data
    :param labels: labels data
    :param network: our neural network
    :return: None
    """

    #prection count holders
    predictionClass1 = [0, 0]
    predictionClass2 = [0, 0]
    predictionClass3 = [0, 0]
    predictionClass4 = [0, 0]

    for index_track in range(attributes.shape[0]):
        classPredicted = predict_class(attributes[index_track],network)

        if compare_prediction(labels[index_track],classPredicted):
            if decode_class(labels[index_track]) == 1:
                predictionClass1[0] += 1
            elif decode_class(labels[index_track]) == 2:
                predictionClass2[0] += 1
            elif decode_class(labels[index_track]) == 3:
                predictionClass3[0] += 1
            elif decode_class(labels[index_track]) == 4:
                predictionClass4[0] += 1
        else:
            if decode_class(labels[index_track]) == 1:
                predictionClass1[1] += 1
            elif decode_class(labels[index_track]) == 2:
                predictionClass2[1] += 1
            elif decode_class(labels[index_track]) == 3:
                predictionClass3[1] += 1
            elif decode_class(labels[index_track]) == 4:
                predictionClass4[1] += 1

    correctPredictionCount      = predictionClass1[0] + predictionClass2[0] + predictionClass3[0] + predictionClass4[0]
    incorrectPredictionCount    = predictionClass1[1] + predictionClass2[1] + predictionClass3[1] + predictionClass4[1]
    accuracy    =   correctPredictionCount/(correctPredictionCount+incorrectPredictionCount)

    print("\n****************************************************************************\n"+
          "----------------------------Accuracy Statistics-----------------------------\n"+
          "****************************************************************************\n")
    print("Accuracy for the testing data is calculated as stated below :")
    print("Prediction Accuracy = no. of correct predictions/total predictions(correct+incorrect)")
    print("Prediction Accuracy Obtained for " + sys.argv[2] + " :",accuracy)
    print("\n****************************************************************************\n")


def print_data(confusion):
    """
    This function prints the classificatin details that are obtained after the classification process.
    Prints the confusion matrix after proper formatting
    :param confusion: confusion matrix
    :return: None
    """
    print("************************************************************************************")
    print("******************************** Confusion Matrix **********************************")
    print("************************************************************************************")

    confusionMatrix = np.array(confusion)

    print("------------------------------------------------------------------------------------")
    print("| Class  ","|\t",
          "Class 1  ", "\t|\t",
          "Class 2  ", "\t|\t",
          "Class 3  ", "\t|\t",
          "Class 4  ", "\t|\ttotal"," |")
    print("------------------------------------------------------------------------------------")

    digit = 5
    for counter in range(1,5):
        print("| Class " + str(counter),"|\t",
              confusionMatrix[counter-1][0], "  \t\t|\t",
              confusionMatrix[counter-1][1], "  \t\t|\t",
              confusionMatrix[counter-1][2], "  \t\t|\t",
              confusionMatrix[counter-1][3], "  \t\t|\t",
              np.sum(confusionMatrix[counter-1]),"     |")
    print("------------------------------------------------------------------------------------")
    print("| Total   |\t",
          np.sum(confusionMatrix[:,0]), "  \t\t|\t",
          np.sum(confusionMatrix[:,1]), "  \t\t|\t",
          np.sum(confusionMatrix[:,2]), "  \t\t|\t",
          np.sum(confusionMatrix[:,3]), "  \t\t|\t",
          np.sum(confusionMatrix[:,:]), "     |")
    print("------------------------------------------------------------------------------------\n\n")

def configure_network_weight(weight_file,mlp):
    """
    This function initializes the weights to the neroun's in the neural network.
    gets the weights from the specified input file.
    :param weight_file: weights file
    :param mlp: network object
    :return:
    """
    file = open(weight_file, "r")
    for line in file:
        pass
    lastline=line
    weight_vector=lastline.strip().split(",")
    weight_counter = 0
    for layer in mlp.neural_network:
        neurons=layer.neurons
        for neuron in neurons:
            weight=[]
            for counter in range(neuron.inward_count):
                weight.append(float(weight_vector[weight_counter]))
                weight_counter+=1
            neuron.weights = np.array(weight)


def predict_class(input,network):
    """
    This function predicts the class for a given input using our mlp network
    :param input: input to be classified
    :param network: neural network
    :return: classification. typically of the format [1,0,0,0] if it is class 1.
    """
    network_value = forward_prop(network,input)
    output=[0,0,0,0]
    max = np.max(network_value)
    for bit in range(len(network_value)):
        if network_value[bit] == max:
            output[bit] = 1
    return output

def decode_class(labels):
    """
    This function outputs the Class(1,2,3 or 4) from the input labels
    :param labels: the label list
    :return: Actual Class(1,2,3 or 4)
    """
    actualClass = 0
    for counter in range(len(labels)):
        if int(labels[counter]) == 1:
            actualClass = counter + 1
    return int(actualClass)

def compare_prediction(labels,prediction):
    """
    This function compares the actual class against the predicted class.
    :param labels: list of lables
    :param prediction: list of the predicted class
    :return: True if the actual class and predicted class are same.False otherwise.
    """
    for counter in range(len(labels)):
        if labels[counter] != prediction[counter]:
            return False
    return True

def decision_boundary(figure, attributes, labels,network):
    """
    It plots a graph of decision boundary and datapoints
    :param network: MLP (list of layer)
    :param datafile data file
    :return: none
    """

    decisionBoundaryPlot = figure.add_subplot(111)
    classLabels     = [1,2,3,4]
    colorLabels     = ['r', 'b','g','w']
    markerLabels    = ['*', 'x', '+', '<']
    objectLabels    = ['bolts', 'nuts', 'rings', 'scrap']
    increment       = .01

    #meshgrid for generating decision
    xv, yv = np.meshgrid(
        np.arange(0, 1, increment),
        np.arange(0, 1, increment))

    predicted = []

    for loop_counter in range(xv.shape[0]):
        predicted.append([])
        for inner_counter in range(xv.shape[1]):
            temp = [float(1),xv[loop_counter][inner_counter],yv[loop_counter][inner_counter]]
            temp_prediction = decode_class(predict_class(np.array(temp),network))
            predicted[loop_counter].append(temp_prediction)

    ## decision_plot.contourf(x1_corr, x2_corr, np.array(Y_predicted))
    # decisionPLot = decisionBoundaryPlot.contourf(xv, yv, np.array(predicted))
    # plt.colorbar(decisionPLot, ticks=[1, 2, 3, 4])

    decisionPLot = decisionBoundaryPlot.contourf(xv, yv, np.array(predicted))
    plt.colorbar(decisionPLot, ticks=[1, 2, 3, 4])

    for loop_counter in classLabels:
        attribute1 = [attributes[i][1] for i in range(len(attributes[:]))
                       if decode_class(labels[i]) == loop_counter]

        attribute2 = [attributes[i][2] for i in range(len(attributes[:]))
                       if decode_class(labels[i]) == loop_counter]

        decisionBoundaryPlot.scatter(attribute1, attribute2, c=colorLabels[loop_counter - 1],
                              marker=markerLabels[loop_counter-1]
                              ,label=objectLabels[loop_counter-1],
                              s=100)

    decisionBoundaryPlot.legend(loc='upper right')
    decisionBoundaryPlot.set_xlabel("Six fold Rotational Symmetry")
    decisionBoundaryPlot.set_ylabel("Eccentricity")
    decisionBoundaryPlot.set_title("Decision Boundary Plot")



    return decisionBoundaryPlot

def confusion_matrix(attributes,labels,network):
    """
    This function builds a confusion matrix by using the class prediction
    from the given dataset and the network
    :param attributes: attributes data
    :param labels: labels data
    :param network: neural network
    :return: confusion matrix
    """

    matrix = []

    for loop_counter in range(len(labels)):
        matrix.append([])
        for inner_counter in range(len(labels)):
            matrix[loop_counter].append(0)

    for label_index in range(attributes.shape[0]):
        originalClass       =   decode_class(labels[label_index])
        predictedClass      =   predict_class(attributes[label_index],network)
        decodedPredictedClass =   decode_class(predictedClass)
        matrix[int(decodedPredictedClass-1)][int(originalClass-1)]+=1

    return matrix

def profit_calculation(confusion_matrix):
    """
    This function calculates the net profit obtained for the given test dataset
    :param confusion_matrix: confusion_matrix
    :return: None
    """

    #profit matrix given in the requirements sheet
    profit_data = [[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]]
    net_profit = 0

    #looping through and calculating the net profit for the given dataset
    for outer_loop in range(len(profit_data)):
        for inner_loop in range(len(profit_data)):
            net_profit += confusion_matrix[outer_loop][inner_loop]*profit_data[outer_loop][inner_loop]

    #printing format
    print("***********************************************************************")
    print("******************************** Profit *******************************")
    print("***********************************************************************")
    print("\nOverall Profit(in cents): ",net_profit)
    print("\n***********************************************************************")

def main():
    """
    Main method that runs the executes the MLP for a test dataset
    :return: None
    """
    if len(sys.argv) != 3:
        print("Usage: python3 executeMLP <Weight_file> <test_file>")
        sys.exit(1)

    FileObject = FileHandling(sys.argv[2])

    attributes,labels=FileObject.read_data()

    network = MultiLayerPerceptron(attributes.shape[1],2)

    configure_network_weight(sys.argv[1],network)


    Classification(attributes,labels,network)

    # confusion_matrix= get_confusion_matrix(network, DATA_File, "Trained data")
    # profit(confusion_matrix)

    confusion= confusion_matrix(attributes,labels,network)

    print_data(confusion)

    profit_calculation(confusion)

    figure = plt.figure()

    decision_boundary(figure, attributes,labels,network )
    plt.show()

if __name__ == '__main__':
    main()
