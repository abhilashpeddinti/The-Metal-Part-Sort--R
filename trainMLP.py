#Authors : Girish Kumar Reddy Veerepalli & Abhilash Peddinti
#File Name : trainMLP.py
#The trainMLP program takes a training data file as input and trains the neural network for a specific
#number of epochs.
#Uses sigmoid function as an activation function.
#Uses batch gradiet descent technique to train the neural network

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

NUM_LAYER = 2
LEARNING_RATE = 0.01

OutputLayerNeuronCount = 4
HiddenLayerNeuronCount = 5

class NeuronUnit:
    """
    Respresents the functionality of a Neuron in the network.
    """

    __slots__ = "inward_count","outward_count","weights","input_dataset","activation"

    def __init__(self, inward_count, outward_count):
        """
        Initializing the neuron unit
        :param inward_count: no. of inward connections.
        :param outward_count: no. of outward connections.
        """
        self.inward_count   = inward_count
        self.outward_count  = outward_count
        self.input_dataset  = None
        self.activation     = None
        self.weights        = self.initializeNeuronWeights()

    def initializeNeuronWeights(self):
        neuronWeights = []
        for loop_counter in range(self.inward_count):
            neuronWeights.append(random.uniform(-1, 1))
        return np.array(neuronWeights)



def sigmoid_value(activationValue):
    """
    A sigmoid function.
    :param activationValue: Activation Value that will be passed as input to the sigmoid function.
    :return: a sigmoid value of the input
    """

    return 1 / (1 + math.exp((-1) * activationValue))


def neuron_activation(inputs,weights):
    """
    The function neuron_activation calculates the activation value for a given input.
    :param inputs: list of inputs to the neurons
    :param weights: weights
    :return: sigmoid value
    """
    activationValue = 0

    for loop_counter in range(len(inputs)):
        activationValue += inputs[loop_counter]*weights[loop_counter]
    sigmoidValue = sigmoid_value(activationValue)

    return sigmoidValue

class NeuronLayer:
    """
    The class NeuronLayer represents a layer in the Multi Layer Perceptron.
    Each Layer contains severak Neuron Units.
    """

    __slots__ = ('input_count', 'output_count', 'neuron_count', 'neurons')

    def __init__(self, input_count=1, output_count=1, neuron_count=1):
        """
        Initialize the Neuron Layer
        :param input_count: No. of inputs/inward connections
        :param output_count: No. of outputs/outward connections
        :param neuron_count: Count of Neurons in the Layer
        """
        self.input_count    = input_count
        self.output_count   = output_count
        self.neuron_count   = neuron_count
        self.neurons        = self.initializeNeuronLayer()

    def initializeNeuronLayer(self):
        """
        Creating and initializing the neuron units for the Neuron Layer
        :return: neurons in the layer
        """
        neuron_list = []

        for _ in range(self.neuron_count):
            neuron_list.append(NeuronUnit(self.input_count, self.output_count))

        return neuron_list

class MultiLayerPerceptron:
    """
    The class MultiLayerPerceptron represents a Neural Network.
    It has Neuron Layers which has several Neuron Units.
    """
    __slots__ = "neural_network"

    def __init__(self, networkInwardCount, networkOutwardCount):
        """
        Creating a Multi Layer Perceptron with input and output channels.
        :param networkInwardCount: No. of inputs to the network.
        :param networkOutwardCount: No. of outputs to the network.
        """
        hiddenLayer = NeuronLayer(networkInwardCount, HiddenLayerNeuronCount+1, HiddenLayerNeuronCount)
        outputLayer = NeuronLayer(HiddenLayerNeuronCount+1, networkOutwardCount, OutputLayerNeuronCount)
        self.neural_network =   list([hiddenLayer,outputLayer])

    def update_network(self,weights):
        """
        This function updates weights of the Multi Layer Perceptron Network.
        :param weights: input weights
        :return: None
        """

        for layer_counter in range(len(self.neural_network)):
            neurons=self.neural_network[layer_counter].neurons
            for neuron_counter in range(len(neurons)):
                neurons[neuron_counter].weights = (weights[layer_counter][neuron_counter])


    def get_weights(self):
        """
        Function to retrieve the network weights.
        :return:
        """
        tempWeights = []
        count = 0
        for layer in self.neural_network:
            tempWeights.append([])
            for neuron in layer.neurons:
                tempWeights[count].append(neuron.weights)
            count += 1
        return tempWeights


def response(input_object,inputs):
    """
    Method for finding activation for a given input
    :param input_object: input is either a Neuron object or a Layer object
    :param inputs: inputs to the neurons
    :return: activation
    """

    if isinstance(input_object,NeuronUnit):
        input_object.input_dataset = inputs
        activation = neuron_activation(inputs,input_object.weights)
        input_object.activation=activation
        return activation
    elif isinstance(input_object,NeuronLayer):
        activation = []
        for neuron in input_object.neurons:
            activation.append(response(neuron, inputs))
        return activation

def back_propogation(network_object,previous_weights,error_obtained):
    """
    This function implements the back propogation
    :param network_object: mlp network object
    :param previous_weights: old weights
    :param error_obtained: error list
    :return: updated weights
    """

    #Output layer.
    networkOutputLayer = network_object.neural_network[-1] # 1 output layer

    #Neurons in the output layer
    output_neurons = networkOutputLayer.neurons

    old_delta = []

    #loop through neurons in the layer
    for neuron_counter in range(len(output_neurons)): #
        activationvalue =   output_neurons[neuron_counter].activation
        input_data   =   output_neurons[neuron_counter].input_dataset #list
        sigmoid    = activationvalue*(1-activationvalue)

        #calculate new delta
        new_delta = error_obtained[neuron_counter]*sigmoid

        #calculate dw
        dw = [ LEARNING_RATE * new_delta* inp for inp in input_data]
        previous_weights[-1][neuron_counter]+=dw #add the dw

        #save it in the list
        old_delta.append(new_delta)

    #Hidden Layer/2nd Layer
    networkHiddenLayer = network_object.neural_network[-2]

    #Get Neurons in the hidden Layer
    hidden_neurons = networkHiddenLayer.neurons


    output_weights = []
    for each_neuron in output_neurons:
        output_weights.append( each_neuron.weights )

    hidden_layer_delta = []
    for neuron_counter in range(len(hidden_neurons)):

        #activation
        hiddenActivation = hidden_neurons[neuron_counter].activation

        #input data
        input_data = hidden_neurons[neuron_counter].input_dataset

        new_delta = 0

        #delta calculation and storing
        for delta_counter in range(len(old_delta)):
            new_delta += output_weights[delta_counter][neuron_counter + 1]*old_delta[delta_counter]
        new_delta = new_delta * hiddenActivation * (1 - hiddenActivation)
        hidden_layer_delta.append(new_delta)

        #calculate dw
        dw = [LEARNING_RATE * hidden_layer_delta[neuron_counter] * inp for inp in
              input_data]
        previous_weights[-2][neuron_counter] += dw

    return previous_weights

def forward_prop(mlp,input):
    """
    Generating the activation of the MLP
    :param input: input to the MLP
    :return: return prediction/activation
    """
    activation=input
    for layer in range(NUM_LAYER):
        activation=response(mlp.neural_network[layer],activation)
        if layer == 0:
            activation.insert(0,1)
    return activation




def gradient_descent(network, filehandler,attributes,label):
    """
    This function implements the batch gradient descent technique
    :param network: neural network
    :param filehandler: filehandler object
    :param attributes: attributes data
    :param label: label data
    :return: neural network and the Sum of Squared Errors History
    """

    #list to store the sum of squared error at each epoch
    SumSqauredErrorData =[]

    #count of samples
    data_count=attributes.shape[0]

    #reading epochs from command line
    epochs = int(sys.argv[2])

    if epochs == 0:
        epochs = 1

    #perform batch gradient descent for the epochs specified.
    for epoch in range(epochs):
        SumSqauredError = 0
        updated_weights = network.get_weights()

        for sample in range(data_count):
            prediction = forward_prop(network,attributes[sample])
            error=[]
            for bit_counter in range(len(label[sample])):
                error.append((label[sample][bit_counter] - prediction[bit_counter]))
                SumSqauredError +=(label[sample][bit_counter] - prediction[bit_counter])**2
            updated_weights = back_propogation( network,updated_weights,error)
        network.update_network(updated_weights)

        #appending the current epoch's SSE to the list
        SumSqauredErrorData.append(SumSqauredError)

        #writing the weights into the file
        filehandler.write_csv(network)

    return network, SumSqauredErrorData


class FileHandling:
    __slots__ = "input_file"

    def __init__(self, input):
        self.input_file = input

    def read_data(self):
        """
        This function reads the data from the input file and classifies it into attributes and labels
        :return: a nummpy array of attributes and the data
        """
        file_name = self.input_file
        data = []
        labels = []

        count = 0
        with open(file_name) as data_file:
            for line in data_file:
                line_list = line.strip().split(",")
                if len(line_list) == 1:
                    continue
                data.append([])
                labels.append([float(0), float(0), float(0), float(0)])
                data[count].append(float(1))
                data[count].append(float(line_list[0]))
                data[count].append(float(line_list[1]))
                if float(line_list[2]) == 1.0:
                    labels[count][0] = 1
                if float(line_list[2]) == 2.0:
                    labels[count][1] = 1
                if float(line_list[2]) == 3.0:
                    labels[count][2] = 1
                if float(line_list[2]) == 4.0:
                    labels[count][3] = 1
                count += 1

        data = np.array(data)
        labels = np.array(labels)

        return data, labels

    def write_csv(self, network):
        """
        It writes the weights in a CSV file
        :param network: A neuron network
        :return: None
        """
        weight_line = ""
        epochs = int(sys.argv[2])
        weights = network.get_weights()
        for layer_counter in range(len(weights)):
            for neuron_counter in range(len(weights[layer_counter])):
                for weight in weights[layer_counter][neuron_counter]:
                    weight_line += str(weight) + ","
        weight_line = weight_line[0:len(weight_line) - 1]
        myStr = "weights_" + str(epochs) + ".csv"
        fp = open(myStr, "a+")
        fp.write(weight_line + "\n")
        fp.close()


def main():
    """
    Main method
    return: None
    """
    if len(sys.argv) != 3:
        print("Usage: python3 trainMLP <train_data file> <epochs count>")
        sys.exit(1)

    #file handling object
    fileHandlingObject = FileHandling(sys.argv[1])

    #get epochs
    epochs = sys.argv[2]

    #clear the weights file if it is already there.
    myStr = "weights_" + str(epochs) + ".csv"
    f = open(myStr,'w+')
    f.close()

    #read the input data and extract the attributes and labels
    attributes, label = fileHandlingObject.read_data()

    #create a MLP Network
    MLPNetwork = MultiLayerPerceptron(3, 4)

    #perform batch gradientdescent
    trained_network, SSE_data = gradient_descent(MLPNetwork, fileHandlingObject, attributes, label)

    print("Network Successfully Trained with " + epochs + " epochs!!!")
    # plotting it
    figure = plt.figure()
    loss__curve = figure.add_subplot(111)
    loss__curve.plot(SSE_data, label='Training')
    loss__curve.set_title("SSE vs Epochs Plot")
    loss__curve.set_xlabel("Epochs count")
    loss__curve.set_ylabel("SSE")
    loss__curve.legend()
    figure.show()
    plt.show()


if __name__ == "__main__":
    main()

