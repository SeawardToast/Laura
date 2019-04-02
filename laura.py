import numpy as nm

class NeuralNetwork():
    def __init__(self):
        # seed the random number generator, so it generates the same numbers
        # every time the program runs
        nm.random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.

        self.synaptic_weights = 2 * nm.random.random((3, 1)) -1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    
    def __sigmoid(self, x):
        return 1 / (1 + nm.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # pass the training set through our neural network (a single neuron)
            output = self.think(training_set_inputs)

            # calculate the error (difference between desired output and predicted input
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = nm.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # adjust the weights
            self.synaptic_weights += adjustment

    # the nerual network thinks
    def think(self, inputs):
        #pass inputs through our neural network (single neuron)
        #if (1 - (self.__sigmoid(nm.dot(inputs, self.synaptic_weights)))) < (.05):
        
        number = self.__sigmoid(nm.dot(inputs, self.synaptic_weights))
        #number = 1-number
        
        #print("in think: ", number)
        for c in number:
            #print("in c : ", c)
            if c > .9999:
                return 1
        return self.__sigmoid(nm.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    #initialise a single neuron network

    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # the training set. We have 4 examples, each consisting of 3 input values and 1 output
    training_set_inputs = nm.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = nm.array([[0, 1, 1, 0]]).T
    
    # train the neural network using a training set
    # do it 10,000 times and make small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # test the neural network with a new situation
    print("Considering new situation [1, 0, 0] -> ?: ")
    print("Laura thinks the answer is : ", neural_network.think(nm.array([1, 0, 0])))


