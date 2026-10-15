import numpy as np

class Neuron:
    def __init__(self, weight, bias, summation = 0, output: float = 0, function_type = 0, entries: list = None, connections: list = None):
        self.weight = weight
        self.bias = bias
        self.summation = summation
        self.output = output
        self._function_type = function_type
        self.entries = entries or []
        self.connections = connections or []

    @property
    def function_type(self): #Default is ReLU as defined above
        return self._function_type

    def activation_function(self):
        if self.function_type == 0: #ReLU
            if self.summation <= 0:
                self.output = 0
                return self.output
            else:
                self.output = self.summation
                return self.output

        elif self.function_type == 1: #Sigmoid
            euler = np.e
            x = -1 * self.summation
            result = 1 / (1 + (np.power(euler, x)))
            self.output = result
            return self.output
        else:
            raise ValueError("ERROR: NO ACTIVATION FUNCTION DEFINED")


    def y_hat(self):
        self.sum_iteration()
        return self.activation_function()

    def sum_iteration(self):
        self.summation = 0
        for i in self.entries:
            result = i * self.weight
            self.summation += result
        self.summation += self.bias




class Layer:
    pass




class NeuralNetwork:
    pass


