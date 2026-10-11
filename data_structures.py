import numpy as np

class Neuron:
    def __init__(self, weight, bias, summation = 0, function_type = 0, entries: list = []):
        self.weight = weight
        self.bias = bias
        self.summation = summation
        self._function_type = function_type
        self.entries = entries

    @property
    def function_type(self): #Default is ReLU as defined above
        return self._function_type

    def activation_function(self):
        if self.function_type == 0: #ReLU
            if self.summation <= 0:
                return 0
            else:
                return self.summation

        elif self.function_type == 1: #Sigmoid
            euler = np.e
            x = -1 * self.summation
            result = 1 / (1 + (np.power(euler, x)))
            return result
        else:
            print("ERROR: NO ACTIVATION FUNCTION")
            return 1

    def output(self):
        return self.activation_function()

    def sum_iteration(self):
        for i in self.entries:
            result = i * self.weight + self.bias
            self.summation += result




class Layer:
    pass




class NeuralNetwork:
    pass


