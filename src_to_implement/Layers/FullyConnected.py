from .Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter 
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def __init__ (self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self._optimizer = None
        self.weights = np.random.rand(input_size + 1, output_size)

    def forward(self, input_tensor):
        self.input = np.column_stack((input_tensor, np.ones(input_tensor.shape[0]))) # input was (m x input), now (m x input + 1)
        self.output = self.input @ self.weights # (m x output) = (m x input+1) @ (input + 1, output)
        return self.output
    
    def backward(self, error_tensor):
        #dL/dW = dL/dy * dy/dW
        self.gradient_weights = self.input.T @ error_tensor # (input + 1 x output) = (input + 1 x m) @ (m x output)

        #dL/dx = dL/dy * dy/dx
        prev_error_tensor = error_tensor @ self.weights[:-1, :].T # (m x input) = (m x output) @ (output x input)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return prev_error_tensor # this is dL/dx, which has dimensions m x input