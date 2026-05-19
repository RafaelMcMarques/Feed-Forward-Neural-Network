from .Base import BaseLayer
import numpy as np

#TODO: this currently calculates Y = X @ W, change to Y = W.t @ X
# also, change optimizer to be a @property instead of attribute
class FullyConnected(BaseLayer):
    optimizer = None

    def get_optimizer(self):
        return self.optimizer
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def __init__ (self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)

    def forward(self, input_tensor):
        self.input = np.column_stack((input_tensor, np.ones(input_tensor.shape[0]))) 
        self.output = self.input @ self.weights
        return self.output
    
    def backward(self, error_tensor):
        #dL/dW = dL/dy * dy/dW
        self.gradient_weights = self.input.T @ error_tensor

        #dL/dx = dL/dy * dy/dx
        prev_error_tensor = error_tensor @ self.weights[:-1, :].T

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return prev_error_tensor 