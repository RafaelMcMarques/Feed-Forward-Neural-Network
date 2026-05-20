from .Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # relu(x) = max(0, x)
        self.input = input_tensor
        self.output = np.where(input_tensor > 0, input_tensor, 0)
        return self.output

    def backward(self, error_tensor):
        prev_error_tensor = np.where(self.input > 0, error_tensor, 0)
        return prev_error_tensor
