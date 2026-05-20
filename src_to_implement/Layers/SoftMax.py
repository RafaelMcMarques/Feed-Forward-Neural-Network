from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        self.input = input_tensor # input is (m x input)
        max_elems = np.max(input_tensor, axis=1, keepdims=True)
        shifted_input = input_tensor - max_elems 
        shifted_input_exp = np.exp(shifted_input)
        self.output =  shifted_input_exp / np.sum(shifted_input_exp, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor):
        product = error_tensor * self.output
        sum_of_products = np.sum(product, axis=1, keepdims=True)
        prev_error_tensor = self.output * (error_tensor - sum_of_products)
        return prev_error_tensor
