import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        pred_for_correct_class = np.sum(prediction_tensor * label_tensor, axis=1)
        neg_log_of_y = -1 * np.log(pred_for_correct_class + np.finfo(float).eps)
        self.input = prediction_tensor
        self.output = np.sum(neg_log_of_y)
        return self.output

    def backward(self, label_tensor):
        denominator = self.input + np.finfo(float).eps
        prev_error_tensor = -1 * label_tensor / denominator
        return prev_error_tensor
