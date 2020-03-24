import torch
import torch.autograd as autograd
import torch.nn as nn


torch.set_default_tensor_type('torch.DoubleTensor')
class Value(nn.Module):
    def __init__(self, num_inputs,hidden_size):
        super(Value, self).__init__()
        self.inputLayer = nn.Linear(num_inputs, hidden_size)
        self.hiddenLayer = nn.Linear(hidden_size, hidden_size)
        self.hiddenLayer2 = nn.Linear(hidden_size, hidden_size)
        self.outputLayer = nn.Linear(hidden_size, 1)
        # TODO: Check effect of this
        # self.outputLayer.weight.data.mul_(0.1)
        # self.outputLayer.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        x = torch.tanh(self.hiddenLayer2(x))
        x = self.outputLayer(x)
        return x
