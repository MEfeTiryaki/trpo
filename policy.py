import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn

import numpy as np

torch.set_default_tensor_type('torch.DoubleTensor')
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs,hidden_size):
        super(Policy, self).__init__()
        self.inputLayer = nn.Linear(num_inputs, hidden_size)
        self.hiddenLayer = nn.Linear(hidden_size, hidden_size)
        self.hiddenLayer2 = nn.Linear(hidden_size, hidden_size)
        self.outputLayer = nn.Linear(hidden_size, num_outputs)
        # TODO: Check effect of this
        # self.outputLayer.weight.data.mul_(0.1)
        # self.outputLayer.bias.data.mul_(0.0)

        self.logStd = nn.Parameter(torch.zeros(1, num_outputs))


    def forward(self, x):
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        x = torch.tanh(self.hiddenLayer2(x))
        action_mean = self.outputLayer(x)
        action_logStd = self.logStd.expand_as(action_mean)
        action_std = torch.exp(self.logStd)

        return action_mean, action_logStd, action_std


    def getLogProbabilityDensity(self,states,actions):
        action_mean, logStd, action_std = self.forward(states)
        var = torch.exp(logStd).pow(2);
        # print("action_mean",action_mean)
        # print("actions",actions.unsqueeze(1))
        # print("actions - action_mean", actions - action_mean)
        # print("var",var)

        logProbablitiesDensity_ = -(actions.unsqueeze(1) - action_mean).pow(2) / (
            2 * var) - 0.5 * np.log(2 * np.pi) - logStd;
        # print("logProbablitiesDensity_",logProbablitiesDensity_)
        return logProbablitiesDensity_.sum(1);

    def meanKlDivergence(self, states, actions, logProbablityOld):
        logProbabilityNew = self.getLogProbabilityDensity(states,actions);
        # print(logProbabilityNew.size(),logProbabilityNew.grad_fn)
        # print("meanKlDivergence __________________")
        # print("logProbabilityNew\n",logProbabilityNew)
        # print("logProbablityOld\n",logProbablityOld)
        # print("(logProbablityOld - logProbabilityNew)\n",(logProbablityOld - logProbabilityNew))
        # print("meanKlDivergence __________________")
        return (torch.exp(logProbablityOld)
                * (logProbablityOld - logProbabilityNew)).mean(); #Tensor kl.mean()

    def get_action(self,state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, action_log_std, action_std = self.forward(state)
        action = torch.normal(action_mean, action_std)
        return action.detach().numpy()

    def get_mean_action(self,state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, action_log_std, action_std = self.forward(state)
        return action_mean.detach().numpy()
