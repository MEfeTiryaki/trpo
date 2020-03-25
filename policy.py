import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn

import numpy as np

torch.set_default_tensor_type('torch.DoubleTensor')
class Policy(nn.Module):
    def __init__(self, num_inputs,num_outputs,hidden_size):
        super(Policy, self).__init__()
        self.inputLayer = nn.Linear(num_inputs, hidden_size)
        self.hiddenLayer = nn.Linear(hidden_size, hidden_size)
        self.hiddenLayer2 = nn.Linear(hidden_size, hidden_size)
        self.outputLayer = nn.Linear(hidden_size, num_outputs)
        self.logStd = nn.Parameter(torch.zeros(1, num_outputs))


    def forward(self, x):
        """
        Parameters:
        states (torch.Tensor): N_state x N_sample

        Returns:
        torch.Tensor:  N_action x N_sample  | mean of the action
        torch.Tensor:  N_action x N_sample  | log(std) of action
        torch.Tensor:  N_action x N_sample  | std of action
        """
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        x = torch.tanh(self.hiddenLayer2(x))
        action_mean = self.outputLayer(x)
        action_logStd = self.logStd.expand_as(action_mean)
        action_std = torch.exp(self.logStd)

        return action_mean, action_logStd, action_std


    def getLogProbabilityDensity(self,states,actions):
        """
        Parameters:
        states (torch.Tensor): N_state x N_sample | The states of the samples
        actions (torch.Tensor): N_action x N_sample | The action taken for this samples

        Returns:
        torch.Tensor: Log probability of the actions calculated by gaussian distribution
        """
        action_mean, logStd, action_std = self.forward(states)
        var = torch.exp(logStd).pow(2);

        logProbablitiesDensity_ = -(actions.unsqueeze(1) - action_mean).pow(2) / (
            2 * var) - 0.5 * np.log(2 * np.pi) - logStd;
        return logProbablitiesDensity_.sum(1);

    def meanKlDivergence(self, states, actions, logProbablityOld):
        """
        Parameters:
        states (torch.Tensor): N_state x N_sample | The states of the samples
        actions (torch.Tensor): N_action x N_sample | The action taken for this samples
        logProbablityOld (torch.Tensor): N_sample |  Log probablility of the action, note that
            this should be detached from the gradient.

        Returns:
        torch.Tensor: Scalar | the mean of KL-divergence
        """
        logProbabilityNew = self.getLogProbabilityDensity(states,actions);
        return (torch.exp(logProbablityOld)
                * (logProbablityOld - logProbabilityNew)).mean(); #Tensor kl.mean()

    def get_action(self,state):
        """
        Parameters:
        states (numpy.array): N_state

        Returns:
        numpy.array: N_action | sampled action
        """
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, action_log_std, action_std = self.forward(state)
        action = torch.normal(action_mean, action_std)
        return action.detach().numpy()

    def get_mean_action(self,state):
        """
        Parameters:
        states (numpy.array): N_state

        Returns:
        numpy.array: N_action | mean action
        """
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, action_log_std, action_std = self.forward(state)
        return action_mean.detach().numpy()
