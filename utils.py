import math
import numpy as np
import torch


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

def saveTensorCsv(t,filename):
    import csv
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(t.size()[0]):
            writer.writerow(t[i].detach().numpy().tolist())
def loadTensorCsv(filename):
    import csv
    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        t = []
        for row in reader:
            t.append([float(x) for x in row])

        return torch.Tensor(t)

def saveParameterCsv(param,filename):
    import csv
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(param.detach().numpy().tolist())

def loadParameterCsv(filename):
    import csv

    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            param = [float(x) for x in row]
    return torch.Tensor(param)
