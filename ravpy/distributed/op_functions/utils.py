import torch
import torch.nn as nn

def get_activation_layer(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softmax':
        return nn.Softmax(dim=-1)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'softsign':
        return nn.Softsign()
    elif activation == 'tanhshrink':
        return nn.Tanhshrink()
    elif activation == 'softmin':
        return nn.Softmin()
    elif activation == 'softshrink':
        return nn.Softshrink()
    elif activation == 'logsigmoid':
        return nn.LogSigmoid()
    elif activation == 'hardshrink':
        return nn.Hardshrink()
    elif activation == 'hardtanh':
        return nn.Hardtanh()
    elif activation == 'threshold':
        return nn.Threshold()
    else:
        return nn.ReLU()

def get_optimizer(model, **kwargs):
    optimizer = kwargs.get('name', None)
    del kwargs['name']
    if optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), **kwargs)
    elif optimizer == 'adadelta':
        return torch.optim.Adadelta(model.parameters(), **kwargs)
    elif optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer == 'adamax':
        return torch.optim.Adamax(model.parameters(), **kwargs)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    else:
        return None
        