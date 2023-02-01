import torch
import numpy as np
from .utils import get_activation_layer, get_optimizer


def square_loss(y, y_pred, params=None):
    if isinstance(y_pred, dict):
        y_pred = y_pred['result']
    if isinstance(y_pred, list) or isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, requires_grad=True)

    y = torch.tensor(np.array(y, dtype=np.float32))
    mse_loss = torch.nn.functional.mse_loss(y_pred, y, reduction='mean')
    return mse_loss

def cross_entropy_loss(y, y_pred, params = None):
    ignore_index = params.get('ignore_index',None)
    reshape_target = params.get('reshape_target',None)
    reshape_label = params.get('reshape_label',None)

    if isinstance(y_pred, dict):
        y_pred = y_pred['result']
    if isinstance(y_pred, list) or isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, requires_grad=True)

    y = torch.tensor(np.array(y))

    if ignore_index is not None:
        if reshape_target is not None and reshape_label is None:
            loss = torch.nn.functional.cross_entropy(y_pred.view(*reshape_target),y,ignore_index=ignore_index)
        if reshape_target is None and reshape_label is not None:
            loss = torch.nn.functional.cross_entropy(y_pred,y.view(*reshape_label),ignore_index=ignore_index)
        if reshape_target is not None and reshape_label is not None:
            loss = torch.nn.functional.cross_entropy(y_pred.view(*reshape_target),y.view(*reshape_label),ignore_index=ignore_index)
        if reshape_target is None and reshape_label is None:
            loss = torch.nn.functional.cross_entropy(y_pred,y,ignore_index=ignore_index)
    else:
        if reshape_target is not None and reshape_label is None:
            loss = torch.nn.functional.cross_entropy(y_pred.view(*reshape_target),y)
        if reshape_target is None and reshape_label is not None:
            loss = torch.nn.functional.cross_entropy(y_pred,y.view(*reshape_label))
        if reshape_target is not None and reshape_label is not None:
            loss = torch.nn.functional.cross_entropy(y_pred.view(*reshape_target),y.view(*reshape_label))
        if reshape_target is None and reshape_label is None:
            loss = torch.nn.functional.cross_entropy(y_pred,y)

    return loss 


def forward_pass_dense(X, params=None):
    n_units = params.get('n_units', None)
    initial_W = params.get('initial_W', None)
    initial_w0 = params.get('initial_w0', None)
    optimizer_dict = params.get('optimizer_dict', None)
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)    
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        lin = previous_forward_pass_instance['instance']
        optimizer = previous_forward_pass_instance['optimizer']
        X1 = X.to(torch.float32)
        result = lin(X1)
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        optimizer = None
        if previous_batch_layer_data is not None:
            lin = previous_batch_layer_data['instance']
            if 'optimizer' in previous_batch_layer_data:
                optimizer = previous_batch_layer_data['optimizer']
        else:
            lin = torch.nn.Linear(in_features=X.shape[-1], out_features=n_units, bias=True)
            if initial_W is not None:
                lin.weight.data = torch.tensor(initial_W, dtype=torch.float32)
            if initial_w0 is not None:
                lin.bias.data = torch.tensor(initial_w0, dtype=torch.float32)

        if optimizer_dict is not None:
            optimizer = get_optimizer(lin, **optimizer_dict)
        
        X1 = X.to(torch.float32)
        result = lin(X1)
        forward_pass_output = {
            'result': result,
            'instance': lin,
            'optimizer': optimizer
        }
        return forward_pass_output

def forward_pass_batchnorm1d(X, params=None):
    momentum = params.get('momentum', 0.1)
    eps = params.get('eps', 0.00001)
    affine = params.get('affine', True)
    training = params.get('training', None)
    initial_gamma = params.get('initial_gamma', None)
    initial_beta = params.get('initial_beta', None)
    initial_running_mean = params.get('initial_running_mean', None)
    initial_running_var = params.get('initial_running_var', None)
    optimizer_dict = params.get('optimizer_dict', None)
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        bn = previous_forward_pass_instance['instance']
        optimizer = previous_forward_pass_instance['optimizer']
        if not training:
            bn.eval()
        result = bn(X.type(torch.float32))
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        optimizer = None
        if previous_batch_layer_data is not None:
            bn = previous_batch_layer_data['instance']
            if 'optimizer' in previous_batch_layer_data:
                optimizer = previous_batch_layer_data['optimizer']
        else:
            bn = torch.nn.BatchNorm1d(num_features=X.shape[-1], momentum=momentum, eps=eps, affine=affine)
            if initial_gamma is not None:
                bn.weight.data = torch.tensor(initial_gamma, dtype=torch.float32)
            if initial_beta is not None:
                bn.bias.data = torch.tensor(initial_beta, dtype=torch.float32)
            if initial_running_mean is not None:
                bn.running_mean.data = torch.tensor(initial_running_mean, dtype=torch.float32)
            if initial_running_var is not None:
                bn.running_var.data = torch.tensor(initial_running_var, dtype=torch.float32)

        if not training:
            bn.eval()
        
        if optimizer_dict is not None:
            optimizer = get_optimizer(bn, **optimizer_dict)

        result = bn(X.type(torch.float32))
        forward_pass_output = {
            'result': result,
            'instance': bn,
            'optimizer': optimizer
        }
        return forward_pass_output

def forward_pass_batchnorm2d(X, params=None):
    num_features = params.get('num_features', None)
    momentum = params.get('momentum', 0.1)
    eps = params.get('eps', 0.00001)
    training = params.get('training', None)
    affine = params.get('affine', True)
    initial_gamma = params.get('initial_gamma', None)
    initial_beta = params.get('initial_beta', None)
    initial_running_mean = params.get('initial_running_mean', None)
    initial_running_var = params.get('initial_running_var', None)
    optimizer_dict = params.get('optimizer_dict', None)
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        bn = previous_forward_pass_instance['instance']
        optimizer = previous_forward_pass_instance['optimizer']
        if not training:
            bn.eval()
        result = bn(X.type(torch.float32))
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        optimizer = None
        if previous_batch_layer_data is not None:
            bn = previous_batch_layer_data['instance']
            if 'optimizer' in previous_batch_layer_data:
                optimizer = previous_batch_layer_data['optimizer']
        else:
            bn = torch.nn.BatchNorm2d(num_features=num_features, momentum=momentum, eps=eps, affine=affine)
            if initial_gamma is not None:
                bn.weight.data = torch.tensor(initial_gamma, dtype=torch.float32)
            if initial_beta is not None:
                bn.bias.data = torch.tensor(initial_beta, dtype=torch.float32)
            if initial_running_mean is not None:
                bn.running_mean.data = torch.tensor(initial_running_mean, dtype=torch.float32)
            if initial_running_var is not None:
                bn.running_var.data = torch.tensor(initial_running_var, dtype=torch.float32)

        if not training:
            bn.eval()
        if optimizer_dict is not None:
            optimizer = get_optimizer(bn, **optimizer_dict)

        result = bn(X.type(torch.float32))
        forward_pass_output = {
            'result': result,
            'instance': bn,
            'optimizer': optimizer
        }
        return forward_pass_output

def forward_pass_layernorm(X, params=None):
    eps = params.get('eps', 1e-05)
    training = params.get('training', None)
    normalized_shape = params.get('normalized_shape', None)
    initial_W = params.get('initial_W', None)
    initial_w0 = params.get('initial_w0', None)
    optimizer_dict = params.get('optimizer_dict', None)
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)    
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        ln = previous_forward_pass_instance['instance']
        optimizer = previous_forward_pass_instance['optimizer']
        if not training:
            ln.eval()
        result = ln(X.type(torch.float32))
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        optimizer = None
        if previous_batch_layer_data is not None:
            ln = previous_batch_layer_data['instance']
            if 'optimizer' in previous_batch_layer_data:
                optimizer = previous_batch_layer_data['optimizer']
        else:
            ln = torch.nn.LayerNorm(normalized_shape=normalized_shape,eps=eps)
            if initial_W is not None:
                ln.weight.data = torch.tensor(initial_W, dtype=torch.float32)
            if initial_w0 is not None:
                ln.bias.data = torch.tensor(initial_w0, dtype=torch.float32)

        if not training:
            ln.eval()

        if optimizer_dict is not None:
            optimizer = get_optimizer(ln, **optimizer_dict)

        result = ln(X.type(torch.float32))
        forward_pass_output = {
            'result': result,
            'instance': ln,
            'optimizer': optimizer
        }
        return forward_pass_output
    
def forward_pass_dropout(X, params=None):
    p = params.get('p',0.5)
    training = params.get('training',None)    
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        drp = previous_forward_pass_instance['instance']
        if not training:
            drp.eval()
        result = drp(X.type(torch.float32))
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        if previous_batch_layer_data is not None:
            drp = previous_batch_layer_data['instance']
        else:
            drp = torch.nn.Dropout(p=p)
        if not training:
            drp.eval()
        result = drp(X.type(torch.float32))
        forward_pass_output = {
            'result': result,
            'instance': drp
        }
        return forward_pass_output

def forward_pass_activation(X, params=None):
    act_data = params.get('act_data',None)
    training = params.get('training',None)    
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        act = previous_forward_pass_instance['instance']
        result = act(X.type(torch.float32))
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        if previous_batch_layer_data is not None:
            act = previous_batch_layer_data['instance']
        else:
            act = get_activation_layer(act_data['name'])
        result = act(X.type(torch.float32))
        forward_pass_output = {
            'result': result,
            'instance': act
        }
        return forward_pass_output

def forward_pass_conv2d(X, params=None):
    in_channels = params.get('in_channels', None)
    out_channels = params.get('out_channels', None)
    kernel_size = params.get('kernel_size', None)
    stride = params.get('stride', 1)
    padding = params.get('padding', 0)
    dilation = params.get('dilation', 1)
    groups = params.get('groups', 1)
    bias = params.get('bias', True)
    padding_mode = params.get('padding_mode', 'zeros')
    initial_W = params.get('initial_W', None)
    initial_w0 = params.get('initial_w0', None)
    optimizer_dict = params.get('optimizer_dict', None)
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)    
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        conv2d = previous_forward_pass_instance['instance']
        optimizer = previous_forward_pass_instance['optimizer']
        X1 = X.to(torch.float32)
        result = conv2d(X1)
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        optimizer = None
        if previous_batch_layer_data is not None:
            conv2d = previous_batch_layer_data['instance']
            if 'optimizer' in previous_batch_layer_data:
                optimizer = previous_batch_layer_data['optimizer']
        else:
            conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
            if initial_W is not None:
                conv2d.weight.data = torch.tensor(initial_W, dtype=torch.float32)
            if initial_w0 is not None:
                conv2d.bias.data = torch.tensor(initial_w0, dtype=torch.float32)
        
        if optimizer_dict is not None:
            optimizer = get_optimizer(conv2d, **optimizer_dict)

        X1 = X.to(torch.float32)
        result = conv2d(X1)
        forward_pass_output = {
            'result': result,
            'instance': conv2d,
            'optimizer': optimizer
        }
        return forward_pass_output

def forward_pass_maxpool2d(X, params=None):
    kernel_size = params.get('kernel_size', None)
    stride = params.get('stride', None)
    padding = params.get('padding', 0)
    dilation = params.get('dilation', 1)
    return_indices = params.get('return_indices', False)
    ceil_mode = params.get('ceil_mode', False)
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)
    previous_forward_pass_instance = params.get('previous_forward_pass', None)

    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        maxpool2d = previous_forward_pass_instance['instance']
        result = maxpool2d(X.type(torch.float32))
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        if previous_batch_layer_data is not None:
            maxpool2d = previous_batch_layer_data['instance']
        else:
            maxpool2d = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
        result = maxpool2d(X.type(torch.float32))
        forward_pass_output = {
            'result': result,
            'instance': maxpool2d
        }
        return forward_pass_output

def forward_pass_flatten(X, params=None):
    start_dim = params.get('start_dim', 1)
    end_dim = params.get('end_dim', -1)
    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)
    
    result = torch.flatten(X, start_dim=start_dim, end_dim=end_dim)
    forward_pass_output = {
        'result': result
    }
    return forward_pass_output

def forward_pass_concat(x1,x2, params=None):
    pass

def forward_pass_add(x1,x2, params=None):
    
    if isinstance(x1, dict):
        x1 = x1['result']
    if isinstance(x1, list) or isinstance(x1, np.ndarray):
        x1 = torch.tensor(x1)
    if isinstance(x2, dict):
        x2 = x2['result']
    if isinstance(x2, list) or isinstance(x2, np.ndarray):
        x2 = torch.tensor(x2)

    result = x1 + x2
    forward_pass_output = {
        'result': result
    }
    return forward_pass_output
  
def forward_pass_subtract(x1,x2, params=None):

    if isinstance(x1, dict):
        x1 = np.array(x1['result'])
    else:
        x1 = np.array(x1)
    if isinstance(x2, dict):
        x2 = np.array(x2['result'])
    else:
        x2 = np.array(x2)
    
    result = np.subtract(x1,x2)
    forward_pass_output = {
        'result': result.tolist()
    }
    return forward_pass_output

def forward_pass_dot(x1,x2, params=None):
    
    if isinstance(x1, dict):
        x1 = x1['result']
    if isinstance(x1, list) or isinstance(x1, np.ndarray):
        x1 = torch.tensor(x1)
    if isinstance(x1, torch.Tensor):
        x1 = x1
    if isinstance(x2, dict):
        x2 = x2['result']
    if isinstance(x2, list) or isinstance(x2, np.ndarray):
        x2 = torch.tensor(x2)
    if isinstance(x2, torch.Tensor):
        x2 = x2

    result = x1 @ x2
    forward_pass_output = {
        'result': result
    }
    
    return forward_pass_output

def forward_pass_reshape(X, params=None):
    shape = params.get('shape', None)
    contiguous = params.get('contiguous', False)
    
    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)
    if isinstance(X, torch.Tensor):
        X = X

    if contiguous:
        result = X.contiguous().view(*shape)
    else:
        result = X.view(*shape)
    forward_pass_output = {
        'result': result
    }
    return forward_pass_output

def forward_pass_transpose(X, params=None):
    axes = params.get('axes', None)
    
    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)
    if isinstance(X, torch.Tensor):
        X = X

    result = X.transpose(*axes)
    forward_pass_output = {
        'result': result
    }
    return forward_pass_output

def forward_pass_power(X, params=None):
    power = params.get('power', None)
    
    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)
    if isinstance(X, torch.Tensor):
        X = X

    result = torch.pow(X, power)
    forward_pass_output = {
        'result': result
    }
    return forward_pass_output

def forward_pass_multiply(x1,x2, params=None):
    
    if isinstance(x1, dict):
        x1 = x1['result']
    if isinstance(x1, list) or isinstance(x1, np.ndarray):
        x1 = torch.tensor(x1)
    if isinstance(x1, torch.Tensor):
        x1 = x1
    if isinstance(x2, dict):
        x2 = x2['result']
    if isinstance(x2, list) or isinstance(x2, np.ndarray):
        x2 = torch.tensor(x2)
    if isinstance(x2, torch.Tensor):
        x2 = x2

    result = x1 * x2
    forward_pass_output = {
        'result': result
    }
    return forward_pass_output

def forward_pass_division(x1,x2, params=None):
    
    if isinstance(x1, dict):
        x1 = x1['result']
    if isinstance(x1, list) or isinstance(x1, np.ndarray):
        x1 = torch.tensor(x1)
    if isinstance(x1, torch.Tensor):
        x1 = x1
    if isinstance(x2, dict):
        x2 = x2['result']
    if isinstance(x2, list) or isinstance(x2, np.ndarray):
        x2 = torch.tensor(x2)
    if isinstance(x2, torch.Tensor):
        x2 = x2

    result = x1 / x2
    forward_pass_output = {
        'result': result
    }
    return forward_pass_output

def forward_pass_embedding(X, params=None):
    vocab_size = params.get('vocab_size', None)
    embed_dim = params.get('embed_dim', None)
    initial_W = params.get('initial_weights', None)
    optimizer_dict = params.get('optimizer_dict', None)
    previous_batch_layer_data = params.get('previous_batch_layer_data', None)    
    previous_forward_pass_instance = params.get('previous_forward_pass', None)


    if isinstance(X, dict):
        X = X['result']
    if isinstance(X, list) or isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if previous_forward_pass_instance is not None:
        emb = previous_forward_pass_instance['instance']
        optimizer = previous_forward_pass_instance['optimizer']
        X1 = X.to(torch.int64)
        result = emb(X1)
        forward_pass_output = {
            'result': result
        }
        return forward_pass_output
    else:
        optimizer = None
        if previous_batch_layer_data is not None:
            emb = previous_batch_layer_data['instance']
            if 'optimizer' in previous_batch_layer_data:
                optimizer = previous_batch_layer_data['optimizer']
        else:
            emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
            if initial_W is not None:
                emb.weight.data = torch.tensor(initial_W, dtype=torch.float32)

        if optimizer_dict is not None:
            optimizer = torch.optim.Adam(emb.parameters(), lr=optimizer_dict['lr'], betas=optimizer_dict['betas'], eps=optimizer_dict['eps'], weight_decay=optimizer_dict['weight_decay'], amsgrad=optimizer_dict['amsgrad'])
        
        result = emb(X.type(torch.int64))
        forward_pass_output = {
            'result': result,
            'instance': emb,
            'optimizer': optimizer
        }
        return forward_pass_output
