import numpy as np
import math

from ..deep_learning.optimizers import *
from ..deep_learning.loss_functions import *
from ..deep_learning.activation_functions import *
from ..deep_learning.layers import *

from ..deep_learning.utils import image_to_column, column_to_image, output_shape, pooling_layer_output_shape 

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'selu': SELU,
    'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    'softplus': SoftPlus
}

def square_loss(y_true, y_pred, params = None):
    y_pred = np.array(y_pred['result'])
    y_true = np.array(y_true)
    return SquareLoss().loss(y_true, y_pred)

def square_loss_gradient(y, y_pred, params = None):
    y_pred = np.array(y_pred['result'])
    y = np.array(y)
    return SquareLoss().gradient(y, y_pred)

def cross_entropy_loss(y_true, y_pred, params = None):
    y_pred = np.array(y_pred['result'])
    y_true = np.array(y_true)
    return CrossEntropy().loss(y_true, y_pred)

def cross_entropy_gradient(y, y_pred, params = None):
    y_pred = np.array(y_pred['result'])
    y = np.array(y)
    return CrossEntropy().gradient(y, y_pred)

def cross_entropy_accuracy(y_true, y_pred, params = None):
    y_pred = np.array(y_pred['result'])
    y_true = np.array(y_true)
    return CrossEntropy().acc(y_true, y_pred)


def forward_pass_dense(X, params=None):#n_units=None, input_shape=None, data=None, input_layer=None):
    n_units = params.get('n_units', None)
    input_shape = params.get('input_shape', None)
    data = params.get('data', None)
    input_layer = params.get('input_layer', None)
    
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])
    
    if data is None: 
        limit = 1 / math.sqrt(input_shape[0])
        W  = np.random.uniform(-limit, limit, (input_shape[0], n_units))
        w0 = np.zeros((1, n_units))
        W_opt_state_dict = None
        w0_opt_state_dict = None
    else:
        W = np.array(data['W'])
        w0 = np.array(data['w0'])
        W_opt_state_dict = data['W_opt_state_dict']
        w0_opt_state_dict = data['w0_opt_state_dict']
    
    result = X.dot(W)+w0
    forward_pass_output = {
        'W': W.tolist(),
        'w0': w0.tolist(),
        'result': result.tolist(),
        'W_opt_state_dict': W_opt_state_dict,
        'w0_opt_state_dict': w0_opt_state_dict
    }
    return forward_pass_output

def backward_pass_dense(accum_grad, params=None):#layer_input=None, optimizer=None,data=None, input_layer=None):
    layer_input = params.get('layer_input', None)
    optimizer = params.get('optimizer', None)
    data = params.get('data', None)
    input_layer = params.get('input_layer', None)

    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:
        accum_grad = np.array(accum_grad['accum_grad'])

    if isinstance(layer_input, dict):
        layer_input = np.array(layer_input['result'])
    else:
        layer_input = np.array(layer_input)

    W_init = np.array(data['W'])
    W = np.array(data['W'])
    w0 = np.array(data['w0'])
        
    # Calculate gradient w.r.t layer weights
    grad_w = layer_input.T.dot(accum_grad)
    grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

    optimizer_data = optimizer
    optimizer_name = optimizer_data['name']
    del optimizer_data['name']
    
    if optimizer_name == "RMSprop": 
        if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
            W_opt = RMSprop(**optimizer_data)
            w0_opt = RMSprop(**optimizer_data)
        else:
            W_opt = RMSprop(**optimizer_data, **data['W_opt_state_dict'])
            w0_opt = RMSprop(**optimizer_data, **data['w0_opt_state_dict'])
    
    if optimizer_name == "Adam": 
        if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
            W_opt = Adam(**optimizer_data)
            w0_opt = Adam(**optimizer_data)
        else:
            W_opt = Adam(**optimizer_data, **data['W_opt_state_dict'])
            w0_opt = Adam(**optimizer_data, **data['w0_opt_state_dict'])

    # Update the layer weights
    W = W_opt.update(W, grad_w)
    w0 = w0_opt.update(w0, grad_w0)

    # Return accumulated gradient for next layer
    # Calculated based on the weights used during the forward pass

    accum_grad = accum_grad.dot(W_init.T)

    backward_pass_output = {
        'W': W.tolist(),
        'w0': w0.tolist(),
        'accum_grad': accum_grad.tolist(),
        'W_opt_state_dict': W_opt.state_dict(),
        'w0_opt_state_dict': w0_opt.state_dict()
    }
    return backward_pass_output

def forward_pass_batchnorm(X, params=None):#input_shape=None, momentum=None, eps=None, training="True", trainable="True", data=None, input_layer=None):
    input_shape = params.get('input_shape', None)
    momentum = params.get('momentum', None)
    eps = params.get('eps', None)
    training = params.get('training', None)
    trainable = params.get('trainable', None)
    data = params.get('data', None)
    input_layer = params.get('input_layer', None)
    
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    if data is None:    
        if len(input_shape) == 1:
            shape = (1, input_shape[0])
        else:
            shape = (1, input_shape[0], 1, 1)

        running_mean = np.zeros(shape)
        running_var = np.ones(shape)
        gamma = np.ones(shape)
        beta = np.zeros(shape)
        gamma_opt_state_dict = None
        beta_opt_state_dict = None
    else:
        running_mean = np.array(data['running_mean'])
        running_var = np.array(data['running_var'])
        gamma = np.array(data['gamma'])
        beta = np.array(data['beta'])
        gamma_opt_state_dict = data['gamma_opt_state_dict']
        beta_opt_state_dict = data['beta_opt_state_dict']

    if training == "True" and trainable == "True":
        if len(input_shape) == 1:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
        else:
            mean = np.mean(X, axis=(0,2,3), keepdims=True)
            var = np.var(X, axis=(0,2,3), keepdims=True)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
    else:
        mean = running_mean
        var = running_var

    # Statistics saved for backward pass
    X_centered = X - mean
    stddev_inv = 1 / np.sqrt(var + eps)

    X_norm = X_centered * stddev_inv
    output = gamma * X_norm + beta

    forward_pass_output = {
        'running_mean': running_mean.tolist(),
        'running_var': running_var.tolist(),
        'X_centered': X_centered.tolist(),
        'stddev_inv': stddev_inv.tolist(),
        'gamma': gamma.tolist(),
        'beta': beta.tolist(),
        'result': output.tolist(),
        'gamma_opt_state_dict': gamma_opt_state_dict,
        'beta_opt_state_dict': beta_opt_state_dict
    }
    return forward_pass_output

def backward_pass_batchnorm(accum_grad, params=None):#input_shape=None, optimizer=None, trainable="True", data=None, input_layer=None):
    input_shape = params.get('input_shape', None)
    optimizer = params.get('optimizer', None)
    trainable = params.get('trainable', None)
    data = params.get('data', None)
    input_layer = params.get('input_layer', None)
    
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])

    # Save parameters used during the forward pass
    gamma_init = np.array(data['gamma'])
    gamma = np.array(data['gamma'])
    beta = np.array(data['beta'])
    X_centered = np.array(data['X_centered'])
    stddev_inv = np.array(data['stddev_inv'])

    # If the layer is trainable the parameters are updated
    if trainable=="True":
        X_norm = X_centered * stddev_inv
        if len(input_shape) == 1:
            grad_gamma = np.sum(accum_grad * X_norm, axis=0, keepdims=True)
            grad_beta = np.sum(accum_grad, axis=0, keepdims=True)
        else:
            grad_gamma = np.sum(accum_grad * X_norm, axis=(0,2,3), keepdims=True)
            grad_beta = np.sum(accum_grad, axis=(0,2,3), keepdims=True)

        optimizer_data = optimizer
        optimizer_name = optimizer_data['name']
        del optimizer_data['name']
        
        if optimizer_name == "RMSprop":
            if data['gamma_opt_state_dict'] is None and data['beta_opt_state_dict'] is None:
                gamma_opt = RMSprop(**optimizer_data)
                beta_opt = RMSprop(**optimizer_data)
            else:
                gamma_opt = RMSprop(**optimizer_data, **data['gamma_opt_state_dict'])
                beta_opt = RMSprop(**optimizer_data, **data['beta_opt_state_dict'])

        if optimizer_name == "Adam":
            if data['gamma_opt_state_dict'] is None and data['beta_opt_state_dict'] is None:
                gamma_opt = Adam(**optimizer_data)
                beta_opt = Adam(**optimizer_data)
            else:
                gamma_opt = Adam(**optimizer_data, **data['gamma_opt_state_dict'])
                beta_opt = Adam(**optimizer_data, **data['beta_opt_state_dict'])


        gamma = gamma_opt.update(gamma, grad_gamma)
        beta = beta_opt.update(beta, grad_beta)

    batch_size = accum_grad.shape[0]

    # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)

    if len(input_shape) == 1:
        accum_grad = (1 / batch_size) * gamma_init * stddev_inv * (batch_size * accum_grad - np.sum(accum_grad, axis=0,keepdims=True) 
                                                                            - X_centered * stddev_inv**2 * np.sum(accum_grad * X_centered, axis=0, keepdims=True))
    else:
        accum_grad = (1 / batch_size) * gamma_init * stddev_inv * (batch_size * accum_grad - np.sum(accum_grad, axis=(0,2,3),keepdims=True) 
                                                                            - X_centered * stddev_inv**2 * np.sum(accum_grad * X_centered, axis=(0,2,3), keepdims=True))

    backward_pass_output = {
        'gamma': gamma.tolist(),
        'beta': beta.tolist(),
        'running_mean': data['running_mean'],
        'running_var': data['running_var'],
        'accum_grad': accum_grad.tolist(),
        'gamma_opt_state_dict': gamma_opt.state_dict(),
        'beta_opt_state_dict': beta_opt.state_dict()
    }
    return backward_pass_output    
    
def forward_pass_dropout(X, params=None):#p=None, training="True", input_layer=None):
    p = params.get('p',None)
    training = params.get('training',None)
    input_layer = params.get('input_layer',None)
    
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    if training == "True":
        _mask = np.random.uniform(size=X.shape) > p
        c = _mask * (1 / (1-p)) 
        output = X * c
        forward_pass_output = {
            '_mask': _mask.tolist(),
            'result': output.tolist()
        }
    else:
        output = X
        forward_pass_output = {
            'result': output.tolist()
        }
    
    return forward_pass_output

def backward_pass_dropout(accum_grad, params=None):#data=None, input_layer=None):
    data = params.get('data', None)
    input_layer = params.get('input_layer', None)

    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])

    _mask = np.array(data['_mask'])
    accum_grad = accum_grad * _mask

    backward_pass_output = {
        'accum_grad': accum_grad.tolist()
    }
    return backward_pass_output

def forward_pass_activation(X, params=None):#act_data=None, input_layer=None):
    act_data = params.get('act_data',None)
    input_layer = params.get('input_layer',None)
    
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    activation_function = activation_functions[act_data['name']]()
    output = activation_function(X)

    forward_pass_output = {
        'result': output.tolist()
    }
    return forward_pass_output

def backward_pass_activation(accum_grad, params=None):#layer_input=None, act_data=None, input_layer=None):
    layer_input = params.get('layer_input',None)
    act_data = params.get('act_data',None)
    input_layer = params.get('input_layer',None)
    
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])

    if isinstance(layer_input, dict):
        layer_input = np.array(layer_input['result'])
    else:
        layer_input = np.array(layer_input)

    activation_function = activation_functions[act_data['name']]()
    accum_grad = activation_function.gradient(layer_input) * accum_grad

    backward_pass_output = {
        'accum_grad': accum_grad.tolist()
    }
    return backward_pass_output

def forward_pass_conv2d(X, params=None):#input_shape=None, n_filters=None, filter_shape=None, stride=None, padding_data=None, data=None, input_layer=None):
    input_shape = params.get('input_shape', None)
    n_filters = params.get('n_filters', None)
    filter_shape = params.get('filter_shape', None)
    stride = params.get('stride', None)
    padding_data = params.get('padding_data', None)
    data = params.get('data', None)
    input_layer = params.get('input_layer', None)
    
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    if data is None: 
        filter_height, filter_width = filter_shape
        channels = input_shape[0]
        limit = 1 / math.sqrt(np.prod(filter_shape))
        W  = np.random.uniform(-limit, limit, size=(n_filters, channels, filter_height, filter_width))
        w0 = np.zeros((n_filters, 1))
        W_opt_state_dict = None
        w0_opt_state_dict = None
        
    else:
        W = np.array(data['W'])
        w0 = np.array(data['w0'])
        W_opt_state_dict = data['W_opt_state_dict']
        w0_opt_state_dict = data['w0_opt_state_dict']
    
    batch_size = X.shape[0]
    X_col = image_to_column(X, filter_shape, stride=stride, output_shape=padding_data['padding'])
    # Turn weights into column shape
    W_col = W.reshape((n_filters, -1))
    # Calculate output
    output = W_col.dot(X_col) + w0
    # Reshape into (n_filters, out_height, out_width, batch_size)
    output = output.reshape(output_shape(input_shape=input_shape, n_filters=n_filters, filter_shape=filter_shape, padding=padding_data['padding'], stride=stride) + (batch_size, ))
    # output = output.reshape(shape=(self.output_shape() + (batch_size, )))
    # Redistribute axises so that batch size comes first
    
    forward_pass_output = {
        'result': output.transpose(3,0,1,2).tolist(),
        'X_col': X_col.tolist(),
        'W_col': W_col.tolist(),
        'W': W.tolist(),
        'w0': w0.tolist(),
        'W_opt_state_dict': W_opt_state_dict,
        'w0_opt_state_dict': w0_opt_state_dict
    }
    
    return forward_pass_output

def backward_pass_conv2d(accum_grad, params=None):#layer_input=None, n_filters=None, filter_shape=None, stride=None, padding_data=None, optimizer=None, data=None, trainable="True", input_layer=None):
    layer_input = params.get('layer_input', None)
    n_filters = params.get('n_filters', None)
    filter_shape = params.get('filter_shape', None)
    stride = params.get('stride', None)
    padding_data = params.get('padding_data', None)
    optimizer = params.get('optimizer',None)
    data = params.get('data', None)
    trainable = params.get('trainable', None)
    input_layer = params.get('input_layer', None)
    
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:
        accum_grad = np.array(accum_grad['accum_grad'])

    if isinstance(layer_input, dict):
        layer_input = np.array(layer_input['result'])
    else:
        layer_input = np.array(layer_input)

    X_col = np.array(data['X_col'])
    W_col = np.array(data['W_col'])
    W = np.array(data['W'])
    w0 = np.array(data['w0'])
    

    accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)

    if trainable=="True":
        grad_w = accum_grad.dot(X_col.T).reshape(W.shape)

        grad_w0 = np.sum(accum_grad, axis=1, keepdims=True)

        optimizer_data = optimizer
        optimizer_name = optimizer_data['name']
        del optimizer_data['name']
        
        if optimizer_name == "RMSprop":
            if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
                W_opt = RMSprop(**optimizer_data)
                w0_opt = RMSprop(**optimizer_data)
            else:
                W_opt = RMSprop(**optimizer_data, **data['W_opt_state_dict'])
                w0_opt = RMSprop(**optimizer_data, **data['w0_opt_state_dict'])

        if optimizer_name == "Adam":
            if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
                W_opt = Adam(**optimizer_data)
                w0_opt = Adam(**optimizer_data)
            else:
                W_opt = Adam(**optimizer_data, **data['W_opt_state_dict'])
                w0_opt = Adam(**optimizer_data, **data['w0_opt_state_dict'])

        # Update the layers weights
        W = W_opt.update(W, grad_w)
        w0 = w0_opt.update(w0, grad_w0)

    # Recalculate the gradient which will be propogated back to prev. layer
    accum_grad = W_col.T.dot(accum_grad)
    # Reshape from column shape to image shape
    accum_grad = column_to_image(accum_grad,
                                layer_input.shape,
                                filter_shape,
                                stride=stride,
                                output_shape=padding_data['padding'])

    backward_pass_output = {
        'accum_grad': accum_grad.tolist(),
        'W': W.tolist(),
        'w0': w0.tolist(),
        'W_opt_state_dict': W_opt.state_dict(),
        'w0_opt_state_dict': w0_opt.state_dict()
    }
    return backward_pass_output

def forward_pass_maxpool2d(X, params=None):#input_shape=None, pool_shape=None, stride=None, padding_data=None, input_layer=None):
    input_shape = params.get('input_shape', None)
    pool_shape = params.get('pool_shape', None)
    stride = params.get('stride', None)
    padding_data = params.get('padding_data', None)
    input_layer = params.get('input_layer', None)
    
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    batch_size, channels, height, width = X.shape

    _, out_height, out_width = pooling_layer_output_shape(
        input_shape=input_shape, pool_shape=pool_shape, stride=stride
    )

    X = X.reshape(batch_size*channels, 1, height, width)
    X_col = image_to_column(X, pool_shape, stride, padding_data['padding'])

    # MaxPool specific method
    arg_max = np.argmax(X_col, axis=0).flatten()
    output = X_col[arg_max, range(arg_max.size)]
    cache = arg_max
    
    output = output.reshape(out_height, out_width, batch_size, channels)
    output = output.transpose(2, 3, 0, 1)

    forward_pass_output = {
        'result': output.tolist(),
        'X_col': X_col.tolist(),
        'cache': cache.tolist()
    }
    
    return forward_pass_output

def backward_pass_maxpool2d(accum_grad, params=None):#input_shape=None, pool_shape=None, stride=None, padding_data=None, data=None, input_layer=None):
    input_shape = params.get('input_shape', None)
    pool_shape = params.get('pool_shape', None)
    stride = params.get('stride', None)
    padding_data = params.get('padding_data', None)
    data = params.get('data',None)
    input_layer = params.get('input_layer', None)
    
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:
        accum_grad = np.array(accum_grad['accum_grad'])

    cache = np.array(data['cache'])

    batch_size, _, _, _ = accum_grad.shape
    channels, height, width = input_shape
    accum_grad = accum_grad.transpose(2, 3, 0, 1).ravel()

    # MaxPool or AveragePool specific method

    accum_grad_col = np.zeros((np.prod(pool_shape), accum_grad.size))
    arg_max = cache
    accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad

    accum_grad = column_to_image(accum_grad_col, (batch_size * channels, 1, height, width), pool_shape, stride, padding_data['padding'])
    accum_grad = accum_grad.reshape((batch_size,) + tuple(input_shape))

    backward_pass_output = {
        'accum_grad': accum_grad.tolist()
    }
    return backward_pass_output

def forward_pass_flatten(X, params=None):#input_layer=None):
    input_layer = params.get('input_layer',None)

    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    forward_pass_output = {
        'result': X.reshape((X.shape[0], -1)).tolist()
    }
    return forward_pass_output

def backward_pass_flatten(accum_grad, params=None):#prev_input=None, input_layer=None):
    prev_input = params.get('prev_input',None)
    input_layer = params.get('input_layer',None)
    
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])
    
    if isinstance(prev_input, dict):
        prev_shape = np.array(prev_input['result']).shape
    else:
        prev_shape = np.array(prev_input).shape

    backward_pass_output = {
        'accum_grad': accum_grad.reshape(prev_shape).tolist()
    }
    return backward_pass_output