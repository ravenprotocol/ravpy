import numpy as np
import math

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def determine_padding(filter_shape, output_shape="same"):
    if output_shape == "valid":
        return (0, 0), (0, 0)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)

def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    return (k, i, j)

def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape

    pad_h, pad_w = determine_padding(filter_shape, output_shape)

    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols

def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.zeros((batch_size, channels, height_padded, width_padded))

    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]

def output_shape(input_shape=None, n_filters=None, filter_shape=None, padding=None, stride=None):
    channels, height, width = input_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape=padding)
    output_height = (height + np.sum(pad_h) - filter_shape[0]) / stride + 1
    output_width = (width + np.sum(pad_w) - filter_shape[1]) / stride + 1
    return n_filters, int(output_height), int(output_width)

def pooling_layer_output_shape(input_shape=None, pool_shape=None, stride=None):
    channels, height, width = input_shape
    out_height = (height - pool_shape[0]) / stride + 1
    out_width = (width - pool_shape[1]) / stride + 1
    assert out_height % 1 == 0
    assert out_width % 1 == 0
    return channels, int(out_height), int(out_width)
