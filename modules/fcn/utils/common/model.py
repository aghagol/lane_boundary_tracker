"""
File: model.py
Author: Brett Jackson <brett.jackson@here.com>
Version: 3.5.0

Helper functions used to build a model

"""

# ==============================================================================
import tensorflow as tf


# ------------------------------------------------------------------------------
def _weights_and_biases(kernel_shape, bias_shape, name=""):
    """
    Generate weights and bias used to construct a model

    Args:
        kernel_shape (list): Shape of the weights tensor
        bias_shape (list): Shape of the biases tensor
        name (str): Name to append to the weighs and bias variables

    Returns:
        tensorflow.Tensor, tensorflow.Tensor: Tuple of the weights and
                                              biases tensors

    """
    # Create variable named "weights".
    weights = tf.get_variable("weights_{}".format(name), kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())

    # Create variable named "biases".
    biases = tf.get_variable("biases_{}".format(name), bias_shape,
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    return weights, biases


# ------------------------------------------------------------------------------
def _conv(inputs, convolution_shape, output_depth):
    """
    Apply convolution layer with stride length of 1

    Args:
        inputs (tensorflow.Tensor): Input tensor
        convolution_shape (list): Shape of the kernel used in the convolution
        output_depth (int): Depth of the output of convolution the layer

    Returns:
        tensorflow.Tensor, tensorflow.Tensor: Tuple of the output of the
                                              ELU layer and the weights

    """
    kernel_shape = [convolution_shape[0],
                    convolution_shape[1],
                    inputs.get_shape()[-1],
                    output_depth]
    weights, biases = _weights_and_biases(kernel_shape, [output_depth])
    conv = tf.nn.conv2d(inputs,
                        weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    return conv + biases


# ------------------------------------------------------------------------------
def _conv_elu(inputs, convolution_shape, output_depth):
    """
    Apply convolution layer with stride length of 1 + ELU layer

    Args:
        inputs (tensorflow.Tensor): Input tensor
        convolution_shape (list): Shape of the kernel used in the convolution
        output_depth (int): Depth of the output of convolution the layer

    Returns:
        tensorflow.Tensor, tensorflow.Tensor: Tuple of the output of the
                                              ELU layer and the weights

    """
    return tf.nn.elu(_conv(inputs, convolution_shape, output_depth))

    # _BJ_ kernel_shape = [convolution_shape[0],
    # _BJ_                 convolution_shape[1],
    # _BJ_                 inputs.get_shape()[-1],
    # _BJ_                 output_depth]
    # _BJ_ weights, biases = _weights_and_biases(kernel_shape, [output_depth])
    # _BJ_ conv = tf.nn.conv2d(inputs,
    # _BJ_                     weights,
    # _BJ_                     strides=[1, 1, 1, 1],
    # _BJ_                     padding='SAME')

    # _BJ_ return tf.nn.elu(conv + biases)


# ------------------------------------------------------------------------------
def _conv_elu_div2(inputs, convolution_shape, output_depth):
    """
    Apply convolution layer with stride length of 4 + ELU layer

    Args:
        inputs (tensorflow.Tensor): Input tensor
        convolution_shape (list): Shape of the kernel used in the convolution
        output_depth (int): Depth of the output of convolution the layer

    Returns:
        tensorflow.Tensor: Tuple of the output of the ELU layer

    """
    kernel_shape = [convolution_shape[0],
                    convolution_shape[1],
                    inputs.get_shape[-1],
                    output_depth]
    weights, biases = _weights_and_biases(kernel_shape,
                                          [output_depth],
                                          name='stride_2')
    conv = tf.nn.conv2d(inputs,
                        weights,
                        strides=[1, 4, 4, 1],
                        padding='SAME')

    return tf.nn.elu(conv + biases)


# ------------------------------------------------------------------------------
def _residual_block(inputs,
                    convolution_shape,
                    output_depth,
                    convolutions,
                    simple_shortcut=True,
                    ):
    """
    This will create a series of convolutions

    Args:
        inputs (tensorflow.Tensor): Input tensor
        kernel_shape (list): Shape of the kernel used in the convolution
        bias_shape (list): Shape of the biases tensor
        convolutions (int): Number of convolutions in the residual block

    Returns:
        tensorflow.Tensor: Output of the residual block

    """
    input_loop = inputs

    for _conv_itr in range(convolutions):
        with tf.variable_scope("Convolution_{}".format(_conv_itr)):
            temp = _conv_elu(input_loop, convolution_shape, output_depth)
            input_loop = temp

    with tf.variable_scope('Shortcut_{}'.format(_conv_itr)):
        if simple_shortcut:
            shortcut = inputs
        else:
            shortcut = _conv_elu(inputs, (1, 1), output_depth)

    return temp + shortcut


# ------------------------------------------------------------------------------
def clip_batch(batch):
    """
    Clip the batch of data to a range [0, 1]

    Args:
        batch (tensorflow.Tensor): Tensor of data to clip

    Returns:
        tensorflow.Tensor: Tensor of clipped data

    """
    batch = tf.cast(batch, tf.float32)
    batch = tf.clip_by_value(batch, 0.0, 1.0)
    batch = tf.cast(batch, tf.uint8)
    return batch


# ------------------------------------------------------------------------------
def random_flip_up_down(layers, seed=None):
    """
    Randomly flip the images along the x-axis. Probability of a flip = 0.5

    Note: If heading_x is included in the layers, it must be modified to
          ensure headings follow original rules of the road.
          i.e. Drivers stay on right side of the road

    Note: Modeled after functions in python.ops.image_ops

    Args:
        layers (dict): Dictionary of layers to be transformed
        seed (int): Seed for random number generator. It's probably
                    reasonable to set this to None unless repeatability
                    is requires (eg. when testing)

    Returns:
        dict: Dictionary of layers that may have been flipped

    """
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)

    # flip all the tiles about x-axis
    mirror = tf.less(tf.pack([uniform_random, 1.0, 1.0]), 0.5)
    result = {_key: tf.reverse(_layer, mirror)
              for _key, _layer in layers.items()}

    # invert the  heading_x tile (i.e. 0 -> 1, 1 -> 0, 0.5 -> 0.5)
    if 'heading_x' in result:
        flip = tf.cast(tf.less([uniform_random], 0.5), tf.float32)
        flip = tf.mul(tf.add(flip, -0.5), -2)
        result['heading_x'] = tf.add(result['heading_x'], -0.5)
        result['heading_x'] = tf.mul(result['heading_x'], flip)
        result['heading_x'] = tf.add(result['heading_x'], 0.5)
        result['heading_x'] = tf.mul(result['heading_x'], result['occupancy'])

    return result


# ------------------------------------------------------------------------------
def random_flip_left_right(layers, seed=None):
    """
    Randomly flip the images along the y-axis. Probability of a flip = 0.5

    Note: If heading_y is included in the layers, it must be modified to
          ensure headings follow original rules of the road.
          i.e. Drivers stay on right side of the road

    Modeled after functions in python.ops.image_ops

    Args:
        layers (dict): Dictionary of layers to be transformed
        seed (int): Seed for random number generator. It's probably
                    reasonable to set this to None unless repeatability
                    is requires (eg. when testing)

    Returns:
        dict: Dictionary of layers that may have been flipped

    """
    uniform_random = tf.random_uniform([], 0, 1., seed=seed)

    # flip all the tiles about y-axis
    mirror = tf.less([1., uniform_random, 1.], 0.5)
    result = {_key: tf.reverse(_layer, mirror)
              for _key, _layer in layers.items()}

    # invert the  heading_y tile (i.e. 0 > 1, 1 > 0, 0.5 > 0.5)
    if 'heading_y' in result:
        flip = tf.cast(tf.less([uniform_random], 0.5), tf.float32)
        flip = tf.mul(tf.add(flip, -0.5), -2)
        result['heading_y'] = tf.add(result['heading_y'], -0.5)
        result['heading_y'] = tf.mul(result['heading_y'], flip)
        result['heading_y'] = tf.add(result['heading_y'], 0.5)
        result['heading_y'] = tf.mul(result['heading_y'], result['occupancy'])

    return result


# ------------------------------------------------------------------------------
def random_flips(layers, seed=None):
    """
    Randomly flip the images along the x- and y-axes

    Args:
        layers (dict): Dictionary of layers to be transformed
        seed (int): Seed for random number generator. It's probably
                    reasonable to set this to None unless repeatability
                    is requires (eg. when testing)

    Returns:
        dict: Dictionary of layers that may have been flipped

    """
    layers = random_flip_up_down(layers, seed)
    layers = random_flip_left_right(layers, seed)

    return layers


# ------------------------------------------------------------------------------
def random_drop_image(layers, seed=None):
    """
    Randomly drop the layers by multiplying it by zero

    Args:
        layers (dict): Dictionary of layers to be transformed
        seed (int): Seed for random number generator. It's probably
                    reasonable to set this to None unless repeatability
                    is requires (eg. when testing)

    Returns:
        dict: Dictionary of layers where some may have been zeroed out

    """
    if not isinstance(layers, tuple) and not isinstance(layers, list):
        layers = (layers)

    with tf.variable_scope('random_drop_image'):
        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
        mask = tf.less(tf.pack([uniform_random for _ in
                                range(layers[0].get_shape()[2])]),
                       0.5)
        mask = tf.cast(mask, tf.float32)
        return (tf.mul(_layer, mask) for _layer in layers)
