"""
File: conv_5_pool_2_2x_res_5_7_pool_2_2x_conv_1.py
Author: Brett Jackson <brett.jackson@here.com>
Version: 3.5.0

Default model for nn_lidar_roads project

"""

# ==============================================================================
from collections import OrderedDict

import tensorflow as tf

from tensorflow.contrib.layers import batch_norm
from ..model import _conv_elu
from ..model import _residual_block



# ------------------------------------------------------------------------------
def build_network(input_tensor, labels_layers, is_training=False):
    """
    Build the network for the classifier

    Args:
        input_tensor (tensorflow.Tensor): Input tensor

    Returns:
        tensorflow.Tensor: Prediction of the network

    """
    print('Building network')
    print('\tInput shape: {}'.format(input_tensor.get_shape()))

    with tf.variable_scope('Prediction'):
        # Consider adding batch norm on input...
        with tf.variable_scope("Convolution_1"):
            num_input_layers = input_tensor.get_shape().as_list()[3]
            elu_1a = _conv_elu(inputs=input_tensor,
                              convolution_shape=[5, 5],
                              output_depth=32)
            elu_1 = batch_norm(elu_1a, updates_collections=None, is_training=is_training, fused=True, scale=True)

            print('\tConvolution 1 shape: {}'.format(elu_1.get_shape()))

        with tf.variable_scope("pooling_2"):
            elu_2 = tf.nn.max_pool(value=elu_1,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID')
            print('\tPooling 2 shape: {}'.format(elu_2.get_shape()))

        with tf.variable_scope("residual_3"):
            elu_3a = _residual_block(inputs=elu_2,
                                    convolution_shape=[5, 5],
                                    output_depth=32,
                                    convolutions=7,
                                    simple_shortcut=True)
            elu_3 = batch_norm(elu_3a, updates_collections=None, is_training=is_training, fused=True, scale=True)
            print('\tResidual 3 shape: {}'.format(elu_3.get_shape()))

        with tf.variable_scope("pooling_4"):
            elu_4 = tf.nn.max_pool(value=elu_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID')

            print('\tPooling 4 shape: {}'.format(elu_4.get_shape()))

        with tf.variable_scope("residual_5"):
            elu_5a = _residual_block(inputs=elu_4,
                                    convolution_shape=[5, 5],
                                    output_depth=32,
                                    convolutions=7,
                                    simple_shortcut=True,)
            elu_5 = batch_norm(elu_5a, updates_collections=None, is_training=is_training, fused=True, scale=True)
            print('\tResidual 5 shape: {}'.format(elu_5.get_shape()))

        with tf.variable_scope("pooling_6"):
            elu_6 = tf.nn.max_pool(value=elu_5,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID')
            print('\tPooling 6 shape: {}'.format(elu_6.get_shape()))

        with tf.variable_scope("Convolution_concat"):
            #shape = input_tensor.get_shape().as_list()
            shape_dyn = tf.shape(input_tensor)
            elu_2_rs = tf.image.resize_bilinear(elu_2, [shape_dyn[1], shape_dyn[2]])
            print('\tResize elu 2: {}'.format(elu_2_rs.get_shape()))
            elu_4_rs = tf.image.resize_bilinear(elu_4, [shape_dyn[1], shape_dyn[2]])
            print('\tResize elu 4: {}'.format(elu_4_rs.get_shape()))
            elu_6_rs = tf.image.resize_bilinear(elu_6, [shape_dyn[1], shape_dyn[2]])
            print('\tResize elu 6: {}'.format(elu_6_rs.get_shape()))
            concat_feats = tf.concat(
                values=[input_tensor,
                        elu_2_rs,
                        elu_4_rs,
                        elu_6_rs,
                        ],
                axis=3,
                name='concat_feats')

            print('\tConcatinated features shape: {}'.format(
                concat_feats.get_shape()))
            concat_depth = concat_feats.get_shape().as_list()[-1]
            print('\tConcatinated features depth: {}'.format(concat_depth))
            elu_concat_a = _conv_elu(inputs=concat_feats,
                                   convolution_shape=[1, 1],
                                   output_depth=concat_depth)
            elu_concat = batch_norm(elu_concat_a, updates_collections=None, is_training=is_training, fused=True, scale=True)
            print('\tConvolution concat shape: {}'.format(
                elu_concat.get_shape()))

        with tf.variable_scope("Convolution_final"):
            result = _conv_elu(inputs=elu_concat,
                               convolution_shape=[1, 1],
                               output_depth=len(labels_layers))
            print('\tConvolution final shape: {}'.format(result.get_shape()))

            result = tf.sigmoid(result)
            print('\tSigmiod shape: {}'.format(result.get_shape()))

        with tf.variable_scope('Re_extract_results'):
            result_dict = OrderedDict()

            # Extract prediction layers matched to provided labels
            for _itr, _layer in enumerate(labels_layers):
                with tf.variable_scope('Extract_{}'.format(_layer.name)):
                    result_dict[_layer.name] = result[:, :, :, _itr:_itr+1]

            print('\tResults shapes:')
            for _key, _result in result_dict.items():
                print('\t\t{}: {}'.format(_key, _result.get_shape()))

        with tf.variable_scope('Result_summary'):
            for _itr, (_label, _result) in enumerate(result_dict.items()):
                _summary = tf.cast(255*_result, tf.uint8)
                tf.summary.image('0_{}_prediction_{}'.format(_itr, _label),
                                 _summary, max_outputs=5)

    return result_dict
