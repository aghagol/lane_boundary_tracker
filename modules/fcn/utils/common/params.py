"""
File: params.py
Author: Brett Jackson <brett.jackson@here.com>
Version: 3.5.0

Parameter container class used to store parameters for the model

"""

# ==============================================================================
import os

from itertools import chain

import tensorflow as tf

import arch_dir


# ------------------------------------------------------------------------------
def get_layer_property(list_of_layers, prop):
    """
    Given a list of layers, return a list of the specified layer property

    Args:
        list_of_layers (list): List of layers to extract the names

    Returns:
        list: List of names for the layers

    """
    prop = prop.lower()

    if len(list_of_layers) == 0:
        return []
    for _layer in list_of_layers:
        assert prop in dir(_layer), (
            'Property [{}] does not exist in the layer. Please check the '
            'property name'.format(prop))

    return [getattr(_layer, prop) for _layer in list_of_layers]


# ------------------------------------------------------------------------------
def get_total_channels(list_of_layers):
    """
    Given a list of layers, return the total number of channels
    contained in all the layers

    Args:
        list_of_layers (list): List of layers to query for the number
            of channels

    Returns:
        int: Total number of channels

    """
    return sum(_layer.number_channels for _layer in list_of_layers)


# ==============================================================================
class Layer(object):
    # --------------------------------------------------------------------------
    def __init__(self,
                 name,
                 data_path,
                 summary_string,
                 rescale_factor=None,
                 dtype=tf.float32,
                 number_channels=1,
                 weights=None,
                 remap=None,
                 png_dtype=tf.uint8):
        """
        Initialize an instance of the Layer class

        Args:
            name (str): Name of the layer
            data_path (str): Path to base directory of this layer in the
                file system
            summary_string (str): Summary string for this layer on the
                TensorBoard
            rescale_factor (float): How much to rescale this tile by when
                processing the image
            dtype (tensorflow.DType): Data type of this object. Typicaly,
                features will be of type tf.float32
            number_channels (int): Number of channels to be read in this
                image

        Returns:
            Nothing

        """
        self.name = name
        self.data_path = data_path
        self.summary_string = summary_string

        if (rescale_factor is None):
            if (png_dtype == tf.uint16):
                self.rescale_factor = 2 ** 16 - 1
            else:
                self.rescale_factor = 2 ** 8 - 1
        else:
            self.rescale_factor = rescale_factor

        self.dtype = dtype
        self.number_channels = number_channels
        self.remap = remap
        self.png_dtype = png_dtype

        if weights is None:
            weights = {}
        if not isinstance(weights, dict):
            print("Weights provided for {} is not a dictionary."
                  "I'm setting weights to \{\}".format(name))
            weights = {}
        self.weights = weights

    # --------------------------------------------------------------------------
    def __str__(self):
        """
        How to display the layer object when "printed"

        Args:
            Nothing

        Returns:
            Nothing

        """
        return '\n'.join(
            ['Layer object:',
             '\tName: {}'.format(self.name),
             '\tData path: {}'.format(self.data_path),
             '\tSummary string: {}'.format(self.summary_string),
             '\tRescale factor: {}'.format(self.rescale_factor),
             '\tDtype: {}'.format(self.dtype),
             '\tNumber channels: {}'.format(self.number_channels),
             ])


# ==============================================================================
class Params_base(object):
    # --------------------------------------------------------------------------
    def __init__(self):
        self.model_tag = None
        self.arch_name = None

        self.train_dir = None
        self.test_dir = None

        self.eval_model_path = None

        self.features_layers = []
        self.labels_layers = []
        self.other_layers = []

        self.train_list_path = None
        self.test_list_path = None

        self.weights = {}

        self.batch_size = None
        self.max_steps = None
        self.num_epochs_per_decay = None
        self.learning_rate_decay_factor = None
        self.initial_learning_rate = None

        self.display_frequency = None
        self.checkpoint_frequency = None
        self.test_interval_secs = None

        self.rand_rotate = False
        self.rand_crop = True
        self.train_size = (256, 256)
        self.test_image_size = (362, 362)
        self.sampling = (1, 1)

    # --------------------------------------------------------------------------
    def __str__(self):
        """
        How to display the object when "printed"

        Args:
            Nothing

        Returns:
            Nothing

        """
        layer_summaries = '\n'.join(
            ['\t\t{}'.format(str(_layer).replace('\n', '\n\t\t'))
             for _layer in self.all_layers()])
        weight_summaries = '\n'.join(['\t\t{}: {}'.format(_key, _value)
                                      for _key, _value in self.weights.items()])
        return '\n'.join(
            ['Parameter container:',
             '\tModel tag: {}'.format(self.model_tag),
             '\tTrain directory: {}'.format(self.train_dir),
             '\tTest directory: {}'.format(self.test_dir),
             '\tEvaluation model path: {}'.format(self.eval_model_path),
             '\tLayers:', layer_summaries,
             '\tTrain list path: {}'.format(self.train_list_path),
             '\tTest list path: {}'.format(self.test_list_path),
             '\tWeights:', weight_summaries,
             '\tBatch size: {}'.format(self.batch_size),
             '\tMax steps: {}'.format(self.max_steps),
             '\tNum epochs per decay: {}'.format(self.num_epochs_per_decay),
             '\tLearning rate decay factor: {}'.format(
                 self.learning_rate_decay_factor),
             '\tInitial learning rate: {}'.format(self.initial_learning_rate),
             '\tDisplay frequency: {}'.format(self.display_frequency),
             '\tCheckpoint frequency: {}'.format(self.checkpoint_frequency),
             '\tTest interval in seconds: {}'.format(self.test_interval_secs),
             ])

    # --------------------------------------------------------------------------
    @property
    def train_dir(self):
        assert self.__train_dir is not None, (
            'Please, provide valid path for train_dir')
        return self.__train_dir

    @train_dir.setter
    def train_dir(self, new_input):
        self.__train_dir = new_input

    # --------------------------------------------------------------------------
    @property
    def test_dir(self):
        assert self.__test_dir is not None, (
            'Please, provide valid path for test_dir')
        return self.__test_dir

    @test_dir.setter
    def test_dir(self, new_input):
        self.__test_dir = new_input

    # --------------------------------------------------------------------------
    @property
    def checkpoint_dir(self):
        return '{}/checkpoints'.format(self.train_dir)

    # --------------------------------------------------------------------------
    @property
    def train_summary_dir(self):
        return '{}/summary'.format(self.train_dir)

    # --------------------------------------------------------------------------
    @property
    def test_summary_dir(self):
        return '{}/summary'.format(self.test_dir)

    # --------------------------------------------------------------------------
    def all_layers(self):
        """
        Get a list of all the layers
            [*features_layers, *labels_layers, *other_layers]

        Args:
            Nothing

        Returns:
            list: List of layer objects

        """
        return [_layer for _layer in chain(self.features_layers,
                                           self.labels_layers,
                                           self.other_layers)]

    # --------------------------------------------------------------------------
    def check_data_dirs_exist(self):
        """
        Check the data directories listed in all layers exist

        Args:
            Nothing

        Returns:
            bool: Do all the data directories exist

        """
        all_found = True

        for _layer in self.all_layers():
            if not os.path.exists(_layer.data_path):
                print('Path for {} does not exist: {}'.format(
                    _layer.name, _layer.data_path))
                all_found = False
        return all_found


        # --------------------------------------------------------------------------


# ==============================================================================
if __name__ == "__main__":
    params = Params()

    root_dir = '/home/bjackson/DeepLearningProbe'

    params.train_dir = '{}/models/tf_train'.format(root_dir)
    params.test_dir = '{}/models/tf_test'.format(root_dir)

    params.model_tag = 0

    data_dir = ('{}/DataPrep/cache/nn_lidar_roads/lidar_tiles_v4/'
                'v4_60cm_6cm'.format(root_dir))

    # Add features layers
    params.features_layers = [
        Layer(name='lidar_height',
              data_path='{}/feats/height'.format(data_dir),
              summary_string='1_0_lidar_height', ),
        Layer(name='lidar_intens',
              data_path='{}/feats/intens'.format(data_dir),
              summary_string='1_1_lidar_intens', ),
        Layer(name='lidar_observed',
              data_path='{}/feats/observed'.format(data_dir),
              summary_string='1_2_lidar_observed', ),
    ]

    # Add labels layers
    params.labels_layers = [
        Layer(name='labels',
              data_path='{}/labels/lane_bnd'.format(data_dir),
              summary_string='2_labels',
              dtype=tf.uint8),
    ]

    # Set model weights
    params.weights['l2_weight'] = 10
    params.weights['negative_class_weight'] = 1 / 10

    params.batch_size = 10
    params.max_steps = 1001
    params.num_epochs_per_decay = 50
    params.learning_rate_decay_factor = 0.1
    params.initial_learning_rate = 0.1

    params.display_frequency = 100
    params.checkpoint_frequency = 100
    params.test_interval_secs = 10 * 60

    # Assert model parameters are as expected
    print(params)

    assert (params.train_dir ==
            '/home/bjackson/DeepLearningProbe/models/tf_train')
    assert params.test_dir == '/home/bjackson/DeepLearningProbe/models/tf_test'
    assert params.model_tag == 0

    assert get_layer_property(
        params.features_layers, 'name') == ['lidar_height',
                                            'lidar_intens',
                                            'lidar_observed']
    assert get_layer_property(
        params.labels_layers, 'name') == ['labels']
    assert get_layer_property(
        params.all_layers(), 'name') == ['lidar_height',
                                         'lidar_intens',
                                         'lidar_observed',
                                         'labels']

    assert params.check_data_dirs_exist()

    assert params.batch_size == 10
    assert params.max_steps == 1001
    assert params.num_epochs_per_decay == 50
    assert params.learning_rate_decay_factor == 0.1
    assert params.initial_learning_rate == 0.1

    assert params.display_frequency == 100
    assert params.checkpoint_frequency == 100
    assert params.test_interval_secs == 600

    assert get_total_channels(params.features_layers) == 3
    assert get_total_channels(params.labels_layers) == 1
    assert get_total_channels(params.all_layers()) == 4
