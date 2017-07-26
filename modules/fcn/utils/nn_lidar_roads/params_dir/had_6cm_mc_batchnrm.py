"""
File: jupiter_v5_5.py
Author: Brett Jackson <brett.jackson@here.com>
Version: 3.5.0

Define the model parameters for the nn_lidar_roads model directory

"""

# ==============================================================================
import os

from ...common.params import Layer
from ..params import Params

import tensorflow as tf


# ------------------------------------------------------------------------------
# Root directory on which paths are based
root_dir = '{}/../../../'.format(os.path.dirname(__file__))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
params = Params()

# Model tag for this model
params.model_tag = 'had_6cm_mc_batchnrm100_rate0.1'

# Model architecture
params.arch_name = 'batchnrm_conv_5_pool_2_2x_res_5_7_pool_2_2x_conv_1'

# Output directories for train and test runs
params.train_dir = '{}/models/nn_lidar_roads/train/{}'.format(
    root_dir, params.model_tag)
params.test_dir = '{}/models/nn_lidar_roads/eval/{}'.format(
    root_dir, params.model_tag)

# Where is the data located
data_dir = ('{}/DataPrep/cache/nn_lidar_roads/had/6cm/'.format(root_dir))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Paths to train and test data lists
params.train_list_path = (
    '{}/tile_list/train.txt'.format(data_dir))
params.test_list_path = (
    '{}/tile_list/test.txt'.format(data_dir))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Add features layers
params.features_layers = [
    Layer(name='lidar_height',
          data_path='{}/feats/height'.format(data_dir),
          summary_string='1_0_lidar_height',),
    Layer(name='lidar_intens',
          data_path='{}/feats/intens'.format(data_dir),
          summary_string='1_1_lidar_intens',),
    ]

# Add labels layers
params.labels_layers = [
    Layer(name='allbnd',
          data_path='{}/labels/all_bnd'.format(data_dir),
          summary_string='2_0_labels_allnd',
          dtype=tf.uint8,
          rescale_factor=127,
          weights={'negative': 1/100,
                   'non_observed': 0/400}),
    Layer(name='roadbnd',
          data_path='{}/labels/road_bnd'.format(data_dir),
          summary_string='2_1_labels_roadbnd',
          dtype=tf.uint8,
          rescale_factor=127,
          weights={'negative': 1/100,
                   'non_observed': 0/400}),
    ]

# Other layers
params.other_layers  = [
    Layer(name='lidar_observed',
          data_path='{}/feats/vis'.format(data_dir),
          summary_string='2_1_lidar_observed',
          dtype=tf.uint8,),
    ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Set model weights
params.weights['l2'] = 1
#params.weights['boundaries'] = 1
#params.weights['center'] = 1/2
#params.weights['negative'] = 1/20
params.weights['non_observed'] = 0/400

params.batch_size = 10
params.max_steps = 1000000001
params.num_epochs_per_decay = 100
params.learning_rate_decay_factor = 0.1
params.initial_learning_rate = 0.1

params.display_frequency = 10
params.checkpoint_frequency = 50
params.test_interval_secs = 30*60

# Training Params
params.rand_rotate = True

# Testing Params
params.crop_when_scoring = False
params.cent_when_scoring = True
params.sampling = (6,6)
params.test_image_size = (512, 512)
params.train_size = (256, 256)
