"""
File: utils.py
Author: Brett Jackson <brett.jackson@here.com>
Version: 3.5.0

Helper functions used to build a model

"""

# ==============================================================================
import os
from importlib import import_module
from argparse import ArgumentParser

import tensorflow as tf


# ------------------------------------------------------------------------------
def get_file_names(prefix):
    """
    Get the relative file name with the prefix directory

    Args:
        prefix (str): Base directory to strip from the full paths

    Returns:
        list: List of relative file paths

    """
    # Get all the files in the directory structure within the prefix path.
    # Strip off the prefix from each of the file names in the file lists.
    # In principle, this results in list that should be identicle.
    # If these lists are not identical, that means tiles are missing in at
    # least one of the layers. We will drop those missing tiles from all
    # layers
    return ['{}/{}'.format(_root, _file).replace(prefix, '').lstrip('/')
            for _root, _, _files in os.walk(prefix)
            for _file in _files
            if '.png' in _file]


# ------------------------------------------------------------------------------
def restore_vars(saver, sess, checkpoint_dir, path=None, quiet=False):
    """
    Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage.

    Args:
        saver (tf.train.Saver): Saver object used to restore the model
        sess (tensorflow.Session): TensorFlow session in  which to
                                   initialize variables
        checkpoint_dir (str): Path to directory where checkpoints are
                              stored
        path (str): Path to model to be restored. If path is not None,
                    Use this path. If the path is None, get the model
                    path from the checkpoint directory.
        quiet (bool): Do we print statements while restoring variables

    Returns:
        str: Tag for the restored step. If no model is restored, return None

    """
    if not quiet:
        print('Attempting to restore model from {}'.format(checkpoint_dir))

    if not quiet:
        print('\tInitializing all variables')
    sess.run(tf.global_variables_initializer())

    if path is None:
        print('\tNo model path provided. I will determine it from the'
              'checkpoint directory')

        if not os.path.exists(checkpoint_dir):
            try:
                if not quiet:
                    print('\tCheckpoint directory does not exist')
                    print('\tMaking {}'.format(checkpoint_dir))
                os.makedirs(checkpoint_dir)

                return False

            except OSError:
                raise

        path = tf.train.get_checkpoint_state(checkpoint_dir)

        if path is None:
            if not quiet:
                print("\tNo model restored")
            return None

        path = path.model_checkpoint_path

        path = '{}/{}'.format(checkpoint_dir, path.split('/')[-1])

    if not quiet:
        print("\tRestored model from {}".format(path))

    saver.restore(sess, path)
    step = path.split('-')[-1]

    if not quiet:
        print('\tModel at step {}'.format(step))

    return int(step)


# ------------------------------------------------------------------------------
def get_file_lists(params, file_list_path=None):
    """
    Get lists of input files

    Args:
        params (Params): Parameter container object
        file_list_path (str): Path to the list of valid files to consider

    Returns:
        dict: Dictionary of the file paths of the various inputs

    """
    # Get lists of input files. These include one file lists for each
    # of the items in params.prefixes

    check_files = True  # This takes forever for big datasets.

    if (check_files):
        file_names = {_layer.name: get_file_names(_layer.data_path)
                      for _layer in params.all_layers()}

        # Keep only tiles that exist in **all** layers
        file_names_set = None
        for _file_names in file_names.values():
            if file_names_set is None:
                file_names_set = set(_file_names)
            file_names_set = file_names_set.intersection(set(_file_names))

    # Keep only tiles within the mask
    if file_list_path is not None and os.path.exists(file_list_path):
        with open(file_list_path, 'r') as valid_list_file:
            valid_list = [_line.rstrip() for
                          _line in valid_list_file.readlines()]

        if (check_files):
            file_names_set = file_names_set.intersection(set(valid_list))
        else:
            file_names_set = set(valid_list)

    # Create list of inputs from our set
    file_names_set = list(file_names_set)
    num_inputs = len(file_names_set)

    # Reconstruct the file lists
    file_names = {_layer.name: ['{}/{}'.format(_layer.data_path, _input)
                                for _input in file_names_set]
                  for _layer in params.all_layers()}

    return file_names


# ------------------------------------------------------------------------------
def get_images_and_labels(params, gen_batch, list_path=None):
    """
    This function was intended to work with the function above with the
    small dataset

    Args:
        params (Params): Parameter container object
        gen_batch (function): Function that generates a batch of data
                              for training/testing. This function should
                              consume a dictionary of tensors (of file
                              names), an optional summary, and the batch
                              size. The function should return a
                              dictionary of batches with the keys:
                              ['features', 'labels']
        list_path (str): Path to the list of valid files to consider

    Returns:
        dict: Dictionary containing the features and label queues. The
              dictionary also contains the length of the file list.

    """
    # Get lists of input files.
    file_names = get_file_lists(params, list_path)
    num_tiles = len(file_names[list(file_names.keys())[0]])
    print('Number of tiles: {}'.format(num_tiles))

    batch = gen_batch(file_names, params, )
    batch['length'] = num_tiles

    return batch
