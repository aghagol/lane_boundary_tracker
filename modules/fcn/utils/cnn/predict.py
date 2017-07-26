from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from ..common.params import get_total_channels
from ..common.utils import restore_vars
from ..common.arch_dir import batchnrm_conv_5_pool_2_2x_res_5_7_pool_2_2x_conv_1

class Prediction(object):
    # --------------------------------------------------------------------------
    def __init__(self, params, mult=16, pad=None):
        self.params = params
        self.mult = mult

        if (pad is None):
            pad = [0, 0]

        self.pad = pad

        print('init session')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.001)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=config)

        with tf.device('/cpu:0'):
            print("Get features and labels")
            self.features_pl = tf.placeholder(
                tf.float32,
                shape=(1,
                       None,
                       None,
                       get_total_channels(params.features_layers)))

            print("Build network")
            build_network = batchnrm_conv_5_pool_2_2x_res_5_7_pool_2_2x_conv_1.build_network
            self.prediction_op = build_network(self.features_pl, params.labels_layers)

            print("Define the param saver")
            self.saver = tf.train.Saver()

        step = restore_vars(self.saver,
                            self.sess,
                            self.params.train_dir,
                            path=self.params.eval_model_path,
                            quiet=True)
        print('Restored model at step {}'.format(step))

    def predict_raw(self, tile_data):
        prediction = self.sess.run(
            [self.prediction_op],
            feed_dict={self.features_pl: tile_data})

        return prediction

    def predict(self, im0, label='boundaries'):
        shape_arr = np.array(im0.shape[:2]) + self.pad

        mult_pad = (np.ceil(shape_arr.astype(float) / self.mult) * self.mult - shape_arr).astype(
            int)  # How much to pad to make it a multiple of self.mult

        total_pad = np.r_[mult_pad + self.pad]

        if ((total_pad > 0).any()):
            pad0 = np.r_[total_pad / 2., 0]  # append 0 for feature channels
            padF = tuple([(int(np.ceil(p)), int(np.floor(p))) for p in pad0])

            im = np.pad(im0, padF, 'constant', constant_values=0)

        else:
            im = im0
            padF = None

        X = self.predict_raw(im[None])

        if (type(label) is str):
            X = X[0][label][0, :, :, 0]
        else:
            Xt = X[0]
            Xout = np.dstack([Xt[lab][0, :, :, 0] for lab in label])
            X = Xout

        # Unpad
        if (padF is not None):
            X = X[padF[0][0]:X.shape[0] - padF[0][1], padF[1][0]:X.shape[1] - padF[1][1]]

        return X
