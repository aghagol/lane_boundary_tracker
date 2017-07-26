from scipy.ndimage.filters import *


def downsample_data(features, ds=[1, 3]):
    # labels_ds = maximum_filter(labels, tuple(ds)+(1,))[::ds[0],::ds[1]]
    if (len(features.shape) == 3):
        features_ds = maximum_filter(features, tuple(ds) + (1,))[::ds[0], ::ds[1]]  # Gaussian filter?
    else:
        features_ds = maximum_filter(features, ds)[::ds[0], ::ds[1]]  # Gaussian filter?

    return features_ds
