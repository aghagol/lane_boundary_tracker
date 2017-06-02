"""
canny.py - Canny Edge detector

Reference: Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter, generate_binary_structure, binary_erosion, label, convolve

from skimage import dtype_limits
from skimage.filters import gabor_kernel

def peaks_clean(image, th, input_as_mag=False):

    T = gaussian_filter(image, 1)
    peak = peaks(T, input_as_mag=input_as_mag)
    peak[T<th] = 0

    # Get rid of noise:
    comps, _ = label(peak, structure=np.ones((3,3)))
    cnts = np.bincount(comps[comps>0])
    peak = peak*(cnts[comps]>10) 
    
    return peak

def peaks(image, sigma=3., low_threshold=0, high_threshold=0, mask=None, input_as_mag=False):
    """Edge filter an image using the Canny algorithm.

    Modified by Ian Endres (HERE) to follow peaks of linear structures rather than edges.

    Parameters
    -----------
    image : 2D array
        Greyscale input image to detect edges on; can be of any dtype.
    sigma : float
        Standard deviation of the Gaussian filter.
    low_threshold : float
        Lower bound for hysteresis thresholding (linking edges).
        If None, low_threshold is set to 10% of dtype's max.
    high_threshold : float
        Upper bound for hysteresis thresholding (linking edges).
        If None, high_threshold is set to 20% of dtype's max.
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.

    Returns
    -------
    output : 2D array (image)
        The binary edge map.

    See also
    --------
    skimage.sobel

    Notes
    -----
    The steps of the algorithm are as follows:

    * Smooth the image using a Gaussian with ``sigma`` width.

    * Apply the horizontal and vertical Sobel operators to get the gradients
      within the image. The edge strength is the norm of the gradient.

    * Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.

    * Perform a hysteresis thresholding: first label all points above the
      high threshold as edges. Then recursively label any point above the
      low threshold that is 8-connected to a labeled point as an edge.

    References
    -----------
    Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986

    William Green's Canny tutorial
    http://dasl.mem.drexel.edu/alumni/bGreen/www.pages.drexel.edu/_weg22/can_tut.html

    Examples
    --------
    >>> from skimage import filter
    >>> # Generate noisy image of a square
    >>> im = np.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>> im += 0.2 * np.random.random(im.shape)
    >>> # First trial with the Canny filter, with the default smoothing
    >>> edges1 = filter.canny(im)
    >>> # Increase the smoothing for better results
    >>> edges2 = filter.canny(im, sigma=3)
    """

    #
    # The steps involved:
    #
    # * Smooth using the Gaussian with sigma above.
    #
    # * Apply the horizontal and vertical Sobel operators to get the gradients
    #   within the image. The edge strength is the sum of the magnitudes
    #   of the gradients in each direction.
    #
    # * Find the normal to the edge at each point using the arctangent of the
    #   ratio of the Y sobel over the X sobel - pragmatically, we can
    #   look at the signs of X and Y and the relative magnitude of X vs Y
    #   to sort the points into 4 categories: horizontal, vertical,
    #   diagonal and antidiagonal.
    #
    # * Look in the normal and reverse directions to see if the values
    #   in either of those directions are greater than the point in question.
    #   Use interpolation to get a mix of points instead of picking the one
    #   that's the closest to the normal.
    #
    # * Label all points above the high threshold as edges.
    # * Recursively label any point above the low threshold that is 8-connected
    #   to a labeled point as an edge.
    #
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    #

    if image.ndim != 2:
        raise TypeError("The input 'image' must be a two-dimensional array.")

    if low_threshold is None:
        low_threshold = 0.1 * dtype_limits(image)[1]

    if high_threshold is None:
        high_threshold = 0.2 * dtype_limits(image)[1]

    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    from skimage.filters import gabor_kernel
    from scipy import ndimage as ndi

    # Build filters:
    frequency = 0.35/sigma

    # For determining the orientation bin
    b_resp = np.zeros(image.shape + (4,))
    b_kernels = [] # Debug only
    for i in range(4):
        theta = i / 4. * np.pi + np.pi/8
        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma/2.))
        b_resp[:,:,i] = convolve(image, kernel, mode='constant')

    # For determining the angle weights
    w_resp = np.zeros(image.shape + (2,))
    for i in range(2):
        theta = i / 2. * np.pi
        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma/2.))
        w_resp[:,:,i] = convolve(image, kernel, mode='constant')


    orientation = np.argmax(b_resp,2)
    if(input_as_mag):
        magnitude = image
    else:
        magnitude = np.max(b_resp,2)
    #resp_clip = np.maximum(w_resp,0)
    isobel = w_resp[:,:,1]
    jsobel = w_resp[:,:,0]
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)
    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    s = generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)
    #
    #--------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    # 
    local_maxima = np.zeros(image.shape, bool)
    #----- 0 to 45 degrees ------
    #pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    #pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    #pts = pts_plus | pts_minus
    pts = orientation==1
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:, :][pts[:-1, :]] # 0
    c2 = magnitude[1:, 1:][pts[:-1, :-1]] # 45
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    #w = resp_clip[:,:,3][pts]/(resp_clip[:,:,3][pts]+resp_clip[:,:,0][pts]+1e-9)
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]] # 0
    c2 = magnitude[:-1, :-1][pts[1:, 1:]] # 180+45
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    #pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    #pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    #pts = pts_plus | pts_minus
    pts = orientation==0
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]] # 90
    c2 = magnitude[1:, 1:][pts[:-1, :-1]] # 45
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts] # Linearly interpolate between vertical and diagonal
    #w = resp_clip[:,:,3][pts]/(resp_clip[:,:,3][pts]+resp_clip[:,:,2][pts]+1e-9)
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    #pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    #pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    #pts = pts_plus | pts_minus
    pts = orientation==3
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]] # 90
    c2 = magnitude[:-1, 1:][pts[1:, :-1]] # 135
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    #w = resp_clip[:,:,1][pts]/(resp_clip[:,:,1][pts]+resp_clip[:,:,2][pts]+1e-9)
    c_plus = c2 * w + c1 * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    #pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    #pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    #pts = pts_plus | pts_minus
    pts = orientation==2
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]] # 180
    c2 = magnitude[:-1, 1:][pts[1:, :-1]] # 135
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    #w = resp_clip[:,:,1][pts]/(resp_clip[:,:,1][pts]+resp_clip[:,:,0][pts]+1e-9)
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #
    #---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                             np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask

