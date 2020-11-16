"""
Utils functions.
"""

from scipy.ndimage import distance_transform_edt as distance
import numpy as np
import tensorflow as tf


def calc_dist_map_2D(seg):
    """
    Computes distance map of input ground truth image using scipy function
    Args:
        seg: 2D or 3D binary array to compute the distance map
    Returns:
        res: distance map
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def calc_dist_map_3D(seg):
    """
    Computes separately 2D distance maps of all single slices of
    input ground truth volume, using scipy function
    Args:
        seg: 3D binary array to compute the distance map
    Returns:
        res: distance maps computed singularly from 2D slices of input
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    for i in range(seg.shape[2]):
        pos = posmask[:, :, i]
        if pos.any():
            neg = ~pos
            res[:, :, i] = distance(neg) * neg - (distance(pos) - 1) * pos
    return res


def calc_dist_map_batch_2D(y_true, numClasses):
    """
    Prepares the input for distance maps computation, and pass it to calc_dist_map
    Args:
        y_true: ground truth tensor of dimensions [class, batch_size, rows, columns, slices]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            dist_batch[c, i] = calc_dist_map_2D(y)
    return np.array(dist_batch).astype(np.float32)


def calc_dist_map_batch_3D(y_true, numClasses):
    """
    Prepares the input for distance maps computation, and pass it to calc_dist_map
    Args:
        y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            dist_batch[c, i] = calc_dist_map_3D(y)
    return np.array(dist_batch).astype(np.float32)


def count_class_voxels(labels, nVoxels, numClasses):
    """
    Counts total number of voxels for each class in the batch size.
    input is supposed to be 4 or 5-dimensional: (class, batch, rows, columns) or
    (class, batch, rows, columns, slices).
    Args:
        labels: ground truth tensor of dimensions (class, batch_size, rows, columns, slices) or
        (class, batch_size, rows, columns)
        nVoxels: total number of voxels
        numClasses: number of classes
    Returns:
        out: list with number of voxel per class
    """
    out = [0] * numClasses
    out[0] = 0
    for c in range(1, numClasses):
        out[c] = tf.math.count_nonzero(labels[c])
    first_term = tf.cast(nVoxels, 'int64')
    second_term = tf.reduce_sum(out)
    out[0] = tf.subtract(first_term, second_term)
    return out


def get_loss_weights(labels, nVoxels, numClasses):
    """
    Compute loss weights for each class.
    Args:
        labels: ground truth tensor of dimensions (class, batch_size, rows, columns, slices) or
        (class, batch_size, rows, columns)
        nVoxels: total number of voxels
        numClasses: number of classes
    Returns:
        1D tf.tensor of len = numClasses containing weights for each class
    """

    numerator_1 = count_class_voxels(labels, nVoxels, numClasses)
    numerator = tf.multiply(1.0 / nVoxels, numerator_1)
    subtract_term = tf.subtract(1.0, numerator)
    return tf.multiply(1.0 / (numClasses - 1), subtract_term)
