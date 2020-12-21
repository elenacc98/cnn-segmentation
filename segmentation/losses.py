"""
The losses submodule implements loss function to be used
in segmentation tasks.
"""

from segmentation.metrics import MeanDice
from segmentation.utils import calc_SDM_batch, calc_DM_batch, calc_SDM, calc_DM
from segmentation.utils import count_class_voxels, get_loss_weights
from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance
from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config


def MeanDice_Loss(numClasses):
    """

    Args:
        numClasses:

    Returns:

    """
    def mean_dice(y_true, y_pred):
        """

        Args:
            y_true:
            y_pred:

        Returns:

        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis = axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis = axisSum), tf.reduce_sum(y_pred_c, axis = axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                       tf.divide(numerator, denominator)))

        return tf.subtract(1.0, mean_over_classes)

    return mean_dice


def CrossEntropyEdge_loss(numClasses):
    """

    Args:
        numClasses: number of classes

    Returns: cross entropy value computed on boundaries of predictions and ground truth

    """

    def cross_entropy_edge_loss(y_true, y_pred):
        """

        Args:
            y_true:
            y_pred:

        Returns:

        """
        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        nEdgeVoxels = tf.math.count_nonzero(y_true)
        epsilon = backend_config.epsilon

        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        wcc_loss = -math_ops.reduce_sum(y_true * math_ops.log(y_pred), axis=(1,2,3,4)) / tf.cast(nEdgeVoxels, tf.float32)
        wcc_loss = tf.reduce_sum(tf.multiply(loss_weights, wcc_loss))

        return wcc_loss

    return cross_entropy_edge_loss


def CrossEntropyRegion_loss(numClasses):
    """

    Args:
        numClasses:

    Returns:

    """

    def cross_entropy_region_loss(y_true, y_pred):
        """

        Args:
            y_true:
            y_pred:

        Returns:

        """
        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        epsilon = backend_config.epsilon

        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        wcc_loss = -math_ops.reduce_sum(y_true * math_ops.log(y_pred), axis=(1, 2, 3, 4)) / tf.cast(nVoxels,
                                                                                                    tf.float32)
        wcc_loss = tf.reduce_sum(tf.multiply(loss_weights, wcc_loss))

        return wcc_loss

    return cross_entropy_region_loss


def Weighted_DiceBoundary_Loss(numClasses, alpha):
    """
    DiceBoundary wrapper function.
    Args:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and boundary loss

    Returns: multiclass_3D_weighted_dice_boundary_loss
    """

    def multiclass_weighted_dice_boundary_loss(y_true, y_pred):
        """
        Compute multiclass class weighted dice loss function.
        Args:
            y_true: ground truth tensor [batch, rows, columns, slices, classes], or [batch, rows, columns, classes]
            y_pred: softmax probabilities predicting classes. Shape must be the same as y_true.
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis = axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis = axisSum), tf.reduce_sum(y_pred_c, axis = axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                       tf.divide(numerator, denominator)))


        SDM = tf.py_function(func=calc_SDM_batch,
                             inp=[y_true, numClasses],
                             Tout=tf.float32)

        boundary_loss = tf.multiply(tf.reduce_sum(tf.multiply(SDM, y_pred)), 1.0/nVoxels)

        return alpha * tf.subtract(1.0, mean_over_classes) + (1-alpha) * boundary_loss

    return multiclass_weighted_dice_boundary_loss


def Weighted_DiceCatCross_Loss_v0(numClasses, alpha):
    """
    Categorical crossentropy wrapper function between y_pred tensor and a target tensor.
    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss

    Returns:
        categorical_cross_entropy function
    Raises:
        ValueError: if `axis` is neither -1 nor one of the axes of `output`.
    """

    def dice_categorical_cross_entropy(y_true, y_pred):
        """
        Computes categorical cross entropy weighted by distance weighted map.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            categorical crossentropy value
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        nVoxels = tf.size(y_true)/numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                                   tf.divide(numerator, denominator)))

        SDM = tf.py_function(func=calc_DM_batch,
                             inp=[y_true, numClasses],
                             Tout=tf.float32)

        epsilon = backend_config.epsilon
        gamma = 8
        sigma = 10

        # Exponential transformation of the Distance transform
        DWM = 1 + gamma * tf.math.exp(tf.math.negative(SDM)/sigma)
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        wcc_loss = -math_ops.reduce_sum(DWM * y_true * math_ops.log(y_pred))/tf.cast(nVoxels, tf.float32)
        return alpha * tf.subtract(1.0, mean_over_classes) + (1-alpha) * wcc_loss

    return dice_categorical_cross_entropy


def Weighted_DiceCatCross_Loss_v1(numClasses, alpha):
    """
    Categorical crossentropy wrapper function between y_pred tensor and a target tensor.
    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss

    Returns:
        categorical_cross_entropy function
    Raises:
        ValueError: if `axis` is neither -1 nor one of the axes of `output`.
    """

    def dice_categorical_cross_entropy(y_true, y_pred):
        """
        Computes categorical cross entropy weighted by distance weighted map.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            categorical crossentropy value
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        nVoxels = tf.size(y_true)/numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                                   tf.divide(numerator, denominator)))

        SDM = tf.py_function(func=calc_DM_batch,
                             inp=[y_true, numClasses],
                             Tout=tf.float32)

        epsilon = backend_config.epsilon
        gamma = 10
        sigma = 5
        DWM_list = []

        # Exponential transformation of the Distance transform
        for index in range(numClasses):
            DWM_list.append(
                loss_weights[index] + gamma * tf.math.exp(-(tf.math.square(SDM[index])) / (2 * sigma * sigma)))
        DWM = tf.stack(DWM_list)

        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        wcc_loss = -math_ops.reduce_sum(DWM * y_true * math_ops.log(y_pred))/tf.cast(nVoxels, tf.float32)
        return alpha * tf.subtract(1.0, mean_over_classes) + (1-alpha) * wcc_loss

    return dice_categorical_cross_entropy


def Weighted_DiceFocal_Loss(numClasses, alpha):
    """
    Dice + Focal loss wrapper function between y_pred tensor and a target tensor.
    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss

    Returns:
        categorical_cross_entropy function
    Raises:
        ValueError: if `axis` is neither -1 nor one of the axes of `output`.
    """

    def dice_focal(y_true, y_pred):
        """
        Computes categorical cross entropy weighted with focal method.
        Voxel classified with less confidence weight more in the function.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            dice + weighted categorical crossentropy value
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        nVoxels = tf.size(y_true)/numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                                   tf.divide(numerator, denominator)))


        epsilon = backend_config.epsilon
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        focal_loss = -math_ops.reduce_sum(tf.math.square(1 - y_pred) * y_true * math_ops.log(y_pred))/tf.cast(nVoxels, tf.float32)

        return alpha * tf.subtract(1.0, mean_over_classes) + (1-alpha) * focal_loss

    return dice_focal


def Exp_Log_Loss(numClasses, alpha, gamma):
    """
    Dice + Focal loss wrapper function between y_pred tensor and a target tensor.
    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss
        gamma: exponential of logaritmic dice and CE

    Returns:
        categorical_cross_entropy function
    Raises:
        ValueError: if `axis` is neither -1 nor one of the axes of `output`.
    """

    def exp_log(y_true, y_pred):
        """
        Computes categorical cross entropy weighted with focal method.
        Voxel classified with less confidence weight more in the function.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            dice + weighted categorical crossentropy value
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        # gamma = ops.convert_to_tensor_v2(gamma)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # gamma = tf.cast(gamma, tf.float32)

        nVoxels = tf.size(y_true)/numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        add_dice = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            # class_loss_weight = loss_weights[c]

            add_dice = tf.add(add_dice, tf.math.pow(-math_ops.log(tf.divide(numerator, denominator)), gamma))

        dice_loss = tf.reduce_mean(add_dice)

        wcc_loss = -math_ops.reduce_sum(tf.math.pow(y_true * math_ops.log(y_pred), gamma),
                                        axis=(1, 2, 3, 4)) / tf.cast(nVoxels,tf.float32)
        wcc_loss = tf.reduce_mean(tf.multiply(loss_weights, wcc_loss))

        return alpha * dice_loss + (1-alpha) * wcc_loss

    return exp_log
