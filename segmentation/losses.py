"""
The losses submodule implements loss function to be used
in segmentation tasks.
"""

from segmentation.metrics import MeanDice
from segmentation.utils import calc_SDM_batch, calc_DM_batch, \
    calc_SDM, calc_DM, calc_DM_batch_edge, calc_DM_edge, calc_DM_batch_edge2, computeContours
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


# 0
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
        Computes categorical cross entropy weighted by an exponential transformation of the
        distance weighted map. Voxels closer to boundaries are weighted more.
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


# 1
def Weighted_DiceCatCross_Loss_v1(numClasses, alpha):
    """
    Categorical crossentropy wrapper function between y_pred tensor and ground truth tensor.
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
        Computes categorical cross entropy weighted by am exponential transformation of the
        distance weighted map. Voxels closer to the boundaries are weighted more.
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


# 2
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
        Compute multiclass weighted dice index, weighted by the euclidean distance transform. Voxels
        further from the boundaries are weighted more.
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


# 3
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
        Voxel classified with less confidence are weighted more in the function.
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


# 4
def Hausdorff_Distance(numClasses, alpha):
    """
    Computes Hausdorff distance from contour groud truth loaded in y_pred by
    some DataGenerator.
    Args:
        numClasses:
        alpha:

    Returns:

    """

    def hausdorff_distance(y_true, y_pred):
        """

        Args:
            y_true:
            y_pred:

        Returns:

        """

        y_true_real = y_true[:,:,:,:,5:10]

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true_real = tf.transpose(y_true_real, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true_real = tf.transpose(y_true_real, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true_real = tf.cast(y_true_real, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true_real) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        SDM = tf.py_function(func=calc_DM_batch_edge,
                             inp=[y_true_real, numClasses],
                             Tout=tf.float32)

        h_dist = tf.multiply(SDM, tf.math.pow(tf.subtract(y_pred, y_true_real), 2))
        h_dist_loss = tf.divide(tf.reduce_sum(h_dist), nVoxels)

        return h_dist_loss

    return hausdorff_distance


def Hausdorff_Distance2(numClasses, alpha):
    """
    Computes Hausdorff distance after the generatoikn of contour ground truth labels from mask ground truth labels.

    Args:
        numClasses:
        alpha:

    Returns:

    """

    def hausdorff_distance(y_true, y_pred):
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

        SDM, contours = tf.py_function(func=calc_DM_batch_edge2,
                             inp=[y_true, numClasses],
                             Tout=[tf.float32, tf.float32])
        gamma = 8
        sigma = 10

        # Exponential transformation of the Distance transform
        DWM = 1 + gamma * tf.math.exp(tf.math.negative(SDM) / sigma)

        h_dist = tf.multiply(DWM, tf.math.pow(tf.subtract(y_pred, contours), 2))
        h_dist_loss = tf.divide(tf.reduce_sum(h_dist), nVoxels)

        return h_dist_loss

    return hausdorff_distance


# 5
def MeanDice_Loss(numClasses):
    """
    Mean dice wrapper to compute loss function with weights to compensate for class imbalance.
    Args:
        numClasses: number of classes

    Returns: mean dice weigthed by class

    """
    def mean_dice(y_true, y_pred):
        """
        Computed mean dice coefficient
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


def MeanDice_Loss2(numClasses):
    """
    Mean Dice where intra-articular slices are weighted more than slices further from the knee joint location
    that are easier to segment.
    Args:
        numClasses: number of classes

    Returns: weigthted mean dice

    """
    def mean_dice(y_true, y_pred):
        """
        Computed mean dice weighting differently both th classes and the slides in the volume.
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
        x = tf.linspace(1, 192, 192)
        vect = tf.cast(1.5 * tf.exp(-tf.square(tf.subtract(x, 85)) / 800) + 0.6, tf.float32)
        # Loop over each class
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis = (0,1,2)))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis = (0,1,2)), tf.reduce_sum(y_pred_c, axis = (0,1,2)))

            numerator_1 = tf.reduce_sum(tf.multiply(vect, numerator))
            denominator_1 = tf.reduce_sum(tf.multiply(vect, denominator))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                       tf.divide(numerator_1, denominator_1)))

        return tf.subtract(1.0, mean_over_classes)

    return mean_dice


# 6
def Exp_Log_Loss(numClasses, gamma=1):
    """
    Exponential logarithmic computation of dice and Cross entropy indexes.
    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss
        gamma: exponential of logaritmic dice and CE

    Returns:
        categorical_cross_entropy function
    """

    def exp_log(y_true, y_pred):
        """
        Computes categorical cross entropy and dice with exponential logarithmic transformations.
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

        epsilon = backend_config.epsilon
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)

        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            # class_loss_weight = loss_weights[c]

            add_dice = tf.add(add_dice,
                              tf.math.pow(-math_ops.log(clip_ops.clip_by_value(
                                  tf.divide(numerator, denominator), epsilon_, 1. - epsilon_)), gamma))

        dice_loss = tf.divide(add_dice, numClasses)

        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        wcc_loss = -math_ops.reduce_sum(tf.math.pow(y_true * math_ops.log(y_pred), gamma),
                                        axis=(1, 2, 3, 4)) / tf.cast(nVoxels,tf.float32)
        wcc_loss = tf.reduce_mean(tf.multiply(loss_weights, wcc_loss))

        return 0.8 * dice_loss + 0.2 * wcc_loss

    return exp_log


# 7
def Boundary_Crossentropy(numClasses):
    """
    Computes "double-faced" boundary cross entropy, after the generation of contours ground truth labels with function
    "computeContours".
    Args:
        numClasses: number fo classes

    Returns: value of "double-faced" cross entropy

    """

    def boundary_crossentropy(y_true, y_pred):
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

        contours = tf.py_function(func=computeContours,
                                  inp=[y_true, numClasses],
                                  Tout=tf.float32)

        nEdgeVoxels = tf.math.count_nonzero(contours)
        nEdgeVoxels = tf.cast(nEdgeVoxels, tf.float32)
        beta = 1 - nEdgeVoxels / nVoxels
        epsilon = backend_config.epsilon
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = beta * tf.multiply(contours, math_ops.log(y_pred))
        second_term = (1 - beta) * tf.multiply((1 - contours), math_ops.log(1 - y_pred))

        bc = - tf.reduce_sum(tf.add(first_term, second_term)) / nEdgeVoxels

        return bc

    return boundary_crossentropy


def Dist_Boundary_Crossentropy(numClasses):
    """
    Computes "double-faced" boundary cross entropy, after the generation of contours ground truth labels
    and generation of euclidean distance transform map with function "calc_DM_batch_edge2".
    Args:
        numClasses: number of classes

    Returns: Value of "double-faced" cross entropy

    """

    def dist_boundary_crossentropy(y_true, y_pred):
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

        SDM, contours = tf.py_function(func=calc_DM_batch_edge2,
                                  inp=[y_true, numClasses],
                                  Tout=[tf.float32, tf.float32])

        nEdgeVoxels = tf.math.count_nonzero(contours)
        nEdgeVoxels = tf.cast(nEdgeVoxels, tf.float32)

        gamma = 8
        sigma = 10
        # Exponential transformation of the Distance transform
        DWM = 1 + gamma * tf.math.exp(tf.math.negative(SDM) / sigma)

        beta = 1 - nEdgeVoxels / nVoxels
        epsilon = backend_config.epsilon
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = beta * tf.multiply(DWM, tf.multiply(contours, math_ops.log(y_pred)))
        second_term = (1 - beta) * tf.multiply(DWM, tf.multiply((1 - contours), math_ops.log(1 - y_pred)))

        bc = - tf.reduce_sum(tf.add(first_term, second_term)) / nEdgeVoxels

        return bc

    return dist_boundary_crossentropy


# 8
def Region_Crossentropy(numClasses):
    """
    Computes the "double-faced" regional cross entropy function.
    Args:
        numClasses: number of classes

    Returns: vaue of the the "double-faced" regional cross entropy

    """

    def region_crossentropy_loss(y_true, y_pred):
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
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = 0.5 * tf.multiply(y_true, math_ops.log(y_pred))
        second_term = 0.5 * tf.multiply((1 - y_true), math_ops.log(1 - y_pred))

        bc_temp = - tf.reduce_sum(tf.add(first_term, second_term), axis=(1, 2, 3, 4))
        bc = tf.reduce_sum(tf.multiply(loss_weights, bc_temp)) / nVoxels

        return bc

    return region_crossentropy_loss


def Dist_Region_Crossentropy(numClasses):
    """
     Computes "double-faced" regional cross entropy weighted with euclidean distance transform map,
     after generation of euclidean distance transform map with function "calc_DM_batch".
    Args:
        numClasses: number of classes

    Returns: value of "double-faced" regional cross entropy

    """

    def dist_region_crossentropy_loss(y_true, y_pred):
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

        SDM = tf.py_function(func=calc_DM_batch,
                             inp=[y_true, numClasses],
                             Tout=tf.float32)
        gamma = 8
        sigma = 10
        DWM = 1 + gamma * tf.math.exp(tf.math.negative(SDM) / sigma)

        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        epsilon = backend_config.epsilon
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = 0.5 * tf.multiply(DWM, tf.multiply(y_true, math_ops.log(y_pred)))
        second_term = 0.5 * tf.multiply(DWM, tf.multiply((1 - y_true), math_ops.log(1 - y_pred)))

        bc_temp = - tf.reduce_sum(tf.add(first_term, second_term), axis=(1, 2, 3, 4))
        bc = tf.reduce_sum(tf.multiply(loss_weights, bc_temp)) / nVoxels

        return bc

    return dist_region_crossentropy_loss



