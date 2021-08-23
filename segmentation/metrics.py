"""
The metrics module defines some classes to be used as metrics
during model training.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import (array_ops, confusion_matrix, init_ops,
                                   math_ops)


class PerClassIoU(Metric):
    """Computes the Intersection-Over-Union metric per-class. This metric
    is supposed to work only with three-dimensional input.
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation, obtained by computing the IOU for each semantic class.
    IOU is defined as follows:
    .. math::
          IOU = \\frac{TP}{TP+FP+FN}

    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Args:
      num_classes: The possible number of labels the prediction task can have.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    Standalone usage:
    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # result = (1 / (2 + 2 - 1) , 1 / (2 + 2 - 1)) = 0.33, 0.33
    >>> m = segmentation.metrics.PerClassIoU(num_classes=2, class_to_return=1)
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result().numpy()
    0.33333334
    >>> m.reset_states()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> m.result().numpy()
    0.14285715

    Usage with `compile()` API:
    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[segmentation.metrics.PerClassIoU(num_classes=2, class_to_return=0)])
    ```
    """

    def __init__(self, num_classes, name=None, dtype=None, class_to_return=0):
        super(PerClassIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64)
        self.class_to_return = class_to_return

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        """
    # Flatten the input if its rank > 1.
    if y_pred.shape.ndims > 1:
      y_pred = array_ops.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
      y_true = array_ops.reshape(y_true, [-1])
    """

        # Select predicted class for each voxel
        y_pred = tf.argmax(y_pred, axis=-1, output_type='int32')
        y_pred = tf.one_hot(
            indices=y_pred, depth=self.num_classes, axis=-1, dtype='int32')
        y_pred = tf.cast(y_pred, 'bool')

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row[self.class_to_return] + \
            sum_over_col[self.class_to_return] - \
            true_positives[self.class_to_return]

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(math_ops.cast(
            math_ops.not_equal(denominator, 0), dtype=self._dtype))

        iou = math_ops.div_no_nan(
            true_positives[self.class_to_return], denominator)

        return math_ops.div_no_nan(
            math_ops.reduce_sum(iou, name=self.name), num_valid_entries)

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros(
            (self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(PerClassIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dice(Metric):
    """Computes the Dice metric per-class.

    Dice is a common evaluation metric for semantic image
    segmentation, obtained by computing the Dice for each semantic class.
    Dice is defined as follows:

    .. math::

        Dice = \\frac{2*TP}{2*TP + FP + FN}

    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Standalone usage: ::

        >>> # cm = [[1, 1],
        >>> #        [1, 1]]
        >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        >>> # dice = 2*true_positives / (sum_row + sum_col))
        >>> # result = (2 / (2 + 2)) = 0.5
        >>> m = segmentation.metrics.Dice(num_classes=2, class_to_return=0)
        >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
        >>> m.result().numpy()
        0.5
        >>> m = segmentation.metrics.Dice(num_classes=2, class_to_return=1)
        >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
        ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
        >>> m.result().numpy()
        0.25

    Usage with `compile()` API: ::

        model.compile(
            optimizer='sgd',
            loss='mse',
            metrics=[segmentation.metrics.Dice(num_classes=2)]
            )

    Args:

        num_classes (int, required): The possible number of labels
            the prediction can have.
        name (str, optional): string name of the metric instance. Defaults to None.
        dtype (dtype, optional): data type of the metric result. Defaults to None.
        class_to_return (int, optional): class for which Dice value is returned. Defaults to 0.
    """

    def __init__(self, num_classes, name=None, dtype=None, class_to_return=0):
        super(Dice, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64)
        self.class_to_return = class_to_return

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to 1. Can be a
                `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
                be broadcastable to `y_true`.
        Returns:

          Update op.
        """

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        """
    # Flatten the input if its rank > 1.
    if y_pred.shape.ndims > 1:
      y_pred = array_ops.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
      y_true = array_ops.reshape(y_true, [-1])
    """

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col
        numerator = true_positives + true_positives
        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(math_ops.cast(
            math_ops.not_equal(denominator, 0), dtype=self._dtype))

        dice = math_ops.div_no_nan(numerator, denominator)
        return dice[self.class_to_return]

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros(
            (self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanDice(Metric):
    """Computes the Dice metric average over classes.
    Dice is a common evaluation metric for semantic image
    segmentation, obtained by computing the Dice for each semantic class
    and then by averaging the values.
    Dice is defined as follows:

    .. math::

      Dice = \\frac{2*TP}{2*TP + FP + FN}

    Standalone usage: ::

      >>> # cm = [[1, 1],
      >>> #        [1, 1]]
      >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
      >>> # dice = 2*true_positives / (sum_row + sum_col - true_positives))
      >>> # result = (1 / (2 + 2 - 1) , 1 / (2 + 2 - 1)) = 0.33, 0.33
      >>> m = tf.keras.metrics.MeanIoU(num_classes=2)
      >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
      >>> m.result().numpy()
      0.33333334, 0.33333334
      >>> m.reset_states()
      >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
      ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
      >>> m.result().numpy()
      0.33333334, 0.14285715

    Usage with ``compile()`` API: ::

      model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[segmentation.metrics.MeanDice(num_classes=2)])

    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:

      num_classes (int, required): the possible number of labels the prediction task can have.
      name (str, optional): string name of the metric instance.
      dtype (dtype, optional): data type of the metric result.
    """

    def __init__(self, num_classes, name=None, dtype=None):
        super(MeanDice, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col

        numerator = true_positives + true_positives
        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(math_ops.cast(
            math_ops.not_equal(denominator, 0), dtype=self._dtype))

        dice = math_ops.div_no_nan(true_positives, denominator)

        return math_ops.div_no_nan(
            math_ops.reduce_sum(dice, name='mean_dice'), num_valid_entries)

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros(
            (self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanDice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IoUPerClass(Metric):
    """
    Compute metric IoU for parameter y_true and y_pred only for the
    specified class.
    Input y_true and y_pred is supposed to be 5-dimensional:
    (batch, x, y, z, softmax_probabilities)
    """

    def __init__(self, numClasses, name=None, dtype=None,
                 class_to_return=0, use_3D=True):
        super(IoUPerClass, self).__init__(name=name, dtype=dtype)
        self.numClasses = numClasses
        self.class_to_return = class_to_return
        self.use_3D = use_3D
        self.tp = 0
        self.fn = 0
        self.fp = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        class_IoU_list = []
        y_true = tf.cast(y_true, 'bool')
        # choose which class the model predicts for each voxel
        y_pred = tf.argmax(y_pred, axis=-1, output_type='int64')
        y_pred = tf.one_hot(
            indices=y_pred, depth=self.numClasses, axis=-1, dtype='int64')
        y_pred = tf.cast(y_pred, 'bool')

        if not self.use_3D:
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])

        # Now dimensions are --> [Classes, Batch, Rows, Columns, Slices]
        # or [Classes, Batch, Rows, Columns]

        y_true_c = y_true[self.class_to_return]
        y_pred_c = y_pred[self.class_to_return]
        self.tp = tf.math.count_nonzero(tf.logical_and(y_true_c, y_pred_c))
        self.fn = tf.math.count_nonzero(tf.logical_and(
            tf.math.logical_xor(y_true_c, y_pred_c), y_true_c))
        self.fp = tf.math.count_nonzero(tf.logical_and(
            tf.math.logical_xor(y_true_c, y_pred_c), y_pred_c))
        return self.tp, self.fn, self.fp

    def result(self):
        """
        This function is only used to assign a name to the given IoU metric.
        """
        return self.tp / (self.tp + self.fn + self.fp)


def count_tp(cl, trueLabel, predictedLabel):
    '''
    Return total number of true positives for the specified class, given
    the true and predicted labels.
    '''
    match = trueLabel[np.nonzero(predictedLabel == cl)]
    return(len(np.nonzero(match == cl)[0]))


def count_fp(cl, trueLabel, predictedLabel):
    """Compute total number of false positives for given class.

    This function returns the total number of false positives
    for the specified class, given the true and predicted labels.

    Args:
        cl (int): The class of interest.
        trueLabel (array): The true label.
        predictedLabel (array): The predicted label.

    Return:
        total number of false positives.
    """
    match = trueLabel[np.nonzero(predictedLabel == cl)]
    return(len(np.nonzero(match != cl)[0]))


def count_tn(cl, trueLabel, predictedLabel):
    '''Compute number of true negatives.

    Return total number of true negatives for the specified class, given
    the true and predicted labels.

    Args:
        cl (int, required): class to be considered in labels.

    '''
    match = trueLabel[np.nonzero(predictedLabel != cl)]
    return(len(np.nonzero(match != cl)[0]))


def count_fn(cl, trueLabel, predictedLabel):
    '''
    Return total number of false negatives for the specified class, given
    the true and predicted labels.
    '''
    match = trueLabel[np.nonzero(predictedLabel != cl)]
    return(len(np.nonzero(match == cl)[0]))


def compute_ppv_class(cl, trueLabel, predictedLabel):
    tp = count_tp(cl, trueLabel, predictedLabel)
    fp = count_fp(cl, trueLabel, predictedLabel)
    ppv = tp / (tp + fp)
    return ppv


def compute_dice_class(cl, trueLabel, predictedLabel):
    tp = count_tp(cl, trueLabel, predictedLabel)
    fp = count_fp(cl, trueLabel, predictedLabel)
    fn = count_fn(cl, trueLabel, predictedLabel)
    dice = 2 * tp / (2 * tp + fp + fn)
    return dice


def compute_dice(trueLabel, predictedLabel, return_average=True, classes=None):
    if (classes == None):
        classes = np.unique(predictedLabel)
    dice_values = np.zeros(len(classes))
    for cl_index, cl_value in enumerate(classes):
        try:
            dice_values[cl_index] = compute_dice_class(cl_value,
                                                       trueLabel,
                                                       predictedLabel)
        except ZeroDivisionError:
            dice_values[cl_index] = 0

    if (return_average):
        return dice_values, np.mean(dice_values)
    else:
        return dice_values


def compute_jaccard(trueLabel, predictedLabel, return_average=True, classes=None):
    if (classes == None):
        classes = np.unique(predictedLabel)
    jaccard_values = np.zeros(len(classes))
    for cl_index, cl_value in enumerate(classes):
        try:
            dice_temp = compute_dice_class(cl_value,
                                           trueLabel,
                                           predictedLabel)

            jaccard_values[cl_index] = dice_temp / (2 - dice_temp)
        except ZeroDivisionError:
            jaccard_values[cl_index] = 0
    if (return_average):
        return jaccard_values, np.mean(jaccard_values)
    else:
        return jaccard_values


def compute_sensitivity(trueLabel, predictedLabel, classes=None):
    if (classes == None):
        classes = np.unique(predictedLabel)
    sensitivity_values = np.zeros(len(classes))
    for cl_index, cl_value in enumerate(classes):
        tp = count_tp(cl_value, trueLabel, predictedLabel)
        fn = count_fn(cl_value, trueLabel, predictedLabel)
        try:
            sensitivity_values[cl_index] = tp / (tp + fn)
        except ZeroDivisionError:
            sensitivity_values[cl_index] = 0
    return sensitivity_values


def compute_precision(trueLabel, predictedLabel, classes=None):
    if (classes == None):
        classes = np.unique(predictedLabel)
    precision_values = np.zeros(len(classes))
    for cl_index, cl_value in enumerate(classes):
        tp = count_tp(cl_value, trueLabel, predictedLabel)
        fp = count_fp(cl_value, trueLabel, predictedLabel)
        try:
            precision_values[cl_index] = tp / (tp + fp)
        except ZeroDivisionError:
            precision_values[cl_index] = 0
    return precision_values


def compute_for(trueLabel, predictedLabel, classes=None):
    if (classes == None):
        classes = np.unique(predictedLabel)
    for_values = np.zeros(len(classes))
    for cl_index, cl_value in enumerate(classes):
        tn = count_tn(cl_value, trueLabel, predictedLabel)
        fn = count_fn(cl_value, trueLabel, predictedLabel)
        try:
            for_values[cl_index] = fn / (tn + fn)
        except ZeroDivisionError:
            for_values[cl_index] = 0
    return for_values
