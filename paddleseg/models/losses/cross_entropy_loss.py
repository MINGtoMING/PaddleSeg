# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        avg_non_ignore (bool, optional): Whether the loss is only averaged over non-ignored value of pixels. Default: True.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 avg_non_ignore=True,
                 data_format='NCHW'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.avg_non_ignore = avg_non_ignore
        self.EPS = 1e-8
        self.data_format = data_format
        if weight is not None:
            self.weight = paddle.to_tensor(weight, dtype='float32')
        else:
            self.weight = None

    def forward(self, logit, label, semantic_weights=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label. Default: None.
        Returns:
            (Tensor): The average loss.
        """
        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[channel_axis]))

        if channel_axis == 1:
            logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')

        loss = F.cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.weight)

        return self._post_process_loss(logit, label, semantic_weights, loss)

    def _post_process_loss(self, logit, label, semantic_weights, loss):
        """
        Consider mask and top_k to calculate the final loss.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label.
            loss (Tensor): Loss tensor which is the output of cross_entropy. If soft_label
                is False in cross_entropy, the shape of loss should be the same as the label.
                If soft_label is True in cross_entropy, the shape of loss should be
                (N, D1, D2,..., Dk, 1).
        Returns:
            (Tensor): The average loss.
        """
        if self.avg_non_ignore:
            mask = paddle.cast(label != self.ignore_index, dtype='float32')
        else:
            mask = paddle.ones(label.shape, dtype='float32')
        mask.stop_gradient = True
        label.stop_gradient = True

        if loss.ndim > mask.ndim:
            loss = paddle.squeeze(loss, axis=-1)
        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label * mask, logit.shape[-1])
            coef = paddle.sum(_one_hot * self.weight, axis=-1)
        else:
            coef = paddle.ones_like(label)

        if self.top_k_percent_pixels == 1.0:
            avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)
        else:
            loss = loss.reshape((-1,))
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, indices = paddle.topk(loss, top_k_pixels)
            coef = coef.reshape((-1,))
            coef = paddle.gather(coef, indices)
            coef.stop_gradient = True
            coef = coef.astype('float32')
            avg_loss = loss.mean() / (paddle.mean(coef) + self.EPS)

        return avg_loss


@manager.LOSSES.add_component
class DistillCrossEntropyLoss(CrossEntropyLoss):
    """
    The implementation of distill cross entropy loss.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
            Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'.
            Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW'):
        super().__init__(weight, ignore_index, top_k_percent_pixels,
                         data_format)

    def forward(self,
                student_logit,
                teacher_logit,
                label,
                semantic_weights=None):
        """
        Forward computation.

        Args:
            student_logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            teacher_logit (Tensor): Logit tensor, the data type is float32, float64. The shape
                is the same as the student_logit.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label. Default: None.
        """

        if student_logit.shape != teacher_logit.shape:
            raise ValueError(
                'The shape of student_logit = {} must be the same as the shape of teacher_logit = {}.'
                .format(student_logit.shape, teacher_logit.shape))

        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and student_logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), student_logit.shape[channel_axis]))

        if channel_axis == 1:
            student_logit = paddle.transpose(student_logit, [0, 2, 3, 1])
            teacher_logit = paddle.transpose(teacher_logit, [0, 2, 3, 1])

        teacher_logit = F.softmax(teacher_logit)

        loss = F.cross_entropy(
            student_logit,
            teacher_logit,
            weight=self.weight,
            reduction='none',
            soft_label=True)

        return self._post_process_loss(student_logit, label, semantic_weights,
                                       loss)


@manager.LOSSES.add_component
class MultiLabelCategoricalCrossEntropyLoss(nn.Layer):
    """
    This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a multi-label classification problems.
    This is particularly useful when you have an unbalanced training set.
    The `input` is expected to contain raw, un-normalized scores for each class.

    `input` has to be a Tensor of size either :math:`(batch_size, C)` or
    :math:`(batch_size, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self, ignore_index=255, data_format='NCHW'):
        super(MultiLabelCategoricalCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-8
        self.data_format = data_format

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is (N, C, H, W).
            label (Tensor): Label tensor, the data type is int64. Shape is (N, C, H, W),
            where each value is {0, 1, ig_index}
        Returns:
            (Tensor): The average loss.
        """

        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if channel_axis == 1:
            logit = paddle.transpose(logit, [0, 2, 3, 1])

        assert label.shape == logit.shape

        mask = (label != self.ignore_index)
        mask = paddle.all(mask, axis=-1, keepdim=True) * mask
        label = paddle.where(mask, label, paddle.zeros_like(label))
        label = paddle.cast(label, 'bool')

        logit = paddle.where(label, -logit, logit)
        logit_pos = paddle.where(paddle.logical_and(label, mask),
                                 logit, paddle.to_tensor(float("-inf")))
        logit_neg = paddle.where(paddle.logical_or(label, ~mask),
                                 paddle.to_tensor(float("-inf")), logit)
        logit_zero = paddle.zeros_like(logit_neg[..., :1])
        logit_pos = paddle.concat([logit_zero, logit_pos], axis=-1)
        logit_neg = paddle.concat([logit_zero, logit_neg], axis=-1)
        neg_loss = paddle.logsumexp(logit_neg, axis=-1)
        pos_loss = paddle.logsumexp(logit_pos, axis=-1)
        loss = neg_loss + pos_loss
        mask = mask.all(-1).astype("float32")
        loss = loss * mask
        avg_loss = loss.sum() / (mask.sum() + self.EPS)
        mask.stop_gradient = True
        label.stop_gradient = True
        return avg_loss
