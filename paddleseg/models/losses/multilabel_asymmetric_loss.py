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
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class MultiLabelAsymmetricLoss(nn.Layer):
    """
    Multi-label asymmetric loss.

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        disable_focal_loss_grad (bool): freeze grad of asymmetric_weight.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self,
                 gamma_pos=1,
                 gamma_neg=4,
                 clip=0.05,
                 disable_focal_loss_grad=True,
                 ignore_index=255):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.disable_focal_loss_grad = disable_focal_loss_grad
        self.ignore_index = ignore_index
        self.EPS = 1e-10

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

        assert len(label.shape) == len(logit.shape)
        logit = logit.transpose([0, 2, 3, 1])
        mask = (label != self.ignore_index)
        mask = paddle.all(mask, axis=-1, keepdim=True) * mask
        label = paddle.where(mask, label, paddle.zeros_like(label))
        mask = paddle.cast(mask, 'float32')
        label = label.astype('float32')

        logit = F.sigmoid(logit)
        # Asymmetric Clipping and Basic CE calculation
        if self.clip and self.clip > 0:
            pt = (1 - logit + self.clip).clip(max=1) \
                 * (1 - label) + logit * label
        else:
            pt = (1 - logit) * (1 - label) + logit * label

        # Asymmetric Focusing
        if self.disable_focal_loss_grad:
            paddle.set_grad_enabled(False)
        asymmetric_weight = (1 - pt).pow(
            self.gamma_pos * label + self.gamma_neg * (1 - label))
        if self.disable_focal_loss_grad:
            paddle.set_grad_enabled(True)

        loss = -paddle.log(pt.clip(min=self.EPS)) * asymmetric_weight
        loss = loss * mask
        loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)
        label.stop_gradient = True
        mask.stop_gradient = True

        return loss
