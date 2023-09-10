# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import paddle
import pycocotools.coco as cocoUtils
import pycocotools.mask as maskUtils

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.utils import logger


@manager.DATASETS.add_component
class COCO(paddle.io.Dataset):

    def __init__(self,
                 transforms=None,
                 image_root=None,
                 json_file=None,
                 mode='train',
                 add_background=True,
                 edge=False):
        super().__init__()
        self.transforms = Compose(transforms)
        self.image_root = image_root
        self.mode = mode.lower()
        self.add_background = add_background
        self.edge = edge
        self.ignore_index = 255

        if mode not in ['train', 'trainval', 'val']:
            raise ValueError("`mode` should be one of ('train', 'trainval', 'val') "
                             "in PascalContext dataset, but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if not os.path.exists(image_root) or not os.path.exists(json_file):
            raise ValueError(
                "The dataset is not Found or the folder structure is non-conformance.")

        self.info_list = list()

        coco_api = cocoUtils.COCO(json_file)

        cat_id_map = {cat_id: i for i, cat_id in enumerate(coco_api.getCatIds())}
        self.num_classes = len(list(cat_id_map.keys()))

        assert 'annotations' in coco_api.dataset, \
            'Annotation file: {} does not contains ground truth!!!'.format(json_file)

        for img_id in sorted(coco_api.getImgIds()):
            img_info = coco_api.loadImgs([img_id])[0]
            img_name = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']
            img_path = os.path.join(image_root, img_name)
            if not os.path.exists(img_path):
                logger.warning('Illegal image file: {}, '
                               'and it will be ignored'.format(img_path))
                continue

            if img_w < 0 or img_h < 0:
                logger.warning(
                    'Illegal width: {} or height: {} in annotation, '
                    'and im_id: {} will be ignored'.format(img_w, img_h, img_id))
                continue

            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            anns = coco_api.loadAnns(ann_ids)
            if len(anns) == 0:
                logger.warning(
                    'No valid annotation, and im_id: {} '
                    'will be ignored'.format(img_w, img_h, img_id))
                continue

            label_info = []
            for ann in anns:
                label_info.append(
                    [cat_id_map[ann['category_id']], ann['segmentation']])

            self.info_list.append([img_info, label_info])

    def __getitem__(self, idx):
        data = {'trans_info': [], 'gt_fields': []}
        image_info, label_info = self.info_list[idx]
        data['img'] = os.path.join(self.image_root, image_info['file_name'])
        data['label'] = self._gen_label(image_info, label_info)
        if self.mode == 'val':
            data = self.transforms(data)
        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
        data['label'] = data['label'].transpose([2, 0, 1])
        return data

    def _gen_label(self, image_info, label_info):
        label = np.zeros([image_info['height'], image_info['width'],
                          self.num_classes], dtype='uint8')
        for sub_label_info in label_info:
            label[..., sub_label_info[0]] += maskUtils.decode(sub_label_info[1])

        if self.add_background:
            bg_label = (label.sum(axis=-1, keepdims=True) == 0).astype('uint8')
            label = np.concatenate([bg_label, label], axis=-1)

        return label

    def __len__(self):
        return len(self.info_list)
