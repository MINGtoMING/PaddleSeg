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
import pycocotools.coco as cocoAPI
import pycocotools.mask as maskUtils
from paddle.io import Dataset
from scipy.ndimage import distance_transform_edt

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.utils import logger


@manager.DATASETS.add_component
class COCODataset(Dataset):
    """
    The API for loading coco format dataset.

    Args:
        transforms (list): Transforms for image.
        image_root (str): The image directory.
        json_file (str): The file path of annotation json.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        add_background (bool, optional): Whether add additional background category. Default: False
        ignore_index (int, optional): Specifies a target value that is ignored. Default: 255.
        edge (bool, optional): Whether to compute edge while training. Default: False.
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.
     """

    def __init__(self,
                 transforms,
                 image_root,
                 json_file,
                 mode='train',
                 add_background=True,
                 ignore_index=255,
                 edge=False,
                 use_multilabel=False):
        super().__init__()
        self.transforms = Compose(transforms)
        self.image_root = image_root
        self.mode = mode.lower()
        self.add_background = add_background
        self.ignore_index = ignore_index
        self.edge = edge
        self.use_multilabel = use_multilabel

        if mode not in ['train', 'trainval', 'val']:
            raise ValueError("`mode` should be one of ('train', 'trainval', 'val') "
                             "in PascalContext dataset, but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if not os.path.exists(image_root) or not os.path.exists(json_file):
            raise ValueError(
                "The dataset is not Found or the folder structure is non-conformance.")

        self.file_list = list()

        coco = cocoAPI.COCO(json_file)

        cat_id_map = {cat_id: i for i, cat_id in enumerate(coco.getCatIds())}
        self.num_classes = len(list(cat_id_map.keys()))
        if self.add_background:
            self.num_classes += 1

        assert 'annotations' in coco.dataset, \
            'Annotation file: {} does not contains ground truth!!!'.format(json_file)

        for img_id in sorted(coco.getImgIds()):
            img_info = coco.loadImgs([img_id])[0]
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

            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            if len(anns) == 0:
                logger.warning(
                    'No valid annotation, and im_id: {} '
                    'will be ignored'.format(img_w, img_h, img_id))
                continue

            label_info = []
            for ann in anns:
                label_info.append(
                    [cat_id_map[ann['category_id']], ann['segmentation']])

            self.file_list.append([img_info, label_info])

    def __getitem__(self, idx):
        data = {'trans_info': [], 'gt_fields': []}
        image_info, label_info = self.file_list[idx]
        data['img'] = os.path.join(self.image_root, image_info['file_name'])
        data['label'] = self.parse_label(image_info, label_info)
        if self.mode == 'val':
            data = self.transforms(data)
        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
            if self.edge:
                data['edge'] = self.gen_edge(data['label'])
        data['label'] = data['label'].transpose([2, 0, 1])
        return data

    def parse_label(self, image_info, label_info):
        if not self.use_multilabel:
            label = np.zeros([image_info['height'], image_info['width']], dtype='uint8')
            for sub_label_info in label_info:
                cat_id = sub_label_info[0]
                mask = maskUtils.decode(sub_label_info[1])
                label = np.where(mask, cat_id, label)
            if not self.add_background:
                label = np.where(label == 0, self.ignore_index, label - 1)
            return label[..., None]
        else:
            label = np.zeros([image_info['height'], image_info['width'],
                              self.num_classes], dtype='uint8')
            for sub_label_info in label_info:
                cat_id = sub_label_info[0]
                if self.add_background:
                    cat_id += 1
                mask = maskUtils.decode(sub_label_info[1])
                label[..., cat_id] = np.where(mask, mask, label[..., cat_id])
            if self.add_background:
                label[..., 0] = (label[..., 1:].sum(axis=-1) == 0).astype('uint8')
            return label

    def gen_edge(self, label, radius=2):
        if not self.use_multilabel:
            mask = [label[0] == i for i in range(self.num_classes)]
            mask = np.array(mask, dtype='uint8')
        else:
            mask = np.array(label, dtype='uint8')

        if radius < 1:
            raise ValueError('`radius` should be greater than or equal to 1')

        padded_mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)),
                             mode='constant', constant_values=0)

        edge = np.zeros_like(mask)
        for i in range(self.num_classes):
            dist = distance_transform_edt(padded_mask[i, :]) + \
                   distance_transform_edt(1.0 - padded_mask[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edge[i, :] = dist

        if not self.use_multilabel:
            edge = np.sum(edge, axis=0, keepdims=True)

        edge = (edge > 0).astype('uint8')
        return edge

    def __len__(self):
        return len(self.file_list)
