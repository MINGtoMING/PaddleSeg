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
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F


@manager.DATASETS.add_component
class COCODataset(Dataset):
    """
    Loading a (custom) dataset organized in COCO format.

    Args:
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        image_root (str): The image directory.
        json_file (str): The file path of annotation json.
        transforms (list): Transforms for image.
        num_classes (int): Number of classes.
        img_channels (int, optional): Number of image channels. Default: 3.
        ignore_index (int, optional): Specifies a target category that is ignored. Default: -1.
        edge (bool, optional): Whether to compute edge while training. Default: False
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.
    """

    def __init__(self,
                 mode,
                 image_root,
                 json_file,
                 transforms,
                 num_classes,
                 img_channels=3,
                 ignore_index=-1,
                 edge=False,
                 use_multilabel=False):
        super().__init__()
        self.image_root = image_root
        self.json_file = json_file
        self.transforms = Compose(transforms, img_channels=img_channels)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.ignore_index = ignore_index
        self.edge = edge
        self.use_multilabel = use_multilabel

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))
        if not os.path.exists(self.image_root):
            raise FileNotFoundError('there is not `image_root`: {}.'.format(
                self.image_root))
        if not os.path.exists(self.json_file):
            raise FileNotFoundError('there is not `json_file`: {}.'.format(
                self.json_file))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if self.num_classes < 1:
            raise ValueError(
                "`num_classes` should be greater than 1, but got {}".format(
                    self.num_classes))
        if self.img_channels not in [1, 3]:
            raise ValueError("`img_channels` should in [1, 3], but got {}".
                             format(self.img_channels))

        coco = cocoAPI.COCO(json_file)
        cat_ids = sorted(coco.getCatIds())
        if 0 <= self.ignore_index < len(cat_ids):
            cat_ids.pop(self.ignore_index)
        self.cat_id_map = {old_cat_id: new_cat_id
                           for new_cat_id, old_cat_id in enumerate(cat_ids)}

        if self.num_classes != len(self.cat_id_map):
            raise ValueError(
                "`num_classes` should be equal to {}, but got {}".format(
                    len(self.cat_id_map), self.num_classes))
        if 'annotations' not in coco.dataset:
            raise ValueError(
                "`json_file`: {} doesn't contains ground truth!!!".format(
                    self.json_file))

        for img_id in sorted(coco.getImgIds()):
            img_info = coco.loadImgs([img_id])[0]
            img_name = img_info['file_name']
            img_path = os.path.join(self.image_root, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    "The image path({}) is incorrect. "
                    "Please carefully check!!!".format(img_path))
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            self.file_list.append([img_info, anns])

    def __getitem__(self, idx):
        data = {'trans_info': [], 'gt_fields': []}
        img_info, anns = self.file_list[idx]
        data['img'] = os.path.join(self.image_root, img_info['file_name'])
        label = np.zeros(
            [img_info['height'], img_info['width'], self.num_classes], dtype='uint8')
        for ann in anns:
            cat_id = self.cat_id_map.get(ann['category_id'], None)
            if cat_id is None:
                continue
            mask = maskUtils.decode(ann['segmentation'])
            label[:, :, cat_id] = np.where(mask, mask, label[:, :, cat_id])
        if not self.use_multilabel:
            data['label'] = np.argmax(label, axis=-1, keepdims=True).astype('uint8')
        else:
            data['label'] = label
        if self.mode == 'val':
            data = self.transforms(data)
        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
            if self.edge and not self.use_multilabel:
                edge_mask = F.mask_to_binary_edge(
                    data['label'].squeeze(-1), radius=2, num_classes=self.num_classes)
                data['edge'] = edge_mask
        data['label'] = data['label'].transpose([2, 0, 1])
        return data

    def __len__(self):
        return len(self.file_list)
