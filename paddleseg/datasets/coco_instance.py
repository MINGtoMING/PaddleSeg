# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.utils import logger
from pycocotools.coco import COCO


@manager.DATASETS.add_component
class COCOInstance(paddle.io.Dataset):
    """
    Load segmentation dataset with COCO instance format.
    1. Support single-label segmentation and multi-label segmentation
    2. Support polygon and RLE

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        transforms (list): Transforms for image.
    """
    NUM_CLASSES = 80
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self, dataset_dir, image_dir, anno_path, transforms):
        super().__init__()
        self.load_image_only = False

        image_dir = os.path.join(dataset_dir, image_dir)
        anno_path = os.path.join(dataset_dir, anno_path)
        assert os.path.exists(image_dir) and os.path.exists(anno_path), \
            ValueError("The dataset is not Found or "
                       "the folder structure is non-conformance.")

        self.transforms = Compose(transforms)
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.file_list = list()

        coco = COCO(anno_path)
        cat_ids = coco.getCatIds()
        self.NUM_CLASSES = len(cat_ids)
        catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in list(sorted(coco.getImgIds())):
            img_info = coco.loadImgs([img_id])[0]
            img_filename = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']

            img_filepath = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_filepath):
                logger.warning('Illegal image file: {}, '
                               'and it will be ignored'.format(img_filepath))
                continue

            if img_w < 0 or img_h < 0:
                logger.warning(
                    'Illegal width: {} or height: {} in annotation, '
                    'and im_id: {} will be ignored'.format(img_w, img_h, img_id))
                continue

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(imgIds=[img_id])
                instances = coco.loadAnns(ins_anno_ids)
                clean_instances = []
                for instance in instances:
                    if instance['area'] <= 0:
                        continue
                    clean_instance = {
                            'gt_class': catid2clsid[instance['category_id']],
                            'gt_poly': instance['segmentation'],
                    }
                    clean_instances.append(clean_instance)
                self.file_list.append([img_filepath, clean_instances])
            else:
                self.file_list.append(img_filepath)

    def __getitem__(self, idx):
        data = {
            'trans_info': [],
            'num_classes': self.NUM_CLASSES,
        }
        if not self.load_image_only:
            image_path, instances = self.file_list[idx]
            data['instances'] = instances
        else:
            image_path = self.file_list[idx]

        data['img'] = image_path

        data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.file_list)
