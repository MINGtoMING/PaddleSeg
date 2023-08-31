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

import numpy as np
from paddleseg.cvlibs import manager
from paddleseg.datasets import Dataset
from paddleseg.transforms import Compose
from paddleseg.utils import seg_env, logger
from paddleseg.utils.download import download_file_and_uncompress
from pycocotools.coco import COCO
from pycocotools.mask import decode as rle_decode
from pycocotools.mask import frPyObjects as rle_frPyObjects
from pycocotools.mask import merge as rle_merge
from scipy.ndimage import distance_transform_edt

URL = "COCO2017 zip url to be provided."


@manager.DATASETS.add_component
class COCOInstance(Dataset):
    """
    COCO dataset `https://cocodataset.org/`.
    `COCOInstance` api used for dataset adopts the annotation format of COCO instance segmentation style.
    Examples of annotation file content are as follows:
    json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [
        {
            'segmentation': polygons or RLE,
            'category_id': category_id,
        }],
        "categories": []
    }

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The COCOInstance dataset directory. Default: None.
        image_dir (str, optional): The image directory. Default: None.
        anno_path (str, optional): The annotation file path. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        allow_overlap (bool, optional): Whether allow overlapping masks between different categories.
            1. allow_overlap=False, each pixel on the image can only correspond to one category at the same time.
                Using at single-label segmentation tasks. data['label']: (1, img_h, img_w) 0 ~ 255
            2. allow_overlap=True, each pixel on the image can correspond to multiple categories at the same time.
                Using at multi-label segmentation tasks. data['label']: (num_class, img_h, img_w) {0, 1}
        add_background (bool, optional): Whether to add add_background category. Default: False
            1. the background category is already included in the annotation file, just keep default.
            2. the background category is not included in the annotation file,
                2.1 add_background=False, background will be replaced with ignore index
                    and will not participate in the supervised calculation process.
                2.2 add_background=True, background will be added as a new category
                    and will participate in the supervised calculation process.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 80
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self, transforms, dataset_root=None, image_dir=None, anno_path=None,
                 mode='train', allow_overlap=False, add_background=False, edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.IGNORE_INDEX
        self.allow_overlap = allow_overlap
        self.add_background = add_background
        self.edge = edge

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'val') in "
                "COCOInstance dataset, but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME,
                extraname='COCO2017')
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)

        self.image_dir = os.path.join(self.dataset_root, image_dir)
        self.anno_path = os.path.join(self.dataset_root, anno_path)
        assert os.path.exists(self.image_dir) and os.path.exists(self.anno_path), \
            ValueError("The dataset is not Found or "
                       "the folder structure is non-conformance.")

        self._parse_anno()

    def __getitem__(self, idx):
        data = {'trans_info': [], 'gt_fields': []}
        if self.mode == 'val':
            if not self.load_image_only:
                image_path, gt_polygon, [img_h, img_w] = self.file_list[idx]
                data['img'] = image_path
                data = self.transforms(data)
                data['label'] = self._gen_label(gt_polygon, img_h, img_w)
            else:
                image_path = self.file_list[idx]
                data['img'] = image_path
                data = self.transforms(data)
        else:
            image_path, gt_polygon, [img_h, img_w] = self.file_list[idx]
            data['img'] = image_path
            data['label'] = self._gen_label(gt_polygon, img_h, img_w)
            data['gt_fields'].append('label')
            data = self.transforms(data)

            if self.edge:
                edge_mask = self._mask2edge(data['label'], radius=2)
                data['edge'] = edge_mask

        if not self.allow_overlap:
            data['label'] = data['label'][None, ...]

        return data

    def _parse_anno(self):
        coco = COCO(self.anno_path)
        cat_ids = coco.getCatIds()
        if self.NUM_CLASSES != len(cat_ids):
            self.NUM_CLASSES = len(cat_ids)
            self.num_classes = self.NUM_CLASSES
        catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(self.anno_path))
        else:
            self.load_image_only = False

        for img_id in list(sorted(coco.getImgIds())):
            img_info = coco.loadImgs([img_id])[0]
            img_filename = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']

            img_filepath = os.path.join(self.image_dir, img_filename)
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
                gt_polygon = [None] * self.NUM_CLASSES
                for instance in instances:
                    if instance['area'] <= 0:
                        continue
                    cls_id = catid2clsid[instance['category_id']]
                    gt_polygon[cls_id] = instance['segmentation']

                if all([poly_by_cls is None for poly_by_cls in gt_polygon]):
                    logger.warning(
                        f'Invalid mask annotation, and im_id: {img_id} will be ignored')
                    continue
                self.file_list.append([img_filepath, gt_polygon, [img_h, img_w]])
            else:
                self.file_list.append(img_filepath)

    @staticmethod
    def _poly2mask(poly, img_h, img_w):
        if isinstance(poly, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rle = rle_merge(rle_frPyObjects(poly, img_h, img_w))
        elif isinstance(poly['counts'], list):
            # uncompressed RLE
            rle = rle_frPyObjects(poly, img_h, img_w)
        else:
            # rle: str
            rle = poly
        mask = rle_decode(rle)
        return mask

    def _gen_label(self, gt_polygon, img_h, img_w):
        # single-label
        if not self.allow_overlap:
            label = np.ones([img_h, img_w], dtype=np.uint8) * self.ignore_index
            for cls_idx, poly in enumerate(gt_polygon):
                if poly is not None:
                    mask = self._poly2mask(poly, img_h, img_w)
                    label = np.where(mask, cls_idx, label)

            if self.add_background:
                label[label == self.ignore_index] = self.num_classes + 1
        # multi-label
        else:
            label = np.zeros([img_h, img_w, self.num_classes], dtype=np.uint8)
            for cls_idx, poly in enumerate(gt_polygon):
                if poly is not None:
                    label[..., cls_idx] = self._poly2mask(poly, img_h, img_w)

            if self.add_background:
                bg_label = (label.sum(axis=-1, keepdims=True) == 0).astype(np.uint8)
                label = np.concatenate([label, bg_label], axis=-1)
            else:
                label[label.sum(-1) == 0] = self.ignore_index

        return label

    def _mask2edge(self, mask, radius=2):
        if not self.allow_overlap:
            if not self.add_background:
                mask = [mask == i for i in range(self.num_classes)]
            else:
                mask = [mask == i for i in range(self.num_classes + 1)]
            mask = np.array(mask).astype(np.uint8)

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

        if not self.allow_overlap:
            edge = np.sum(edge, axis=0, keepdims=True)

        edge = (edge > 0).astype(np.uint8)

        return edge
