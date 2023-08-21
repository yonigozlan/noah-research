import json
import os.path as osp

import cv2
from common.imutils import process_image
from common.utils import estimate_focal_length
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    def __init__(self, root_dir, annotation_path):
        self.root_dir = root_dir
        self.annotations = json.load(open(osp.join(root_dir, annotation_path), "r"))

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, idx):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y, det_conf, nms_conf, category_id]
        :param idx:
        :return:
        """

        item = {}
        img_bgr = cv2.imread(
            osp.join(self.root_dir, self.annotations["images"][idx]["img_path"])
        )
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        bbox = self.annotations["annotations"][idx]["bbox"]
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)

        item["norm_img"] = norm_img
        item["center"] = center
        item["scale"] = scale
        item["crop_ul"] = crop_ul
        item["crop_br"] = crop_br
        item["img_h"] = img_h
        item["img_w"] = img_w
        item["focal_length"] = focal_length
        item["img_path"] = self.annotations["images"][idx]["img_path"]
        item["id"] = self.annotations["annotations"][idx]["id"]

        item["keypoints"] = self.annotations["annotations"][idx]["keypoints"]
        item["coco_keypoints"] = self.annotations["annotations"][idx]["coco_keypoints"]

        return item
