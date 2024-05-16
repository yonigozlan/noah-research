
import os
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
import tqdm
from bedlam import constants
from bedlam.config import update_hparams
# from bedlam.models.head.smplx_cam_head import SMPLXCamHead
from bedlam.models.hmr import HMR
from loguru import logger
from torchvision.transforms import Normalize


def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        logger.warning(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        logger.warning(f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                                       f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}')

                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [state_dict[pk], state_dict[pk][:,-7:]], dim=-1
                            )
                            logger.warning(f'Updated \"{pk}\" param to {updated_pretrained_state_dict[pk].shape} ')
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
    return model



class Tester:
    def __init__(self, cfg, ckpt):
        self.model_cfg = update_hparams(cfg)
        self.ckpt = ckpt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.bboxes_dict = {}

        self.model = self._build_model()
        # self.smplx_cam_head = SMPLXCamHead(img_res=self.model_cfg.DATASET.IMG_RES).to(self.device)
        self._load_pretrained_model()
        self.model.eval()

    def _build_model(self):
        self.hparams = self.model_cfg
        model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        ).to(self.device)
        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {self.ckpt}')
        ckpt = torch.load(self.ckpt)['state_dict']
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        logger.info(f'Loaded pretrained weights from \"{self.ckpt}\"')


    # @torch.no_grad()
    # def run_on_image_folder(self, all_image_folder, detections, output_folder, visualize_proj=True):
    #     for fold_idx, image_folder in enumerate(all_image_folder):
    #         image_file_names = [
    #             os.path.join(image_folder, x)
    #             for x in os.listdir(image_folder)
    #             if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
    #         ]
    #         image_file_names = (sorted(image_file_names))
    #         for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):

    #             dets = detections[fold_idx][img_idx]
    #             if len(dets) < 1:
    #                 continue

    #             img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
    #             orig_height, orig_width = img.shape[:2]
    #             inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
    #                                      self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

    #             batch_size = inp_images.shape[0]
    #             bbox_scale = []
    #             bbox_center = []

    #             for det_idx, det in enumerate(dets):
    #                 bbox = det
    #                 bbox_scale.append(bbox[2] / 200.)
    #                 bbox_center.append([bbox[0], bbox[1]])
    #                 rgb_img = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
    #                 rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
    #                 rgb_img = torch.from_numpy(rgb_img)
    #                 norm_img = self.normalize_img(rgb_img)
    #                 inp_images[det_idx] = norm_img.float().to(self.device)

    #             bbox_center = torch.tensor(bbox_center).cuda().float()
    #             bbox_scale = torch.tensor(bbox_scale).cuda().float()
    #             img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
    #             img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
    #             focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
    #             hmr_output = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)

    #             focal_length = (img_w * img_w + img_h * img_h) ** 0.5
    #             pred_vertices_array = (hmr_output['vertices'] + hmr_output['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
    #             # renderer = Renderer(focal_length=focal_length[0], img_w=img_w[0], img_h=img_h[0],
    #             #                     faces=self.smplx_cam_head.smplx.faces,
    #             #                     same_mesh_color=False)
    #             # front_view = renderer.render_front_view(pred_vertices_array,
    #             #                                         bg_img_rgb=img.copy())

    #             # # save rendering results
    #             # basename = img_fname.split('/')[-1]
    #             # filename = basename + "pred_%s.jpg" % 'bedlam'
    #             # filename_orig = basename + "orig_%s.jpg" % 'bedlam'
    #             # front_view_path = os.path.join(output_folder, filename)
    #             # orig_path = os.path.join(output_folder, filename_orig)
    #             # logger.info(f'Writing output files to {output_folder}')
    #             # cv2.imwrite(front_view_path, front_view[:, :, ::-1])
    #             # cv2.imwrite(orig_path, img[:, :, ::-1])
    #             # renderer.delete()
