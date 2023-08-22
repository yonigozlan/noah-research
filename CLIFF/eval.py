# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import argparse
import glob
import os
import os.path as osp

import cv2
import numpy as np
import smplx
import torch
import torchgeometry as tgm
from common import constants
from common.pose_dataset import PoseDataset
from common.utils import cam_crop2full, estimate_focal_length, strip_prefix_if_present
from constants import AUGMENTED_VERTICES_INDEX_DICT
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from omegaconf import OmegaConf
from smplx import build_layer
from torch.utils.data import DataLoader
from tqdm import tqdm
from xtcocotools.coco import COCO

from mmpose.evaluation.metrics.infinity_metric import InfinityAnatomicalMetric
from smplx_local.transfer_model.config.defaults import conf as default_conf
from smplx_local.transfer_model.losses import build_loss
from smplx_local.transfer_model.optimizers import build_optimizer, minimize
from smplx_local.transfer_model.transfer_model import (
    build_edge_closure,
    build_vertex_closure,
    get_variables,
    summary_closure,
)
from smplx_local.transfer_model.utils import (
    batch_rodrigues,
    get_vertices_per_edge,
    read_deformation_transfer,
)
from smplx_local.transfer_model.utils.def_transfer import apply_deformation_transfer

CKPT_PATH = "data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
BACKBONE = "hr48"
BATCH_SIZE = 64


def run_fitting(
    exp_cfg,
    vertices,
    faces,
    body_model,
    def_matrix,
    mask_ids=None,
):
    """Runs fitting"""

    batch_size = len(vertices)
    dtype, device = vertices.dtype, vertices.device
    summary_steps = exp_cfg.get("summary_steps")
    interactive = exp_cfg.get("interactive")

    # Get the parameters from the model
    var_dict = get_variables(batch_size, body_model)

    # Build the optimizer object for the current batch
    optim_cfg = exp_cfg.get("optim", {})

    def_vertices = apply_deformation_transfer(def_matrix, vertices, faces)

    if mask_ids is None:
        f_sel = np.ones_like(body_model.faces[:, 0], dtype=np.bool_)
    else:
        f_per_v = [[] for _ in range(body_model.get_num_verts())]
        [
            f_per_v[vv].append(iff)
            for iff, ff in enumerate(body_model.faces)
            for vv in ff
        ]
        f_sel = list(set(tuple(sum([f_per_v[vv] for vv in mask_ids], []))))
    vpe = get_vertices_per_edge(
        body_model.v_template.detach().cpu().numpy(), body_model.faces[f_sel]
    )

    def log_closure():
        return summary_closure(def_vertices, var_dict, body_model, mask_ids=mask_ids)

    edge_fitting_cfg = exp_cfg.get("edge_fitting", {})
    edge_loss = build_loss(
        type="vertex-edge", gt_edges=vpe, est_edges=vpe, **edge_fitting_cfg
    )
    edge_loss = edge_loss.to(device=device)

    vertex_fitting_cfg = exp_cfg.get("vertex_fitting", {})
    vertex_loss = build_loss(**vertex_fitting_cfg)
    vertex_loss = vertex_loss.to(device=device)

    per_part = edge_fitting_cfg.get("per_part", True)
    # Optimize edge-based loss to initialize pose
    if per_part:
        for key, var in tqdm(var_dict.items(), desc="Parts"):
            if "pose" not in key:
                continue

            for jidx in tqdm(range(var.shape[1]), desc="Joints"):
                part = torch.zeros(
                    [batch_size, 3], dtype=dtype, device=device, requires_grad=True
                )
                # Build the optimizer for the current part
                optimizer_dict = build_optimizer([part], optim_cfg)
                closure = build_edge_closure(
                    body_model,
                    var_dict,
                    edge_loss,
                    optimizer_dict,
                    def_vertices,
                    per_part=per_part,
                    part_key=key,
                    jidx=jidx,
                    part=part,
                )

                minimize(
                    optimizer_dict["optimizer"],
                    closure,
                    params=[part],
                    summary_closure=log_closure,
                    summary_steps=summary_steps,
                    interactive=interactive,
                    **optim_cfg,
                )
                with torch.no_grad():
                    var[:, jidx] = part
    else:
        optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
        closure = build_edge_closure(
            body_model,
            var_dict,
            edge_loss,
            optimizer_dict,
            def_vertices,
            per_part=per_part,
        )

        minimize(
            optimizer_dict["optimizer"],
            closure,
            params=var_dict.values(),
            summary_closure=log_closure,
            summary_steps=summary_steps,
            interactive=interactive,
            **optim_cfg,
        )

    if "translation" in var_dict:
        optimizer_dict = build_optimizer([var_dict["translation"]], optim_cfg)
        closure = build_vertex_closure(
            body_model,
            var_dict,
            optimizer_dict,
            def_vertices,
            vertex_loss=vertex_loss,
            mask_ids=mask_ids,
            per_part=False,
            params_to_opt=[var_dict["translation"]],
        )
        # Optimize translation
        minimize(
            optimizer_dict["optimizer"],
            closure,
            params=[var_dict["translation"]],
            summary_closure=log_closure,
            summary_steps=summary_steps,
            interactive=interactive,
            **optim_cfg,
        )

    #  Optimize all model parameters with vertex-based loss
    optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
    closure = build_vertex_closure(
        body_model,
        var_dict,
        optimizer_dict,
        def_vertices,
        vertex_loss=vertex_loss,
        per_part=False,
        mask_ids=mask_ids,
    )
    minimize(
        optimizer_dict["optimizer"],
        closure,
        params=list(var_dict.values()),
        summary_closure=log_closure,
        summary_steps=summary_steps,
        interactive=interactive,
        **optim_cfg,
    )

    param_dict = {}
    for key, var in var_dict.items():
        # Decode the axis-angles
        if "pose" in key or "orient" in key:
            param_dict[key] = batch_rodrigues(var.reshape(-1, 3)).reshape(
                len(var), -1, 3, 3
            )
        else:
            # Simply pass the variable
            param_dict[key] = var

    body_model_output = body_model(return_full_pose=True, get_skin=True, **param_dict)
    var_dict.update(body_model_output)
    var_dict["faces"] = body_model.faces

    return var_dict


def get_smplx_tools(device):
    exp_cfg = default_conf.copy()
    exp_cfg.merge_with(OmegaConf.load("smplx_local/config_files/smpl2smplx.yaml"))
    deformation_transfer_path = osp.join(
        "smplx_local", exp_cfg.get("deformation_transfer_path", "")
    )
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)
    # body_model = build_layer("../../models/", **exp_cfg.body_model)
    body_model = build_layer("/scratch/users/yonigoz/", **exp_cfg.body_model)
    body_model.to(device=device)

    return exp_cfg, def_matrix, body_model


def smpl_to_smplx(vertices, faces, exp_cfg, body_model, def_matrix, device):
    vertices.to(device=device)
    faces.to(device=device)
    return run_fitting(
        exp_cfg,
        vertices,
        faces,
        body_model,
        def_matrix,
        mask_ids=None,
    )


def eval_dataset(root_dir, annotation_path):
    infinity_metric = InfinityAnatomicalMetric(
        osp.join(root_dir, annotation_path), use_area=False
    )
    coco = COCO(osp.join(root_dir, annotation_path))
    # ann_ids = coco.getAnnIds(imgIds=img_id)
    infinity_metric.dataset_meta = {"CLASSES": coco.loadCats(coco.getCatIds())}
    infinity_metric.dataset_meta["num_keypoints"] = 36
    infinity_metric.dataset_meta["sigmas"] = np.array(
        [
            0.026,
            0.025,
            0.025,
            0.035,
            0.035,
            0.079,
            0.079,
            0.072,
            0.072,
            0.062,
            0.062,
            0.107,
            0.107,
            0.087,
            0.087,
            0.089,
            0.089,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ]
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    exp_cfg, def_matrix, body_model = get_smplx_tools(device)

    print("--------------------------- 3D HPS estimation ---------------------------")
    # Create the model instance
    cliff = eval("cliff_" + BACKBONE)
    cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
    # Load the pretrained model
    print("Load the CLIFF checkpoint from path:", CKPT_PATH)
    state_dict = torch.load(CKPT_PATH)["model"]
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Setup the SMPL model
    smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)
    # Setup the SMPL-X model

    pose_dataset = PoseDataset(root_dir, annotation_path)
    pose_data_loader = DataLoader(pose_dataset, batch_size=BATCH_SIZE, num_workers=0)
    j = 0
    for batch in tqdm(pose_data_loader):
        if j >= 3:
            break
        j += 1
        norm_img = batch["norm_img"].to(device).float()
        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()
        ann_ids = coco.getAnnIds(imgIds=batch["id"].numpy())
        anns = coco.loadAnns(ann_ids)
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = (
            bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        )  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (
            0.06 * focal_length
        )  # [-1, 1]

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_cam_full = cam_crop2full(
            pred_cam_crop, center, scale, full_img_shape, focal_length
        )
        pred_output = smpl_model(
            betas=pred_betas,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            pose2rot=False,
            transl=pred_cam_full,
        )
        pred_vertices = pred_output.vertices

        var_dict = smpl_to_smplx(
            pred_vertices,
            torch.tensor(
                smpl_model.faces.astype(np.int32), dtype=torch.long, device=device
            ),
            exp_cfg,
            body_model,
            def_matrix,
            device,
        )
        data_samples = []
        for i in range(len(batch["img_path"])):
            data_sample = {}
            data_sample["ori_shape"] = (
                torch.stack((img_h, img_w), dim=-1)[0].cpu().numpy()
            )
            data_sample["id"] = int(batch["id"][i].cpu().numpy())
            data_sample["img_id"] = int(batch["id"][i].cpu().numpy())
            data_sample["raw_ann_info"] = anns[i]
            data_sample["gt_instances"] = {}

            img_ori_path = osp.join(root_dir, batch["img_path"][i])
            img_ori = cv2.imread(img_ori_path)
            focal_length = estimate_focal_length(img_h[i], img_w[i])
            intrinsic_matrix = torch.tensor(
                [
                    [focal_length, 0, img_w[i] / 2],
                    [0, focal_length, img_h[i] / 2],
                    [0, 0, 1],
                ]
            )
            anatomical_vertices = var_dict["vertices"][
                i, list(AUGMENTED_VERTICES_INDEX_DICT.values())
            ]
            # anatomical_vertices = var_dict["vertices"][i]
            projected_vertices = np.matmul(
                intrinsic_matrix.cpu().detach().numpy(),
                anatomical_vertices.cpu().detach().numpy().T,
            ).T
            projected_vertices[:, :2] /= projected_vertices[:, 2:]
            projected_vertices = projected_vertices[:, :2]
            # add dimension to axis 0:
            projected_vertices = np.expand_dims(projected_vertices, axis=0)
            data_sample["pred_instances"] = {}
            data_sample["pred_instances"]["bbox_scores"] = np.ones(
                len(projected_vertices)
            )
            coco_kps = np.zeros((len(projected_vertices), 17, 2))
            keypoints = np.concatenate((coco_kps, projected_vertices), axis=1)
            data_sample["pred_instances"]["keypoints"] = keypoints
            data_sample["pred_instances"]["keypoint_scores"] = np.ones(
                (1, len(projected_vertices[0]) + 17)
            )
            # print(
            #     "keypoint_scores shape:",
            #     data_sample["pred_instances"]["keypoint_scores"].shape,
            # )
            # render vertices on image and save it
            # for x, y in keypoints[0, 17:, :]:
            #     cv2.circle(img_ori, (int(x), int(y)), 1, (0, 0, 255))
            # for name in data_sample["raw_ann_info"]["keypoints"]:
            #     cv2.circle(
            #         img_ori,
            #         (
            #             int(data_sample["raw_ann_info"]["keypoints"][name]["x"]),
            #             int(data_sample["raw_ann_info"]["keypoints"][name]["y"]),
            #         ),
            #         10,
            #         (255, 0, 0),
            #     )
            # filename = osp.basename(img_ori_path).split(".")[0]
            # filename = filename + "_vertices_cliff_%s.jpg" % BACKBONE
            # # create folder if not exists
            # if not osp.exists("eval_test"):
            #     os.makedirs("eval_test")
            # vertices_path = osp.join("eval_test", filename)
            # cv2.imwrite(vertices_path, img_ori)
            data_samples.append(data_sample)
        infinity_metric.process([], data_samples)
        print("results:", infinity_metric.results)
        # results = infinity_metric.compute_metrics(infinity_metric.results)
    infinity_metric.evaluate(size=len(infinity_metric.results))


if __name__ == "__main__":
    # eval_dataset("../../", "combined_dataset_15fps/test/annotations.json")
    eval_dataset("/scratch/users/yonigoz/RICH/full_test/", "val_annotations.json")
