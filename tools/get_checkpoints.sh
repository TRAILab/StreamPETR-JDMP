#!/usr/bin/env bash

wget "https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth" -P ckpts
wget "https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth" -P ckpts
wget "https://github.com/exiawsh/storage/releases/download/v1.0/eva02_L_coco_det_sys_o365_remapped.pth" -P ckpts