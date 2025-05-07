# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Sandro Papais
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import torch
from nuscenes.eval.common.utils import Quaternion
from nuscenes.eval.prediction.data_classes import Prediction
from mmcv.parallel import DataContainer as DC
import random
import math
import mmcv
import os.path as osp
import json
import time
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, eval_mod=['detection'], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        self.forecast_match_threshold = 1 # Match threshold for forecast
        self.forecast_classes = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian'] # Filter classes for forecast eval
        self.eval_mod =  ['detection', 'forecast', 'forecast_uniad'] # Evaluation vizualization and metrics
        self.detection_conf_thresh = None # Result filtering detection confidence threshold
        self.foreval_detection_conf_thresh = 0.4 # Forecast evaluation detection confidence threshold
        self.foreval_future_seconds = 6 # Forecast evaluation future seconds
        self.deteval_range = None # Detection evaluation rectangular range (m), overwrites default circular range, ex [30,15]
        for mod in self.eval_mod:
            assert mod in ['viz', 'detection', 'detection_ext', 'forecast', 'forecast_uniad'], f"Invalid evaluation metric: {mod}"
        if self.deteval_range is not None:
            assert len(self.deteval_range) == 2, "deteval_range must be a list of two values"
            assert self.deteval_range[0] >= self.deteval_range[1], "deteval_range[0] must be greater than deteval_range[1]"
            for cls in self.eval_detection_configs.class_range.keys():
                self.eval_detection_configs.class_range[cls] = self.deteval_range[0] 
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def visualize_forecasts(self, jsonfile_prefix, gts_full_dict):
        """Visualize detection and forecast results.
        
        Args:
            jsonfile_prefix (str): Path prefix to the results json files
            gts_full_dict (dict): Dictionary mapping sample tokens to ground truth trajectories
        """
        from projects.mmdet3d_plugin.datasets.nuscenes_viz import NuScenesVisualizer
        from tqdm import tqdm
        
        # Initialize visualizer
        nuscViz = NuScenesVisualizer()
        
        # Load detection results
        path = osp.join(jsonfile_prefix, 'pts_bbox', 'results_nusc.json')
        with open(path, 'rb') as f:
            predictions = json.load(f)['results']
            
        # Load and process forecast results
        path = osp.join(jsonfile_prefix, 'forecast', 'results_nusc_full.json')
        with open(path, 'rb') as f:
            for_preds_raw = json.load(f)
            
        # Organize forecasts by sample
        for_preds = {}
        for_scores = {}
        for pred in for_preds_raw:
            if pred['sample'] not in for_preds:
                for_preds[pred['sample']] = [pred['prediction']]
                for_scores[pred['sample']] = [float(pred['instance'])]
            else:
                for_preds[pred['sample']].append(pred['prediction'])
                for_scores[pred['sample']].append(float(pred['instance']))
                
        # Render visualizations
        for sample_id, (token, preds) in enumerate(tqdm(predictions.items(), desc="Rendering boxes")):
            if token not in for_preds:
                print("No forecast for", token, "skipping")
                continue
            file_name = f'{self.version[5:]}_{str(sample_id).zfill(5)}.png'
            out_path = f'output/viz/forecast_eval/{file_name}'
            nuscViz.render_boxes(self.nusc, token, preds, for_preds[token], 
                               for_scores[token], gts_full_dict[token], out_path)

    def evaluate(self, results, metric=['bbox'], logger=None,
                jsonfile_prefix=None, result_names=['pts_bbox'], show=False,
                out_dir=None, pipeline=None):
        """Evaluation in nuScenes protocol.
        
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'forecast'. Use 'bbox' for detection only evaluation.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        
        # Filter results based on confidence threshold if specified
        if self.detection_conf_thresh is not None:
            results = self.detection_conf_filter_results(results, self.detection_conf_thresh)
        
        # Visualize results
        if 'viz' in self.eval_mod and 'forecast_results' in results:
            preds, gts, for_gts = self.forecast_format(results['forecast_results'], results['bbox_results'], jsonfile_prefix)
            self.visualize_forecasts(jsonfile_prefix, for_gts)
        
        # Evaluate results
        else:
            results_dict = dict()
            
            # Evaluate detection metrics
            if 'detection' in self.eval_mod or 'detection_ext' in self.eval_mod:
                start_time = time.time()
                if 'bbox_results' in results:
                    results_dict.update(super().evaluate(results['bbox_results'], metric, logger, jsonfile_prefix, result_names, show, out_dir, pipeline))            
                else:
                    results_dict.update(super().evaluate(results, metric, logger, jsonfile_prefix, result_names, show, out_dir, pipeline))            
                print('Format and eval time: ', round(time.time()-start_time,1), 's')
            
            # Evaluate forecast metrics
            if 'forecast_results' in results:
                if 'forecast' in self.eval_mod:
                    preds, gts, for_gts = self.forecast_format(results['forecast_results'], results['bbox_results'], jsonfile_prefix)
                    num_forecasts = len(results['forecast_results'][0]['pts_forecast']['trajs_2d'])
                    results_dict.update(self.forecast_evaluate(preds, gts, jsonfile_prefix, num_forecasts))
                    del preds, gts, for_gts
                if 'forecast_uniad' in self.eval_mod:
                    result_files, tmp_dir = self.forecast_format_uniad(results, jsonfile_prefix)
                    results_dict.update(self.forecast_evaluate_uniad(result_files))
                    if tmp_dir is not None:
                        tmp_dir.cleanup()
                    
        del self.nusc
        return results_dict
    
    def detection_conf_filter_results(self, results, conf_thresh=0.4):
        """Filter results based on confidence threshold.
        
        Args:
            results (list[dict]): Testing results of the dataset.

        Returns:
            list[dict]: Filtered results.
        """
        filtered_results = {'bbox_results': [], 'forecast_results': []}
        for bbox_result, forecast_result in zip(results['bbox_results'], results['forecast_results']):
            mask = bbox_result['pts_bbox']['scores_3d'] >= conf_thresh
            filtered_bbox_result = { 'pts_bbox': {
                'boxes_3d': bbox_result['pts_bbox']['boxes_3d'][mask],
                'scores_3d': bbox_result['pts_bbox']['scores_3d'][mask],
                'labels_3d': bbox_result['pts_bbox']['labels_3d'][mask]
                }
            }
            filtered_forecast_result = { 'pts_forecast': {
                    'trajs_2d': forecast_result['pts_forecast']['trajs_2d'][mask],
                    'scores_2d': forecast_result['pts_forecast']['scores_2d'][mask],
                    'refs_2d': forecast_result['pts_forecast']['refs_2d'][mask]
                }
            }
            filtered_results['bbox_results'].append(filtered_bbox_result)
            filtered_results['forecast_results'].append(filtered_forecast_result)
        results = filtered_results
        return results

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        if 'forecast_results' in results:
            forecast_results = results['forecast_results']
            results = results['bbox_results']
            self.forecast_format(forecast_results, jsonfile_prefix)
        result_files, tmp_dir = super().format_results(results, jsonfile_prefix)
        return result_files, tmp_dir

    def forecast_format_uniad(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        from nuscenes.prediction import convert_local_coords_to_global
        import tempfile
        import copy
        print('\nFormatting forecast uniad')
        start_time = time.time()

        for k in results.keys():
            assert isinstance(results[k], list), 'results must be a list'
            assert len(results[k]) == len(self), (
                'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results[k]), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        nusc_annos = {}
        mapped_class_names = self.CLASSES

        for sample_id, _ in enumerate(mmcv.track_iter_progress(results['forecast_results'])):
            annos = []
            sample_token = self.data_infos[sample_id]['token']
            det = results['bbox_results'][sample_id]['pts_bbox']
            pred = results['forecast_results'][sample_id]['pts_forecast']

            if 'boxes_3d' not in det:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                                       mapped_class_names,
                                                       self.eval_detection_configs,
                                                       self.deteval_range,
                                                       self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                # center_ = box.center.tolist()
                # change from ground height to center height
                # center_[2] = center_[2] + (box.wlh.tolist()[2] / 2.0)
                if name not in self.forecast_classes: 
                    continue

                box_ego = boxes_ego[keep_idx[i]]
                trans = box_ego.center

                if 'trajs_2d' in pred:
                    traj_local = pred['trajs_2d'][keep_idx[i]].numpy()[..., :2]
                    traj_scores = pred['scores_2d'][keep_idx[i]].numpy()
                else:
                    traj_local = np.zeros((0,))
                    traj_scores = np.zeros((0,))
                traj_ego = np.zeros_like(traj_local)
                rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi/2)
                for kk in range(traj_ego.shape[0]):
                    traj_ego[kk] = convert_local_coords_to_global(
                        traj_local[kk], trans, rot)

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.token,
                    predict_traj=traj_ego,
                    predict_traj_score=traj_scores,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos
        }

        jsonfile_prefix = osp.join(jsonfile_prefix,'forecast_uniad')
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Forecast results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        print('Format time: ', round(time.time()-start_time,1), 's')
        return res_path, tmp_dir


    def forecast_format(self, forecast_results, det_results, jsonfile_prefix=None):
        """Format the forecast results to json.

        Args:
            forecast_results (list[dict]): Forecast testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        """
        print('\nFormatting forecasts')
        preds = []
        preds_full = []
        gts_full_dict = {}
        gts = []
        start_time = time.time()
        for sample_id, forecast in enumerate(forecast_results):
            # Get gt positions
            sample_token = self.data_infos[sample_id]['token']
            if self.data_infos[sample_id]['gt_forecasting_locs'].size == 0:
                continue
            gt = self.data_infos[sample_id]['gt_forecasting_locs'][:,:,:2]
            gt_cur_positions = gt[:,0]
            gt_pred_positions = gt[:,1:]

            # Get forecast positions
            forecast_det_scores = det_results[sample_id]['pts_bbox']['scores_3d'].cpu().numpy()
            labels = det_results[sample_id]['pts_bbox']['labels_3d'].cpu().numpy()
            forecast_classes = [self.CLASSES[label] for label in labels]
            forecast_pred_positions = forecast['pts_forecast']['trajs_2d'].numpy()
            forecast_probs = forecast['pts_forecast']['scores_2d'].numpy()
            forecast_cur_positions = forecast['pts_forecast']['refs_2d'].numpy()
            forecast_pred_positions = forecast_pred_positions + forecast_cur_positions[:, None, None, :]
            num_modes = forecast_pred_positions.shape[1]

            # Get all top predictions 
            if 'viz' in self.eval_mod:
                gts_full_dict[sample_token] = gt
                top_indices = np.argmax(forecast_probs, axis=1)  # Shape: (300,)
                forecast_top_probs = np.max(forecast_probs, axis=1)  # Shape: (300,)
                forecast_top_pred_positions = forecast_pred_positions[np.arange(300), top_indices][:, np.newaxis, :, :]  # Shape: (300, 1, 12, 2)
                forecast_pred_positions_full = np.concatenate(
                    (forecast_cur_positions[:, None, None, :], 
                    forecast_top_pred_positions), axis=2)
                pred_ids = np.arange(len(forecast_pred_positions))
                for pred_id in pred_ids:
                    instance_token = str(forecast_det_scores[pred_id]) # store detections scores instead
                    sample_token = self.data_infos[sample_id]['token']
                    pred = forecast_pred_positions_full[pred_id]
                    prob = np.array([forecast_top_probs[pred_id]])
                    if num_modes == 1:
                        prob = prob[0]
                    if not isinstance(prob, np.ndarray):
                        prob = np.array([prob])
                    preds_full.append(Prediction(instance_token, sample_token, pred, prob).serialize())

            # Match forecast to gt
            delta = gt_cur_positions.reshape(-1,1,2) - forecast_cur_positions.reshape(1,-1,2)
            dist = np.sqrt(delta[:,:,0]**2 + delta[:,:,1]**2)
            min_gt_idx, min_dist = np.argmin(dist, axis=0), np.min(dist, axis=0)
            match_true = min_dist < self.forecast_match_threshold
            gt_ids = min_gt_idx[match_true]
            pred_ids = np.arange(len(forecast_pred_positions))[match_true]

            # Get matched gt and predictions
            for pred_id, gt_id in zip(pred_ids, gt_ids):
                if str(forecast_classes[pred_id]) not in self.forecast_classes:
                    continue
                det = forecast_cur_positions[pred_id]
                if self.deteval_range is not None:
                    if abs(det[0]) > self.deteval_range[0] or abs(det[1]) > self.deteval_range[1]:
                        continue
                else:
                    range_limit = self.eval_detection_configs.class_range[str(forecast_classes[pred_id])]
                    det_range = np.sqrt(np.sum((det)**2))
                    if det_range > range_limit:
                        continue
                gt_pred_mask = self.data_infos[sample_id]['gt_forecasting_masks'][gt_id][1:]
                if gt_pred_mask.sum() == 0:
                    continue
                gt = gt_pred_positions[gt_id][gt_pred_mask]
                gts.append(gt.tolist())
                instance_token = str(forecast_classes[pred_id]) # store classes instead
                sample_token = self.data_infos[sample_id]['token']
                pred = forecast_pred_positions[pred_id][:,gt_pred_mask]
                prob = forecast_probs[pred_id]
                if num_modes == 1:
                    prob = prob[0]
                if not isinstance(prob, np.ndarray):
                    prob = np.array([prob])
                preds.append(Prediction(instance_token, sample_token, pred, prob).serialize())

        # Write results to file
        if jsonfile_prefix is not None:
            print('Forecast results writes to', jsonfile_prefix)
            jsonfile_prefix = osp.join(jsonfile_prefix,'forecast')
            mmcv.mkdir_or_exist(jsonfile_prefix)
            path = osp.join(jsonfile_prefix, 'results_nusc.json')
            json.dump(preds, open(path, "w"), indent=2)
            if 'viz' in self.eval_mod:
                path = osp.join(jsonfile_prefix, 'results_nusc_full.json')
                json.dump(preds_full, open(path, "w"), indent=2)
        print('Format time: ', round(time.time()-start_time,1), 's')    

        return preds, gts, gts_full_dict


    def forecast_evaluate_uniad(self, result_path):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from .nuscenes_eval_motion import MotionEval
        print('Evaluating forecast uniad')
        start_time = time.time()
        output_dir = osp.join(*osp.split(result_path)[:-1])
        mmcv.mkdir_or_exist(output_dir)

        eval_set_map = {
            'v1.0-mini': 'mini_train',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval_motion = MotionEval(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=False,
            data_infos=self.data_infos,
            category_convert_type='motion_category',
            conf_thresh=self.foreval_detection_conf_thresh,
            future_seconds=self.foreval_future_seconds,
            deteval_range=self.deteval_range,
        )

        detail = dict()
        metrics_summary = self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='standard')
        detail['forecast/MinADEU'] = metrics_summary['label_tp_errors']['car']['min_ade_err']
        detail['forecast/MinFDEU'] = metrics_summary['label_tp_errors']['car']['min_fde_err']
        detail['forecast/MissRateU'] = metrics_summary['label_tp_errors']['car']['miss_rate_err']
        
        metrics_summary = self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='motion_map')
        detail['forecast/mAPf'] = metrics_summary['mean_dist_aps']['car']

        metrics_summary = self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='epa')
        detail['forecast/EPA'] = metrics_summary['label_tp_errors']['car']['epa']
        
        results_str = json.dumps(detail, indent=2)[2:-2].replace(" ", "").replace("\"", "").replace(":", ": ").replace("forecast/", "").replace(",", "")
        print(results_str)
        print('Eval time: ', round(time.time()-start_time,1), 's')

        return detail

    def forecast_evaluate(self, preds, gts, jsonfile_prefix=None, num_forecasts=300):
        """Evaluation for a single forecast model in nuScenes protocol.

        Args:
            preds (list[dict]): List of prediction dictionaries.
            gts (list[list[list]]): List of ground truth trajectories (n_gt, n_times, n_states).
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            num_forecasts (int): Number of forecasts. Default: 300.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes.eval.prediction.config import PredictionConfig
        from nuscenes.prediction import PredictHelper

        print("Evaluating forecast")
        start_time = time.time()
        
        # Setup
        eval_class_sets = {'all': self.CLASSES, 
            'vehicle': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle'],
            'static': ['barrier', 'traffic_cone'],
            'pedestrian': ['pedestrian'],
        }
        
        config_name = 'predict_eval.json'
        helper = PredictHelper(self.nusc)
        this_dir = osp.dirname(osp.abspath(__file__))
        cfg_path = osp.join(this_dir, config_name)
        assert osp.exists(cfg_path), f'Requested unknown configuration {cfg_path}'
        config = json.load(open(cfg_path, 'r'))
        config = PredictionConfig.deserialize(config, helper)
        
        # Create class set mapping for each prediction
        pred_class_sets = {}
        for i, pred_dict in enumerate(preds):
            pred = Prediction.deserialize(pred_dict)
            pred_class_sets[i] = []
            for class_set_name, class_set in eval_class_sets.items():
                if pred.instance in class_set:
                    pred_class_sets[i].append(class_set_name)
        
        # Compute metrics for all predictions once
        n_preds = len(preds)
        metric_results = {}
        for forecast_metric in config.metrics:
            metric_results[forecast_metric.name] = np.zeros((n_preds, forecast_metric.shape))
        
        # Compute metrics for each prediction
        for i, (pred_dict, gt_array) in enumerate(zip(preds, gts)):
            pred = Prediction.deserialize(pred_dict)
            gt = np.array(gt_array)
            for forecast_metric in config.metrics:
                metric_results[forecast_metric.name][i] = forecast_metric(gt, pred)
        
        # Aggregate results by class set
        all_results = {}
        for class_set_name, class_set in eval_class_sets.items():
            # Get indices of predictions that belong to this class set
            indices = [i for i, class_sets in pred_class_sets.items() if class_set_name in class_sets]
            if not indices:
                print(f"  No predictions found for {class_set_name} class set")
                continue
            
            # Aggregate metrics for this class set
            results = {}
            for forecast_metric in config.metrics:
                metric_name_prefix = f'forecast/{class_set_name}'
                class_set_results = metric_results[forecast_metric.name][indices]
                
                for agg in forecast_metric.aggregators:
                    if hasattr(forecast_metric, 'k_to_report'):
                        for i, k in enumerate(forecast_metric.k_to_report):
                            metric_name = f'{metric_name_prefix}/{forecast_metric.name.replace("K", str(k))}'
                            results[metric_name] = agg(class_set_results)[i]
                    else:
                        metric_name = f'{metric_name_prefix}/{forecast_metric.name}'
                        results[metric_name] = agg(class_set_results)[0]
            
            results[f'{metric_name_prefix}/AvgMatchRate_2'] = len(indices) / len(self.data_infos)
            
            for result in results:
                results[result] = round(results[result], 4)
            all_results.update(results)
        
        # Format results for table
        all_metrics = set()
        class_sets_with_results = set()
        for key in all_results.keys():
            parts = key.split('/')
            class_set_name = parts[1]
            metric_name = parts[2]
            all_metrics.add(metric_name)
            class_sets_with_results.add(class_set_name)
        all_metrics = sorted(list(all_metrics))
        class_sets_with_results = sorted(list(class_sets_with_results))
        
        # Print table rows
        header = "Metric".ljust(20)
        for class_set in class_sets_with_results:
            header += class_set.ljust(15)
        print(header)
        print("-" * (20 + 15 * len(class_sets_with_results)))
        for metric in all_metrics:
            row = metric.ljust(20)
            for class_set in class_sets_with_results:
                key = f"forecast/{class_set}/{metric}"
                if key in all_results:
                    row += f"{all_results[key]:.4f}".ljust(15)
                else:
                    row += "N/A".ljust(15)
            print(row)
        
        # Write results to file
        if jsonfile_prefix is not None:
            jsonfile_prefix = osp.join(jsonfile_prefix,'forecast')
            mmcv.mkdir_or_exist(jsonfile_prefix)
            path = osp.join(jsonfile_prefix, 'results_nusc.json')
            json.dump(preds, open(path, "w"), indent=2)
            path = osp.join(jsonfile_prefix, 'metrics_summary.json')
            json.dump(all_results, open(path, "w"), indent=2)
        
        print(f'Eval time: {time.time() - start_time:.2f}s')

        return all_results

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        if 'detection_ext' in self.eval_mod:
            from projects.mmdet3d_plugin.datasets.nuscenes_eval_detection import NuScenesEval
        else:
            from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        eval_set_map = {
            'v1.0-mini': 'mini_train', # switched from val to train
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # Record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        
        # Original per-class metrics
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']

        if 'detection_ext' in self.eval_mod:
            # Add distance-based metrics
            for dist_range, dist_metrics in metrics['distance_metrics'].items():
                for name in self.CLASSES:
                    if name in dist_metrics['label_aps']:
                        for k, v in dist_metrics['label_aps'][name].items():
                            val = float('{:.4f}'.format(v))
                            detail['{}/{}_AP_{}_{}'.format(metric_prefix, name, dist_range, k)] = val
                # Add mean metrics for this distance range
                detail['{}/mAP_{}'.format(metric_prefix, dist_range)] = float('{:.4f}'.format(dist_metrics['mean_ap']))
                detail['{}/mAR_{}'.format(metric_prefix, dist_range)] = float('{:.4f}'.format(dist_metrics['mean_ar']))

            # Add point-based metrics
            for point_range, point_metrics in metrics['point_metrics'].items():
                for name in self.CLASSES:
                    if name in point_metrics['label_aps']:
                        for k, v in point_metrics['label_aps'][name].items():
                            val = float('{:.4f}'.format(v))
                            detail['{}/{}_AP_{}_{}'.format(metric_prefix, name, point_range, k)] = val
                # Add mean metrics for this point range
                detail['{}/mAP_{}'.format(metric_prefix, point_range)] = float('{:.4f}'.format(point_metrics['mean_ap']))
                detail['{}/mAR_{}'.format(metric_prefix, point_range)] = float('{:.4f}'.format(point_metrics['mean_ar']))

            # Add visibility-based metrics
            for vis_range, vis_metrics in metrics['visibility_metrics'].items():
                for name in self.CLASSES:
                    if name in vis_metrics['label_aps']:
                        for k, v in vis_metrics['label_aps'][name].items():
                            val = float('{:.4f}'.format(v))
                            detail['{}/{}_AP_{}_{}'.format(metric_prefix, name, vis_range, k)] = val
                # Add mean metrics for this visibility range
                detail['{}/mAP_{}'.format(metric_prefix, vis_range)] = float('{:.4f}'.format(vis_metrics['mean_ap']))
                detail['{}/mAR_{}'.format(metric_prefix, vis_range)] = float('{:.4f}'.format(vis_metrics['mean_ar']))
        
        return detail

@DATASETS.register_module()
class JDMPCustomNuScenesDataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, eval_mod=['detection', 'forecast'], *args, **kwargs):
        super().__init__(collect_keys, seq_mode, seq_split_num, num_frame_losses, queue_length, random_length, eval_mod, *args, **kwargs)

    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 
                        'depths', 'gt_forecasting_bboxes_3d', 'gt_forecasting_masks']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            # Work around for modifying get_ann_info
            if self.use_valid_flag:
                mask = info['valid_flag']
            else:
                mask = info['num_lidar_pts'] > 0
            gt_forecasting_bboxes_3d = info['gt_forecasting_boxes'][mask]
            if self.with_velocity:
                if np.sum(mask)>0:
                    gt_forecasting_velocity = info['gt_forecasting_velocity'][mask]
                    nan_mask = np.isnan(gt_forecasting_velocity[:, :, 0])
                    gt_forecasting_velocity[nan_mask] = [0.0, 0.0]
                    gt_forecasting_bboxes_3d = np.concatenate([gt_forecasting_bboxes_3d, gt_forecasting_velocity], axis=-1)
                    # Stack all trajectories together
                    gt_forecasting_bboxes_3d = gt_forecasting_bboxes_3d.reshape(-1, gt_forecasting_bboxes_3d.shape[-1])
                box_dim = 7
            else:
                box_dim = 9
            # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
            # the same as KITTI (0.5, 0.5, 0)
            gt_forecasting_bboxes_3d = LiDARInstance3DBoxes(gt_forecasting_bboxes_3d,
                9, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            annos.update(dict(gt_forecasting_bboxes_3d=gt_forecasting_bboxes_3d))
            annos.update(dict(gt_forecasting_masks=info['gt_forecasting_masks'][mask]))
            input_dict['ann_info'] = annos
            
        return input_dict


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    if 'track_ids' in detection:
        ids = detection['track_ids'].numpy()
    else:
        ids = np.ones_like(labels)

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box.token = ids[i]
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_range,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.
    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'
    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    keep_idx = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        if eval_range is not None: 
            x_distance, y_distance = box.center[0], box.center[1]
            if abs(x_distance) > eval_range[0] or abs(y_distance) > eval_range[1]:
                continue
        else:
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = eval_configs.class_range[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        keep_idx.append(i)
    return box_list, keep_idx
