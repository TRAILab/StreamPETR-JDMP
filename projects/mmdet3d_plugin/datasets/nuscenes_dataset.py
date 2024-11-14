# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import torch
from nuscenes import NuScenes
from nuscenes.eval.common.utils import Quaternion
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.eval.prediction.config import PredictionConfig
from nuscenes.prediction import PredictHelper
from mmcv.parallel import DataContainer as DC
import random
import math
import mmcv
import os.path as osp
import json
import time

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
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

    def evaluate(self,
                 results,
                 metric=['bbox', 'forecast'],
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
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
        results_dict = dict()
        if 'forecast_results' in results:
            forecast_results = results['forecast_results']
            results = results['bbox_results']
            preds, gts = self.forecast_format(forecast_results, jsonfile_prefix)
            results_dict.update(self.forecast_evaluate(forecast_results, preds, gts, jsonfile_prefix))

        if 'bbox' in metric:
            results_dict.update(super().evaluate(results, metric, logger, jsonfile_prefix, result_names, show, out_dir, pipeline))            

        if 'bbox' not in metric and 'forecast' not in metric:
            raise ValueError(f'Invalid metric type {metric}.')
        
        return results_dict

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

    def forecast_format(self, forecast_results, jsonfile_prefix=None, match_threshold=1):
        """Format the forecast results to json.

        Args:
            forecast_results (list[dict]): Forecast testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        """
        print('\nFormatting forecasts')
        export_full_preds = False
        preds = []
        preds_full = []
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
            if forecast.dim() == 4:
                forecast_pred_positions = forecast[..., :2].detach().cpu().numpy()
                forecast_probs = forecast[..., 0, 2].detach().cpu().numpy()
                forecast_cur_positions = forecast[:, 0, 0, 3:5].detach().cpu().numpy()
                forecast_pred_positions = forecast_pred_positions + forecast_cur_positions[:, None, None, :]
            else:
                raise ValueError('Forecast dim should be 4')

            if export_full_preds:
                forecast_pred_positions_full = np.concatenate((forecast_cur_positions[:, None, None, :], forecast_pred_positions),axis=2)
                pred_ids = np.arange(len(forecast))
                for pred_id in pred_ids:
                    instance_token = '0' # Not needed
                    sample_token = self.data_infos[sample_id]['token']
                    pred = forecast_pred_positions_full[pred_id]
                    prob = forecast_probs[pred_id]
                    preds_full.append(Prediction(instance_token, sample_token, pred, prob).serialize())

            # Match forecast to gt
            delta = gt_cur_positions.reshape(-1,1,2) - forecast_cur_positions.reshape(1,-1,2)
            dist = np.sqrt(delta[:,:,0]**2 + delta[:,:,1]**2)
            min_gt_idx, min_dist = np.argmin(dist, axis=0), np.min(dist, axis=0)
            match_true = min_dist < match_threshold
            gt_ids = min_gt_idx[match_true]
            pred_ids = np.arange(len(forecast))[match_true]

            # Get matched gt and predictions and apply gt masks
            for pred_id, gt_id in zip(pred_ids, gt_ids):
                gt_pred_mask = self.data_infos[sample_id]['gt_forecasting_masks'][gt_id][1:]
                if gt_pred_mask.sum() == 0:
                    continue
                gt = gt_pred_positions[gt_id][gt_pred_mask]
                gts.append(gt.tolist())
                instance_token = '0' # Not needed
                sample_token = self.data_infos[sample_id]['token']
                pred = forecast_pred_positions[pred_id][:,gt_pred_mask]
                prob = forecast_probs[pred_id]
                preds.append(Prediction(instance_token, sample_token, pred, prob).serialize())
        print('Format time: ', round(time.time()-start_time,1), 's')

        # Write results to file
        if jsonfile_prefix is not None:
            print('Forecast results writes to', jsonfile_prefix)
            jsonfile_prefix = osp.join(jsonfile_prefix,'forecast')
            mmcv.mkdir_or_exist(jsonfile_prefix)
            path = osp.join(jsonfile_prefix, 'results_nusc.json')
            json.dump(preds, open(path, "w"), indent=2)
            if export_full_preds:
                path = osp.join(jsonfile_prefix, 'results_nusc_full.json')
                json.dump(preds, open(path, "w"), indent=2)

        return preds, gts

    def forecast_evaluate(self, forecast_results, preds, gts, jsonfile_prefix=None):
        """Evaluation for a single forecast model in nuScenes protocol.

        Args:
            forecast_results (list[dict]): Forecast testing results of the dataset.
            preds (list[dict]): List of prediction dictionaries.
            gts (list[list[list]]): List of ground truth trajectories (n_gt, n_times, n_states).
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict: Dictionary of evaluation details.
        """
        print("Evaluating forecast")
        start_time = time.time()
        config_name = 'predict_eval.json'
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        helper = PredictHelper(nusc)
        this_dir = osp.dirname(osp.abspath(__file__))
        cfg_path = osp.join(this_dir, config_name)
        assert osp.exists(cfg_path), f'Requested unknown configuration {cfg_path}'
        config = json.load(open(cfg_path, 'r'))
        config = PredictionConfig.deserialize(config, helper)

        # Aggregate metrics
        n_preds = len(preds)
        containers = {metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics}
        for i in range(n_preds):
            pred = Prediction.deserialize(preds[i]) # [n_modes, n_timesteps, n_states]
            gt = np.array(gts[i])  # [n_timesteps, n_states]
            for forecast_metric in config.metrics:
                containers[forecast_metric.name][i] = forecast_metric(gt, pred)

        # Format results
        results = {}
        for forecast_metric in config.metrics:
            for agg in forecast_metric.aggregators:
                if hasattr(forecast_metric, 'k_to_report'):
                    for i, k in enumerate(forecast_metric.k_to_report):
                        metric_name = 'forecast/'+forecast_metric.name.replace('K', str(k))
                        results[metric_name] = agg(containers[forecast_metric.name])[i]
                else:
                    metric_name = 'forecast/'+forecast_metric.name
                    results[metric_name] = agg(containers[forecast_metric.name])[0]
        num_forecasts = forecast_results[0].shape[0]
        num_matches_avg = n_preds / len(self.data_infos)
        results['forecast/AvgMatchRate_2'] = num_matches_avg / num_forecasts
        for result in results:
            results[result] = round(results[result], 4)

        # Print results
        results_str = json.dumps(results, indent=2)[2:-2]
        results_str = results_str.replace(" ", "").replace("\"", "").replace(":", ": ").replace("forecast/", "").replace(",", "")
        print(results_str)
        print('Eval time: ', round(time.time()-start_time,1), 's')

        # Write results to file
        if jsonfile_prefix is not None:
            jsonfile_prefix = osp.join(jsonfile_prefix,'forecast')
            mmcv.mkdir_or_exist(jsonfile_prefix)
            path = osp.join(jsonfile_prefix, 'results_nusc.json')
            json.dump(preds, open(path, "w"), indent=2)
            path = osp.join(jsonfile_prefix, 'metrics_summary.json')
            json.dump(results, open(path, "w"), indent=2)
        
        return results

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
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
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
        return detail
    
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
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_train', # switched from val to train
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
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
        return detail


@DATASETS.register_module()
class JDMPCustomNuScenesDataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, *args, **kwargs):
        super().__init__(collect_keys, seq_mode, seq_split_num, num_frame_losses, queue_length, random_length, *args, **kwargs)

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