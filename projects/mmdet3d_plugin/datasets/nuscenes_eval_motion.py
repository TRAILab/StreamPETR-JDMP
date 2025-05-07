import argparse
import copy
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import Quaternion
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, \
    DetectionMetricDataList
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, dist_pr_curve, visualize_sample
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

from .nuscenes_eval_motion_utils import load_prediction, load_gt, accumulate, accumulate_motion, \
    DetectionMotionBox, DetectionMotionBox_modified, DetectionMotionMetrics, DetectionMotionMetricDataList

TP_METRICS = [
    'trans_err',
    'scale_err',
    'orient_err',
    'vel_err',
    'attr_err',
    'min_ade_err',
    'min_fde_err',
    'miss_rate_err']
TP_TRAJ_METRICS = ['min_ade_err', 'min_fde_err', 'miss_rate_err']
Axis = Any


def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.

    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are
        # plotted.
        rel_metrics = [
            m for m in TP_METRICS if not np.isnan(
                metrics.get_label_tp(
                    detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1])
                     for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(
            title=PRETTY_DETECTION_NAMES[detection_name],
            xlabel='Recall',
            ylabel='Error',
            xlim=1,
            min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind +
                                      1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(
                PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        if metric == 'trans_err':
            label += f' ({md.max_recall_ind})'  # add recall
            print(f'Recall: {detection_name}: {md.max_recall_ind/100}')
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def center_in_image(box,
                    intrinsic: np.ndarray,
                    imsize: Tuple[int,
                                  int],
                    vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    center_3d = box.center.reshape(3, 1)
    center_img = view_points(center_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(
        center_img[0, :] > 0, center_img[0, :] < imsize[0])
    visible = np.logical_and(visible, center_img[1, :] < imsize[1])
    visible = np.logical_and(visible, center_img[1, :] > 0)
    visible = np.logical_and(visible, center_3d[2, :] > 1)

    # True if a corner is at least 0.1 meter in front of the camera.
    in_front = center_3d[2, :] > 0.1

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def exist_corners_in_image_but_not_all(box,
                                       intrinsic: np.ndarray,
                                       imsize: Tuple[int,
                                                     int],
                                       vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible in images but not all corners in image .
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(
        corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    # True if a corner is at least 0.1 meter in front of the camera.
    in_front = corners_3d[2, :] > 0.1

    if any(visible) and not all(visible) and all(in_front):
        return True
    else:
        return False


def filter_eval_boxes_by_id(nusc: NuScenes,
                            eval_boxes: EvalBoxes,
                            id=None,
                            verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.token in id:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes


def filter_eval_boxes_by_visibility(
        ori_eval_boxes: EvalBoxes,
        visibility=None,
        verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.visibility == visibility:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After visibility based filtering: %d" % anns_filter)

    return eval_boxes


def filter_by_sample_token(
        ori_eval_boxes,
        valid_sample_tokens=[],
        verbose=False):
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    for sample_token in eval_boxes.sample_tokens:
        if sample_token not in valid_sample_tokens:
            eval_boxes.boxes.pop(sample_token)
    return eval_boxes


def filter_eval_boxes_by_overlap(nusc: NuScenes,
                                 eval_boxes: EvalBoxes,
                                 verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. basedon overlap .
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    cams = ['CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT']

    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        sample_record = nusc.get('sample', sample_token)
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            count = 0
            for cam in cams:
                '''
                copy-paste form nuscens
                '''
                sample_data_token = sample_record['data'][cam]
                sd_record = nusc.get('sample_data', sample_data_token)
                cs_record = nusc.get(
                    'calibrated_sensor',
                    sd_record['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                imsize = (sd_record['width'], sd_record['height'])
                new_box = Box(
                    box.translation,
                    box.size,
                    Quaternion(
                        box.rotation),
                    name=box.detection_name,
                    token='')

                # Move box to ego vehicle coord system.
                new_box.translate(-np.array(pose_record['translation']))
                new_box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                new_box.translate(-np.array(cs_record['translation']))
                new_box.rotate(Quaternion(cs_record['rotation']).inverse)

                if center_in_image(
                        new_box,
                        cam_intrinsic,
                        imsize,
                        vis_level=BoxVisibility.ANY):
                    count += 1
                # if exist_corners_in_image_but_not_all(new_box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                #    count += 1

            if count > 1:
                with open('center_overlap.txt', 'a') as f:
                    try:
                        f.write(box.token + '\n')
                    except BaseException:
                        pass
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    verbose = True

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes


class MotionEval(NuScenesEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 overlap_test=False,
                 eval_mask=False,
                 data_infos=None,
                 category_convert_type='motion_category',
                 conf_thresh=0.4,
                 future_seconds=6,
                 deteval_range=None):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """

        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.overlap_test = overlap_test
        self.eval_mask = eval_mask
        self.data_infos = data_infos
        self.conf_thresh = conf_thresh
        self.future_seconds = future_seconds
        self.deteval_range = deteval_range
        # Check result file exists.
        assert os.path.exists(
            result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionMotionBox,
                                                     verbose=verbose, category_convert_type=category_convert_type, conf_thresh=self.conf_thresh, seconds=self.future_seconds)
        self.gt_boxes = load_gt(
            self.nusc,
            self.eval_set,
            DetectionMotionBox_modified,
            verbose=verbose,
            category_convert_type=category_convert_type,
            seconds=self.future_seconds)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).

        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(
            nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if self.deteval_range is not None:
            for ind, sample_token in enumerate(self.pred_boxes.sample_tokens):
                self.pred_boxes.boxes[sample_token] = [box for box in self.pred_boxes.boxes[sample_token] if abs(box.ego_translation[0]) < self.deteval_range[0] and abs(box.ego_translation[1]) < self.deteval_range[1]]
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(
            nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        if self.deteval_range is not None:
            for ind, sample_token in enumerate(self.gt_boxes.sample_tokens):
                self.gt_boxes.boxes[sample_token] = [box for box in self.gt_boxes.boxes[sample_token] if abs(box.ego_translation[:2][0]) < self.deteval_range[0] and abs(box.ego_translation[:2][1]) < self.deteval_range[1]]

        if self.overlap_test:
            self.pred_boxes = filter_eval_boxes_by_overlap(
                self.nusc, self.pred_boxes)

            self.gt_boxes = filter_eval_boxes_by_overlap(
                self.nusc, self.gt_boxes, verbose=True)

        self.all_gt = copy.deepcopy(self.gt_boxes)
        self.all_preds = copy.deepcopy(self.pred_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

        self.index_map = {}
        for scene in nusc.scene:
            first_sample_token = scene['first_sample_token']
            sample = nusc.get('sample', first_sample_token)
            self.index_map[first_sample_token] = 1
            index = 2
            while sample['next'] != '':
                sample = nusc.get('sample', sample['next'])
                self.index_map[sample['token']] = index
                index += 1

    def update_gt(self, type_='vis', visibility='1', index=1):
        if type_ == 'vis':
            self.visibility_test = True
            if self.visibility_test:
                '''[{'description': 'visibility of whole object is between 0 and 40%',
                'token': '1',
                'level': 'v0-40'},
                {'description': 'visibility of whole object is between 40 and 60%',
                'token': '2',
                'level': 'v40-60'},
                {'description': 'visibility of whole object is between 60 and 80%',
                'token': '3',
                'level': 'v60-80'},
                {'description': 'visibility of whole object is between 80 and 100%',
                'token': '4',
                'level': 'v80-100'}]'''

                self.gt_boxes = filter_eval_boxes_by_visibility(
                    self.all_gt, visibility, verbose=True)

        elif type_ == 'ord':

            valid_tokens = [
                key for (
                    key,
                    value) in self.index_map.items() if value == index]
            # from IPython import embed
            # embed()
            self.gt_boxes = filter_by_sample_token(self.all_gt, valid_tokens)
            self.pred_boxes = filter_by_sample_token(
                self.all_preds, valid_tokens)
        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate_forcast(self) -> Tuple[DetectionMotionMetrics,
                                DetectionMotionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating forecasting metric data...', 'dist_ths', self.cfg.dist_ths, 'dist_th_tp', self.cfg.dist_th_tp)
        metric_data_list = DetectionMotionMetricDataList()

        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md, _, _, _ = accumulate(
                    self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating forecasting metrics...')
        metrics = DetectionMotionMetrics(self.cfg)

        traj_metrics = {}
        for class_name in self.cfg.class_names:
            # Compute APs.
            # for dist_th in self.cfg.dist_ths:
            #     metric_data = metric_data_list[(class_name, dist_th)]
            #     ap = calc_ap(
            #         metric_data,
            #         self.cfg.min_recall,
            #         self.cfg.min_precision)
            #     metrics.add_label_ap(class_name, dist_th, ap)
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(
                    class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in [
                        'attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                    if metric_name in TP_TRAJ_METRICS:
                        if class_name not in traj_metrics:
                            traj_metrics[class_name] = {}
                        traj_metrics[class_name][metric_name] = tp
                metrics.add_label_tp(class_name, metric_name, tp)
        # print_traj_metrics(traj_metrics)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def evaluate_mapf(
            self) -> Tuple[DetectionMotionMetrics, DetectionMotionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        self.cfg.dist_ths = [1.0]
        self.cfg.dist_th_tp = 1.0  # center dist for detection
        traj_dist_th = 2.0  # FDE for traj

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating mAP-minFDE metric data...', 'dist_ths', self.cfg.dist_ths, 'dist_th_tp', self.cfg.dist_th_tp)
        metric_data_list = DetectionMotionMetricDataList()

        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md, _, _, _ = accumulate_motion(
                    self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, traj_fde, dist_th, traj_dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating mAP-minFDE metrics...')
        metrics = DetectionMotionMetrics(self.cfg)

        traj_metrics = {}
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(
                    metric_data,
                    self.cfg.min_recall,
                    self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def evaluate_epa(
            self) -> Tuple[DetectionMotionMetrics, DetectionMotionMetricDataList]:
        """
        Performs the EPA evaluation.
        """
        start_time = time.time()
        
        self.cfg.dist_ths = [2.0]
        self.cfg.dist_th_tp = 2.0
        traj_dist_th = 2.0
        
        if self.verbose:
            print('Accumulating EPA metric data...', 'dist_ths', self.cfg.dist_ths, 'dist_th_tp', self.cfg.dist_th_tp)
        metric_data_list = DetectionMotionMetricDataList()
        metrics = DetectionMotionMetrics(self.cfg)
        
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md, N_det_tp, N_det_fp, N_det_gt = accumulate(
                    self.gt_boxes, self.pred_boxes, class_name, 
                    self.cfg.dist_fcn_callable, dist_th)
                md, N_det_traj_tp, N_det_traj_fp, N_det_traj_gt = accumulate_motion(
                    self.gt_boxes, self.pred_boxes, class_name,
                    self.cfg.dist_fcn_callable, traj_fde, dist_th, traj_dist_th)
                metric_data_list.set(class_name, dist_th, md)
                epa = (N_det_traj_tp - 0.5 * N_det_fp) / (N_det_gt + 1e-5)
                metrics.add_label_tp(class_name, 'epa', epa)                
                if self.verbose:
                    print(f'EPA for {class_name}: {epa:.4f}')
                    print(f'Details: TP={N_det_traj_tp}, FP={N_det_fp}, GT={N_det_gt}')

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)
        
        return metrics, metric_data_list

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True,
             eval_mode: str = 'standard') -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        if eval_mode == 'motion_map':
            metrics, metric_data_list = self.evaluate_mapf()
        elif eval_mode == 'standard':
            metrics, metric_data_list = self.evaluate_forcast()
        elif eval_mode == 'epa':
            metrics, metric_data_list = self.evaluate_epa()
        else:
            raise NotImplementedError
        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # print('Eval time: %.1fs' % metrics_summary['eval_time'])

        return metrics_summary

    def render(self, metrics: DetectionMetrics,
               md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(
            md_list,
            metrics,
            min_precision=self.cfg.min_precision,
            min_recall=self.cfg.min_recall,
            dist_th_tp=self.cfg.dist_th_tp,
            savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(
                md_list,
                metrics,
                detection_name,
                self.cfg.min_precision,
                self.cfg.min_recall,
                savepath=savepath(
                    detection_name +
                    '_pr'))

            class_tp_curve(
                md_list,
                metrics,
                detection_name,
                self.cfg.min_recall,
                self.cfg.dist_th_tp,
                savepath=savepath(
                    detection_name +
                    '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(
                md_list,
                metrics,
                dist_th,
                self.cfg.min_precision,
                self.cfg.min_recall,
                savepath=savepath(
                    'dist_pr_' +
                    str(dist_th)))


def print_traj_metrics(metrics):
    class_names = metrics.keys()
    x = PrettyTable()
    x.field_names = ["class names"] + TP_TRAJ_METRICS
    for class_name in metrics.keys():
        row_data = [class_name]
        for m in TP_TRAJ_METRICS:
            row_data.append('%.4f' % metrics[class_name][m])
        x.add_row(row_data)
    print(x)


def traj_fde(gt_box, pred_box, final_step):
    if gt_box.traj.shape[0] <= 0:
        return np.inf
    final_step = min(gt_box.traj.shape[0], final_step)
    gt_final = gt_box.traj[None, final_step-1]
    pred_final = np.array(pred_box.traj)[:,final_step-1,:]
    err = gt_final - pred_final
    err = np.sqrt(np.sum(np.square(gt_final - pred_final), axis=-1))
    return np.min(err)