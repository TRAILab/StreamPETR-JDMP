import json
import os
import random
import time
from typing import Tuple, Dict, Any, Union
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from nuscenes.eval.detection.utils import category_to_detection_name
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


class ExtendedDetectionMetrics(DetectionMetrics):
    """Extended version of DetectionMetrics that includes recall calculations."""
    
    def __init__(self, cfg: DetectionConfig):
        super().__init__(cfg)
        self._label_recalls = defaultdict(float)

    def add_label_recall(self, detection_name: str, recall: float) -> None:
        """Add recall value for a detection class."""
        self._label_recalls[detection_name] = recall

    def get_label_recall(self, detection_name: str) -> float:
        """Get recall value for a detection class."""
        return self._label_recalls[detection_name]

    @property
    def mean_recall(self) -> float:
        """Calculate mean recall across all classes."""
        return float(np.mean(list(self._label_recalls.values())))

    def serialize(self):
        """Serialize with additional recall information."""
        base = super().serialize()
        base.update({
            'label_recalls': self._label_recalls,
            'mean_ar': self.mean_recall
        })
        return base


def load_gt_with_visibility(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Modified version of load_gt that includes visibility information.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=float(sample_annotation['visibility_token']),  # Store visibility token as float
                        attribute_name=attribute_name
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


class NuScenesEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
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

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data with visibility information
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                   verbose=verbose)
        # Use modified load_gt function that includes visibility information
        self.gt_boxes = load_gt_with_visibility(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        print(self.pred_boxes[self.pred_boxes.sample_tokens[0]][0].num_pts)
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = set(self.gt_boxes.sample_tokens)

        # Add distance ranges for evaluation
        self.distance_ranges = [
            (0, 15),  # 0-15m
            (15, 30), # 15-30m
            (30, float('inf')), # 30+
        ]

        # Add point ranges for evaluation
        self.point_ranges = {
            'dense': (6, float('inf')),  # 6 <= p <= inf
            'sparse': (0, 5)             # 0 <= p <= 5 points
        }

        # Pre-compute distance ranges masks
        self.distance_masks = self._precompute_distance_masks()
        
        # Add point counts to prediction boxes
        if verbose:
            print('Computing point counts for prediction boxes')
        self._add_point_counts_to_predictions()

        # Add point thresholds for evaluation
        self.point_masks = self._precompute_point_masks()

        # Add visibility ranges for evaluation
        self.visibility_ranges = {
            'v0-40': '1',
            'v40-60': '2',
            'v60-80': '3',
            'v80-100': '4'
        }

        # Pre-compute visibility masks
        if verbose:
            print('Computing visibility masks for ground truth boxes')
        self.visibility_masks = self._precompute_visibility_masks()
        
    def _precompute_distance_masks(self):
        """Pre-compute distance masks for each range to avoid repeated calculations"""
        masks = {}
        for min_d, max_d in self.distance_ranges:
            range_masks = defaultdict(dict)
            for sample_token in self.sample_tokens:
                gt_boxes = self.gt_boxes[sample_token]
                pred_boxes = self.pred_boxes[sample_token]
                
                gt_mask = np.array([min_d <= box.ego_dist < max_d for box in gt_boxes])
                pred_mask = np.array([min_d <= box.ego_dist < max_d for box in pred_boxes])
                
                range_masks[sample_token]['gt'] = gt_mask
                range_masks[sample_token]['pred'] = pred_mask
            
            masks[f"{min_d}_{max_d}m"] = range_masks
        return masks

    def _precompute_point_masks(self):
        """Pre-compute point count masks"""
        masks = {}
        for point_range, (min_pts, max_pts) in self.point_ranges.items():
            range_masks = defaultdict(dict)
            for sample_token in self.sample_tokens:
                gt_boxes = self.gt_boxes[sample_token]
                pred_boxes = self.pred_boxes[sample_token]
                
                gt_mask = np.array([min_pts <= box.num_pts <= max_pts for box in gt_boxes])
                pred_mask = np.array([min_pts <= box.num_pts <= max_pts for box in pred_boxes])
                
                range_masks[sample_token]['gt'] = gt_mask
                range_masks[sample_token]['pred'] = pred_mask
            
            masks[point_range] = range_masks
        return masks

    def _precompute_visibility_masks(self):
        """Pre-compute visibility masks for each visibility level"""
        masks = {}
        for vis_name, vis_token in self.visibility_ranges.items():
            range_masks = defaultdict(dict)
            for sample_token in self.sample_tokens:
                gt_boxes = self.gt_boxes[sample_token]
                pred_boxes = self.pred_boxes[sample_token]
                
                # Compare as integers but use float for storage
                gt_mask = np.array([int(box.detection_score) == int(vis_token) for box in gt_boxes])
                # All predictions are considered for each visibility range
                pred_mask = np.ones(len(pred_boxes), dtype=bool)
                
                range_masks[sample_token]['gt'] = gt_mask
                range_masks[sample_token]['pred'] = pred_mask
            
            masks[vis_name] = range_masks
        return masks

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList, Dict[str, ExtendedDetectionMetrics], 
                               Dict[str, ExtendedDetectionMetrics], Dict[str, ExtendedDetectionMetrics]]:
        """Optimized evaluation method with visibility-based metrics"""
        start_time = time.time()
        
        metric_data_list = DetectionMetricDataList()
        distance_metrics = {f"{min_d}_{max_d}m": ExtendedDetectionMetrics(self.cfg) 
                          for min_d, max_d in self.distance_ranges}
        point_metrics = {k: ExtendedDetectionMetrics(self.cfg) for k in self.point_masks.keys()}
        visibility_metrics = {k: ExtendedDetectionMetrics(self.cfg) for k in self.visibility_ranges.keys()}

        def accumulate_metrics(class_name, dist_th):
            # Regular metrics accumulation
            md = accumulate_test(self.gt_boxes, self.pred_boxes, class_name, 
                           self.cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)
            
            # Distance-based metrics accumulation
            for (min_d, max_d) in self.distance_ranges:
                range_key = f"{min_d}_{max_d}m"
                masks = self.distance_masks[range_key]
                
                filtered_gt_boxes = EvalBoxes()
                filtered_pred_boxes = EvalBoxes()
                
                for sample_token in self.sample_tokens:
                    gt_mask = masks[sample_token]['gt']
                    pred_mask = masks[sample_token]['pred']
                    
                    if np.any(gt_mask):
                        filtered_gt_boxes.boxes[sample_token] = [box for i, box in enumerate(self.gt_boxes[sample_token]) if gt_mask[i]]
                    if np.any(pred_mask):
                        filtered_pred_boxes.boxes[sample_token] = [box for i, box in enumerate(self.pred_boxes[sample_token]) if pred_mask[i]]
                
                if filtered_gt_boxes.boxes or filtered_pred_boxes.boxes:
                    md = accumulate(filtered_gt_boxes, filtered_pred_boxes, class_name,
                                  self.cfg.dist_fcn_callable, dist_th)
                    ap = calc_ap(md, self.cfg.min_recall, self.cfg.min_precision)
                    distance_metrics[range_key].add_label_ap(class_name, dist_th, ap)
                    
                    # Add average recall calculation for the max distance threshold
                    if dist_th == max(self.cfg.dist_ths):
                        if isinstance(distance_metrics[range_key], ExtendedDetectionMetrics):
                            # Calculate average recall across all recall points
                            avg_recall = float(np.mean(md.recall[:md.max_recall_ind + 1]))
                            distance_metrics[range_key].add_label_recall(class_name, avg_recall)
            
            # Point-based metrics accumulation
            for point_range, masks in self.point_masks.items():
                filtered_gt_boxes = EvalBoxes()
                filtered_pred_boxes = EvalBoxes()
                
                for sample_token in self.sample_tokens:
                    gt_mask = masks[sample_token]['gt']
                    pred_mask = masks[sample_token]['pred']
                    
                    if np.any(gt_mask):
                        filtered_gt_boxes.boxes[sample_token] = [box for i, box in enumerate(self.gt_boxes[sample_token]) if gt_mask[i]]
                    if np.any(pred_mask):
                        filtered_pred_boxes.boxes[sample_token] = [box for i, box in enumerate(self.pred_boxes[sample_token]) if pred_mask[i]]
                
                if filtered_gt_boxes.boxes or filtered_pred_boxes.boxes:
                    md = accumulate(filtered_gt_boxes, filtered_pred_boxes, class_name,
                                  self.cfg.dist_fcn_callable, dist_th)
                    ap = calc_ap(md, self.cfg.min_recall, self.cfg.min_precision)
                    point_metrics[point_range].add_label_ap(class_name, dist_th, ap)
                    
                    # Add average recall calculation for the max distance threshold
                    if dist_th == max(self.cfg.dist_ths):
                        if isinstance(point_metrics[point_range], ExtendedDetectionMetrics):
                            # Calculate average recall across all recall points
                            avg_recall = float(np.mean(md.recall[:md.max_recall_ind + 1]))
                            point_metrics[point_range].add_label_recall(class_name, avg_recall)

            # Visibility-based metrics accumulation
            for vis_range, masks in self.visibility_masks.items():
                filtered_gt_boxes = EvalBoxes()
                filtered_pred_boxes = EvalBoxes()
                
                for sample_token in self.sample_tokens:
                    gt_mask = masks[sample_token]['gt']
                    pred_mask = masks[sample_token]['pred']
                    
                    if np.any(gt_mask):
                        filtered_gt_boxes.boxes[sample_token] = [box for i, box in enumerate(self.gt_boxes[sample_token]) if gt_mask[i]]
                    if np.any(pred_mask):
                        filtered_pred_boxes.boxes[sample_token] = [box for i, box in enumerate(self.pred_boxes[sample_token]) if pred_mask[i]]
                
                if filtered_gt_boxes.boxes or filtered_pred_boxes.boxes:
                    md = accumulate(filtered_gt_boxes, filtered_pred_boxes, class_name,
                                  self.cfg.dist_fcn_callable, dist_th)
                    ap = calc_ap(md, self.cfg.min_recall, self.cfg.min_precision)
                    visibility_metrics[vis_range].add_label_ap(class_name, dist_th, ap)
                    
                    # Add average recall calculation for the max distance threshold
                    if dist_th == max(self.cfg.dist_ths):
                        if isinstance(visibility_metrics[vis_range], ExtendedDetectionMetrics):
                            avg_recall = float(np.mean(md.recall[:md.max_recall_ind + 1]))
                            visibility_metrics[vis_range].add_label_recall(class_name, avg_recall)

        # Parallel execution of metric accumulation
        tasks = [(class_name, dist_th) 
                for class_name in self.cfg.class_names 
                for dist_th in self.cfg.dist_ths]
        
        # Create progress bar first
        pbar = tqdm(total=len(tasks), desc="Accumulating metrics")
        
        def accumulate_with_progress(args):
            result = accumulate_metrics(*args)
            pbar.update(1)
            return result

        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
            list(executor.map(accumulate_with_progress, tasks))
        
        pbar.close()

        # Calculate final metrics
        metrics = DetectionMetrics(self.cfg)
        self._calculate_metrics(metrics, metric_data_list)
        
        # Calculate distance-based metrics
        for dist_range, dist_md_list in distance_metrics.items():
            self._calculate_metrics(distance_metrics[dist_range], dist_md_list)

        # Calculate point-based metrics
        for point_range, point_md_list in point_metrics.items():
            self._calculate_metrics(point_metrics[point_range], point_md_list)

        # Calculate visibility-based metrics
        for vis_range, vis_md_list in visibility_metrics.items():
            self._calculate_metrics(visibility_metrics[vis_range], vis_md_list)

        metrics.add_runtime(time.time() - start_time)
        
        return metrics, metric_data_list, distance_metrics, point_metrics, visibility_metrics

    def _calculate_metrics(self, metrics: Union[DetectionMetrics, ExtendedDetectionMetrics], metric_data_list: DetectionMetricDataList):
        """Optimized metric calculation"""
        # Pre-compute class metrics in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            class_futures = {
                class_name: executor.submit(self._calculate_class_metrics, 
                                         class_name, 
                                         metrics, 
                                         metric_data_list)
                for class_name in self.cfg.class_names
            }
            
            # Wait for all calculations to complete
            for future in tqdm(class_futures.values(), 
                             desc="Calculating class metrics",
                             disable=not self.verbose):
                future.result()

    def _calculate_class_metrics(self, class_name, metrics, metric_data_list):
        """Calculate metrics for a single class"""
        # Calculate APs
        for dist_th in self.cfg.dist_ths:
            # Handle both DetectionMetricDataList and ExtendedDetectionMetrics cases
            if isinstance(metric_data_list, DetectionMetricDataList):
                key = (class_name, float(dist_th))
                if key in metric_data_list.md:
                    metric_data = metric_data_list.md[key]
                    ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                    metrics.add_label_ap(class_name, dist_th, ap)
                    
                    # Add average recall calculation for the max distance threshold
                    if dist_th == max(self.cfg.dist_ths):
                        if isinstance(metrics, ExtendedDetectionMetrics):
                            # Calculate average recall across all recall points
                            avg_recall = float(np.mean(metric_data.recall[:metric_data.max_recall_ind + 1]))
                            metrics.add_label_recall(class_name, avg_recall)

            # Calculate TP metrics
            if dist_th == self.cfg.dist_th_tp:  # Only calculate TP metrics once
                for metric_name in TP_METRICS:
                    if isinstance(metric_data_list, DetectionMetricDataList):
                        key = (class_name, float(self.cfg.dist_th_tp))
                        if key in metric_data_list.md:
                            metric_data = metric_data_list.md[key]
                            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                                tp = np.nan
                            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                                tp = np.nan
                            else:
                                tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                            metrics.add_label_tp(class_name, metric_name, tp)

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
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

        # Run evaluation
        metrics, metric_data_list, distance_metrics, point_metrics, visibility_metrics = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        metrics_summary['distance_metrics'] = {
            dist_range: dist_metrics.serialize() 
            for dist_range, dist_metrics in distance_metrics.items()
        }
        metrics_summary['point_metrics'] = {
            point_range: point_metrics.serialize() 
            for point_range, point_metrics in point_metrics.items()
        }
        metrics_summary['visibility_metrics'] = {
            vis_range: vis_metrics.serialize() 
            for vis_range, vis_metrics in visibility_metrics.items()
        }
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics with total predictions
        total_predictions = sum(len(boxes) for boxes in self.pred_boxes.boxes.values())
        print(f'Total predictions: {total_predictions}')
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err']))

        # Count boxes in different ranges
        distance_stats = self._count_distance_range_boxes()
        point_stats = self._count_point_range_boxes()

        # Print distance-based metrics and statistics with percentages
        print('\nDistance-based evaluation results:')
        for dist_range, dist_metrics in metrics_summary['distance_metrics'].items():
            stats = distance_stats[dist_range]
            pred_percentage = (stats["pred"] / total_predictions) * 100 if total_predictions > 0 else 0
            print(f'{dist_range}: GT={stats["gt"]}, Pred={stats["pred"]} ({pred_percentage:.1f}% of total), '
                  f'mAP={dist_metrics["mean_ap"]:.4f}, NDS={dist_metrics["nd_score"]:.4f}, '
                  f'AR={dist_metrics["mean_ar"]:.4f}')

        # Print point-based metrics and statistics with percentages
        print('\nPoint-based evaluation results:')
        for point_range, point_metrics in metrics_summary['point_metrics'].items():
            stats = point_stats[point_range]
            pred_percentage = (stats["pred"] / total_predictions) * 100 if total_predictions > 0 else 0
            print(f'{point_range}: GT={stats["gt"]}, Pred={stats["pred"]} ({pred_percentage:.1f}% of total), '
                  f'mAP={point_metrics["mean_ap"]:.4f}, NDS={point_metrics["nd_score"]:.4f}, '
                  f'AR={point_metrics["mean_ar"]:.4f}')
            
        # Print visibility-based metrics and statistics with prediction counts
        print('\nVisibility-based evaluation results:')
        visibility_stats = self._count_visibility_boxes()  # Add this helper method
        for vis_range, vis_metrics in metrics_summary['visibility_metrics'].items():
            stats = visibility_stats[vis_range]
            pred_percentage = (stats["pred"] / total_predictions) * 100 if total_predictions > 0 else 0
            print(f'{vis_range}: GT={stats["gt"]}, Pred={stats["pred"]} ({pred_percentage:.1f}% of total), '
                  f'mAP={vis_metrics["mean_ap"]:.4f}, NDS={vis_metrics["nd_score"]:.4f}, '
                  f'AR={vis_metrics["mean_ar"]:.4f}')

        return metrics_summary

    def _count_distance_range_boxes(self):
        """Count number of boxes in each distance range"""
        stats = {}
        for (min_d, max_d) in self.distance_ranges:
            range_key = f"{min_d}_{max_d}m"
            masks = self.distance_masks[range_key]
            
            gt_count = 0
            pred_count = 0
            for sample_token in self.sample_tokens:
                gt_mask = masks[sample_token]['gt']
                pred_mask = masks[sample_token]['pred']
                
                gt_count += np.sum(gt_mask)
                pred_count += np.sum(pred_mask)
            
            stats[range_key] = {
                "gt": int(gt_count),
                "pred": int(pred_count)
            }
        
        return stats

    def _count_point_range_boxes(self):
        """Count number of boxes in each point range"""
        stats = {}
        for point_range, masks in self.point_masks.items():
            gt_count = 0
            pred_count = 0
            for sample_token in self.sample_tokens:
                gt_mask = masks[sample_token]['gt']
                pred_mask = masks[sample_token]['pred']
                
                gt_count += np.sum(gt_mask)
                pred_count += np.sum(pred_mask)
            
            stats[point_range] = {
                "gt": int(gt_count),
                "pred": int(pred_count)
            }
        
        return stats

    def _add_point_counts_to_predictions(self):
        """Compute the number of points in each predicted box."""
        if self.verbose:
            print("Computing point counts for all prediction boxes...")
        
        # Process each sample
        for sample_token in tqdm(self.pred_boxes.sample_tokens, disable=not self.verbose):
            # Get sample data
            sample = self.nusc.get('sample', sample_token)
            sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            
            # Load point cloud
            lidar_path = self.nusc.get_sample_data_path(sample_data['token'])
            pc = LidarPointCloud.from_file(lidar_path)
            
            # Transform points to global coordinates
            cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            
            # First transform to ego vehicle coord system
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pc.translate(np.array(cs_record['translation']))
            
            # Then transform to global coord system
            pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
            pc.translate(np.array(pose_record['translation']))
            
            # For each box in the sample
            for box in self.pred_boxes[sample_token]:
                # Convert DetectionBox to Box
                box3d = Box(
                    center=box.translation,
                    size=box.size,
                    orientation=Quaternion(box.rotation)
                )
                # Get points inside box
                mask = points_in_box(box3d, pc.points[:3, :])
                box.num_pts = int(mask.sum())

    def _count_visibility_boxes(self):
        """Count number of boxes in each visibility range"""
        stats = {}
        for vis_range, masks in self.visibility_masks.items():
            gt_count = 0
            pred_count = 0
            for sample_token in self.sample_tokens:
                gt_mask = masks[sample_token]['gt']
                pred_mask = masks[sample_token]['pred']
                
                gt_count += np.sum(gt_mask)
                pred_count += np.sum(pred_mask)
            
            stats[vis_range] = {
                "gt": int(gt_count),
                "pred": int(pred_count)
            }
        
        return stats

from typing import Callable
from nuscenes.eval.detection.data_classes import DetectionMetricData
def accumulate_test(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    import numpy as np
    from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    print(class_name, dist_th, "npos=", npos, "tp=", np.sum(tp), "fp=", np.sum(fp))
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])
