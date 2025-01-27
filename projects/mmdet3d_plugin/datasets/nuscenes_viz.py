import os
import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import warnings

class NuScenesVisualizer:
    def __init__(self):
        self.category_mapping = {
            'barrier': 'movable_object.barrier',
            'traffic_cone': 'movable_object.barrier',
            'bicycle': 'vehicle.bicycle',
            'motorcycle': 'vehicle.bicycle',
            'bus': 'vehicle.car',
            'trailer': 'vehicle.car',
            'construction_vehicle': 'vehicle.car',
            'truck': 'vehicle.car',
            'car': 'vehicle.car',
            'pedestrian': 'human.pedestrian.adult',
        }
        self.legend_nusc_categories = ['vehicle.car', 'vehicle.bicycle', 'human.pedestrian.adult', 'movable_object.barrier']
        self.legend_names = ['vehicle', 'bicycle', 'pedestrian', 'barrier']
        # self.ignore_classes = ["barrier", "traffic_cone"]
        self.ignore_classes = []

    def render_boxes(self, nusc: NuScenes, token, dets, for_preds, for_scores, for_gts, out_path, axes_limit=50,
                     eval_filter=True, with_gt_boxes=True, with_map=True,
                     with_det_boxes=True, with_forecast=True):
        legend_fontsize = 'large'
        plot_linewidth = 1.5
        score_thresh = 0.2
        
        lidar_sample_token = nusc.get('sample', token)['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', lidar_sample_token)
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        ref_sd_token = sample_rec['data']['LIDAR_TOP']

        ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_translation = ego_pose['translation']
        ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
        ego_quat = Quaternion(scalar=np.cos(ego_yaw / 2),
                              vector=[0, 0, np.sin(ego_yaw / 2)])
        
        if eval_filter:
            cfg = det_configs("detection_cvpr_2019")
            max_dist = cfg.class_range

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        if with_map:
            nusc.explorer.render_ego_centric_map(
                sample_data_token=lidar_sample_token, axes_limit=axes_limit, ax=ax)

        ax.plot(0, 0, 'x', color='red')

        if with_gt_boxes:
            _, gt_boxes, _ = nusc.get_sample_data(ref_sd_token,
                                                       use_flat_vehicle_coordinates=True)
            num_pts = [nusc.get('sample_annotation', box.token)['num_lidar_pts']
                       for box in gt_boxes]
            if eval_filter:
                gt_boxes, _ = self.filter_boxes(gt_boxes, max_dist, None, num_pts)
            for box in gt_boxes:
                box.render(ax, view=np.eye(4), colors=('black', 'black', 'black'),
                           linewidth=plot_linewidth)
                # TODO: add gt future here


        if with_det_boxes:
            self.render_detection_boxes(nusc, dets, ax, ego_translation, ego_quat, score_thresh, plot_linewidth)

        if with_forecast:
            self.render_forecast_points(for_preds, for_scores, ax, score_thresh, plot_linewidth)

        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
        ax.set_aspect('equal')
        handles = self.get_legend_handles(nusc)
        ax.legend(handles=handles, fontsize=legend_fontsize, loc='upper right', frameon=True)
        # Add colorbar for motion forecast
        norm = plt.Normalize(0, 12)
        sm = plt.cm.ScalarMappable(cmap='Blues_r', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Motion Forecast (s)', fontsize=legend_fontsize)

        if out_path:
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.close()

    def render_forecast_points(self, for_preds, for_scores, ax, threshold, linewidth):
        for pred, score in zip(for_preds, for_scores):
            if score < threshold:
                continue  # Skip rendering if the score is below the threshold
            pred_array = np.array(pred)
            rotation_matrix = np.array([[0, -1], [1, 0]])
            pred_array[:, :, :2] = np.dot(pred_array[:, :, :2], rotation_matrix)
            pred_array[:, :, 0] += 1
            if np.linalg.norm(pred_array[0,0]) > 50: #TODO: add class based filter
                continue
            dense_pred_array = self.interpolate_forecast_points(pred_array)
            color_intensity = [1 - (j / dense_pred_array.shape[1])
                            for j in range(dense_pred_array.shape[1])]
            self.colored_line(dense_pred_array[0, :, 0], dense_pred_array[0, :, 1],
                            color_intensity, ax, cmap='Blues', linewidth=linewidth * 2)

    def render_detection_boxes(self, nusc, dets, ax, ego_translation, ego_quat, score_thresh, linewidth):
        for det in dets:
            if det['detection_score'] < score_thresh or det['detection_name'] in self.ignore_classes:
                continue
            color = np.array(
                nusc.colormap[self.category_mapping[det['detection_name']]]) / 255.0
            box = Box(det['translation'], det['size'], Quaternion(det['rotation']),
                      name=det['detection_name'])
            box.translate(-np.array(ego_translation))
            box.rotate(ego_quat.inverse)
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=linewidth)

    @staticmethod
    def interpolate_forecast_points(pred_array):
        dense_pred_array = np.zeros((pred_array.shape[0], pred_array.shape[1] * 2 - 1, pred_array.shape[2]))
        for j in range(pred_array.shape[1] - 1):
            dense_pred_array[:, 2 * j] = pred_array[:, j]
            dense_pred_array[:, 2 * j + 1] = (pred_array[:, j] + pred_array[:, j + 1]) / 2
        dense_pred_array[:, -1] = pred_array[:, -1]
        return dense_pred_array

    @staticmethod
    def colored_line(x, y, c, ax, **lc_kwargs):
        segments = NuScenesVisualizer.create_segments(x, y)
        lc = LineCollection(segments, **lc_kwargs)
        lc.set_array(c)
        ax.add_collection(lc)

    @staticmethod
    def create_segments(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
        y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
        coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
        coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
        coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
        return np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    def filter_boxes(self, boxes, max_dist, track_histories=None, num_pts=None):
        gt_names_to_tracking_names = {
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'vehicle.car': 'car',
            'vehicle.motorcycle': 'motorcycle',
            'vehicle.bicycle': 'bicycle',
            'vehicle.truck': 'truck',
        }
        filtered_boxes = []
        filtered_histories = []
        for i, box in enumerate(boxes):
            name = gt_names_to_tracking_names.get(box.name, 'ignore')
            if name == 'ignore':
                continue
            if num_pts and num_pts[i] == 0:
                continue
            if np.sqrt(np.sum(box.center[:2] ** 2)) >= max_dist.get(name, float('inf')):
                continue
            filtered_boxes.append(box)
            if track_histories:
                filtered_histories.append(track_histories[i])
        return filtered_boxes, filtered_histories

    def get_legend_handles(self, nusc):
        handles = [mpatches.Patch(color='grey', label='Driveable Area'),
                   Line2D([0], [0], label='Ground Truth', color='k'),]
        for (name, cat) in zip(self.legend_names, self.legend_nusc_categories):
            handles.append(Line2D([0], [0], label=name, color=np.array(nusc.colormap[cat])/255.0))
        return handles
    
