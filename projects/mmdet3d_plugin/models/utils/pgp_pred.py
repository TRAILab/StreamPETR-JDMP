import os
import pickle
import numpy as np
from typing import Dict, Tuple, Union, List
from pyquaternion import Quaternion
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial.distance import cdist
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

class TrajectoryPredictor:
    def __init__(self,
                 nusc_dataroot) -> None:
        self.dataroot = nusc_dataroot
        nusc = NuScenes(version='v1.0-trainval', dataroot=self.dataroot, verbose=True)
        self.nusc_tables = {'sample': nusc.sample, 'scene': nusc.scene, 'log': nusc.log}
        self.nusc_token2ind = {'sample': nusc._token2ind['sample'], 
                               'scene': nusc._token2ind['scene'],
                               'log': nusc._token2ind['log']}
        del nusc
        map_locs = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.maps = {i: NuScenesMap(map_name=i, dataroot=self.dataroot) for i in map_locs}
        self.polyline_resolution = 1
        self.polyline_length = 20
        self.max_nodes = 256
        self.max_nbr_nodes = 14
        self.radius = 100
        self.feat_size = 6
            
    def get_map_representation(self, target_agent, sample_token, plot=False):
        # Get map elements around target agent
        global_pose = (target_agent[0], target_agent[1], target_agent[-1])
        nusc_map = self.maps[self.sample_token_to_map_location(sample_token)]
        lanes = self.get_lanes_around_agent(global_pose, nusc_map, self.radius)
        polygons = self.get_polygons_around_agent(global_pose, nusc_map, self.radius, ['stop_line', 'ped_crossing'])

        # Generate map representation
        lane_node_feats, lane_ids = self.get_lane_node_feats(global_pose, lanes, polygons)
        lane_node_feats, lane_ids = self.topk_lane_poses(lane_node_feats, lane_ids, self.max_nodes)
        e_succ = self.get_successor_edges(lane_ids, nusc_map)        
        lane_node_feats = self.add_boundary_flag(e_succ, lane_node_feats)
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, self.feat_size))]
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, self.feat_size)
        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
        }

        # Debugging plots
        if plot:
            text_pose = '_'.join(map(lambda x: str(round(x)), global_pose))
            out_path = '/proj/output/viz/map_debug_new/sample_'+sample_token+'_map_encoder_map_patch_'+text_pose+'.png'
            self.plot_map_representation(global_pose, nusc_map, out_path)

        return map_representation

    def sample_token_to_map_location(self, sample_token):
        sample_ind = self.nusc_token2ind['sample'][sample_token]
        scene_token = self.nusc_tables['sample'][sample_ind]['scene_token']
        scene_ind = self.nusc_token2ind['scene'][scene_token]
        log_token = self.nusc_tables['scene'][scene_ind]['log_token']
        log_ind = self.nusc_token2ind['log'][log_token]
        log_location = self.nusc_tables['log'][log_ind]['location']

        return log_location

    def get_lanes_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap, radius) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return lanes: Dictionary of lane polylines
        """
        x, y, _ = global_pose
        lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lane_polylines = map_api.discretize_lanes(lanes['lane'], self.polyline_resolution)
        lane_connector_polylines = map_api.discretize_lanes(lanes['lane_connector'], self.polyline_resolution)

        return {
            'lanes': lane_polylines,
            'lane_connectors': lane_connector_polylines
        }

    def get_polygons_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap, radius, 
                                  layer_names = ['stop_line', 'ped_crossing']) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
        """
        x, y, _ = global_pose
        record_tokens = map_api.get_records_in_radius(x, y, radius, layer_names)
        polygons = {k: [] for k in record_tokens.keys()}
        for k, v in record_tokens.items():
            for record_token in v:
                polygon_token = map_api.get(k, record_token)['polygon_token']
                polygons[k].append(map_api.extract_polygon(polygon_token))

        return polygons

    def get_lane_node_feats(self, origin: Tuple, lanes: Dict[str, List[Tuple]],
                            polygons: Dict[str, List[Polygon]]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generates vector HD map representation in the agent centric frame of reference
        :param origin: (x, y, yaw) of target agent in global co-ordinates
        :param lanes: lane centerline poses in global co-ordinates
        :param polygons: stop-line and cross-walk polygons in global co-ordinates
        :return:
        """
        lanes = {**lanes['lanes'], **lanes['lane_connectors']}
        # Convert lanes to list
        lane_ids = [k for k, v in lanes.items()]
        lanes = [v for k, v in lanes.items()]
        # Get flags indicating whether a lane lies on stop lines or crosswalks
        lane_flags = self.get_lane_flags(lanes, polygons)
        # Convert lane polylines to local coordinates:
        lanes = [np.asarray([self.global_to_local(origin, pose) for pose in lane]) for lane in lanes]
        # Concatenate lane poses and lane flags
        lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]
        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(lane_node_feats, self.polyline_length, lane_ids)

        return lane_node_feats, lane_node_ids

    def get_lane_flags(self, lanes: List[List[Tuple]], polygons: Dict[str, List[Polygon]]) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """
        lane_flags = [np.zeros((len(lane), len(polygons.keys()))) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                for n, k in enumerate(polygons.keys()):
                    polygon_list = polygons[k]
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][n] = 1
                            break

        return lane_flags
    
    def global_to_local(self, origin: Tuple, global_pose: Tuple) -> Tuple:
        """
        Converts pose in global co-ordinates to local co-ordinates.
        :param origin: (x, y, yaw) of origin in global co-ordinates
        :param global_pose: (x, y, yaw) in global co-ordinates
        :return local_pose: (x, y, yaw) in local co-ordinates
        """
        # Unpack
        global_x, global_y, global_yaw = global_pose
        origin_x, origin_y, origin_yaw = origin
        # Translate
        local_x = global_x - origin_x
        local_y = global_y - origin_y
        # Rotate
        local_yaw = np.arctan2(-np.sin(global_yaw-origin_yaw), np.cos(global_yaw-origin_yaw))
        r = np.asarray([[np.cos(origin_yaw), np.sin(origin_yaw)],
                        [-np.sin(origin_yaw), np.cos(origin_yaw)]])
        local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())
        local_pose = (local_x, local_y, local_yaw)

        return local_pose
    
    def split_lanes(self, lanes: List[np.ndarray], max_len: int, lane_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses: (n+1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    def topk_lane_poses(self, pose_set: List[np.ndarray],
                                    ids: List[str] = None, topk = 128) -> Union[List[np.ndarray],
                                                                    Tuple[List[np.ndarray], List[str]]]:
        """
        Keep topk lanes based on distnace in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []
        # Sort the pose set based on distance from the target agent
        pose_set.sort(key=lambda pose: np.linalg.norm(pose[:2]))
        # Keep only the top k poses
        updated_pose_set = pose_set[:topk]
        # Update the corresponding ids if provided
        if ids is not None:
            updated_ids = ids[:topk]
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set

    def get_successor_edges(self, lane_ids: List[str], map_api: NuScenesMap) -> List[List[int]]:
        """
        Returns successor edge list for each node
        """
        e_succ = []
        for node_id, lane_id in enumerate(lane_ids):
            e_succ_node = []
            if node_id + 1 < len(lane_ids) and lane_id == lane_ids[node_id + 1]:
                e_succ_node.append(node_id + 1)
            else:
                outgoing_lane_ids = map_api.get_outgoing_lane_ids(lane_id)
                for outgoing_id in outgoing_lane_ids:
                    if outgoing_id in lane_ids:
                        e_succ_node.append(lane_ids.index(outgoing_id))

            e_succ.append(e_succ_node)

        return e_succ

    def add_boundary_flag(self, e_succ: List[List[int]], lane_node_feats: np.ndarray):
        """
        Adds a binary flag to lane node features indicating whether the lane node has any successors.
        Serves as an indicator for boundary nodes.
        """
        for n, lane_node_feat_array in enumerate(lane_node_feats):
            flag = 1 if len(e_succ[n]) == 0 else 0
            lane_node_feats[n] = np.concatenate((lane_node_feat_array, flag * np.ones((len(lane_node_feat_array), 1))),
                                                axis=1)

        return lane_node_feats
    
    def list_to_tensor(self, feat_list: List[np.ndarray], max_num: int, max_len: int,
                    feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = feats
            mask_array[n, :len(feats), :] = 0

        return feat_array, mask_array

    def plot_map_representation(self, global_pose, nusc_map, file_name):
        # Plot map patch using api
        import matplotlib.pyplot as plt
        my_patch = (global_pose[0]-self.radius, global_pose[1]-self.radius, 
                    global_pose[0]+self.radius, global_pose[1]+self.radius)
        nusc_map.render_map_patch(my_patch, nusc_map.non_geometric_layers, figsize=(10, 10))
        plt.scatter(global_pose[0], global_pose[1], c='red', s=10)
        plt.arrow(global_pose[0], global_pose[1], 5*np.cos(global_pose[2]), 5*np.sin(global_pose[2]), 
                    color='red', width=0.5)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()
        # Plot map patch manually for extracted features
        # plt.figure()
        # # Plot polygons
        # layer_names = ['stop_line', 'ped_crossing', 'walkway']
        # poly_colors = ['red', 'orange', 'blue']
        # polygons = self.get_polygons_around_agent(global_pose, nusc_map, self.radius, layer_names) 
        # for n, k in enumerate(polygons.keys()):
        #     for polygon in polygons[k]:
        #         x_exterior, y_exterior = polygon.exterior.xy
        #         plt.fill(x_exterior, y_exterior, alpha=0.5, edgecolor=poly_colors[n], facecolor=poly_colors[n])
        #         for interior in polygon.interiors:
        #             x_interior, y_interior = interior.xy
        #             plt.fill(x_interior, y_interior, alpha=1, edgecolor=poly_colors[n], facecolor='white')
        # # Plot lanes
        # lanes = self.get_lanes_around_agent(global_pose, nusc_map, self.radius)
        # lanes = {**lanes['lanes'], **lanes['lane_connectors']}
        # lanes = [v for k, v in lanes.items()]
        # for lane in lanes:
        #     lane = np.array(lane)
        #     plt.plot(lane[:, 0], lane[:, 1], 'green', linewidth=0.5)
        # # Save figure
        # plt.savefig(file_name, dpi=300)
        # plt.close()