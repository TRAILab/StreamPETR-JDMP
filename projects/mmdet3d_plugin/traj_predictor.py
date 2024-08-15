import yaml
import numpy as np
from typing import Dict, Tuple, Union, List
from pyquaternion import Quaternion
import os
import pickle
import torch
import imageio
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from scipy.spatial.distance import cdist

# from train_eval.initialization import initialize_prediction_model

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.input_representation.static_layers import correct_yaw

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

from nuscenes.prediction import PredictHelper
# from datasets.interface import TrajectoryDataset
# from datasets.nuScenes.nuScenes_raster import NuScenesRaster
# from datasets.nuScenes.nuScenes_vector import NuScenesVector
# from datasets.nuScenes.nuScenes_graphs import NuScenesGraphs

# Import models
from projects.mmdet3d_plugin.forecast_models.model import PredictionModel
from projects.mmdet3d_plugin.forecast_models.encoders.raster_encoder import RasterEncoder
from projects.mmdet3d_plugin.forecast_models.encoders.polyline_subgraph import PolylineSubgraphs
from projects.mmdet3d_plugin.forecast_models.encoders.pgp_encoder import PGPEncoder
from projects.mmdet3d_plugin.forecast_models.aggregators.concat import Concat
from projects.mmdet3d_plugin.forecast_models.aggregators.global_attention import GlobalAttention
from projects.mmdet3d_plugin.forecast_models.aggregators.goal_conditioned import GoalConditioned
from projects.mmdet3d_plugin.forecast_models.aggregators.pgp import PGP
from projects.mmdet3d_plugin.forecast_models.decoders.mtp import MTP
from projects.mmdet3d_plugin.forecast_models.decoders.multipath import Multipath
from projects.mmdet3d_plugin.forecast_models.decoders.covernet import CoverNet
from projects.mmdet3d_plugin.forecast_models.decoders.lvm import LVM

# Import metrics
from projects.mmdet3d_plugin.metrics.mtp_loss import MTPLoss
from projects.mmdet3d_plugin.metrics.min_ade import MinADEK
from projects.mmdet3d_plugin.metrics.min_fde import MinFDEK
from projects.mmdet3d_plugin.metrics.miss_rate import MissRateK
from projects.mmdet3d_plugin.metrics.covernet_loss import CoverNetLoss
from projects.mmdet3d_plugin.metrics.pi_bc import PiBehaviorCloning
from projects.mmdet3d_plugin.metrics.goal_pred_nll import GoalPredictionNLL

# Import helper functions
from projects.mmdet3d_plugin.forecast_utils.forecast_helper import *

from typing import List, Dict, Union
import torch.nn as nn


class TrajectoryPredictor:
    def __init__(self,
                 nusc_dataroot,
                 config_path,
                 checkpoint_path,
                 stats_path,
                 history_length,
                 traversal_horizon,) -> None:
        self.dataroot = nusc_dataroot
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.stats_path = stats_path
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.dataroot, verbose=True)
        self.predict_helper = PredictHelper(self.nusc)
        self.history_length = history_length
        self.map_extent = [-50, 50, -20, 80]
        map_locs = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.maps = {i: NuScenesMap(map_name=i, dataroot=self.predict_helper.data.dataroot) for i in map_locs}
        self.polyline_resolution = 1
        self.polyline_length = 20
        self.traversal_horizon = traversal_horizon
        self.load_config(config_path)
        self.stats = self.load_stats()
        self.max_nodes = self.stats['num_lane_nodes']
        self.max_nbr_nodes = self.stats['max_nbr_nodes']
        self.radius = max(self.map_extent)
        self.model = self.initialize_prediction_model(self.config['encoder_type'], self.config['aggregator_type'], self.config['decoder_type'],
                                        self.config['encoder_args'], self.config['aggregator_args'], self.config['decoder_args'])
        self.model.to('cuda')
        self.model.training = False
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.map_bounds = {
            'singapore-onenorth': [24.945872116655877, 384.71099361022164, 1242.8193836005826, 1947.158585268475],
            'singapore-hollandvillage':[413.6408728325073, 808.5324105448427, 2569.6650879605004, 2893.832714678619],
            'singapore-queenstown':[296.95191633850794, 829.2811651573319, 3094.273578911854, 3457.4887792210247],
            'boston-seaport':[141.3897547891008, 106.05110587259979, 2957.6066195724247, 2028.8744920767883]
        }
        
        self.map_token_to_name = {
            '53992ee3023e5494b90c316c183be829': 'singapore-onenorth',
            '37819e65e09e5547b8a3ceaefba56bb2': 'singapore-hollandvillage',
            '93406b464a165eaba6d9de76ca09f5da': 'singapore-queenstown',
            '36092f0b03a857c6a3403e25b4b7aab3': 'boston-seaport'
        }

        # Debug visualizations
        self.debug_nusc_map = None
        self.debug_global_pose = None
        self.debug_lanes = None
        self.debug_polygons = None
        self.debug_e_succ = None
        self.debug_e_prox = None
        self.debug_lane_ids = None
        self.debug_target_agent = None
        
    def load_config(self, config_path):
        with open(config_path, 'r') as yaml_file:
            cfg = yaml.safe_load(yaml_file)
        self.config = cfg
    
    def load_stats(self) -> Dict[str, int]:
        """
        Function to load dataset statistics like max surrounding agents, max nodes, max edges etc.
        """
        filename = os.path.join(self.stats_path, 'stats.pickle')
        if not os.path.isfile(filename):
            raise Exception('Could not find dataset statistics. Please run the dataset in compute_stats mode')

        with open(filename, 'rb') as handle:
            stats = pickle.load(handle)

        return stats
    
    def initialize_prediction_model(self, encoder_type: str, aggregator_type: str, decoder_type: str,
                                encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
        """
        Helper function to initialize appropriate encoder, aggegator and decoder models
        """
        encoder = self.initialize_encoder(encoder_type, encoder_args)
        aggregator = self.initialize_aggregator(aggregator_type, aggregator_args)
        decoder = self.initialize_decoder(decoder_type, decoder_args)
        model = self.PredictionModel(encoder, aggregator, decoder)

        return model
    
    def initialize_encoder(self, encoder_type: str, encoder_args: Dict):
        """
        Initialize appropriate encoder by type.
        """
        # TODO: Update as we add more encoder types
        encoder_mapping = {
            'raster_encoder': RasterEncoder,
            'polyline_subgraphs': PolylineSubgraphs,
            'pgp_encoder': PGPEncoder
        }

        return encoder_mapping[encoder_type](encoder_args)

    def initialize_aggregator(self, aggregator_type: str, aggregator_args: Union[Dict, None]):
        """
        Initialize appropriate aggregator by type.
        """
        # TODO: Update as we add more aggregator types
        aggregator_mapping = {
            'concat': Concat,
            'global_attention': GlobalAttention,
            'gc': GoalConditioned,
            'pgp': PGP
        }

        if aggregator_args:
            return aggregator_mapping[aggregator_type](aggregator_args)
        else:
            return aggregator_mapping[aggregator_type]()

    def initialize_decoder(self, decoder_type: str, decoder_args: Dict):
        """
        Initialize appropriate decoder by type.
        """
        # TODO: Update as we add more decoder types
        decoder_mapping = {
            'mtp': MTP,
            'multipath': Multipath,
            'covernet': CoverNet,
            'lvm': LVM
        }

        return decoder_mapping[decoder_type](decoder_args)
    
    class PredictionModel(nn.Module):
        """
        Single-agent prediction model
        """
        def __init__(self, encoder, aggregator, decoder):

            """
            Initializes model for single-agent trajectory prediction
            """
            super().__init__()
            self.encoder = encoder
            self.aggregator = aggregator
            self.decoder = decoder

        def forward(self, inputs: Dict) -> Union[torch.Tensor, Dict]:
            """
            Forward pass for prediction model
            :param inputs: Dictionary with
                'target_agent_representation': target agent history
                'surrounding_agent_representation': surrounding agent history
                'map_representation': HD map representation
            :return outputs: K Predicted trajectories and/or their probabilities
            """
            encodings = self.encoder(inputs)
            agg_encoding = self.aggregator(encodings)
            outputs = self.decoder(agg_encoding)

            return outputs
    
    def sample_token_to_map_location(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        return log['location']
    
    def quaternion_to_yaw(self, w, x, y, z):
        # Calculate yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def get_agent_node_masks(self, hd_map: Dict, agents: Dict, dist_thresh=10) -> Dict:
        """
        Returns key/val masks for agent-node attention layers. All agents except those within a distance threshold of
        the lane node are masked. The idea is to incorporate local agent context at each lane node.
        """

        lane_node_feats = hd_map['lane_node_feats']
        lane_node_masks = hd_map['lane_node_masks']
        vehicle_feats = agents['vehicles']
        vehicle_masks = agents['vehicle_masks']
        ped_feats = agents['pedestrians']
        ped_masks = agents['pedestrian_masks']

        vehicle_node_masks = np.ones((len(lane_node_feats), len(vehicle_feats)))
        ped_node_masks = np.ones((len(lane_node_feats), len(ped_feats)))

        for i, node_feat in enumerate(lane_node_feats):
            if (lane_node_masks[i] == 0).any():
                node_pose_idcs = np.where(lane_node_masks[i][:, 0] == 0)[0]
                node_locs = node_feat[node_pose_idcs, :2]

                for j, vehicle_feat in enumerate(vehicle_feats):
                    if (vehicle_masks[j] == 0).any():
                        vehicle_loc = vehicle_feat[-1, :2]
                        dist = np.min(np.linalg.norm(node_locs - vehicle_loc, axis=1))
                        if dist <= dist_thresh:
                            vehicle_node_masks[i, j] = 0

                for j, ped_feat in enumerate(ped_feats):
                    if (ped_masks[j] == 0).any():
                        ped_loc = ped_feat[-1, :2]
                        dist = np.min(np.linalg.norm(node_locs - ped_loc, axis=1))
                        if dist <= dist_thresh:
                            ped_node_masks[i, j] = 0

        agent_node_masks = {'vehicles': vehicle_node_masks, 'pedestrians': ped_node_masks}
        return agent_node_masks
    
    def visualize_map_with_target_agent(self, best_trajectory=None, ax=None, target_agent_history_states=None, plot_all_map_attributes=False, radius=None, name=None):
        if radius is None:
            patch_box = (self.debug_global_pose[0] - self.radius, self.debug_global_pose[1] - self.radius, self.debug_global_pose[0] + self.radius, self.debug_global_pose[1] + self.radius)
        else:
            patch_box = (self.debug_global_pose[0] - radius, self.debug_global_pose[1] - radius, self.debug_global_pose[0] + radius, self.debug_global_pose[1] + radius)
        layer_names = None if plot_all_map_attributes else []
        if ax is None:
            fig, ax = self.debug_nusc_map.render_map_patch(patch_box, figsize=(15, 15), layer_names=layer_names)
        # else: 
        #     self.debug_nusc_map.render_map_patch(patch_box, layer_names=layer_names)
        ax.set_aspect('equal', adjustable='box')
        # Plot lanes
        lane_node_positions = {}
        for lane_id, lane_points in self.debug_lanes['lanes'].items():
            lane_points = np.array(lane_points)
            ax.plot(lane_points[:, 0], lane_points[:, 1], color='yellow', linewidth=2, label='Lane' if lane_id == list(self.debug_lanes['lanes'].keys())[0] else "")
            lane_node_positions[lane_id] = lane_points

        # Plot lane connectors
        for lane_connector_id, lane_connector_points in self.debug_lanes['lane_connectors'].items():
            lane_connector_points = np.array(lane_connector_points)
            ax.plot(lane_connector_points[:, 0], lane_connector_points[:, 1], color='#a08679', linewidth=2, linestyle='dashed', label='Lane Connector' if lane_connector_id == list(self.debug_lanes['lane_connectors'].keys())[0] else "")
            lane_node_positions[lane_connector_id] = lane_connector_points

        # Plot polygons
        for polygon_type, polygon_list in self.debug_polygons.items():
            color = 'blue' if polygon_type == 'ped_crossing' else 'red'  # Use different colors for different types
            for polygon in polygon_list:
                patch = plt.Polygon(list(polygon.exterior.coords), color=color, alpha=0.5, label=polygon_type if polygon == polygon_list[0] else "")
                ax.add_patch(patch)

        # Draw an 'X' at the query point
        ax.plot(self.debug_global_pose[0], self.debug_global_pose[1], 'x', color='blue', markersize=12, mew=3)
        # colors = [plt.cm.cool(i/16) for i in range(17)]
        # colors = np.array(colors)
        # ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'b-', color=colors, linewidth=2)
        
        if best_trajectory is not None:
            # colors = [plt.cm.cool(i/15) for i in range(16)]
            colors = ['#2b0057', '#3a007a', '#4b009e', '#5c00c2', '#6e00e5', '#7f0ef1', '#901ef9', '#a42ffb', '#b741fd', '#c952ff', '#db66ff', '#eb79ff', '#f28eff', '#f7a3ff', '#fbb9ff', '#ffe0ff']
            for i in range(best_trajectory.shape[0]-1):
                # print('Plotting trajectory', i)
                ax.plot(best_trajectory[i:i+2, 0], best_trajectory[i:i+2, 1], 'b-', color=colors[i], linewidth=2)
        
        # Plot target agent history states
        # if target_agent_history_states is not None:
        #     # colors = ['magenta', 'purple', 'violet', 'blueviolet', 'blue']
        #     colors = ['#00FF00', '#00CC00', '#009900', '#006600', '#003300']
        #     # targent_agent_x = [state['translation'][0] for state in target_agent_history_states]
        #     # targent_agent_y = [state['translation'][1] for state in target_agent_history_states]
        #     targent_agent_x = [state[0] for state in target_agent_history_states]
        #     targent_agent_y = [state[1] for state in target_agent_history_states]
        #     target_agent_yaw = [state[2] for state in target_agent_history_states]
        #     for i in range(len(targent_agent_x)-1):
        #         ax.plot(targent_agent_x[i:i+2], targent_agent_y[i:i+2], color=colors[i], linewidth=3)
        #     # plt.plot(targent_agent_x, targent_agent_y, marker='o', linestyle='-', color='black', linewidth=3, label='Agent History')        
        
        #     # Plot the bounding box for the current state
        #     current_state = target_agent_history_states[-1]
        #     # cx, cy = current_state['translation'][0], current_state['translation'][1]
        #     cx, cy = current_state[0], current_state[1]
        #     # target_agent_yaw = self.quaternion_to_yaw(*current_state['rotation'])
        #     target_agent_yaw = current_state[2]
        #     # width, length = current_state['size'][0], current_state['size'][1]
        #     # rect = patches.Rectangle(
        #     #     (cx - width / 2, cy - length / 2), width, length,
        #     #     linewidth=2, edgecolor='r', facecolor='none'
        #     # )
        #     # t = plt.gca().transData
        #     # rot = patches.transforms.Affine2D().rotate_around(cx, cy, target_agent_yaw - np.pi / 2)
        #     # rect.set_transform(rot + t)
        #     # ax.add_patch(rect)
        
        # Get center of x and y axis
        x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        # Set new limits to be +/- 50 from the center
        ax.set_xlim(x_center - 50, x_center + 50)
        ax.set_ylim(y_center - 50, y_center + 50)
        
        # Plot successor edges in blue
        # for node_id, successors in enumerate(self.debug_e_succ):
        #     for succ_id in successors:
        #         start_point = lane_node_positions[self.debug_lane_ids[node_id]][-1]
        #         end_point = lane_node_positions[self.debug_lane_ids[succ_id]][0]
        #         ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green', linestyle='-.', linewidth=3, label='Successor Edge' if node_id == 0 and succ_id == 0 else "")

        # # Plot proximal edges in orange
        # for node_id, proximals in enumerate(self.debug_e_prox):
        #     for prox_id in proximals:
        #         start_point = lane_node_positions[self.debug_lane_ids[node_id]][-1]
        #         end_point = lane_node_positions[self.debug_lane_ids[prox_id]][0]
        #         ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='orange', linestyle='-.', linewidth=3, label='Proximal Edge' if node_id == 0 and prox_id == 0 else "")

        # Add dummy plot entries for legend
        # ax.plot([], [], color='green', linestyle='-.', label='Successor Edge')
        # ax.plot([], [], color='orange', linestyle='-.', label='Proximal Edge')

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.suptitle('Predicted Trajectories for All Valid Agents', fontsize=20, y=1.2)
        # plt.tight_layout()
        plt.savefig(f'trajectory_deviation_{name}.png')
        # print('Trajectory saved to trajectory.png')
        # breakpoint()
        return ax
    
    # Write a function, where given a coordinate, it will return the closest map bounds, if the point is outside the map bounds, it will still return the closest map bounds
    def get_closest_map_bounds(self, x, y):
        min_dist = 1e10
        closest_map = None
        for k, v in self.map_bounds.items():
            dist = (x - v[0])**2 + (y - v[1])**2
            if dist < min_dist:
                min_dist = dist
                closest_map = k
        return closest_map
    
    def get_map_representation(self, target_agent, sample_token):
        # yaw = quaternion_yaw(Quaternion(target_agent['rotation']))
        yaw = target_agent[-1]
        yaw = correct_yaw(yaw)
        global_pose = (target_agent[0], target_agent[1], yaw) # self.quaternion_to_yaw(*target_agent['rotation']))
        nusc_map = self.maps[self.sample_token_to_map_location(sample_token)]
        # nusc_map = self.maps[self.get_closest_map_bounds(target_agent[0], target_agent[1])]
        lanes = self.get_lanes_around_agent(global_pose, nusc_map, self.radius)
        polygons = self.get_polygons_around_agent(global_pose, nusc_map, self.radius)
        lane_node_feats, lane_ids = self.get_lane_node_feats(global_pose, lanes, polygons)
        lane_node_feats, lane_ids = self.discard_poses_outside_extent(lane_node_feats, lane_ids)
        e_succ = self.get_successor_edges(lane_ids, nusc_map)
        e_prox = self.get_proximal_edges(lane_node_feats, e_succ)
        # print(e_succ)
        # print(e_prox)
        
        # Debug visualizations
        self.debug_nusc_map = nusc_map
        self.debug_global_pose = global_pose
        self.debug_lanes = lanes
        self.debug_polygons = polygons
        self.debug_e_succ = e_succ
        self.debug_e_prox = e_prox
        self.debug_lane_ids = lane_ids
        self.debug_target_agent = target_agent
        
        lane_node_feats = self.add_boundary_flag(e_succ, lane_node_feats)
        # Add dummy node (0, 0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, 6))]
            e_succ = [[]]
            e_prox = [[]]


        num_nbrs = [len(e_succ[i]) + len(e_prox[i]) for i in range(len(e_succ))]
        max_nbrs = max(num_nbrs) if len(num_nbrs) > 0 else 0
        num_nodes = len(lane_node_feats)

        s_next, edge_type = self.get_edge_lookup(e_succ, e_prox, self.max_nodes, self.max_nbr_nodes)
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 6)
        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
            's_next': s_next,
            'edge_type': edge_type
        }
        return map_representation
    
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
    
    def get_polygons_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap, radius) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
        """
        x, y, _ = global_pose
        # radius = max(map_extent)
        record_tokens = map_api.get_records_in_radius(x, y, radius, ['stop_line', 'ped_crossing'])
        polygons = {k: [] for k in record_tokens.keys()}
        for k, v in record_tokens.items():
            for record_token in v:
                polygon_token = map_api.get(k, record_token)['polygon_token']
                polygons[k].append(map_api.extract_polygon(polygon_token))

        return polygons

    def get_edge_lookup(self, e_succ: List[List[int]], e_prox: List[List[int]], max_nodes, max_nbr_nodes):
        """
        Returns edge look up tables
        :param e_succ: Lists of successor edges for each node
        :param e_prox: Lists of proximal edges for each node
        :return:

        s_next: Look-up table mapping source node to destination node for each edge. Each row corresponds to
        a source node, with entries corresponding to destination nodes. Last entry is always a terminal edge to a goal
        state at that node. shape: [max_nodes, max_nbr_nodes + 1]. Last

        edge_type: Look-up table of the same shape as s_next containing integer values for edge types.
        {0: No edge exists, 1: successor edge, 2: proximal edge, 3: terminal edge}
        """

        s_next = np.zeros((max_nodes, max_nbr_nodes + 1))
        edge_type = np.zeros((max_nodes, max_nbr_nodes + 1), dtype=int)

        for src_node in range(len(e_succ)):
            nbr_idx = 0
            successors = e_succ[src_node]
            prox_nodes = e_prox[src_node]

            # Populate successor edges
            for successor in successors:
                s_next[src_node, nbr_idx] = successor
                edge_type[src_node, nbr_idx] = 1
                nbr_idx += 1

            # Populate proximal edges
            for prox_node in prox_nodes:
                s_next[src_node, nbr_idx] = prox_node
                edge_type[src_node, nbr_idx] = 2
                nbr_idx += 1

            # Populate terminal edge
            s_next[src_node, -1] = src_node + max_nodes
            edge_type[src_node, -1] = 3

        return s_next, edge_type

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
        global_yaw = correct_yaw(global_yaw)
        theta = np.arctan2(-np.sin(global_yaw-origin_yaw), np.cos(global_yaw-origin_yaw))

        r = np.asarray([[np.cos(np.pi/2 - origin_yaw), np.sin(np.pi/2 - origin_yaw)],
                        [-np.sin(np.pi/2 - origin_yaw), np.cos(np.pi/2 - origin_yaw)]])
        local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

        local_pose = (local_x, local_y, theta)

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
    
    def discard_poses_outside_extent(self, pose_set: List[np.ndarray],
                                    ids: List[str] = None) -> Union[List[np.ndarray],
                                                                    Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if self.map_extent[0] <= pose[0] <= self.map_extent[1] and \
                        self.map_extent[2] <= pose[1] <= self.map_extent[3]:
                    flag = True

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        if ids is not None:
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

    def get_proximal_edges(self, lane_node_feats: List[np.ndarray], e_succ: List[List[int]],
                            dist_thresh=4, yaw_thresh=np.pi/4) -> List[List[int]]:
        """
        Returns proximal edge list for each node
        """
        e_prox = [[] for _ in lane_node_feats]
        for src_node_id, src_node_feats in enumerate(lane_node_feats):
            for dest_node_id in range(src_node_id + 1, len(lane_node_feats)):
                if dest_node_id not in e_succ[src_node_id] and src_node_id not in e_succ[dest_node_id]:
                    dest_node_feats = lane_node_feats[dest_node_id]
                    pairwise_dist = cdist(src_node_feats[:, :2], dest_node_feats[:, :2])
                    min_dist = np.min(pairwise_dist)
                    if min_dist <= dist_thresh:
                        yaw_src = np.arctan2(np.mean(np.sin(src_node_feats[:, 2])),
                                                np.mean(np.cos(src_node_feats[:, 2])))
                        yaw_dest = np.arctan2(np.mean(np.sin(dest_node_feats[:, 2])),
                                                np.mean(np.cos(dest_node_feats[:, 2])))
                        yaw_diff = np.arctan2(np.sin(yaw_src-yaw_dest), np.cos(yaw_src-yaw_dest))
                        if np.absolute(yaw_diff) <= yaw_thresh:
                            e_prox[src_node_id].append(dest_node_id)
                            e_prox[dest_node_id].append(src_node_id)

        return e_prox
    
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
    
    def assign_pose_to_node(self, node_poses, query_pose, dist_thresh=5, yaw_thresh=np.pi/3, return_multiple=False):
        """
        Assigns a given agent pose to a lane node. Takes into account distance from the lane centerline as well as
        direction of motion.
        """
        dist_vals = []
        yaw_diffs = []

        for i in range(len(node_poses)):
            distances = np.linalg.norm(node_poses[i][:, :2] - query_pose[:2], axis=1)
            dist_vals.append(np.min(distances))
            idx = np.argmin(distances)
            yaw_lane = node_poses[i][idx, 2]
            yaw_query = query_pose[2]
            yaw_diffs.append(np.arctan2(np.sin(yaw_lane - yaw_query), np.cos(yaw_lane - yaw_query)))

        idcs_yaw = np.where(np.absolute(np.asarray(yaw_diffs)) <= yaw_thresh)[0]
        idcs_dist = np.where(np.asarray(dist_vals) <= dist_thresh)[0]
        idcs = np.intersect1d(idcs_dist, idcs_yaw)

        if len(idcs) > 0:
            if return_multiple:
                return idcs
            assigned_node_id = idcs[int(np.argmin(np.asarray(dist_vals)[idcs]))]
        else:
            assigned_node_id = np.argmin(np.asarray(dist_vals))
            if return_multiple:
                assigned_node_id = np.asarray([assigned_node_id])

        return assigned_node_id

    def get_initial_node(self, lane_graph: Dict) -> np.ndarray:
        """
        Returns initial node probabilities for initializing the graph traversal policy
        :param lane_graph: lane graph dictionary with lane node features and edge look-up tables
        """

        # Unpack lane node poses
        node_feats = lane_graph['lane_node_feats']
        node_feat_lens = np.sum(1 - lane_graph['lane_node_masks'][:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[:int(node_feat_lens[i]), :3])

        assigned_nodes = self.assign_pose_to_node(node_poses, np.asarray([0, 0, 0]), dist_thresh=3,
                                                    yaw_thresh=np.pi / 4, return_multiple=True)

        max_nodes = self.stats['num_lane_nodes']
        init_node = np.zeros(max_nodes)
        init_node[assigned_nodes] = 1/len(assigned_nodes)
        return init_node
    
    def propagate_trajectory_for_all_agents(self, tracked_objects, outputs):
        assert len(tracked_objects) > 0, "No tracked objects"
        # Select target agent
        valid_agents = find_all_valid_agents(outputs, tracked_objects)
        if len(valid_agents) == 0:
            # print('No valid agent found!')
            return
        
        num_agents = len(valid_agents)
        num_cols = min(num_agents, 3)
        num_rows = 1 if num_agents <= 3 else (num_agents + 2) // 3
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 10, num_rows * 10))
        
        for agent_idx, target_agent in enumerate(valid_agents):
            if num_agents == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[agent_idx % num_cols]
            else:
                ax = axes[agent_idx // num_cols, agent_idx % num_cols]
            # Get target agent representation
            target_agent_histories = tracked_objects[target_agent['tracking_id']][-5:]
            # Also get the history of the target agent from only the 6th frame separately for later yaw rate interpolation
            target_agent_histories_6th = tracked_objects[target_agent['tracking_id']][-6]
            target_agent_histories_7th = tracked_objects[target_agent['tracking_id']][-7]
            # oldest_velocity = math.sqrt(target_agent_histories_6th['velocity'][0] ** 2 + target_agent_histories_6th['velocity'][1] ** 2)
            oldest_yaw = quaternion_to_yaw(*target_agent_histories_6th['rotation'])
            target_agent_history_yaws = []
            assert len(target_agent_histories) == 5, "Target agent history length is not 5"
            target_agent_history_states = []
            
            # Calculate past velocity and acceleration
            velocity_histories = [0 for _ in range(5)]
            velocity_histories[0] = get_velocity(target_agent_histories[0]['translation'], target_agent_histories_6th['translation'], 0.5)
            velocity_histories[1] = get_velocity(target_agent_histories[1]['translation'], target_agent_histories[0]['translation'], 0.5)
            velocity_histories[2] = get_velocity(target_agent_histories[2]['translation'], target_agent_histories[1]['translation'], 0.5)
            velocity_histories[3] = get_velocity(target_agent_histories[3]['translation'], target_agent_histories[2]['translation'], 0.5)
            velocity_histories[4] = get_velocity(target_agent_histories[4]['translation'], target_agent_histories[3]['translation'], 0.5)
            velocity_histories = np.array(velocity_histories)
            
            # Also calculate the oldest velocity for acceleration estimation
            oldest_velocity = get_velocity(target_agent_histories_6th['translation'], target_agent_histories_7th['translation'], 0.5)
            
            # Then estimate the acceleration
            acceleration_histories = [0 for _ in range(5)]
            acceleration_histories[0] = (velocity_histories[0] - oldest_velocity) / 0.5
            acceleration_histories[1] = (velocity_histories[1] - velocity_histories[0]) / 0.5
            acceleration_histories[2] = (velocity_histories[2] - velocity_histories[1]) / 0.5
            acceleration_histories[3] = (velocity_histories[3] - velocity_histories[2]) / 0.5
            acceleration_histories[4] = (velocity_histories[4] - velocity_histories[3]) / 0.5
            acceleration_histories = np.array(acceleration_histories)
            
            for state in target_agent_histories:
                target_agent_state = [state['translation'][0], state['translation'][1], 0.0, 0.0, quaternion_to_yaw(*state['rotation'])]
                target_agent_history_states.append(target_agent_state)
                target_agent_history_yaws.append(quaternion_to_yaw(*state['rotation']))
                
            target_agent_history_yaws = np.array(target_agent_history_yaws) # Just for debugging purposes
            
            for i in range(len(target_agent_history_states)):
                target_agent_history_states[i][0] -= target_agent_history_states[-1][0]
                target_agent_history_states[i][1] -= target_agent_history_states[-1][1]
            target_agent_history_states = np.array(target_agent_history_states)
            # breakpoint()
            # Yaw to yaw rate interpolation
            target_agent_history_yaw_rates = np.zeros(5)
            target_agent_history_yaw_rates[4] = (angle_diff(target_agent_history_states[4, -1], target_agent_history_states[3, -1])) / 0.5
            target_agent_history_yaw_rates[3] = (angle_diff(target_agent_history_states[3, -1], target_agent_history_states[2, -1])) / 0.5
            target_agent_history_yaw_rates[2] = (angle_diff(target_agent_history_states[2, -1], target_agent_history_states[1, -1])) / 0.5
            target_agent_history_yaw_rates[1] = (angle_diff(target_agent_history_states[1, -1], target_agent_history_states[0, -1])) / 0.5
            target_agent_history_yaw_rates[0] = (angle_diff(target_agent_history_states[0, -1], oldest_yaw)) / 0.5
            target_agent_history_states[:, -1] = target_agent_history_yaw_rates
            
            target_agent_history_states[:, 2] = velocity_histories
            target_agent_history_states[:, 3] = acceleration_histories
            
            # Velocity to acceleration interpolation
            # target_agent_history_accelerations = np.zeros(5)
            # target_agent_history_accelerations[4] = (target_agent_history_states[4, -2] - target_agent_history_states[3, -2]) / 0.5
            # target_agent_history_accelerations[3] = (target_agent_history_states[3, -2] - target_agent_history_states[2, -2]) / 0.5
            # target_agent_history_accelerations[2] = (target_agent_history_states[2, -2] - target_agent_history_states[1, -2]) / 0.5
            # target_agent_history_accelerations[1] = (target_agent_history_states[1, -2] - target_agent_history_states[0, -2]) / 0.5
            # target_agent_history_accelerations[0] = (target_agent_history_states[0, -2] - oldest_velocity) / 0.5
            # target_agent_history_states[:, -2] = target_agent_history_accelerations
            
            # Transform target agent velocity from global frame to agent frame
            # rotation_matrix = rotation_matrix_2d(-quaternion_to_yaw(*target_agent['rotation']))
            # vx_global, vy_global = target_agent_history_states[-1, 3], target_agent_history_states[-1, 4]
            # target_agent_history_states[:, 3:5] = np.dot(rotation_matrix, target_agent_history_states[:, 3:5].T).T
            
            if len(target_agent_history_states) < 5:
                num_missing_rows = 5 - len(target_agent_history_states)
                missing_rows = np.zeros((num_missing_rows, target_agent_history_states.shape[1]))
                target_agent_history_states = np.concatenate((missing_rows, target_agent_history_states), axis=0)
            # target_agent_history_states = torch.from_numpy(target_agent_history_states).float()
            # target_agent_history_states = target_agent_history_states.unsqueeze(0)
            # print("Target agent history states: ", target_agent_history_states)
            
            # Get surrounding agents representation for vehicles
            surrounding_agents_vehicle = []
            surrounding_agents_vehicle_masks = []
            max_vehicles = 84
            for agent in outputs:
                if agent['tracking_id'] == target_agent['tracking_id'] or agent['detection_name'] != 'car' or (agent['velocity'][0] == 0 and agent['velocity'][1] == 0):
                    continue
                if len(surrounding_agents_vehicle) >= max_vehicles:
                    break
                agent_histories = tracked_objects[agent['tracking_id']][-6:]
                if len(agent_histories) < 6:
                    continue
                agent_history_states = []
                agent_history_states_mask = []
                agent_histories = agent_histories[-5:]
                agent_history_6th = agent_histories[0]
                agent_history_6th_velocity = math.sqrt(agent_history_6th['velocity'][0] ** 2 + agent_history_6th['velocity'][1] ** 2)
                agent_history_6th_yaw = quaternion_to_yaw(*agent_history_6th['rotation'])
                for state in agent_histories:
                    state_velocity = math.sqrt(state['velocity'][0] ** 2 + state['velocity'][1] ** 2)
                    agent_state = [state['translation'][0], state['translation'][1], state_velocity, state_velocity, quaternion_to_yaw(*state['rotation'])]
                    agent_history_states.append(agent_state)
                    agent_history_states_mask.append(np.zeros(5))
                for i in range(len(agent_history_states)):
                    agent_history_states[i][0] -= target_agent['translation'][0]
                    agent_history_states[i][1] -= target_agent['translation'][1]
                agent_history_states = np.array(agent_history_states)
                
                if len(agent_history_states) < 5:
                    num_missing_rows = 5 - len(agent_history_states)
                    missing_rows = np.zeros((num_missing_rows, agent_history_states.shape[1]))
                    missing_rows_mask = np.ones((num_missing_rows, agent_history_states.shape[1]))
                    agent_history_states = np.concatenate((missing_rows, agent_history_states), axis=0)
                    agent_history_states_mask = np.concatenate((missing_rows_mask, agent_history_states_mask), axis=0)
                
                # Yaw to yaw rate interpolation for surrounding vehicles
                agent_history_yaw_rates = np.zeros(5)
                agent_history_yaw_rates[4] = (angle_diff(agent_history_states[4, -1], agent_history_states[3, -1])) / 0.5
                agent_history_yaw_rates[3] = (angle_diff(agent_history_states[3, -1], agent_history_states[2, -1])) / 0.5
                agent_history_yaw_rates[2] = (angle_diff(agent_history_states[2, -1], agent_history_states[1, -1])) / 0.5
                agent_history_yaw_rates[1] = (angle_diff(agent_history_states[1, -1], agent_history_states[0, -1])) / 0.5
                agent_history_yaw_rates[0] = (angle_diff(agent_history_states[0, -1], agent_history_6th_yaw)) / 0.5
                agent_history_states[:, -1] = agent_history_yaw_rates
                
                # Velocity to acceleration interpolation for surrounding vehicles
                agent_history_accelerations = np.zeros(5)
                agent_history_accelerations[4] = (agent_history_states[4, -2] - agent_history_states[3, -2]) / 0.5
                agent_history_accelerations[3] = (agent_history_states[3, -2] - agent_history_states[2, -2]) / 0.5
                agent_history_accelerations[2] = (agent_history_states[2, -2] - agent_history_states[1, -2]) / 0.5
                agent_history_accelerations[1] = (agent_history_states[1, -2] - agent_history_states[0, -2]) / 0.5
                agent_history_accelerations[0] = (agent_history_states[0, -2] - agent_history_6th_velocity) / 0.5
                agent_history_states[:, -2] = agent_history_accelerations
                
                # Transform surrounding vehicle velocity from global frame to target agent frame
                # rotation_matrix = rotation_matrix_2d(-quaternion_to_yaw(*target_agent['rotation']))
                # agent_history_states[:, 3:5] = np.dot(rotation_matrix, agent_history_states[:, 3:5].T).T
                
                surrounding_agents_vehicle.append(agent_history_states)
                surrounding_agents_vehicle_masks.append(agent_history_states_mask)
                
            surrounding_agents_vehicle = np.array(surrounding_agents_vehicle)
            surrounding_agents_vehicle_masks = np.array(surrounding_agents_vehicle_masks)
            if surrounding_agents_vehicle.shape[0] < max_vehicles:
                num_missing_rows = max_vehicles - surrounding_agents_vehicle.shape[0]
                missing_rows = np.zeros((num_missing_rows, 5, 5))
                missing_rows_mask = np.ones((num_missing_rows, 5, 5))
                if num_missing_rows == max_vehicles:
                    surrounding_agents_vehicle = missing_rows
                    surrounding_agents_vehicle_masks = missing_rows_mask
                else:  
                    surrounding_agents_vehicle = np.concatenate((surrounding_agents_vehicle, missing_rows), axis=0)
                    surrounding_agents_vehicle_masks = np.concatenate((surrounding_agents_vehicle_masks, missing_rows_mask), axis=0)
            
            # Get surrounding agents representation for pedestrians
            surrounding_agents_pedestrian = []
            surrounding_agents_pedestrian_masks = []
            max_pedestrians = 77
            for agent in outputs:
                if agent['tracking_id'] == target_agent['tracking_id'] or agent['detection_name'] != 'pedestrian' or (agent['velocity'][0] == 0 and agent['velocity'][1] == 0):
                    continue
                if len(surrounding_agents_pedestrian) >= max_pedestrians:
                    break
                agent_histories = tracked_objects[agent['tracking_id']][-6:]
                if len(agent_histories) < 6:
                    continue
                agent_history_states = []
                agent_history_states_mask = []
                agent_histories = agent_histories[-5:]
                agent_history_6th = agent_histories[0]
                agent_history_6th_velocity = math.sqrt(agent_history_6th['velocity'][0] ** 2 + agent_history_6th['velocity'][1] ** 2)
                agent_history_6th_yaw = quaternion_to_yaw(*agent_history_6th['rotation'])
                for state in agent_histories:
                    state_velocity = math.sqrt(state['velocity'][0] ** 2 + state['velocity'][1] ** 2)
                    agent_state = [state['translation'][0], state['translation'][1], state_velocity, state_velocity, quaternion_to_yaw(*state['rotation'])]
                    agent_history_states.append(agent_state)
                    agent_history_states_mask.append(np.zeros(5))
                for i in range(len(agent_history_states)):
                    agent_history_states[i][0] -= target_agent['translation'][0]
                    agent_history_states[i][1] -= target_agent['translation'][1]
                agent_history_states = np.array(agent_history_states)
                if len(agent_history_states) < 5:
                    num_missing_rows = 5 - len(agent_history_states)
                    missing_rows = np.zeros((num_missing_rows, agent_history_states.shape[1]))
                    missing_rows_mask = np.ones((num_missing_rows, agent_history_states.shape[1]))
                    agent_history_states = np.concatenate((missing_rows, agent_history_states), axis=0)
                    agent_history_states_mask = np.concatenate((missing_rows_mask, agent_history_states_mask), axis=0)

                # Yaw to yaw rate interpolation for surrounding pedestrians
                agent_history_yaw_rates = np.zeros(5)
                agent_history_yaw_rates[4] = (angle_diff(agent_history_states[4, -1], agent_history_states[3, -1])) / 0.5
                agent_history_yaw_rates[3] = (angle_diff(agent_history_states[4, -1], agent_history_states[2, -1])) / 0.5
                agent_history_yaw_rates[2] = (angle_diff(agent_history_states[3, -1], agent_history_states[1, -1])) / 0.5
                agent_history_yaw_rates[1] = (angle_diff(agent_history_states[2, -1], agent_history_states[0, -1])) / 0.5
                agent_history_yaw_rates[0] = (angle_diff(agent_history_states[1, -1], agent_history_6th_yaw)) / 0.5
                agent_history_states[:, -1] = agent_history_yaw_rates
                
                # Velocity to acceleration interpolation for surrounding pedestrians
                agent_history_accelerations = np.zeros(5)
                agent_history_accelerations[4] = (agent_history_states[4, -2] - agent_history_states[3, -2]) / 0.5
                agent_history_accelerations[3] = (agent_history_states[3, -2] - agent_history_states[2, -2]) / 0.5
                agent_history_accelerations[2] = (agent_history_states[2, -2] - agent_history_states[1, -2]) / 0.5
                agent_history_accelerations[1] = (agent_history_states[1, -2] - agent_history_states[0, -2]) / 0.5
                agent_history_accelerations[0] = (agent_history_states[0, -2] - agent_history_6th_velocity) / 0.5
                agent_history_states[:, -2] = agent_history_accelerations
                
                # Transform surrounding pedestrian velocity from global frame to target agent frame
                # rotation_matrix = rotation_matrix_2d(-quaternion_to_yaw(*target_agent['rotation']))
                # agent_history_states[:, 3:5] = np.dot(rotation_matrix, agent_history_states[:, 3:5].T).T
                
                surrounding_agents_pedestrian.append(agent_history_states)
                surrounding_agents_pedestrian_masks.append(agent_history_states_mask)
            surrounding_agents_pedestrian = np.array(surrounding_agents_pedestrian)
            surrounding_agents_pedestrian_masks = np.array(surrounding_agents_pedestrian_masks)
            
            if surrounding_agents_pedestrian.shape[0] < max_pedestrians:
                num_missing_rows = max_pedestrians - surrounding_agents_pedestrian.shape[0]
                missing_rows = np.zeros((num_missing_rows, 5, 5))
                missing_rows_mask = np.ones((num_missing_rows, 5, 5))
                if num_missing_rows == max_pedestrians:
                    surrounding_agents_pedestrian = missing_rows
                    surrounding_agents_pedestrian_masks = missing_rows_mask
                else:
                    surrounding_agents_pedestrian = np.concatenate((surrounding_agents_pedestrian, missing_rows), axis=0)
                    surrounding_agents_pedestrian_masks = np.concatenate((surrounding_agents_pedestrian_masks, missing_rows_mask), axis=0)
            
            surrounding_agent_representation = {
                'vehicles': surrounding_agents_vehicle, 
                'vehicle_masks': surrounding_agents_vehicle_masks,
                'pedestrians': surrounding_agents_pedestrian,
                'pedestrian_masks': surrounding_agents_pedestrian_masks,
            }

            map_representation = self.get_map_representation(target_agent)
            ax = self.visualize_map_with_target_agent(ax, target_agent_histories, plot_all_map_attributes=True, radius=50)
            init_node = self.get_initial_node(map_representation)
            data = {}
            data['inputs'] = {
                'target_agent_representation': target_agent_history_states,
                'map_representation': map_representation,
                'surrounding_agent_representation': surrounding_agent_representation,
                'init_node': init_node,
            }
            data['inputs']['agent_node_masks'] = self.get_agent_node_masks(map_representation, surrounding_agent_representation)
            data = convert_to_torch_and_expand(data)
            data = send_to_device(convert_double_to_float(data))
            
            # Shape checks
            # print("===== Shape Checks =====")
            # print("Target agent history states shape: ", target_agent_history_states.shape)
            # print("Surrounding agents vehicle shape: ", surrounding_agents_vehicle.shape)
            # print("Surrounding agents vehicle masks shape: ", surrounding_agents_vehicle_masks.shape)
            # print("Surrounding agents pedestrian shape: ", surrounding_agents_pedestrian.shape)
            # print("Surrounding agents pedestrian masks shape: ", surrounding_agents_pedestrian_masks.shape)
            # print("Lane node feats shape: ", map_representation['lane_node_feats'].shape)
            # print("Lane node masks shape: ", map_representation['lane_node_masks'].shape)
            # print("s_next shape: ", map_representation['s_next'].shape)
            # print("edge_type shape: ", map_representation['edge_type'].shape)

            prediction = self.model(data['inputs'])
            
            # Visualization
            trajectories = prediction['traj'].detach().cpu().numpy()[0]
            probs = prediction['probs'].detach().cpu().numpy()[0]
            max_prob_index = np.argmax(probs)

            rotation_matrix = rotation_matrix_2d(-np.pi/2 + quaternion_to_yaw(*target_agent['rotation']))
            # rotation_matrix = np.eye(2)
            degrees_rotated_by = np.degrees(np.arccos(rotation_matrix[0, 0]))
            
            # If you want to plot all trajectories
            for i in range(trajectories.shape[0]):
                trajectories[i] = np.dot(rotation_matrix, trajectories[i].T).T
                trajectories[i] += target_agent['translation'][:2]
                complete_traj = np.concatenate((np.array([target_agent['translation'][:2]]), trajectories[i]), axis=0)
                color = '#001A00' if i == max_prob_index else 'brown'
                linewidth = 3 if i == max_prob_index else 1
                if i == max_prob_index:
                    ax.plot(complete_traj[:, 0], complete_traj[:, 1], 'b-', color=color, linewidth=linewidth, label=f'Trajectory {i+1}')    
            # plt.title(f'Target agent velocity: {target_agent["velocity"][0]:.2f}, {target_agent["velocity"][1]:.2f}, yaw: {quaternion_to_yaw(*target_agent["rotation"]):.2f}, degrees rotated by: {degrees_rotated_by} \n Trajectory confidence: {probs[max_prob_index]}, Inference time: {execution_time:.2f} seconds')
            formatted_vel, formatted_accel, formatted_yaw_rate, formatted_yaws = format_states_string(target_agent_history_states, target_agent_history_yaws)
            formatted_title = f'Agent ID: {target_agent["tracking_id"]}, Agent Type: {target_agent["detection_name"]} \n Agent Vel: [{formatted_vel}], Agent Accel: [{formatted_accel}], \n Agent Yaws: [{formatted_yaws}] Agent Yaw Rate: [{formatted_yaw_rate}]'
            ax.set_title(formatted_title, fontsize=10)
        plt.suptitle('Predicted Trajectories for All Valid Agents', fontsize=20, y=1.2)
        plt.tight_layout()
        plt.savefig('trajectory.png')
        return trajectories[max_prob_index]