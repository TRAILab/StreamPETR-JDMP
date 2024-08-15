from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import math
import torch
from typing import Dict, Tuple, Union, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits

def convert_to_torch_and_expand(data):
    data['inputs']['target_agent_representation'] = torch.unsqueeze(torch.from_numpy(data['inputs']['target_agent_representation']), 0)
    data['inputs']['init_node'] = torch.unsqueeze(torch.from_numpy(data['inputs']['init_node']), 0)
    # data['inputs']['node_seq_gt'] = torch.unsqueeze(torch.from_numpy(data['inputs']['node_seq_gt']), 0)
    for k, v in data['inputs']['surrounding_agent_representation'].items():
        data['inputs']['surrounding_agent_representation'][k] = torch.unsqueeze(torch.from_numpy(data['inputs']['surrounding_agent_representation'][k]), 0)
    for k, v in data['inputs']['map_representation'].items():
        data['inputs']['map_representation'][k] = torch.unsqueeze(torch.from_numpy(data['inputs']['map_representation'][k]), 0)
    for k, v in data['inputs']['agent_node_masks'].items():
        data['inputs']['agent_node_masks'][k] = torch.unsqueeze(torch.from_numpy(data['inputs']['agent_node_masks'][k]), 0)
    return data

def send_to_device(data: Union[Dict, torch.Tensor]):
    """
    Utility function to send nested dictionary with Tensors to GPU
    """
    if type(data) is torch.Tensor:
        return data.to('cuda')
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v)
        return data
    else:
        return data

def convert_double_to_float(data: Union[Dict, torch.Tensor]):
    """
    Utility function to convert double tensors to float tensors in nested dictionary with Tensors
    """
    if type(data) is torch.Tensor and data.dtype == torch.float64:
        return data.float()
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert_double_to_float(v)
        return data
    else:
        return data

def get_surrounding_agent_history_states(agent_idx, valid_agents, valid_vehicles, valid_pedestrians, target_agent_global_pose, all_agent_positions, all_agent_vels, all_agent_rotations, all_agent_classes, max_vehicles=84, max_pedestrians=77):
    # Confirm everything is on cpu devide, if not, convert to cpu
    if all_agent_positions.device != torch.device('cpu'):
        all_agent_positions = all_agent_positions.cpu()
    if all_agent_vels.device != torch.device('cpu'):
        all_agent_vels = all_agent_vels.cpu()
    if all_agent_rotations.device != torch.device('cpu'):
        all_agent_rotations = all_agent_rotations.cpu()
    all_agent_positions = all_agent_positions.detach().numpy()
    all_agent_vels = all_agent_vels.detach().numpy()
    all_agent_rotations = all_agent_rotations.detach().numpy()
    valid_agents[agent_idx] = 0 # Mask the target agent
    valid_vehicles[agent_idx] = 0 # Mask the target agent
    valid_pedestrians[agent_idx] = 0 # Mask the target agent
    
    surrounding_vehicles, surrounding_pedestrians = [], []
    surrounding_vehicle_masks, surrounding_pedestrian_masks = [], []
    
    # Get the surrounding vehicles
    if torch.sum(valid_vehicles) == 0:
        surrounding_vehicles = np.zeros((max_vehicles, 5, 5))
        surrounding_vehicle_masks = np.ones((max_vehicles, 5, 5))
    else:
        all_vehicle_positions, all_vehicle_vels, all_vehicle_rotations = all_agent_positions[valid_vehicles.bool()], all_agent_vels[valid_vehicles.bool()], all_agent_rotations[valid_vehicles.bool()]    
        all_vehicle_history_xy = all_vehicle_positions[..., 1:, :-1]
        all_vehicle_velocities = np.sqrt(np.sum(all_vehicle_vels**2, axis=2)).reshape(-1, 6, 1)
        all_vehicle_history_velocities = all_vehicle_velocities[:, 1:]
        all_vehicle_history_accels = np.diff(all_vehicle_velocities, axis=1) / 0.5
        all_vehicle_history_yaw_rate = np.diff(all_vehicle_rotations, axis=1) / 0.5
        try:
            all_vehicle_history_states = np.concatenate([all_vehicle_history_xy, all_vehicle_history_velocities, all_vehicle_history_accels, all_vehicle_history_yaw_rate], axis=2)
        except:
            breakpoint()
        all_vehicle_history_states[:, :, 0] -= target_agent_global_pose[0]
        all_vehicle_history_states[:, :, 1] -= target_agent_global_pose[1]
        if all_vehicle_history_states.shape[0] < max_vehicles:
            remaining_vehicles = max_vehicles - all_vehicle_history_states.shape[0]
            surrounding_vehicles = np.concatenate([all_vehicle_history_states, np.zeros((remaining_vehicles, 5, 5))], axis=0)
            surrounding_vehicle_masks = np.concatenate([np.zeros((all_vehicle_history_states.shape[0], 5, 5)), np.ones((remaining_vehicles, 5, 5))], axis=0)
    
    # Get the surrounding pedestrians
    if torch.sum(valid_pedestrians) == 0:
        surrounding_pedestrians = np.zeros((max_pedestrians, 5, 5))
        surrounding_pedestrian_masks = np.ones((max_pedestrians, 5, 5))
    else:
        all_pedestrian_positions, all_pedestrian_vels, all_pedestrian_rotations = all_agent_positions[valid_pedestrians.bool()], all_agent_vels[valid_pedestrians.bool()], all_agent_rotations[valid_pedestrians.bool()]    
        all_pedestrian_history_xy = all_pedestrian_positions[..., 1:, :-1]
        all_pedestrian_velocities = np.sqrt(np.sum(all_pedestrian_vels**2, axis=2)).reshape(-1, 1)[np.newaxis, ...]
        all_pedestrian_history_velocities = all_pedestrian_velocities[:, 1:]
        all_pedestrian_history_accels = np.diff(all_pedestrian_history_velocities, axis=1) / 0.5
        all_pedestrian_history_yaw_rate = np.diff(all_pedestrian_rotations, axis=1) / 0.5
        all_pedestrian_history_states = np.concatenate([all_pedestrian_history_xy, all_pedestrian_history_velocities, all_pedestrian_history_accels, all_pedestrian_history_yaw_rate], axis=2)
        all_pedestrian_history_states[:, :, 0] -= target_agent_global_pose[0]
        all_pedestrian_history_states[:, :, 1] -= target_agent_global_pose[1]
        if all_pedestrian_history_states.shape[0] < max_pedestrians:
            remaining_pedestrians = max_pedestrians - all_pedestrian_history_states.shape[0]
            surrounding_pedestrians = np.concatenate([all_pedestrian_history_states, np.zeros((remaining_pedestrians, 5, 5))], axis=0)
            surrounding_pedestrian_masks = np.concatenate([np.zeros((all_pedestrian_history_states.shape[0], 5, 5)), np.ones((remaining_pedestrians, 5, 5))], axis=0)
    
    surrounding_agent_representation = {
        'vehicles': surrounding_vehicles, 
        'vehicle_masks': surrounding_vehicle_masks,
        'pedestrians': surrounding_pedestrians,
        'pedestrian_masks': surrounding_pedestrian_masks,
    }
    
    return surrounding_agent_representation

def get_target_agent_history_states(target_agent_position, target_agent_vel, target_agent_rotation):
    
    '''
    Input:
        target_agent_position: (N, 3) torch tensor, 3 is x, y, z (global frame)
        target_agent_vel: (N, 2) torch tensor, 2 is vx, vy
        target_agent_rotation: (N, 1) torch tensor, 1 is yaw
    Output:
        target_agent_history_states (agent-centric frame): (5, 5) torch tensor, 5 is x, y, v, a, yaw_rate, get the most recent 5 states,
        the earliest state is just for velocity and yaw_rate estimation.
    '''
    # Confirm everything is on cpu devide, if not, convert to cpu
    if target_agent_position.device != torch.device('cpu'):
        target_agent_position = target_agent_position.cpu()
    if target_agent_vel.device != torch.device('cpu'):
        target_agent_vel = target_agent_vel.cpu()
    if target_agent_rotation.device != torch.device('cpu'):
        target_agent_rotation = target_agent_rotation.cpu()
    target_agent_position = target_agent_position.detach().numpy()
    target_agent_vel = target_agent_vel.detach().numpy()
    target_agent_rotation = target_agent_rotation.detach().numpy()
    # print('***** Shape Checks *****\n')
    # print('target_agent_position:', target_agent_position.shape)
    # print('target_agent_vel:', target_agent_vel.shape)
    # print('target_agent_rotation:', target_agent_rotation.shape)
    # print('\n***** End Shape Checks *****')
    target_agent_history_xy = target_agent_position[1:, :-1]
    target_agent_velocities = np.sqrt(np.sum(target_agent_vel**2, axis=1)).reshape(-1, 1)
    target_agent_history_velocities = target_agent_velocities[1:]
    target_agent_history_accels = np.diff(target_agent_velocities, axis=0) / 0.5
    target_agent_history_yaw_rate = np.diff(target_agent_rotation, axis=0) / 0.5
    target_agent_history_states = np.concatenate([target_agent_history_xy, target_agent_history_velocities, target_agent_history_accels, target_agent_history_yaw_rate], axis=1)
    target_agent_history_states[..., 0] -= target_agent_history_states[-1, 0]
    target_agent_history_states[..., 1] -= target_agent_history_states[-1, 1]
    return target_agent_history_states

def convert_to_torch_and_expand(data):
    data['inputs']['target_agent_representation'] = torch.unsqueeze(torch.from_numpy(data['inputs']['target_agent_representation']), 0)
    data['inputs']['init_node'] = torch.unsqueeze(torch.from_numpy(data['inputs']['init_node']), 0)
    # data['inputs']['node_seq_gt'] = torch.unsqueeze(torch.from_numpy(data['inputs']['node_seq_gt']), 0)
    for k, v in data['inputs']['surrounding_agent_representation'].items():
        try:
            data['inputs']['surrounding_agent_representation'][k] = torch.unsqueeze(torch.from_numpy(data['inputs']['surrounding_agent_representation'][k]), 0)
        except:
            if k == 'vehicles':
                data['inputs']['surrounding_agent_representation'][k] = np.zeros((84, 5, 5))
            elif k == 'vehicle_masks':
                data['inputs']['surrounding_agent_representation'][k] = np.ones((84, 5, 5)) 
            elif k == 'pedestrians':
                data['inputs']['surrounding_agent_representation'][k] = np.zeros((77, 5, 5))
            elif k == 'pedestrian_masks':
                data['inputs']['surrounding_agent_representation'][k] = np.ones((77, 5, 5))
    for k, v in data['inputs']['map_representation'].items():
        data['inputs']['map_representation'][k] = torch.unsqueeze(torch.from_numpy(data['inputs']['map_representation'][k]), 0)
    for k, v in data['inputs']['agent_node_masks'].items():
        data['inputs']['agent_node_masks'][k] = torch.unsqueeze(torch.from_numpy(data['inputs']['agent_node_masks'][k]), 0)
    return data

def send_to_device(data: Union[Dict, torch.Tensor]):
    """
    Utility function to send nested dictionary with Tensors to GPU
    """
    if type(data) is torch.Tensor:
        return data.to('cuda')
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v)
        return data
    else:
        return data

def convert_double_to_float(data: Union[Dict, torch.Tensor]):
    """
    Utility function to convert double tensors to float tensors in nested dictionary with Tensors
    """
    if type(data) is torch.Tensor and data.dtype == torch.float64:
        return data.float()
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert_double_to_float(v)
        return data
    else:
        return data

def quaternion_to_yaw(w, x, y, z):
    # Calculate yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def rotation_matrix_2d(theta):
    """
    Returns a 2D rotation matrix for a given angle in radians.

    Parameters:
    theta (float): The rotation angle in radians.

    Returns:
    np.ndarray: The 2x2 rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def find_valid_agent(outputs, tracked_objects):
    # Define a function to check the conditions for the target agent
    def is_valid_agent(agent):
        is_vehicle = agent['detection_name'] == 'car'
        has_enough_history = len(tracked_objects.get(agent['tracking_id'], [])) > 6
        is_tracked = agent['tracking_id'] in tracked_objects
        has_nonzero_velocity = agent['velocity'][0] >= 2.0 or agent['velocity'][1] >= 2.0

        return is_vehicle and has_enough_history and is_tracked and has_nonzero_velocity

    # Iterate through the outputs list to find a valid agent
    for agent in outputs:
        if is_valid_agent(agent):
            return agent

    # If no valid agent is found, return None
    return None

def angle_diff(x: float, y: float, period=2*np.pi) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return abs(diff)

def get_velocity(current_translation, prev_translation, dt):
    current_translation = np.array(current_translation)
    prev_translation = np.array(prev_translation)
    diff = (current_translation - prev_translation) / dt
    velocity = np.linalg.norm(diff[:2])
    return velocity

def format_states_string(target_agent_history_states, target_agent_history_yaws):
    # Extract the relevant rows
    vel = target_agent_history_states[:, 2]
    agent_accel = target_agent_history_states[:, 3]
    agent_yaw_rate = target_agent_history_states[:, 4]

    # Format each element in the arrays
    formatted_vel = ", ".join([f"{v:.2f}" for v in vel])
    formatted_accel = ", ".join([f"{a:.2f}" for a in agent_accel])
    formatted_yaw_rate = ", ".join([f"{y:.2f}" for y in agent_yaw_rate])
    
    if target_agent_history_yaws is not None:
        formatted_yaws = ", ".join([f"{y:.2f}" for y in target_agent_history_yaws])
        return formatted_vel, formatted_accel, formatted_yaw_rate, formatted_yaws
    
    return formatted_vel, formatted_accel, formatted_yaw_rate, None

def find_all_valid_agents(outputs, tracked_objects):
    # Define a function to check the conditions for the target agent
    def is_valid_agent(agent):
        is_vehicle = agent['detection_name'] == 'car'
        has_enough_history = len(tracked_objects.get(agent['tracking_id'], [])) > 7 # need 7 because we need to accurately estimate the past velocity and acceleration
        is_tracked = agent['tracking_id'] in tracked_objects
        has_nonzero_velocity = agent['velocity'][0] >= 0.2 or agent['velocity'][1] >= 0.2

        return is_vehicle and has_enough_history and is_tracked and has_nonzero_velocity

    # Iterate through the outputs list to find a valid agent
    valid_agents = []
    for agent in outputs:
        if is_valid_agent(agent):
            valid_agents.append(agent)

    # If no valid agent is found, return None
    return valid_agents


def update_tracked_objects(tracked_objects, outputs):
    for item in outputs:
        if item['active'] == 0:
            continue 
        if item['tracking_id'] not in tracked_objects:
            tracked_objects[item['tracking_id']] = [{
                'translation': item['translation'],
                'size': item['size'],
                'rotation': item['rotation'],
                'velocity': item['velocity'],
                'tracking_id': item['tracking_id'],
                'detection_name': item['detection_name'],
                'detection_score': item['detection_score'],
            }]
        else:
            tracked_objects[item['tracking_id']].append({
                'translation': item['translation'],
                'size': item['size'],
                'rotation': item['rotation'],
                'velocity': item['velocity'],
                'tracking_id': item['tracking_id'],
                'detection_name': item['detection_name'],
                'detection_score': item['detection_score'],
            })
