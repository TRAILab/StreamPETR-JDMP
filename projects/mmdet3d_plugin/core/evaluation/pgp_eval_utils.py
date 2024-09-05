import torch.optim
from typing import Dict, Union
import torch
import numpy as np
import os
import pickle

from ...metrics.mtp_loss import MTPLoss
from ...metrics.covernet_loss import CoverNetLoss
from ...metrics.min_ade import MinADEK
from ...metrics.min_fde import MinFDEK
from ...metrics.miss_rate import MissRateK
from ...metrics.pi_bc import PiBehaviorCloning
from ...metrics.goal_pred_nll import GoalPredictionNLL

# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        'mtp_loss': MTPLoss,
        'covernet_loss': CoverNetLoss,
        'min_ade_k': MinADEK,
        'min_fde_k': MinFDEK,
        'miss_rate_k': MissRateK,
        'pi_bc': PiBehaviorCloning,
        'goal_pred_nll': GoalPredictionNLL
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metrics = ['min_ade_k','min_ade_k', 'miss_rate_k', 'miss_rate_k'] #, 'pi_bc']
metric_configs = [{'k': 5}, {'k': 10}, {'k': 5, 'dist_thresh': 2}, {'k': 10, 'dist_thresh': 2}] #, {}]
metrics = [initialize_metric(metric, config) for metric, config in zip(metrics, metric_configs)]

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


def send_to_device(data: Union[Dict, torch.Tensor]):
    """
    Utility function to send nested dictionary with Tensors to GPU
    """
    if type(data) is torch.Tensor:
        return data.to(device)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v)
        return data
    else:
        return data


def convert2tensors(data):
    """
    Converts data (dictionary of nd arrays etc.) to tensor with batch_size 1
    """
    if type(data) is np.ndarray:
        return torch.as_tensor(data).unsqueeze(0)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert2tensors(v)
        return data
    else:
        return data



def initialize_aggregate_metrics():
    """
    Initialize aggregate metrics for test set.
    """
    agg_metrics = {'sample_count': 0}
    metrics = []
    val_metrics = ['min_ade_k','min_ade_k', 'miss_rate_k', 'miss_rate_k', 'pi_bc']
    val_metric_args = [{'k': 5}, {'k': 10}, {'k': 5, 'dist_thresh': 2}, {'k': 10, 'dist_thresh': 2}, {}]
    for metric, args in zip(val_metrics, val_metric_args):
        metrics.append(initialize_metric(metric, args))
    for metric in metrics:
        agg_metrics[metric.name] = 0

    return agg_metrics


def aggregate_metrics(agg_metrics: Dict, model_outputs: Dict, ground_truth: Dict):
        """
        Aggregates metrics for evaluation
        """
        minibatch_metrics = {}
        for metric in metrics:
            minibatch_metrics[metric.name] = metric.compute(model_outputs, ground_truth).item()

        batch_size = ground_truth['traj'].shape[0]
        agg_metrics['sample_count'] += batch_size

        for metric in metrics:
            agg_metrics[metric.name] += minibatch_metrics[metric.name] * batch_size

        return agg_metrics

def print_progress(minibatch_count: int, gt_data):
        """
        Prints progress bar
        """
        epoch_progress = minibatch_count / len(gt_data) * 100
        print('\rEvaluating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar, format(epoch_progress, '0.2f'), '%', end="\n" if epoch_progress == 100 else " ")

def evaluate(gt_file_path: str, pred_file_path: str, output_dir: str):
        """
        Evaluate the model predictions against the ground truth.
        :param gt_file_path: Path to the ground truth file.
        :param pred_file_path: Path to the prediction file.
        :param output_dir: Directory to save evaluation results.
        """
        # Initialize aggregate metrics
        agg_metrics = initialize_aggregate_metrics()
        total_num_of_samples = 0

        # Load ground truth and predictions
        with open(gt_file_path, 'rb') as f:
            gt_data = pickle.load(f)

        with open(pred_file_path, 'rb') as f:
            pred_data = pickle.load(f)

        gt_data_forecasting_locs = [gt_data['infos'][i]['gt_forecasting_locs'][..., 1:] for i in range(len(gt_data['infos']))]
        gt_data_forecasting_type = [gt_data['infos'][i]['gt_forecasting_types'][..., 1:] for i in range(len(gt_data['infos']))]
        gt_data_forecasting_masks = [gt_data['infos'][i]['gt_forecasting_masks'][..., 1:] for i in range(len(gt_data['infos']))]
        gt_data_list = []
        for i in range(len(gt_data['infos'])):
            try:
                gt_traj = {'traj': torch.from_numpy(gt_data_forecasting_locs[i])[:, 1:, :]}
                mask = torch.from_numpy(gt_data_forecasting_masks[i])
                gt_traj['traj'] *= mask.unsqueeze(-1)
            except:
                gt_traj = {'traj': torch.zeros(1, 12, 2)}
            gt_data_list.append(gt_traj)
            
        pred_data_list = []
        for i in range(len(pred_data)):
            prediction = pred_data[i][:gt_data_list[i]['traj'].shape[0], 1:, :].unsqueeze(dim=1).cpu().numpy()
            mask = torch.from_numpy(gt_data_forecasting_masks[i])
            if len(mask) != 0:
                prediction *= mask.unsqueeze(-1).unsqueeze(1).cpu().numpy()
            pred_traj = {'traj': prediction, 'probs': torch.ones(prediction.shape[0], prediction.shape[1])}
            pred_data_list.append(pred_traj)
        
        # Ensure that both files have the same number of samples
        assert len(gt_data_list) == len(pred_data), f"{len(gt_data_list)}, {len(pred_data)} Ground truth and predictions have different lengths."

        with torch.no_grad():
            for i, (gt, pred) in enumerate(zip(gt_data_list, pred_data_list)):

                # Convert data to appropriate device
                # gt = send_to_device(gt)
                # pred = send_to_device(pred)
                # Count samples
                total_num_of_samples += gt['traj'].shape[0]

                # Aggregate metrics
                agg_metrics = aggregate_metrics(agg_metrics, pred, gt)

                # print_progress(i, gt_data)

        print(f'total number of samples: {total_num_of_samples}')

        # Compute and print average metrics
        print_progress(len(gt_data), gt_data)

        with open(os.path.join(output_dir, 'results', "results.txt"), "w") as out_file:
            for metric in metrics:
                avg_metric = agg_metrics[metric.name] / agg_metrics['sample_count']
                output = metric.name + ': ' + format(avg_metric, '0.2f')
                print(output)
                out_file.write(output + '\n')
                
