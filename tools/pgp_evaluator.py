import torch.utils.data as torch_data
from typing import Dict
from train_eval.initialization import initialize_prediction_model, initialize_metric, initialize_dataset, get_specific_args
import torch
import os
import pgp_eval_utils as u
import numpy as np
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.data_classes import Prediction
import json
import pickle


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    """
    Class for evaluating trained models
    """
    def __init__(self, cfg: Dict, data_root: str, data_dir: str, checkpoint_path: str):
        """
        Initialize evaluator object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        """

        # Initialize metrics
        self.metrics = [initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
                        for i in range(len(cfg['val_metrics']))]

    def evaluate(self, gt_file_path: str, pred_file_path: str, output_dir: str):
        """
        Evaluate the model predictions against the ground truth.
        :param gt_file_path: Path to the ground truth file.
        :param pred_file_path: Path to the prediction file.
        :param output_dir: Directory to save evaluation results.
        """
        # Initialize aggregate metrics
        agg_metrics = self.initialize_aggregate_metrics()
        total_num_of_samples = 0

        # Load ground truth and predictions
        with open(gt_file_path, 'rb') as f:
            gt_data = pickle.load(f)

        with open(pred_file_path, 'rb') as f:
            pred_data = pickle.load(f)

        # Ensure that both files have the same number of samples
        assert len(gt_data) == len(pred_data), "Ground truth and predictions have different lengths."

        with torch.no_grad():
            for i, (gt, pred) in enumerate(zip(gt_data, pred_data)):

                # Convert data to appropriate device
                gt = u.send_to_device(gt)
                pred = u.send_to_device(pred)

                # Count samples
                total_num_of_samples += gt['traj'].shape[0]

                # Aggregate metrics
                agg_metrics = self.aggregate_metrics(agg_metrics, pred, gt)

                self.print_progress(i)

        print(f'total number of samples: {total_num_of_samples}')

        # Compute and print average metrics
        self.print_progress(len(gt_data))

        with open(os.path.join(output_dir, 'results', "results.txt"), "w") as out_file:
            for metric in self.metrics:
                avg_metric = agg_metrics[metric.name] / agg_metrics['sample_count']
                output = metric.name + ': ' + format(avg_metric, '0.2f')
                print(output)
                out_file.write(output + '\n')

    def initialize_aggregate_metrics(self):
        """
        Initialize aggregate metrics for test set.
        """
        agg_metrics = {'sample_count': 0}
        for metric in self.metrics:
            agg_metrics[metric.name] = 0

        return agg_metrics

    def aggregate_metrics(self, agg_metrics: Dict, model_outputs: Dict, ground_truth: Dict):
        """
        Aggregates metrics for evaluation
        """
        minibatch_metrics = {}
        for metric in self.metrics:
            minibatch_metrics[metric.name] = metric.compute(model_outputs, ground_truth).item()

        batch_size = ground_truth['traj'].shape[0]
        agg_metrics['sample_count'] += batch_size

        for metric in self.metrics:
            agg_metrics[metric.name] += minibatch_metrics[metric.name] * batch_size

        return agg_metrics

    def print_progress(self, minibatch_count: int):
        """
        Prints progress bar
        """
        epoch_progress = minibatch_count / len(self.gt_data) * 100
        print('\rEvaluating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar, format(epoch_progress, '0.2f'), '%', end="\n" if epoch_progress == 100 else " ")

    def generate_nuscenes_benchmark_submission(self, gt_file_path: str, pred_file_path: str, output_dir: str):
        """
        Sets up list of Prediction objects for the nuScenes benchmark.
        """

        # Load ground truth and predictions
        with open(gt_file_path, 'rb') as f:
            gt_data = pickle.load(f)

        with open(pred_file_path, 'rb') as f:
            pred_data = pickle.load(f)

        # Ensure that both files have the same number of samples
        assert len(gt_data) == len(pred_data), "Ground truth and predictions have different lengths."

        # NuScenes prediction helper
        helper = self.gt_data.dataset.helper

        # List of predictions
        preds = []

        with torch.no_grad():
            for i, (gt, pred) in enumerate(zip(gt_data, pred_data)):

                # Convert data to appropriate device
                gt = u.send_to_device(gt)
                pred = u.send_to_device(pred)

                traj = pred['traj']
                probs = pred['probs']

                # Load instance and sample tokens for batch
                instance_tokens = gt['inputs']['instance_token']
                sample_tokens = gt['inputs']['sample_token']

                # Create prediction object and add to list of predictions
                for n in range(traj.shape[0]):

                    traj_local = traj[n].detach().cpu().numpy()
                    probs_n = probs[n].detach().cpu().numpy()
                    starting_annotation = helper.get_sample_annotation(instance_tokens[n], sample_tokens[n])
                    traj_global = np.zeros_like(traj_local)
                    for m in range(traj_local.shape[0]):
                        traj_global[m] = convert_local_coords_to_global(traj_local[m],
                                                                        starting_annotation['translation'],
                                                                        starting_annotation['rotation'])

                    preds.append(Prediction(instance=instance_tokens[n], sample=sample_tokens[n],
                                            prediction=traj_global, probabilities=probs_n).serialize())

                # Print progress bar
                self.print_progress(i)

            # Save predictions to json file
            json.dump(preds, open(os.path.join(output_dir, 'results', "evalai_submission.json"), "w"))
            self.print_progress(len(gt_data))
