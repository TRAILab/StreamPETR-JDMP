from projects.mmdet3d_plugin.core.evaluation.pgp_eval_utils import evaluate


if __name__ == '__main__':
    gt_path = '/home/robert/Desktop/trail/StreamPETR-JDMP/data/nuscenes/nuscenes2d_temporal_infos_val_new.pkl'
    pred_path = '/home/robert/Desktop/trail/StreamPETR-JDMP/forecast_preds.pkl'
    evaluate(gt_path, pred_path, output_dir='/home/robert/Desktop/trail/StreamPETR-JDMP/')