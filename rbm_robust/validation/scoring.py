from empkins_io.datasets.d03.micro_gapvii._dataset import MicroBaseDataset

import numpy as np

from emrad_analysis.validation.PairwiseHeartRate import PairwiseHeartRate
from emrad_analysis.validation.RPeakF1Score import RPeakF1Score
from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline


# Use cases are:
# 1. Good heart rate extimation over the long run
#   - Calculate beat-to-beat heart rate and take the mean for an experiment
# 2. Good beat-to-beat accuracy important for HRV computation
#   - Calculate beat-to-beat heart rates and then calculate the MAE between the instantaneous heart rates


def cnnPipelineScoring(pipeline: CnnPipeline, dataset: D02Dataset):
    pipeline = pipeline.clone()

    pipeline.run(dataset)

    labels = pipeline.feature_extractor.generate_training_labels(dataset).input_labels_

    # normalize predictions and labels between 0 and 1
    result = (
        (pipeline.result_ - np.min(pipeline.result_, axis=1)[:, None])
        / (np.max(pipeline.result_, axis=1)[:, None] - np.min(pipeline.result_, axis=1)[:, None])
    )[::20, :].flatten()
    labels = (
        (labels - np.min(labels, axis=1)[:, None]) / (np.max(labels, axis=1)[:, None] - np.min(labels, axis=1)[:, None])
    )[::20, :].flatten()

    f1_score = (
        RPeakF1Score(max_deviation_ms=100)
        .compute(predicted_r_peak_signal=result, ground_truth_r_peak_signal=labels)
        .f1_score_
    )

    # Compute beat-to-beat heart rates
    heart_rate_prediction = PairwiseHeartRate().compute(result).heart_rate_
    heart_rate_ground_truth = PairwiseHeartRate().compute(labels).heart_rate_

    # 1. heart rate estimation over long run
    hr_pred = heart_rate_prediction.mean()
    hr_g_t = heart_rate_ground_truth.mean()

    absolute_hr_error = abs(hr_pred - hr_g_t)

    # 2. beat_to_beat_accuracy
    assert len(heart_rate_ground_truth) == len(
        heart_rate_prediction
    ), "The heart_rate_ground_truth and heart_rate_prediction should be equally long."

    instantaneous_abs_hr_diff = np.abs(np.subtract(heart_rate_prediction, heart_rate_ground_truth))

    mean_instantaneous_abs_hr_diff = instantaneous_abs_hr_diff.mean()

    mean_relative_error_hr = (
        1 / len(heart_rate_ground_truth) * np.sum(instantaneous_abs_hr_diff / heart_rate_ground_truth)
    )

    mean_absolute_error = np.mean(np.sum(np.square(instantaneous_abs_hr_diff)))

    # Scoring results
    return {
        "abs_hr_error": absolute_hr_error,
        "mean_instantaneous_error": mean_instantaneous_abs_hr_diff,
        "f1_score": f1_score,
        "mean_relative_error_hr": mean_relative_error_hr,
        "mean_absolute_error": mean_absolute_error,
    }
