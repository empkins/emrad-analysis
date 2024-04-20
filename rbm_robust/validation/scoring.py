from datetime import datetime

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pathlib
from emrad_analysis.validation.PairwiseHeartRate import PairwiseHeartRate
from emrad_analysis.validation.RPeakF1Score import RPeakF1Score
from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.cnn import CNN
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline


class Scoring:
    heart_rate_prediction: np.ndarray
    heart_rate_ground_truth: np.ndarray
    f1_score: float
    mean_relative_error_hr: float
    mean_absolute_error: float
    abs_hr_error: float
    mean_instantaneous_abs_hr_diff: float
    mean_relative_error_hr: float
    mean_absolute_error: float
    cnn: CNN
    time_stamps: dict

    def __init__(
        self,
        heart_rate_prediction: np.ndarray,
        heart_rate_ground_truth: np.ndarray,
        f1_score: float,
        mean_relative_error_hr: float,
        mean_absolute_error: float,
        abs_hr_error: float,
        mean_instantaneous_abs_hr_diff: float,
        cnn: CNN,
        time_stamps: dict,
    ):
        self.heart_rate_prediction = heart_rate_prediction
        self.heart_rate_ground_truth = heart_rate_ground_truth
        self.f1_score = f1_score
        self.mean_relative_error_hr = mean_relative_error_hr
        self.mean_absolute_error = mean_absolute_error
        self.abs_hr_error = abs_hr_error
        self.mean_instantaneous_abs_hr_diff = mean_instantaneous_abs_hr_diff
        self.mean_relative_error_hr = mean_relative_error_hr
        self.mean_absolute_error = mean_absolute_error
        self.cnn = cnn
        self.time_stamps = time_stamps

    def save_myself(self):
        path = pathlib.Path("~/Dumps")
        path.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().isoformat(sep="-", timespec="seconds") + "_scoring.pkl"
        file_path = path.joinpath(filename)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_myself(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def cnnPipelineScoring(pipeline: CnnPipeline, dataset: D02Dataset):
    pipeline = pipeline.clone()

    time_stamps = {}

    # Split Data
    train_data, val_data = train_test_split(dataset.subjects, test_size=0.2, random_state=42)
    training_dataset = dataset.get_subset(participant=train_data)
    testing_dataset = dataset.get_subset(participant=val_data)

    time_stamps["Start"] = datetime.now().isoformat(sep="-", timespec="seconds")
    print("Start Training")
    pipeline.self_optimize(training_dataset)

    time_stamps["AfterTraining"] = datetime.now().isoformat(sep="-", timespec="seconds")
    print("Training done")
    pipeline.run(testing_dataset)
    time_stamps["AfterTestRun"] = datetime.now().isoformat(sep="-", timespec="seconds")

    labels = pipeline.feature_extractor.generate_training_labels(testing_dataset).input_labels_
    time_stamps["AfterTestingLabelGeneration"] = datetime.now().isoformat(sep="-", timespec="seconds")

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

    scoring = Scoring(
        heart_rate_prediction=heart_rate_prediction,
        heart_rate_ground_truth=heart_rate_ground_truth,
        f1_score=f1_score,
        mean_relative_error_hr=mean_relative_error_hr,
        mean_absolute_error=mean_absolute_error,
        abs_hr_error=absolute_hr_error,
        mean_instantaneous_abs_hr_diff=mean_instantaneous_abs_hr_diff,
        model=pipeline.cnn,
        time_stamps=time_stamps,
    )
    # scoring.save_myself()

    # Scoring results
    return {
        "abs_hr_error": absolute_hr_error,
        "mean_instantaneous_error": mean_instantaneous_abs_hr_diff,
        "f1_score": f1_score,
        "mean_relative_error_hr": mean_relative_error_hr,
        "mean_absolute_error": mean_absolute_error,
    }
