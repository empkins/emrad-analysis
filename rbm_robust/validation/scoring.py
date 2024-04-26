from datetime import datetime

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pathlib
from rbm_robust.validation.RPeakF1Score import RPeakF1Score
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
    training_dataset, validation_dataset = train_test_split(training_dataset, test_size=0.2, random_state=42)
    testing_dataset = dataset.get_subset(participant=val_data)

    time_stamps["Start"] = datetime.now().isoformat(sep="-", timespec="seconds")
    print("Start Training")
    pipeline.self_optimize(training_dataset, validation_dataset)

    time_stamps["AfterTraining"] = datetime.now().isoformat(sep="-", timespec="seconds")
    print("Training done")
    pipeline.run(testing_dataset)
    time_stamps["AfterTestRun"] = datetime.now().isoformat(sep="-", timespec="seconds")

    label_base_path = pipeline.feature_extractor.generate_training_labels(testing_dataset)
    time_stamps["AfterTestingLabelGeneration"] = datetime.now().isoformat(sep="-", timespec="seconds")

    true_positives = 0
    total_gt_peaks = 0
    total_pred_peaks = 0

    for subject in label_base_path.iterdir():
        if not subject.is_dir():
            continue
        for phase in subject.iterdir():
            if not phase.is_dir():
                continue
            prediction_path = phase / "predictions"
            label_path = phase / "labels"
            prediction_files = sorted(path.name for path in prediction_path.iterdir() if path.is_file())
            f1RPeakScore = RPeakF1Score(max_deviation_ms=100)
            for prediction_file in prediction_files:
                prediction = np.load(prediction_path / prediction_file)
                label = np.load(label_path / prediction_file)

                f1RPeakScore.compute_predictions(prediction, label)
                true_positives += f1RPeakScore.tp_
                total_gt_peaks += f1RPeakScore.total_peaks_
                total_pred_peaks += f1RPeakScore.pred_peaks_

    precision = true_positives / total_pred_peaks
    recall = true_positives / total_gt_peaks
    f1_score = 2 * (precision * recall) / (precision + recall)

    scoring = Scoring(
        heart_rate_prediction=0,
        heart_rate_ground_truth=0,
        f1_score=f1_score,
        mean_relative_error_hr=0,
        mean_absolute_error=0,
        abs_hr_error=0,
        mean_instantaneous_abs_hr_diff=0,
        model=pipeline.cnn,
        time_stamps=time_stamps,
    )
    # scoring.save_myself()

    print(f"f1 Score {f1_score}")
    # Scoring results
    return {
        "abs_hr_error": 0,
        "mean_instantaneous_error": 0,
        "f1_score": f1_score,
        "mean_relative_error_hr": 0,
        "mean_absolute_error": 0,
    }
