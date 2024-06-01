import os
from datetime import datetime

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pathlib
from pathlib import Path
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
        path = Path("~/Dumps")
        path.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().isoformat(sep="-", timespec="seconds") + "_scoring.pkl"
        file_path = path.joinpath(filename)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_myself(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def cnnPipelineScoring(
    pipeline: CnnPipeline,
    dataset: D02Dataset,
    training_and_validation_path: str = "/home/woody/iwso/iwso116h/Data",
    testing_path: str = "/home/woody/iwso/iwso116h/TestData",
):
    pipeline = pipeline.clone()

    time_stamps = {}
    data_path = Path(training_and_validation_path)
    possible_subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    dataset = dataset.get_subset(participant=possible_subjects)
    # Split Data
    # To always get the same subjects
    subjects = dataset.subjects
    subjects.sort()
    train_data, validation_data = train_test_split(subjects, test_size=0.2, random_state=42)
    training_dataset = dataset.get_subset(participant=train_data)
    validation_dataset = dataset.get_subset(participant=validation_data)

    print(f"Training Subjects: {training_dataset.subjects}")
    print(f"Validation Subjects: {validation_dataset.subjects}")
    print(f"Testing Subjects: {testing_subjects}")

    time_stamps["Start"] = datetime.now().isoformat(sep="-", timespec="seconds")
    print("Start Training")
    pipeline.self_optimize(training_dataset, validation_dataset, training_and_validation_path)

    time_stamps["AfterTraining"] = datetime.now().isoformat(sep="-", timespec="seconds")
    print("Training done")
    pipeline.run(testing_subjects, testing_path)
    time_stamps["AfterTestRun"] = datetime.now().isoformat(sep="-", timespec="seconds")

    label_base_path = Path(testing_path)
    time_stamps["AfterTestingLabelGeneration"] = datetime.now().isoformat(sep="-", timespec="seconds")

    true_positives = 0
    total_gt_peaks = 0
    total_pred_peaks = 0

    for subject in label_base_path.iterdir():
        if not subject.is_dir():
            continue
        if subject.name not in testing_subjects:
            continue
        print(f"subject {subject}")
        for phase in subject.iterdir():
            if not phase.is_dir():
                continue
            if phase.name == "logs" or phase.name == "raw":
                continue
            print(f"phase {phase}")
            prediction_path = phase
            prediction_path = Path(str(prediction_path).replace("TestData", "Predictions/predictions_kl_25_epochs"))
            label_path = phase / "labels_gaussian"
            prediction_files = sorted(path.name for path in prediction_path.iterdir() if path.is_file())
            f1RPeakScore = RPeakF1Score(max_deviation_ms=100)
            for prediction_file in prediction_files:
                prediction = np.load(prediction_path / prediction_file)
                label = np.load(label_path / prediction_file)
                f1RPeakScore.compute_predictions(prediction, label)
                true_positives += f1RPeakScore.tp_
                total_gt_peaks += f1RPeakScore.total_peaks_
                total_pred_peaks += f1RPeakScore.pred_peaks_

    # Save the Model
    pipeline.cnn.save_model()

    if total_pred_peaks == 0:
        print("No Peaks detected")
        return {
            "abs_hr_error": 0,
            "mean_instantaneous_error": 0,
            "f1_score": 0,
            "mean_relative_error_hr": 0,
            "mean_absolute_error": 0,
        }

    precision = true_positives / total_pred_peaks
    recall = true_positives / total_gt_peaks
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"f1 Score {f1_score}")

    # Scoring results
    return {
        "abs_hr_error": 0,
        "mean_instantaneous_error": 0,
        "f1_score": f1_score,
        "mean_relative_error_hr": 0,
        "mean_absolute_error": 0,
    }
