import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rbm_robust.validation.RPeakF1Score import RPeakF1Score
from rbm_robust.validation.validationBase import ValidationBase


class ScoreCalculator(ValidationBase):
    def __init__(self, prediction_path, label_path, fs=200, overlap=0.4, label_suffix="labels_ecg", prominence=0.3):
        self.max_heart_rate = 180
        if isinstance(prediction_path, str):
            prediction_path = Path(prediction_path)
        if isinstance(label_path, str):
            label_path = Path(label_path)
        self.prediction_path = prediction_path
        self.label_path = label_path
        self.overlap = overlap
        self.fs = fs
        self.label_suffix = label_suffix
        self.prominence = prominence

    def _calculate_f1_score_for_subject_and_phase(
        self, subject_name, phase, prediction: np.array = None, label: np.array = None
    ):
        if prediction is None or label is None:
            # Build the path for the predictions and labels
            prediction_path = self.prediction_path / subject_name / phase
            label_path = self.label_path / subject_name / phase / self.label_suffix

            # Check if the paths exist
            if not prediction_path.exists() or not label_path.exists():
                return None

            prediction = self._get_collected_array(prediction_path)
            label = self._get_collected_array(label_path)

        # Calculate the F1 score
        return self._calculate_f1_score(prediction, label)

    def _calculate_correlation_for_subject_and_phase(
        self, subject_name, phase, prediction: np.array = None, label: np.array = None
    ):
        if prediction is None or label is None:
            # Build the path for the predictions and labels
            prediction_path = self.prediction_path / subject_name / phase
            label_path = self.label_path / subject_name / phase / self.label_suffix

            # Check if the paths exist
            if not prediction_path.exists() or not label_path.exists():
                return None

            prediction = self._get_collected_array(prediction_path)
            label = self._get_collected_array(label_path)
        if len(prediction) != len(label):
            print(f"Prediction and label length do not match for {subject_name} in phase {phase}")
            return None
        return np.corrcoef(prediction, label)[0, 1]

    def _calculate_ihr_for_subject_and_phase(
        self, subject_name, phase, prediction: np.array = None, label: np.array = None
    ):
        if prediction is None or label is None:
            # Build the path for the predictions and labels
            prediction_path = self.prediction_path / subject_name / phase
            label_path = self.label_path / subject_name / phase / self.label_suffix

            # Check if the paths exist
            if not prediction_path.exists() or not label_path.exists():
                return None

            prediction = self._get_collected_array(prediction_path)
            label = self._get_collected_array(label_path)

        minimal_distance_between_peaks = int(1 / (self.max_heart_rate / 60) * self.fs)

        # Find the peaks
        peaks_predicted, _ = find_peaks(prediction, distance=minimal_distance_between_peaks, prominence=0.3)
        peaks_label, _ = find_peaks(label, distance=minimal_distance_between_peaks, prominence=0.3)

        # Calculate the difference
        diff_predicted = np.diff(peaks_predicted)
        diff_label = np.diff(peaks_label)

        # Calculate the instantaneous heart rate
        diff_predicted = diff_predicted / self.fs
        diff_label = diff_label / self.fs

        median_diff_predicted = np.median(diff_predicted)
        median_diff_label = np.median(diff_label)
        median_hr_predicted = 60 / median_diff_predicted
        median_hr_label = 60 / median_diff_label

        instantaneous_heart_rate_predicted = 60 / diff_predicted
        instantaneous_heart_rate_label = 60 / diff_label

        return instantaneous_heart_rate_predicted, instantaneous_heart_rate_label

    def calculate_scores_already_collected_arrays(self, base_path):
        scores = {}
        for subject in base_path.iterdir():
            if not subject.is_dir():
                continue
            subject_name = subject.name
            scores[subject_name] = {}
            for phase in subject.iterdir():
                if not phase.is_dir():
                    continue
                phase_name = phase.name
                prediction = np.load(phase / "prediction.npy")
                label = np.load(phase / "label.npy")
                scores[subject_name][phase_name] = self._get_scores_for_subject_and_phase_already_collected(
                    subject_name, phase_name, prediction, label
                )
        df = pd.DataFrame.from_dict(
            {(subject, phase): value for subject, inner_dict in scores.items() for phase, value in inner_dict.items()},
            orient="index",
        )
        df.columns = ["F1_score_100", "F1_score_50", "correlation", "ihr_predicted_median", "ihr_label_median"]
        return df

    def calculate_scores(self):
        scores = {}
        for subject in self.prediction_path.iterdir():
            if not subject.is_dir():
                continue
            subject_name = subject.name
            scores[subject_name] = {}
            for phase in subject.iterdir():
                if not phase.is_dir():
                    continue
                phase_name = phase.name
                scores[subject_name][phase_name] = self._get_scores_for_subject_and_phase(subject_name, phase_name)
        print(scores)
        df = pd.DataFrame.from_dict(
            {(subject, phase): value for subject, inner_dict in scores.items() for phase, value in inner_dict.items()},
            orient="index",
        )
        if len(df.columns) != 5:
            print(df)
            return df
        df.columns = ["F1_score_100", "F1_score_50", "correlation", "ihr_predicted_median", "ihr_label_median"]
        return df

    def _get_scores_for_subject_and_phase(self, subject_name, phase):
        prediction, label = self.save_collected_array(subject_name, phase)
        if prediction is None or label is None:
            return None
        f1_score_100, f1_score_50 = self._calculate_f1_score_for_subject_and_phase(
            subject_name, phase, prediction, label
        )
        correlation = self._calculate_correlation_for_subject_and_phase(subject_name, phase, prediction, label)
        ihr_predicted, ihr_labels = self._calculate_ihr_for_subject_and_phase(subject_name, phase, prediction, label)
        median_ihr_predicted = np.median(ihr_predicted)
        median_ihr_label = np.median(ihr_labels)
        return [f1_score_100, f1_score_50, correlation, median_ihr_predicted, median_ihr_label]

    def _get_collected_array(self, path: Path):
        peak_files = self._get_ordered_file_paths(path)
        beats = np.array([])
        for i in range(len(peak_files)):
            beat = np.load(peak_files[i])
            if i == 0:
                # Start to end of middle
                middle = self._get_first_interval(beat)
            elif i == len(peak_files) - 1:
                # Start of the middle to end
                middle = self._get_last_interval(beat)
            else:
                middle = self._get_middle_of_interval(beat)
            beats = np.append(beats, middle)
        return beats

    def _get_first_interval(self, array: np.array):
        if array.ndim != 1:
            raise ValueError("Array must be 1-dimensional")
        percentile = self.overlap / 2 * len(array)
        end = int(len(array) - percentile)
        return array[:end]

    def _get_last_interval(self, array: np.array):
        if array.ndim != 1:
            raise ValueError("Array must be 1-dimensional")
        start = int(self.overlap / 2 * len(array))
        return array[start:]

    def _calculate_f1_score(self, prediction, label):
        f1RPeakScore_100 = RPeakF1Score(max_deviation_ms=100, prominence=self.prominence)
        f1RPeakScore_50 = RPeakF1Score(max_deviation_ms=50, prominence=self.prominence)

        # Calculate the F1 score
        f1_score_100 = f1RPeakScore_100.compute(prediction, label).f1_score_
        f1_score_50 = f1RPeakScore_50.compute(prediction, label).f1_score_

        return f1_score_100, f1_score_50

    def save_collected_array(self, subject_name, phase):
        prediction_path = self.prediction_path / subject_name / phase
        label_path = self.label_path / subject_name / phase / self.label_suffix

        print(f"in Save collected array: Pred path {prediction_path},Label path {label_path}")

        # Check if the paths exist
        if not prediction_path.exists() or not label_path.exists():
            return None, None

        prediction = self._get_collected_array(prediction_path)
        label = self._get_collected_array(label_path)

        if os.getenv("WORK") is None:
            save_path = (
                Path("/Users/simonmeske/Desktop/Masterarbeit/CollectedArrays")
                / self.prediction_path.stem
                / subject_name
                / phase
            )
        else:
            save_path = Path(os.getenv("WORK")) / "CollectedArrays" / self.prediction_path.stem / subject_name / phase
        if not save_path.exists():
            save_path.mkdir(parents=True)

        np.save(save_path / "prediction.npy", prediction)
        np.save(save_path / "label.npy", label)
        return prediction, label

    def _get_scores_for_subject_and_phase_already_collected(self, subject_name, phase, prediction, label):
        f1_score_100, f1_score_50 = self._calculate_f1_score_for_subject_and_phase(
            subject_name, phase, prediction, label
        )
        correlation = self._calculate_correlation_for_subject_and_phase(subject_name, phase, prediction, label)
        ihr_predicted, ihr_labels = self._calculate_ihr_for_subject_and_phase(subject_name, phase, prediction, label)
        median_ihr_predicted = np.median(ihr_predicted)
        median_ihr_label = np.median(ihr_labels)
        return [f1_score_100, f1_score_50, correlation, median_ihr_predicted, median_ihr_label]
