from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from rbm_robust.validation.RPeakF1Score import RPeakF1Score
from rbm_robust.validation.validationBase import ValidationBase


class ScoreCalculator(ValidationBase):
    def __init__(self, prediction_path, label_path, fs=200, overlap=0.4, label_suffix="labels_ecg"):
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

    def _calculate_f1_score_for_subject_and_phase(self, subject_name, phase):
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

    def _calculate_correlation_for_subject_and_phase(self, subject_name, phase):
        # Build the path for the predictions and labels
        prediction_path = self.prediction_path / subject_name / phase
        label_path = self.label_path / subject_name / phase / self.label_suffix

        # Check if the paths exist
        if not prediction_path.exists() or not label_path.exists():
            return None

        prediction = self._get_collected_array(prediction_path)
        label = self._get_collected_array(label_path)
        return np.corrcoef(prediction, label)[0, 1]

    def _calculate_ihr_for_subject_and_phase(self, subject_name, phase):
        # Build the path for the predictions and labels
        prediction_path = self.prediction_path / subject_name / phase
        label_path = self.label_path / subject_name / phase / self.label_suffix

        # Check if the paths exist
        if not prediction_path.exists() or not label_path.exists():
            return None

        peaks_predicted = self._get_collected_array(prediction_path)
        peaks_label = self._get_collected_array(label_path)

        minimal_distance_between_peaks = int(1 / (self.max_heart_rate / 60) * self.fs)

        # Find the peaks
        peaks_predicted, _ = find_peaks(peaks_predicted, distance=minimal_distance_between_peaks)
        peaks_label, _ = find_peaks(peaks_label, distance=minimal_distance_between_peaks)

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
        return scores

    def _get_scores_for_subject_and_phase(self, subject_name, phase):
        f1_score_100, f1_score_50 = self._calculate_f1_score_for_subject_and_phase(subject_name, phase)
        correlation = self._calculate_correlation_for_subject_and_phase(subject_name, phase)
        ihr_predicted, ihr_labels = self._calculate_ihr_for_subject_and_phase(subject_name, phase)
        median_ihr_predicted = np.median(ihr_predicted)
        median_ihr_label = np.median(ihr_labels)
        return {
            "F1_score_100": f1_score_100,
            "F1_score_50": f1_score_50,
            "correlation": correlation,
            "ihr_predicted_median": median_ihr_predicted,
            "ihr_label_median": median_ihr_label,
        }

    def _get_collected_array(self, path: Path):
        peak_files = self._get_ordered_file_paths(path)
        beats = np.array([])
        for peak_file in peak_files:
            beat = np.load(peak_file)
            middle = self._get_middle_of_interval(beat)
            beats = np.append(beats, middle)
        return beats

    def _calculate_f1_score(self, prediction, label):
        f1RPeakScore_100 = RPeakF1Score(max_deviation_ms=100)
        f1RPeakScore_50 = RPeakF1Score(max_deviation_ms=50)

        # Get the collected Arrays
        # rpeaks_pred = self._get_collected_array(prediction)
        # rpeaks_label = self._get_collected_array(label)

        # Calculate the F1 score
        f1_score_100 = f1RPeakScore_100.compute(prediction, label).f1_score_
        f1_score_50 = f1RPeakScore_50.compute(prediction, label).f1_score_

        return f1_score_100, f1_score_50
