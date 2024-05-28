import numpy as np
from rbm_robust.validation.validationBase import ValidationBase
from pathlib import Path
import pathlib
from scipy.signal import find_peaks


class HeartRate(ValidationBase):
    phase: str
    subject: str
    prediction_path: Path
    label_path: Path
    overlap: float
    fs: int

    def __init__(
        self, phase: str, subject: str, prediction_path: Path, label_path: Path, overlap: int = 0.4, fs: int = 200
    ):
        self.phase = phase
        self.subject = subject
        self.prediction_path = prediction_path
        self.label_path = label_path
        self.overlap = overlap
        self.fs = fs

    def _calculate_heart_rate(self, path: Path, peak_height: float = 0.8):
        # Get all files in the path
        beats = self._get_collected_array(path)
        # Calculate the heart rate
        r_peaks = find_peaks(beats, height=peak_height)
        total_time = len(beats) / self.fs
        total_time = total_time / 60
        heart_rate = len(r_peaks[0]) / total_time
        return heart_rate

    def calculate_heart_rate_prediction(self):
        return self._calculate_heart_rate(self.prediction_path, peak_height=0.1)

    def calculate_heart_rate_label(self):
        return self._calculate_heart_rate(self.label_path, peak_height=0.5)

    def calculate_heart_rate_difference(self):
        return np.abs(self.calculate_heart_rate_prediction() - self.calculate_heart_rate_label())
