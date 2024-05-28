from pathlib import Path
from scipy.signal import find_peaks
from rbm_robust.validation.validationBase import ValidationBase
import numpy as np


class InterbeatInterval(ValidationBase):
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

    def _calculate_interbeat_interval(self, path: Path, peak_height: float = 0.5):
        # Get all files in the path
        beats = self._get_collected_array(path)
        # Calculate the interbeat interval
        r_peaks = find_peaks(beats, height=peak_height)[0]
        if len(r_peaks) == 0:
            print(f"No peaks found in {path}")
            return np.array([])
        inter_beat_intervals = np.diff(r_peaks)
        # Convert in seconds
        inter_beat_intervals = inter_beat_intervals / self.fs
        return inter_beat_intervals

    def calculate_interbeat_interval_prediction(self):
        return self._calculate_interbeat_interval(self.prediction_path, peak_height=0.1)

    def calculate_interbeat_interval_label(self):
        return self._calculate_interbeat_interval(self.label_path, peak_height=0.9)

    def calculate_interbeat_interval_difference(self):
        prediction = self.calculate_interbeat_interval_prediction()
        label = self.calculate_interbeat_interval_label()
        diffs = np.abs(prediction - label)
        return diffs
