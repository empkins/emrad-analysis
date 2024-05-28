from pathlib import Path
import pathlib
import numpy as np
from scipy.signal import find_peaks


class ValidationBase:
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

    def _get_collected_array(self, path: Path):
        peak_files = self._get_ordered_file_paths(path)
        beats = np.array([])
        for peak_file in peak_files:
            beat = np.load(peak_file)
            middle = self._get_middle_of_interval(beat)
            beats = np.append(beats, middle)
        return beats

    def _get_middle_of_interval(self, array: np.array):
        if array.ndim != 1:
            raise ValueError("Array must be 1-dimensional")
        percentile = (1 - self.overlap) / 2
        start = int(len(array) / 2 - percentile * len(array))
        end = int(len(array) / 2 + percentile * len(array))
        return array[start:end]

    def _get_ordered_file_paths(self, path: Path):
        array_path = path / self.subject / self.phase
        return sorted(array_path.iterdir(), key=lambda x: int(x.name.split(".")[0]))
