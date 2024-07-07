from pathlib import Path
import numpy as np


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
        percentile = (1 - self.overlap) / 2
        end = int(len(array) / 2 + percentile * len(array))
        return array[:end]

    def _get_last_interval(self, array: np.array):
        if array.ndim != 1:
            raise ValueError("Array must be 1-dimensional")
        percentile = (1 - self.overlap) / 2
        start = int(len(array) / 2 - percentile * len(array))
        return array[start:]

    def _get_middle_of_interval(self, array: np.array):
        if array.ndim != 1:
            raise ValueError("Array must be 1-dimensional")
        percentile = self.overlap / 2 * len(array)
        start = int(percentile)
        end = int(len(array) - percentile)
        return array[start:end]

    def _get_ordered_file_paths(self, path: Path):
        # array_path = path / self.subject / self.phase
        array_path = path
        print("Array path: ", array_path)
        if not array_path.exists():
            raise ValueError(f"Path {array_path} does not exist")
        file_paths = list(array_path.iterdir())
        file_paths = [file_path for file_path in file_paths if file_path.is_file() and file_path.suffix == ".npy"]
        return sorted(file_paths, key=lambda x: int(x.name.split(".")[0]))
