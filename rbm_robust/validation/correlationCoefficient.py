from rbm_robust.validation.validationBase import ValidationBase
from pathlib import Path
import numpy as np


class Correlation(ValidationBase):
    def __init__(
        self, phase: str, subject: str, prediction_path: Path, label_path: Path, overlap: int = 0.4, fs: int = 200
    ):
        self.phase = phase
        self.subject = subject
        self.prediction_path = prediction_path
        self.label_path = label_path
        self.overlap = overlap
        self.fs = fs

    def calculate_correlation(self):
        prediction = self._get_collected_array(self.prediction_path)
        label = self._get_collected_array(self.label_path)
        return np.corrcoef(prediction, label)[0, 1]


class CorrelationAllSubjects(ValidationBase):
    def __init__(self, prediction_path: Path, label_path: Path, overlap: int = 0.4, fs: int = 200):
        self.prediction_path = prediction_path
        self.label_path = label_path
        self.overlap = overlap
        self.fs = fs

    def calculate_correlation_all_subjects_for_phase(self, phase: str):
        subject_corr_dict = {}
        for subject_path in self.label_path.iterdir():
            if not subject_path.is_dir():
                continue
            subject = subject_path.name
            if not (subject_path / phase).exists() or not (self.prediction_path / subject / phase).exists():
                continue
            correlation_calculator = Correlation(
                phase=phase, subject=subject, prediction_path=self.prediction_path, label_path=self.label_path
            )
            subject_corr_dict[subject] = correlation_calculator.calculate_correlation()
        return subject_corr_dict
