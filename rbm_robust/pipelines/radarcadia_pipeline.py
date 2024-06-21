from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
from tpcp import OptimizablePipeline
import tensorflow as tf
from tpcp._dataset import DatasetT

from rbm_robust.data_loading.tf_datasets import DatasetFactory
from rbm_robust.models.waveletModel import UNetWaveletTF
from rbm_robust.validation.RPeakF1Score import RPeakF1Score


def _get_dataset(
    data_path,
    subjects,
    batch_size: int = 8,
    breathing_type: str = "all",
    wavelet_type: str = "morl",
    ecg_labels: bool = False,
    log_transform: bool = False,
    image_based: bool = False,
    dual_channel: bool = False,
    identity: bool = False,
) -> (tf.data.Dataset, int):
    ds_factory = DatasetFactory()
    dataset, steps = ds_factory.get_wavelet_dataset_for_subjects_radarcadia(
        base_path=data_path,
        subjects=subjects,
        batch_size=batch_size,
        breathing_type=breathing_type,
        wavelet_type=wavelet_type,
        ecg_labels=ecg_labels,
        log_transform=log_transform,
        image_based=image_based,
        dual_channel=dual_channel,
        identity=identity,
    )
    return dataset, steps


class RadarcadiaPipeline(OptimizablePipeline):
    wavelet_model: UNetWaveletTF
    result_ = np.ndarray
    training_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset
    testing_subjects: list
    epochs: int
    learning_rate: float
    data_path: str
    testing_path: str
    ecg_labels: bool
    log_transform: bool
    breathing_type: str
    training_subjects: list
    validation_subjects: list
    wavelet_type: str
    batch_size: int
    image_based: bool
    prediction_folder_name: str
    dual_channel: bool
    identity: bool
    loss: str

    def __init__(
        self,
        learning_rate: float = 0.0001,
        data_path: str = "/home/woody/iwso/iwso116h/Data",
        testing_path: str = "/home/woody/iwso/iwso116h/TestData",
        epochs: int = 50,
        training_subjects: list = None,
        validation_subjects: list = None,
        testing_subjects: list = None,
        wavelet_type: str = "morl",
        breathing_type: str = "all",
        ecg_labels: bool = False,
        log_transform: bool = False,
        batch_size: int = 8,
        image_based: bool = False,
        dual_channel: bool = False,
        identity: bool = False,
        loss: str = "bce",
    ):
        # Set the different fields
        self.loss = loss
        self.identity = identity
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.data_path = data_path
        self.testing_path = testing_path
        self.testing_subjects = testing_subjects
        self.image_based = image_based
        self.dual_channel = dual_channel
        self.training_ds, self.training_steps = _get_dataset(
            data_path=data_path,
            subjects=training_subjects,
            breathing_type=breathing_type,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            image_based=image_based,
            dual_channel=dual_channel,
            identity=identity,
        )
        self.validation_ds, self.validation_steps = _get_dataset(
            data_path=data_path,
            subjects=validation_subjects,
            breathing_type=breathing_type,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            image_based=image_based,
            dual_channel=dual_channel,
            identity=identity,
        )
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.breathing_type = breathing_type
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size

        learning_rate_txt = str(learning_rate).replace(".", "_")
        model_name = f"radarcadia_{wavelet_type}_{breathing_type}_{epochs}_{learning_rate_txt}_{loss}"
        if ecg_labels:
            model_name += "_ecg"
        if log_transform:
            model_name += "_log"
        if image_based:
            model_name += "_image"
        if dual_channel:
            model_name += "_dual"
        if identity:
            model_name += "_identity"

        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prediction_folder_name = f"predictions_{model_name}_{time}"

        # Initialize the model
        self.wavelet_model = UNetWaveletTF(
            learning_rate=learning_rate,
            epochs=epochs,
            model_name=model_name,
            training_steps=self.training_steps,
            validation_steps=self.validation_steps,
            training_ds=self.training_ds,
            validation_ds=self.validation_ds,
            batch_size=self.batch_size,
            image_based=image_based,
            loss=loss,
            dual_channel=dual_channel,
        )

    def self_optimize(self):
        self.wavelet_model.self_optimize()
        return self

    def run(self, path_to_save_predictions: str, image_based: bool = False, identity: bool = False):
        input_folder_name = f"inputs_wavelet_array_{self.wavelet_type}"
        if self.log_transform and not self.dual_channel:
            input_folder_name += "_log"
        if self.image_based:
            input_folder_name = input_folder_name.replace("array", "image")
        if identity:
            input_folder_name = input_folder_name.replace("wavelet", "identity")

        self.wavelet_model.predict(
            testing_subjects=self.testing_subjects,
            data_path=self.testing_path,
            input_folder_name=input_folder_name,
            prediction_folder_name=self.prediction_folder_name,
        )
        return self

    def score(self, datapoint: DatasetT) -> Union[float, dict[str, float]]:
        true_positives = 0
        total_gt_peaks = 0
        total_pred_peaks = 0

        test_data_folder_name = Path(self.testing_path).name
        label_folder_name = "labels_gaussian" if not self.ecg_labels else "labels_ecg"
        test_path = Path(self.testing_path)

        for subject in test_path.iterdir():
            if not subject.is_dir():
                continue
            if subject.name not in self.testing_subjects:
                continue
            print(f"subject {subject}")
            for phase in subject.iterdir():
                if not phase.is_dir():
                    continue
                if phase.name == "logs" or phase.name == "raw":
                    continue
                print(f"phase {phase}")
                prediction_path = phase
                prediction_path = Path(
                    str(prediction_path).replace(test_data_folder_name, f"Predictions/{self.prediction_folder_name}")
                )
                label_path = phase / label_folder_name
                prediction_files = sorted(path.name for path in prediction_path.iterdir() if path.is_file())
                f1RPeakScore = RPeakF1Score(max_deviation_ms=100)
                for prediction_file in prediction_files:
                    prediction = np.load(prediction_path / prediction_file)
                    label = np.load(label_path / prediction_file)
                    f1RPeakScore.compute_predictions(prediction, label)
                    true_positives += f1RPeakScore.tp_
                    total_gt_peaks += f1RPeakScore.total_peaks_
                    total_pred_peaks += f1RPeakScore.pred_peaks_

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
