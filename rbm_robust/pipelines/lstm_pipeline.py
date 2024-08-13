import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tpcp import OptimizablePipeline
from tpcp._dataset import DatasetT

from rbm_robust.data_loading.tf_datasets import DatasetFactory
from rbm_robust.models.lstmModel import LSTM
from rbm_robust.validation.score_calculator import ScoreCalculator


def _get_dataset(
    data_path,
    subjects,
    batch_size: int = 8,
) -> (tf.data.Dataset, int):
    ds_factory = DatasetFactory()
    dataset, steps = ds_factory.get_time_power_dataset_for_subjects(
        base_path=data_path,
        subjects=subjects,
        batch_size=batch_size,
    )
    return dataset, steps


class MagPipeline(OptimizablePipeline):
    lstm_model: LSTM
    result_: np.ndarray
    training_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset
    testing_subjects: List[str]
    epochs: int
    learning_rate: float
    data_path: str
    testing_path: Path
    ecg_labels: bool
    log_transform: bool
    breathing_type: str
    training_subjects: List[str]
    validation_subjects: List[str]
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
        testing_path: Path = Path("/home/woody/iwso/iwso116h/TestData"),
        epochs: int = 50,
        training_subjects: Optional[List[str]] = None,
        validation_subjects: Optional[List[str]] = None,
        testing_subjects: Optional[List[str]] = None,
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
        self.loss = loss
        self.identity = identity
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.data_path = data_path
        self.testing_path = testing_path
        self.testing_subjects = testing_subjects or []
        self.image_based = image_based
        self.dual_channel = dual_channel
        self.training_ds, self.training_steps = _get_dataset(
            data_path=data_path,
            subjects=training_subjects or [],
            batch_size=batch_size,
        )
        self.validation_ds, self.validation_steps = _get_dataset(
            data_path=data_path,
            subjects=validation_subjects or [],
            batch_size=batch_size,
        )
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.breathing_type = breathing_type
        self.training_subjects = training_subjects or []
        self.validation_subjects = validation_subjects or []
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size

        self.prediction_folder_name = self._generate_prediction_folder_name()

        self.lstm_model = LSTM(
            learning_rate=learning_rate,
            epochs=epochs,
            model_name=self.prediction_folder_name,
            training_steps=self.training_steps,
            validation_steps=self.validation_steps,
            training_ds=self.training_ds,
            validation_ds=self.validation_ds,
            batch_size=self.batch_size,
            loss=loss,
            dual_channel=dual_channel,
        )

    def _generate_prediction_folder_name(self) -> str:
        learning_rate_txt = str(self.learning_rate).replace(".", "_")
        model_name = (
            f"time_power_{self.wavelet_type}_{self.breathing_type}_{self.epochs}_{learning_rate_txt}_{self.loss}"
        )
        if self.ecg_labels:
            model_name += "_ecg"
        if self.log_transform:
            model_name += "_log"
        if self.image_based:
            model_name += "_image"
        if self.dual_channel:
            model_name += "_dual"
        if self.identity:
            model_name += "_identity"
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"predictions_{model_name}_{time}"

    def self_optimize(self) -> "MagPipeline":
        self.lstm_model.self_optimize()
        return self

    def run(self, path_to_save_predictions: str, image_based: bool = False, identity: bool = False) -> "MagPipeline":
        input_folder_name = self._generate_input_folder_name(image_based, identity)
        self.lstm_model.predict(
            testing_subjects=self.testing_subjects,
            data_path=self.testing_path,
            input_folder_name=input_folder_name,
            prediction_folder_name=self.prediction_folder_name,
        )
        return self

    def _generate_input_folder_name(self, image_based: bool, identity: bool) -> str:
        input_folder_name = "filtered_radar"
        if self.log_transform and not self.dual_channel:
            input_folder_name += "_log"
        if image_based:
            input_folder_name = input_folder_name.replace("array", "image")
        if identity:
            input_folder_name = input_folder_name.replace("wavelet", "identity")
        return input_folder_name

    def score(self, datapoint: DatasetT):
        test_data_folder_name = self.testing_path.name
        label_folder_name = "labels_gaussian" if not self.ecg_labels else "labels_ecg"
        prediction_path = self._get_prediction_path(test_data_folder_name)

        score_calculator = ScoreCalculator(
            prediction_path=prediction_path,
            label_path=self.testing_path,
            overlap=0.4,
            fs=200,
            label_suffix=label_folder_name,
        )

        save_path = self._get_save_path()
        scores = score_calculator.calculate_scores()
        self._save_scores(scores, save_path)
        self._tar_predictions(prediction_path)
        shutil.rmtree(prediction_path)

        print(f"Scores: {scores}")
        return scores

    def _get_prediction_path(self, test_data_folder_name: str) -> Path:
        return Path(str(self.testing_path).replace(test_data_folder_name, f"Predictions/{self.prediction_folder_name}"))

    def _get_save_path(self) -> Path:
        if os.getenv("WORK") is None:
            return Path("~")
        return Path(os.getenv("WORK"))

    def _save_scores(self, scores: pd.DataFrame, save_path: Path):
        score_path = save_path / "Scores"
        score_path.mkdir(parents=True, exist_ok=True)
        scores.to_csv(score_path / f"scores_{self.prediction_folder_name}.csv")

    def _tar_predictions(self, prediction_path: Path):
        output_filename = str(prediction_path) + ".tar"
        with tarfile.open(output_filename, "w") as tar:
            tar.add(prediction_path, arcname=os.path.basename(prediction_path))
