import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from tpcp import OptimizablePipeline
from tpcp._dataset import DatasetT

from rbm_robust.data_loading.tf_datasets import DatasetFactory
from rbm_robust.models.lstmModel import LSTM
from rbm_robust.validation.instantenous_heart_rate import ScoreCalculator


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
    biLSTM_model: LSTM
    result_ = np.ndarray
    training_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset
    testing_subjects: list
    epochs: int
    learning_rate: float
    data_path: str
    testing_path: Path
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
        testing_path: Path = Path("/home/woody/iwso/iwso116h/TestData"),
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
            batch_size=batch_size,
        )
        self.validation_ds, self.validation_steps = _get_dataset(
            data_path=data_path,
            subjects=validation_subjects,
            batch_size=batch_size,
        )
        print(f"Training steps: {self.training_steps}")
        print(f"Validation steps: {self.validation_steps}")
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.breathing_type = breathing_type
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size

        learning_rate_txt = str(learning_rate).replace(".", "_")
        model_name = f"time_power_{wavelet_type}_{breathing_type}_{epochs}_{learning_rate_txt}_{loss}"
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

        num_samples = 100  # number of samples in the dataset
        x_data = np.random.random((num_samples, 1000, 5)).astype(np.float32)
        y_data = np.random.random((num_samples, 1000)).astype(np.float32)

        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.batch(32)  # batch size of 32

        # Initialize the model
        self.biLSTM_model = LSTM(
            learning_rate=learning_rate,
            epochs=epochs,
            model_name=model_name,
            training_steps=3,
            validation_steps=3,
            training_ds=self.training_ds,
            validation_ds=self.training_ds,
            batch_size=8,
            loss=loss,
            dual_channel=dual_channel,
        )

    def self_optimize(self):
        self.biLSTM_model.self_optimize()
        return self

    def run(self, path_to_save_predictions: str, image_based: bool = False, identity: bool = False):
        input_folder_name = f"filtered_radar"
        self.biLSTM_model.predict(
            testing_subjects=self.testing_subjects,
            data_path=self.testing_path,
            input_folder_name=input_folder_name,
            prediction_folder_name=self.prediction_folder_name,
        )
        return self

    def score(self, datapoint: DatasetT) -> Union[float, dict[str, float]]:
        test_data_folder_name = Path(self.testing_path).name
        label_folder_name = "labels_gaussian" if not self.ecg_labels else "labels_ecg"
        test_path = Path(self.testing_path)

        label_path = test_path
        prediction_path = Path(
            str(label_path).replace(test_data_folder_name, f"Predictions/{self.prediction_folder_name}")
        )

        score_calculator = ScoreCalculator(
            prediction_path=prediction_path,
            label_path=label_path,
            overlap=0.4,
            fs=200,
            label_suffix=label_folder_name,
        )

        if os.getenv("WORK") is None:
            save_path = Path("/Users/simonmeske/Desktop/Masterarbeit")
        else:
            save_path = Path(os.getenv("WORK"))

        scores = score_calculator.calculate_scores()
        # Save the scores as a csv file
        score_path = save_path / "Scores"
        if not score_path.exists():
            score_path.mkdir(parents=True)
        scores.to_csv(score_path / f"scores_{self.prediction_folder_name}.csv")

        # Tar the predictions
        self.tar_predictions(prediction_path)

        # Delete the prediction Directory
        shutil.rmtree(prediction_path)

        print(f"Scores: {scores}")
        return scores

    def tar_predictions(self, prediction_path):
        output_filename = str(prediction_path) + ".tar"
        with tarfile.open(output_filename, "w") as tar:
            tar.add(prediction_path, arcname=os.path.basename(prediction_path))
