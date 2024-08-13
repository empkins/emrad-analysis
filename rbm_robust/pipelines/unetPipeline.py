import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Union
from tpcp._dataset import DatasetT
import tensorflow as tf
import numpy as np
from tpcp import OptimizablePipeline
from rbm_robust.data_loading.tf_datasets import DatasetFactory
from rbm_robust.models.waveletModel import UNetWaveletTF
from rbm_robust.validation.score_calculator import ScoreCalculator


def _get_dataset(
    data_path,
    subjects,
    batch_size: int = 8,
    wavelet_type: str = "morl",
    ecg_labels: bool = False,
    log_transform: bool = False,
    single_channel: bool = True,
) -> (tf.data.Dataset, int):
    ds_factory = DatasetFactory()
    if single_channel:
        return ds_factory.get_single_channel_wavelet_dataset_for_subjects(
            base_path=data_path,
            training_subjects=subjects,
            batch_size=batch_size,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
        )
    else:
        return ds_factory.get_dual_channel_wavelet_dataset_for_subjects(
            base_path=data_path,
            training_subjects=subjects,
            batch_size=batch_size,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
        )


class PreTrainedPipeline(OptimizablePipeline):
    wavelet_model: UNetWaveletTF
    result_ = np.ndarray
    testing_subjects: list
    testing_path: Path
    ecg_labels: bool
    log_transform: bool
    wavelet_type: str
    batch_size: int
    image_based: bool
    prediction_folder_name: str
    dual_channel: bool

    def __init__(
        self,
        wavelet_type: str = "morl",
        ecg_labels: bool = False,
        log_transform: bool = False,
        batch_size: int = 8,
        image_based: bool = False,
        dual_channel: bool = False,
        model_path: str = None,
        testing_subjects: list = None,
        testing_path: Path = Path("/home/woody/iwso/iwso116h/TestData"),
    ):
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size
        self.image_based = image_based
        self.dual_channel = dual_channel
        self.model_path = model_path
        self.testing_subjects = testing_subjects
        self.testing_path = testing_path
        model_name = Path(model_path).stem

        self.prediction_folder_name = f"predictions_pretrained_{model_name}"

        # Initialize the model
        self.wavelet_model = UNetWaveletTF(
            model_name=model_name,
            batch_size=batch_size,
            image_based=image_based,
            dual_channel=dual_channel,
            model_path=model_path,
        )

    def run(self, path_to_save_predictions: str):
        input_folder_name = f"inputs_wavelet_array_{self.wavelet_type}"
        if self.log_transform and not self.dual_channel:
            input_folder_name += "_log"
        if self.image_based:
            input_folder_name = input_folder_name.replace("array", "image")

        self.wavelet_model.predict(
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

        prominences = [round(i, 2) for i in np.arange(0.05, 0.36, 0.05)]
        for prominence in prominences:
            score_calculator = ScoreCalculator(
                prediction_path=prediction_path,
                label_path=label_path,
                overlap=int(0.4),
                fs=200,
                label_suffix=label_folder_name,
                prominence=prominence,
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
            scores.to_csv(score_path / f"scores_{self.prediction_folder_name}_{prominence}.csv")

        # Tar the predictions
        self.tar_predictions(prediction_path)

        # Delete the prediction Directory
        shutil.rmtree(prediction_path)

        return scores

    def tar_predictions(self, prediction_path):
        output_filename = str(prediction_path) + ".tar"
        with tarfile.open(output_filename, "w") as tar:
            tar.add(prediction_path, arcname=os.path.basename(prediction_path))


class D02Pipeline(OptimizablePipeline):
    wavelet_model: UNetWaveletTF
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
    diff: bool

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
        ecg_labels: bool = False,
        log_transform: bool = False,
        batch_size: int = 8,
        image_based: bool = False,
        dual_channel: bool = False,
        identity: bool = False,
        loss: str = "bce",
        diff: bool = False,
    ):
        # Set the different fields
        self.diff = diff
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
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            single_channel=not self.dual_channel,
        )

        self.validation_ds, self.validation_steps = _get_dataset(
            data_path=data_path,
            subjects=validation_subjects,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            single_channel=not self.dual_channel,
        )
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size

        learning_rate_txt = str(learning_rate).replace(".", "_")
        model_name = f"d02_{wavelet_type}_{epochs}_{learning_rate_txt}_{loss}"
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
        if diff:
            model_name += "_diff"

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

        if self.diff:
            input_folder_name = "inputs_wavelet_array_diff"

        self.wavelet_model.predict(
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
            save_path = Path("~/Masterarbeit")
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
