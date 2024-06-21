import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional
import keras
import numpy as np
from keras import Sequential, layers
from keras.src.utils import load_img, img_to_array
from keras_unet_collection import models
from tpcp import Algorithm, OptimizableParameter
from rbm_robust.data_loading.tf_datasets import DatasetFactory
import tensorflow as tf


class UNetWaveletTF(Algorithm):
    learning_rate: OptimizableParameter[float]
    training_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset
    prediction_base_path: str = "/home/woody/iwso/iwso116h/Predictions"
    prediction_folder_name: str = "predictions_cwt_bce_75_0001_sigmoid"
    model_name: str
    epochs: int
    training_steps: int
    validation_steps: int
    _model = Optional[keras.Sequential]
    batch_size: int
    image_based: bool
    loss: str
    dual_channel: bool

    def __init__(
        self,
        learning_rate: float = 0.0001,
        training_ds: tf.data.Dataset = None,
        validation_ds: tf.data.Dataset = None,
        model_name: str = None,
        epochs: int = 50,
        training_steps: int = 0,
        validation_steps: int = 0,
        batch_size: int = 8,
        image_based=False,
        loss: str = "bce",
        dual_channel: bool = False,
    ):
        self.learning_rate = learning_rate
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.model_name = model_name
        self.epochs = epochs
        self.training_steps = training_steps
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self.image_based = image_based
        self.loss = loss
        self.dual_channel = dual_channel

    def self_optimize(
        self,
    ):
        self._create_model()
        # log_dir = os.getenv("HPCVAULT") + f"/Runs/logs/fit/{self.model_name}"
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        #
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, update_freq="epoch")
        print("Fitting")
        history = self._model.fit(
            self.training_ds,
            epochs=self.epochs,
            steps_per_epoch=self.training_steps,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=self.validation_ds,
            validation_steps=self.validation_steps,
            verbose=1,
            # callbacks=[tensorboard_callback],
        )

        # history_path = os.getenv("HPCVAULT") + "/Runs/History/"
        # if not os.path.exists(history_path):
        #     os.makedirs(history_path)
        # history_path += self.model_name + "_history.pkl"
        # pickle.dump(history.history, open(history_path, "wb"))
        self.save_model()
        return self

    def _create_model(self):
        self._model = Sequential()
        channel_number = self._get_channel_number()

        self._model.add(
            models.unet_2d(
                (256, 1000, channel_number),
                filter_num=[16, 32, 64],
                weights=None,
                freeze_backbone=False,
                freeze_batch_norm=False,
                output_activation=None,
                n_labels=channel_number,
            )
        )
        self._model.add(layers.Conv2D(filters=1, kernel_size=(256, 1), activation="sigmoid"))
        self._model.add(layers.Flatten())
        loss_func = None
        if self.loss == "bce":
            loss_func = keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
        else:
            loss_func = keras.losses.MeanSquaredError(reduction="none")
        self._model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=loss_func)
        return self

    def _get_channel_number(self):
        if self.dual_channel and not self.image_based:
            return 2
        elif self.dual_channel and self.image_based:
            return 6
        elif not self.dual_channel and not self.image_based:
            return 1
        elif not self.dual_channel and self.image_based:
            return 3
        else:
            raise ValueError("Invalid combination of dual_channel and image_based")

    def save_model(self):
        name = self.model_name + datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists("Models"):
            os.makedirs("Models")
        self._model.save("Models/" + name + ".keras")
        with open("Models/" + name + "_history.pkl", "wb") as f:
            pickle.dump(self._model.history, f)
        return self

    def predict(
        self,
        testing_subjects: list[str] = None,
        data_path: str = "/home/woody/iwso/iwso116h/TestData",
        input_folder_name: str = "inputs",
        prediction_folder_name: str = "predictions_unnnamed",
    ):
        print("Prediction started")
        data_path = Path(data_path)
        data_folder_name = data_path.name
        print(data_path)
        input_file_type = "npy" if not self.image_based else "png"
        subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
        if testing_subjects is not None:
            subjects = [subject for subject in subjects if subject in testing_subjects]
        for subject_id in subjects:
            print(subject_id)
            subject_path = data_path / subject_id
            for phase_path in subject_path.iterdir():
                if not phase_path.is_dir():
                    continue
                input_path = phase_path / input_folder_name
                prediction_path = phase_path
                prediction_path = Path(
                    str(prediction_path).replace(data_folder_name, f"Predictions/{prediction_folder_name}")
                )
                prediction_path.mkdir(parents=True, exist_ok=True)
                input_files = sorted(input_path.glob(f"*.{input_file_type}"))
                for input_file in input_files:
                    if not self.image_based:
                        img_input = np.load(input_file)
                    else:
                        img_input = img_to_array(load_img(input_file, target_size=(256, 1000))) / 255
                    img_input = self._get_input_array(input_file)
                    img_input = np.array([img_input])
                    pred = self._model.predict(img_input, verbose=0)
                    pred = pred.flatten()
                    filename = input_file.stem + ".npy"
                    save_path = Path(str(prediction_path)) / filename
                    np.save(save_path, pred)
        return self

    def _get_input_array(self, input_file):
        if not self.image_based and not self.dual_channel:
            return np.load(input_file)
        elif self.image_based and not self.dual_channel:
            return img_to_array(load_img(input_file, target_size=(256, 1000))) / 255
        elif self.dual_channel and self.image_based:
            img = img_to_array(load_img(input_file, target_size=(256, 1000))) / 255
            second_channel_path = Path(str(input_file.parent) + "_log") / input_file.name
            second_channel = img_to_array(load_img(second_channel_path, target_size=(256, 1000))) / 255
            img = np.dstack((img, second_channel))
            return img
        elif self.dual_channel and not self.image_based:
            img = np.load(input_file)
            second_channel_path = Path(str(input_file.parent) + "_log") / input_file.name
            second_channel = np.load(second_channel_path)
            img = np.dstack((img, second_channel))
            return img


class UNetWavelet(Algorithm):
    _action_methods = "predict"

    # Input Parameters
    learning_rate: OptimizableParameter[float]
    batch_size: OptimizableParameter[int]
    training_subjects: list = None
    validation_subjects: list = None
    base_path: str = "/home/woody/iwso/iwso116h/Data"

    # Model
    _model = Optional[keras.Sequential]

    # Results
    predictions_: np.ndarray

    def __init__(
        self,
        learning_rate: float = 0.0001,
        num_epochs: int = 75,
        batch_size: int = 16,
        _model=None,
        image_based: bool = False,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_based = image_based
        self._model = _model

    def predict(
        self,
        data_path: str = "/home/woody/iwso/iwso116h/TestData",
        testing_subjects: list = None,
    ):
        print("Prediction started")
        data_path = Path(data_path)
        subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
        if testing_subjects is not None:
            subjects = [subject for subject in subjects if subject in testing_subjects]
        for subject_id in subjects:
            subject_path = data_path / subject_id
            for phase_path in subject_path.iterdir():
                if not phase_path.is_dir():
                    continue
                input_path = phase_path / "inputs"
                prediction_path = phase_path
                prediction_path = Path(
                    str(prediction_path).replace("TestData", "Predictions/predictions_cwt_bce_75_0001_sigmoid")
                )
                prediction_path.mkdir(parents=True, exist_ok=True)
                input_files = sorted(input_path.glob("*.png"))
                for input_file in input_files:
                    img_input = img_to_array(load_img(input_file, target_size=(256, 1000))) / 255
                    img_input = np.array([img_input])
                    pred = self._model.predict(img_input, verbose=0)
                    pred = pred.flatten()
                    filename = input_file.stem + ".npy"
                    np.save(prediction_path / filename, pred)
        return self

    def self_optimize(
        self,
        base_path: str = "/home/woody/iwso/iwso116h/Data",
        image_based: bool = False,
        training_subjects: list = None,
        validation_subjects: list = None,
        model_path: str = None,
        start_epoch: int = 0,
        remaining_epochs: int = 0,
    ):
        self.base_path = base_path
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects

        if not image_based and model_path is None:
            self._create_model()
        elif model_path is None and image_based:
            self._image_model()
        else:
            self._model = keras.saving.load_model(model_path)

        log_dir = os.getenv("WORK") + "/Runs/logs/fit/"
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir += time
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, update_freq="epoch")

        print("Before Generators")
        dataset_factory = DatasetFactory()
        training_dataset, training_steps = dataset_factory.get_wavelet_dataset_for_subjects(
            base_path, training_subjects, batch_size=self.batch_size
        )
        validation_dataset, validation_steps = dataset_factory.get_wavelet_dataset_for_subjects(
            base_path, validation_subjects, batch_size=self.batch_size
        )

        print("Fitting")
        if model_path is None:
            history = self._model.fit(
                training_dataset,
                epochs=self.num_epochs,
                steps_per_epoch=training_steps,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=validation_dataset,
                validation_steps=validation_steps,
                verbose=1,
                callbacks=[tensorboard_callback],
            )
        else:
            history = self._model.fit(
                training_dataset,
                epochs=remaining_epochs + start_epoch,
                steps_per_epoch=training_steps,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=validation_dataset,
                validation_steps=validation_steps,
                verbose=1,
                callbacks=[tensorboard_callback],
                initial_epoch=start_epoch,
            )

        history_path = os.getenv("WORK") + "/Runs/History/"
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        history_path += time + "_history.pkl"
        pickle.dump(history.history, open(history_path, "wb"))
        return self

    def _create_model(self):
        self._model = Sequential()
        self._model.add(
            models.unet_2d(
                (256, 1000, 3),
                filter_num=[16, 32, 64],
                weights=None,
                freeze_backbone=False,
                freeze_batch_norm=False,
                output_activation=None,
                n_labels=3,
            )
        )
        self._model.add(layers.Conv2D(filters=1, kernel_size=(256, 1), activation="sigmoid"))
        # self._model.add(layers.Dense(1000, activation="linear"))
        self._model.add(layers.Flatten())
        loss_func_bce = keras.losses.BinaryCrossentropy(from_logits=False, reduction="sum_over_batch_size")
        # loss_func_mse = keras.losses.MeanSquaredError(reduction="sum_over_batch_size")
        self._model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=loss_func_bce)
        return self

    def save_model(self):
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists("Models"):
            os.makedirs("Models")
        self._model.save("Models/" + name + ".keras")
        with open("Models/" + name + "_history.pkl", "wb") as f:
            pickle.dump(self._model.history, f)
        return self
