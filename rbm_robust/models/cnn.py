import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import keras
import numpy as np
from keras import Sequential, layers
from keras_unet_collection import models
from tpcp import Algorithm, OptimizableParameter
from itertools import groupby
import tensorflow as tf
from rbm_robust.data_loading.tf_datasets import DatasetFactory


class CNN(Algorithm):
    _action_methods = "predict"

    # Input Parameters
    kernel_size: int
    strides: Tuple[int]
    padding: OptimizableParameter[str]
    dilation_rate: OptimizableParameter[Tuple[int]]
    groups: OptimizableParameter[int]
    activation: OptimizableParameter[str]
    use_bias: OptimizableParameter[bool]
    kernel_initializer: OptimizableParameter[str]
    bias_initializer: OptimizableParameter[str]
    learning_rate: OptimizableParameter[float]
    batch_size: OptimizableParameter[int]
    filters: OptimizableParameter[int]
    overlap: int
    training_subjects: list = None
    validation_subjects: list = None
    base_path: str = "/home/woody/iwso/iwso116h/Data"

    # Model
    _model = Optional[keras.Sequential]

    # Results
    predictions_: np.ndarray

    def __init__(
        self,
        filters: int = 64,
        kernel_size: Tuple[int] = (5, 5),
        strides: Tuple[int] = (1, 1),
        padding: str = "valid",
        dilation_rate: Tuple[int] = (1, 1),
        groups: int = 1,
        activation: str = "relu",
        use_bias: bool = True,
        kernel_initializer: str = "he_normal",
        bias_initializer: str = "zeros",
        learning_rate: float = 0.0001,
        num_epochs: int = 25,
        batch_size: int = 8,
        _model=None,
        overlap: int = 0.8,
        image_based: bool = False,
    ):
        self.groups = groups
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.overlap = overlap
        self.image_based = image_based
        self._model = _model

    def _get_middle(self, array):
        percentile = (1 - self.overlap) / 2
        start = int(len(array) / 2 - percentile)
        end = int(len(array) / 2 + percentile)
        return array[start:end]

    def get_steps_per_epoch(self, base_path, training_subjects: list = None):
        base_path = Path(base_path)
        steps = 0
        for subject_path in base_path.iterdir():
            if not subject_path.is_dir():
                continue
            if training_subjects is not None and subject_path.name not in training_subjects:
                continue
            for phase_path in subject_path.iterdir():
                if not phase_path.is_dir():
                    continue
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels_gaussian"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                grouped_inputs = {k: list(g) for k, g in groupby(input_names, key=lambda s: s.split("_")[0])}
                steps += len(grouped_inputs.keys())
        return int(steps / self.batch_size)

    def predict(
        self,
        data_path: str = "/home/woody/iwso/iwso116h/TestData",
        testing_subjects: list = None,
        grouped: bool = False,
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
                prediction_path = Path(str(prediction_path).replace("TestData", "Predictions/predictions_kl_25_epochs"))
                prediction_path.mkdir(parents=True, exist_ok=True)
                input_files = sorted(input_path.glob("*.npy"))
                if grouped:
                    grouped_inputs = {k: list(g) for k, g in groupby(input_files, key=lambda s: s.stem.split("_")[0])}
                    for key, group in grouped_inputs.items():
                        inputs = [np.load(file) for file in group]
                        inputs = np.stack(inputs, axis=0)
                        inputs = np.transpose(inputs)
                        inputs = np.array([inputs])
                        if inputs.shape != (1, 1000, 256, 5):
                            padded = np.zeros((1, 1000, 256, 5))
                            padded[:, : inputs.shape[1], : inputs.shape[2], : inputs.shape[3]] = inputs
                            inputs = padded
                        pred = self._model.predict(inputs)
                        pred = pred.flatten()
                        np.save(prediction_path / f"{key}.npy", pred)
                else:
                    for input_file in input_files:
                        inputs = np.load(input_file)
                        inputs = np.array([inputs])
                        if inputs.shape != (1000, 256, 5):
                            padded = np.zeros((1, 1000, 256, 5))
                            padded[: inputs.shape[0], : inputs.shape[1], : inputs.shape[2], : inputs.shape[3]] = inputs
                            inputs = padded
                        pred = self._model.predict(inputs, verbose=0)
                        pred = pred.flatten()
                        np.save(prediction_path / input_file.name, pred)
        return self

    def self_optimize(
        self,
        base_path: str = "/home/woody/iwso/iwso116h/Data",
        image_based: bool = False,
        training_subjects: list = None,
        validation_subjects: list = None,
    ):
        self.base_path = base_path
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects

        if not image_based:
            self._create_model()
        else:
            self._image_model()

        log_dir = os.getenv("WORK") + "/Runs/logs/fit/"
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir += time
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, update_freq="epoch")

        print("Before Generators")
        dataset_factory = DatasetFactory()
        training_dataset, training_steps_alt = dataset_factory.get_dataset_for_subjects(
            base_path, training_subjects, batch_size=self.batch_size
        )
        validation_dataset, validation_steps_alt = dataset_factory.get_dataset_for_subjects(
            base_path, validation_subjects, batch_size=self.batch_size
        )

        print("Getting steps per epoch")
        training_steps = self.get_steps_per_epoch(base_path, training_subjects)
        validation_steps = self.get_steps_per_epoch(base_path, validation_subjects)
        print("Got the step count")

        if training_steps_alt != training_steps:
            print(f"Training steps: {training_steps} vs {training_steps_alt} Alt")
        if validation_steps_alt != validation_steps:
            print(f"Validation steps: {validation_steps} vs {validation_steps_alt} Alt")

        print("Fitting")
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

        history_path = os.getenv("WORK") + "/Runs/History/"
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        history_path += time + "_history.pkl"
        pickle.dump(history.history, open(history_path, "wb"))

        return self

    def _image_model(self):
        input_layer = keras.layers.Input(shape=(3, 1000, 150, 5), batch_size=self.batch_size)

        processed_images = []
        for i in range(5):
            img = input_layer[:, i, :, :, :]
            resnet_model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
            resnet_model = keras.models.Model(
                inputs=resnet_model.input, outputs=resnet_model.output, name=f"resnet50v2_{i}"
            )
            features = resnet_model(img)
            features = keras.layers.Flatten()(features)
            processed_images.append(features)

        # Combine the feature vectors from each image
        combined = keras.layers.Concatenate()(processed_images)

        # Pass the combined feature vector through a fully connected layer
        fc_layer = keras.layers.Dense(1024, activation="relu")(combined)

        # Define the output layer
        output_layer = keras.layers.Dense(1000, activation="linear")(fc_layer)

        # Create the model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self._model = model
        self._model.compile(optimizer="adam", loss="mse")

    def _create_model(self):
        self._model = Sequential()
        self._model.add(
            models.unet_2d(
                (1000, 256, 5),
                filter_num=[32, 64, 128],
                weights=None,
                freeze_backbone=False,
                freeze_batch_norm=False,
                output_activation=None,
                n_labels=5,
            )
        )
        # self._model.add(layers.TimeDistributed(layers.Flatten()))
        # self._model.add(layers.TimeDistributed(layers.Dense(units=1)))
        self._model.add(layers.Conv2D(filters=1, kernel_size=(1, 256), activation="linear"))
        # loss_func = keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
        loss_func = keras.losses.MeanAbsolutePercentageError(reduction="none", name="mean_squared_logarithmic_error")
        # self._model.compile(optimizer="adam", loss=loss_func)
        self._model.compile(optimizer="adam", loss="mse")
        return self

    def save_model(self):
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists("Models"):
            os.makedirs("Models")
        self._model.save("Models/" + name + ".keras")
        with open("Models/" + name + "_history.pkl", "wb") as f:
            pickle.dump(self._model.history, f)
        return self
