import concurrent
import gc
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
from keras.preprocessing.image import load_img, img_to_array
from itertools import groupby, zip_longest
import tensorflow as tf


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
        learning_rate: float = 0.001,
        num_epochs: int = 1,
        batch_size: int = 32,
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

    def batch_generator(self):
        base_path = Path(self.base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        if self.training_subjects is not None:
            subjects = [subject for subject in subjects if subject in self.training_subjects]
        while True:
            # yield from self._get_inputs_and_labels_for_subjects_grouped(base_path, subjects)
            # yield from self._get_inputs_and_labels_for_subjects_improved(base_path, subjects)
            yield from self._get_inputs_and_labels_for_subjects(base_path, subjects)

    def _get_inputs_and_labels_for_subjects_improved(self, base_path, subjects):
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                if phase == "logs" or phase == "raw":
                    continue
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels"
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                grouped_inputs = {
                    int(k): list(filter(lambda x: "png" in x if self.image_based else "npy" in x, list(g)))
                    for k, g in groupby(input_names, key=lambda s: s.split("_")[0])
                }
                megabatches = self.grouper(grouped_inputs.keys(), 20 * self.batch_size)
                for megabatch in megabatches:
                    if all(v is None for v in megabatch):
                        continue
                    batch = self._read_parallel(input_path, label_path, grouped_inputs, megabatch)
                    for i in range(0, len(batch), self.batch_size):
                        inputs = np.stack([batch[i + j][0] for j in range(self.batch_size)], axis=0)
                        labels = np.stack([batch[i + j][1] for j in range(self.batch_size)], axis=0)
                        np.nan_to_num(inputs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        np.nan_to_num(labels, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        yield inputs, labels
                    del batch
                del megabatches

    def _read_parallel(self, input_path, label_path, grouped_inputs, group):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_in_and_labels, input_path, label_path, grouped_inputs, number)
                for number in group
            ]
            return [fut.result() for fut in futures]

    def _load_in_and_labels(self, input_path, label_path, grouped_inputs, number):
        in_files = grouped_inputs[number] if number is not None else None
        inputs = self.get_input(input_path, in_files)
        labels = self.get_labels(label_path, number)
        return inputs, labels

    def _get_inputs_and_labels_for_subjects(self, base_path, subjects):
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                if phase == "logs" or phase == "raw":
                    continue
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                input_names = list(filter(lambda x: "png" in x if self.image_based else "npy" in x, input_names))
                groups = self.grouper(input_names, self.batch_size)
                for group in groups:
                    inputs = np.stack(
                        [
                            np.load(input_path / number) if number is not None else self.get_input(input_path, None)
                            for number in group
                        ],
                        axis=0,
                    )
                    labels = np.stack(
                        [
                            np.load(label_path / number) if number is not None else self.get_labels(input_path, None)
                            for number in group
                        ],
                        axis=0,
                    )
                    yield inputs, labels
                    del inputs, labels
                    gc.collect()

    def _get_inputs_and_labels_for_subjects_grouped(self, base_path, subjects):
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                if phase == "logs" or phase == "raw":
                    continue
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                grouped_inputs = {
                    int(k): list(filter(lambda x: "png" in x if self.image_based else "npy" in x, list(g)))
                    for k, g in groupby(input_names, key=lambda s: s.split("_")[0])
                }
                inputs_sorted = list(grouped_inputs.keys())
                inputs_sorted.sort()
                groups = self.grouper(inputs_sorted, self.batch_size)
                for group in groups:
                    inputs = np.stack(
                        [
                            self.get_input(input_path, grouped_inputs[number])
                            if number is not None
                            else self.get_input(input_path, None)
                            for number in group
                        ],
                        axis=0,
                    )
                    labels = np.stack([self.get_labels(label_path, number) for number in group], axis=0)
                    yield inputs, labels
                    del inputs, labels
                    gc.collect()

    def grouper(self, iterable, n):
        iterators = [iter(iterable)] * n
        return zip_longest(*iterators)

    def get_input(self, base_path, imfs):
        if imfs is not None:
            return np.transpose(np.stack([self._load_input(base_path / imf) for imf in imfs], axis=0))
        elif not self.image_based and imfs is None:
            return np.transpose(np.stack([np.zeros((256, 1000)) for _ in range(5)], axis=0))
        elif self.image_based and imfs is None:
            return np.transpose(np.stack([np.zeros((256, 1000, 3)) for _ in range(5)], axis=0))

    def get_labels(self, base_path, number):
        if number is not None and "." not in number:
            return np.load(base_path / f"{number}.npy")
        elif number is not None and ".npy" in number:
            return np.load(base_path / f"{number}")
        else:
            return np.zeros((1000))

    def _get_middle(self, array):
        percentile = (1 - self.overlap) / 2
        start = int(len(array) / 2 - percentile)
        end = int(len(array) / 2 + percentile)
        return array[start:end]

    def validation_generator(self):
        base_path = Path(self.base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        if self.validation_subjects is not None:
            subjects = [subject for subject in subjects if subject in self.validation_subjects]
        # yield from self._get_inputs_and_labels_for_subjects_grouped(base_path, subjects)
        yield from self._get_inputs_and_labels_for_subjects(base_path, subjects)

    def _load_input(self, path):
        if path.suffix == ".png" and self.image_based:
            arr = img_to_array(load_img(path, target_size=(256, 1000)))
            if arr.shape != (256, 1000, 3):
                padded = np.zeros((256, 1000, 3))
                padded[: arr.shape[0], : arr.shape[1], : arr.shape[2]] = arr
                arr = padded
        elif path.suffix == ".npy" and not self.image_based:
            try:
                arr = self._pad_array(np.load(path))
            except Exception as e:
                print(f"Exception: {e}")
                arr = np.zeros((256, 1000))
        return arr

    def _pad_array(self, array):
        if array.shape != (256, 1000):
            padded = np.zeros((256, 1000))
            padded[: array.shape[0], : array.shape[1]] = array
            arr = padded
        return arr

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
                label_path = phase_path / "labels"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                grouped_inputs = {k: list(g) for k, g in groupby(input_names, key=lambda s: s.split("_")[0])}
                steps += len(grouped_inputs.keys())
        return int(steps / self.batch_size)

    def predict(self, data_path: str, testing_subjects: list = None, grouped: bool = False):
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
                prediction_path = phase_path / "predictions"
                prediction_path.mkdir(exist_ok=True)
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
                        pred = self._model.predict(inputs)
                        pred = pred.flatten()
                        np.save(prediction_path / input_file, pred)
        return self

    def _identity_batch_generator(self, base_path, subjects: list = None):
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                if phase == "logs" or phase == "raw":
                    continue
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                input_names = list(filter(lambda x: "png" in x if self.image_based else "npy" in x, input_names))
                groups = self.grouper(input_names, self.batch_size)
                for group in groups:
                    inputs = np.stack(
                        [
                            np.load(input_path / number) if number is not None else self.get_input(input_path, None)
                            for number in group
                        ],
                        axis=0,
                    )
                    yield inputs, inputs

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

        log_dir = "Runs/logs/fit/"
        log_dir += datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print("Before Generators")
        batch_generator = self.batch_generator
        training_dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
                tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
            ),
        )
        training_dataset.batch(self.batch_size).repeat()
        validation_generator = self.validation_generator
        validation_dataset = tf.data.Dataset.from_generator(
            validation_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
                tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
            ),
        )
        validation_dataset.batch(self.batch_size).repeat()

        print("Getting steps per epoch")
        training_steps = self.get_steps_per_epoch(base_path, training_subjects)
        validation_steps = self.get_steps_per_epoch(base_path, validation_subjects)
        print("Got the step count")

        print("Fitting")
        self._model.fit(
            training_dataset,
            epochs=self.num_epochs,
            steps_per_epoch=training_steps,
            batch_size=self.batch_size,
            shuffle=False,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            verbose=1,
        )
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
        self._model.add(models.unet_plus_2d((1000, 256, 5), filter_num=[16, 32, 64], n_labels=5, weights=None))
        time_layers = Sequential()
        time_layers.add(layers.Flatten())
        time_layers.add(layers.Dense(1))
        self._model.add(layers.TimeDistributed(time_layers))
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
