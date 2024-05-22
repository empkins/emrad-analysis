import gc
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Optional
import keras
import keras_unet_collection.base
import numpy as np
from keras import Sequential
from tpcp import Algorithm
from keras_unet_collection import models
import tensorflow as tf
import pickle


class IdentityModel(Algorithm):
    _action_methods = "predict"

    # Parameters
    num_epochs: int = 10
    batch_size: int = 16
    training_subjects: list = None
    validation_subjects: list = None
    base_path: str = "/home/woody/iwso/iwso116h/Data"

    # Model
    _model: Optional[keras.Sequential]

    # Results
    predictions_: np.ndarray

    def __init__(self, num_epochs: int = 3, batch_size: int = 8):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self._model = None

    def batch_generator(self):
        base_path = Path(self.base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        if self.training_subjects is not None:
            subjects = [subject for subject in subjects if subject in self.training_subjects]
        while True:
            yield from self._get_inputs_and_labels_for_subjects(base_path, subjects)

    def validation_generator(self):
        base_path = Path(self.base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        if self.validation_subjects is not None:
            subjects = [subject for subject in subjects if subject in self.validation_subjects]
        yield from self._get_inputs_and_labels_for_subjects(base_path, subjects)

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
                steps += len(input_names)
        return int(steps / self.batch_size)

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
                input_names = list(filter(lambda x: "npy" in x, input_names))
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
                    del inputs
                    gc.collect()

    def get_input(self, base_path, imfs):
        if imfs is not None:
            return np.transpose(np.stack([self._load_input(base_path / imf) for imf in imfs], axis=0))
        else:
            return np.transpose(np.stack([np.zeros((256, 1000)) for _ in range(5)], axis=0))

    def grouper(self, iterable, n):
        iterators = [iter(iterable)] * n
        return zip_longest(*iterators)

    def predict(self, data_path: str, testing_subjects: list = None):
        print("Prediction started")
        data_path = Path(data_path)
        truths = []
        preds = []
        squares = 0
        count = 0
        subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
        if testing_subjects is not None:
            subjects = [subject for subject in subjects if subject in testing_subjects]
        for subject_id in subjects:
            subject_path = data_path / subject_id
            for phase_path in subject_path.iterdir():
                if not phase_path.is_dir():
                    continue
                input_path = phase_path / "inputs"
                labels_path = phase_path / "labels"
                prediction_path = phase_path / "identity_predictions"
                prediction_path.mkdir(exist_ok=True)
                input_files = sorted(input_path.glob("*.npy"))
                for input_file in input_files:
                    inputs = np.load(input_file)
                    inputs = np.array([inputs])
                    if inputs.shape != (1, 1000, 256, 5):
                        padded = np.zeros((1, 1000, 256, 5))
                        padded[: inputs.shape[0], : inputs.shape[1], : inputs.shape[2], : inputs.shape[3]] = inputs
                        inputs = padded
                    pred = self._model.predict(inputs)
                    truth = np.load(labels_path / input_file)
                    squares += (truth - pred) ** 2
                    count += 1
                    np.save(prediction_path / input_file.name, pred)
        mse = np.sum(squares) / (count * 1000 * 256 * 5)
        print(f"Mean Squared Error: {mse}")
        return self

    def self_optimize(
        self,
        base_path: str = "/home/woody/iwso/iwso116h/Data",
        training_subjects: list = None,
        validation_subjects: list = None,
    ):
        print("Before Generators")
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects
        self.base_path = base_path

        batch_generator = self.batch_generator
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
                tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
            ),
        )
        dataset.batch(self.batch_size).repeat()
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
        if self._model is None:
            self.create_model()

        self._model.fit(
            dataset,
            epochs=self.num_epochs,
            steps_per_epoch=training_steps,
            batch_size=self.batch_size,
            shuffle=False,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            verbose=1,
        )
        print("Fitting done")
        print("Save Model")
        model_file_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_identity_model.h5"
        model_history_file_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_identity_model_history.pkl"
        with open(model_history_file_name, "wb") as file:
            pickle.dump(self._model.history.history, file)
        keras.models.save_model(self._model, model_file_name, save_format="h5")
        return self

    def create_model(self):
        IN = keras.layers.Input((1000, 256, 5))
        #
        # OUT = keras_unet_collection.models.unet_2d(
        #     IN,
        #     filter_num=[16, 32, 64],
        #     weights=None,
        #     freeze_backbone=False,
        #     freeze_batch_norm=False,
        # )
        # self._model = keras.Model(
        #     inputs=[
        #         IN,
        #     ],
        #     outputs=[
        #         OUT,
        #     ],
        # )
        #
        #
        # self._model = Sequential()
        #
        self._model = keras_unet_collection.models.unet_2d(
            (1000, 256, 5),
            filter_num=[16, 32, 64],
            weights=None,
            freeze_backbone=False,
            freeze_batch_norm=False,
            output_activation=None,
            n_labels=5,
        )
        #

        self._model.compile(optimizer="adam", loss="mse")
        return self
