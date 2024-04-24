import os
from datetime import datetime
from typing import Optional, Tuple
import tensorflow as tf
import pickle
import keras
import numpy as np
from tpcp import Algorithm, OptimizableParameter
from keras.preprocessing.image import load_img, img_to_array
from itertools import groupby


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
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        learning_rate: float = 0.001,
        num_epochs: int = 12,
        batch_size: int = 1,
        _model=None,
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
        self._model = _model

    def batch_generator_images(self, base_path):
        subjects = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        i = 0
        while i < len(subjects):
            if i == len(subjects):
                i = 0
            subject_id = subjects[i]
            subject_path = os.path.join(base_path, subject_id)
            phases = [name for name in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, name))]
            for phase in phases:
                phase_path = os.path.join(subject_path, phase)
                input_path = os.path.join(phase_path, "inputs")
                label_path = os.path.join(phase_path, "labels")
                if not os.path.exists(input_path) or not os.path.exists(label_path):
                    continue
                input_names = [
                    name for name in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, name))
                ]
                input_names.sort()
                grouped_inputs = {k: list(g) for k, g in groupby(input_names, key=lambda s: s.split("_")[0])}
                sorted_keys = sorted(grouped_inputs.keys())
                for key in sorted_keys:
                    label = np.load(os.path.join(label_path, f"{key}.npy"))
                    if ".png" in grouped_inputs[key][0]:
                        inputs = [
                            img_to_array(load_img(os.path.join(input_path, f"{in_path}"), target_size=(255, 1000)))
                            for in_path in grouped_inputs[key]
                        ]
                    elif ".npy" in grouped_inputs[key][0]:
                        inputs = [np.load(os.path.join(input_path, f"{in_path}")) for in_path in grouped_inputs[key]]
                    yield inputs, label

    def batch_generator(self, base_path):
        subjects = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        i = 0
        batch_size = self.batch_size
        while True:
            if i == len(subjects):
                i = 0  # reset the counter if we've gone through all subjects
            try:
                subject_id = subjects[i]
            except Exception as _:
                print(f"i is {i}")
                print(subjects)
                raise ValueError("oh no")
            phases = [
                name
                for name in os.listdir(os.path.join(base_path, subject_id))
                if os.path.isdir(os.path.join(base_path, subject_id, name))
            ]

            for phase in phases:
                phase_path = os.path.join(base_path, subject_id, phase)
                input_path = os.path.join(phase_path, "inputs")
                label_path = os.path.join(phase_path, "labels")

                if not os.path.exists(input_path) or not os.path.exists(label_path):
                    continue

                input_paths = [
                    os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(".pkl")
                ]
                label_paths = [
                    os.path.join(label_path, file) for file in os.listdir(label_path) if file.endswith(".pkl")
                ]

                # Order the paths
                input_paths.sort()
                label_paths.sort()

                # zip the paths
                paths = zip(input_paths, label_paths)

                for element in paths:
                    # Load inputs
                    with open(element[0], "rb") as f:
                        inputs = pickle.load(f)
                    # Load labels
                    with open(element[1], "rb") as w:
                        labels = pickle.load(w)

                    ins = np.stack(inputs, axis=0)
                    labs = np.stack(labels, axis=0)
                    labs = np.transpose(labs)
                    # print(f"labels len {len(labels)}")
                    # print(f"labs shape {labs.shape}")
                    for j in range(ins.shape[0]):
                        sub_ins = ins[j : j + 1, :, :, :]
                        sub_lab = labs[:, j : j + 1]
                        # print(f"input shape: {sub_ins.shape} sub_lab shape: {sub_lab.shape}")
                        yield sub_ins, sub_lab
            i += 1

    def get_steps_per_epoch(self, base_path):
        steps = 0
        subjects = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        for subject_id in subjects:
            phases = [
                name
                for name in os.listdir(os.path.join(base_path, subject_id))
                if os.path.isdir(os.path.join(base_path, subject_id, name))
            ]

            for phase in phases:
                phase_path = os.path.join(base_path, subject_id, phase)
                input_path = os.path.join(phase_path, "inputs")
                label_path = os.path.join(phase_path, "labels")

                if not os.path.exists(input_path) or not os.path.exists(label_path):
                    continue

                input_paths = [
                    os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(".pkl")
                ]
                for element in input_paths:
                    # Load inputs
                    with open(element, "rb") as f:
                        inputs = pickle.load(f)
                    steps += len(inputs)
        return steps

    def predict(self, data_path: str):
        subjects = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
        predictions = {}

        for subject_id in subjects:
            subject_dict = {}
            subject_path = os.path.join(data_path, subject_id)
            phases = [name for name in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, name))]
            phase_predictions = []
            for phase in phases:
                phase_path = os.path.join(data_path, subject_id, phase)
                prediction_path = os.path.join(phase_path, "predictions")
                if not os.path.exists(prediction_path):
                    os.makedirs(prediction_path)
                input_path = os.path.join(phase_path, "inputs")
                input_paths = [
                    os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(".pkl")
                ]
                input_paths.sort()
                for element in input_paths:
                    with open(element, "rb") as f:
                        inputs = pickle.load(f)
                    ins = np.stack(inputs, axis=0)
                    for j in range(ins.shape[0]):
                        sub_ins = ins[j : j + 1, :, :, :]
                        pred = self._model.predict(sub_ins)
                        print(f"Predictions for {ins.shape} are {pred.shape} shape")
                        phase_predictions.append(self._model.predict(sub_ins))
                subject_dict[phase] = phase_predictions
            predictions[subject_id] = subject_dict
        self.predictions_ = predictions
        return self

    def self_optimize(self, training_data_path: str, label_path: str):
        """Use the training data and the corresponding labels to train the model with the hyperparameters passed in the init

        Args:
            training_data (list): training data, multiple inputs
            labels (np.ndarray): corresponding labels
        """

        if self._model is None:
            self._create_model()

        log_dir = "Runs/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        batch_generator = self.batch_generator_images("Data")
        steps = self.get_steps_per_epoch("Data")

        # assert (
        #     self._model.layers[0].input_shape[1] == training_data.shape[1]
        # ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"
        # assert (
        #     self._model.layers[0].input_shape[2] == training_data.shape[2]
        # ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"
        print(f"Is compiled {self._model._is_compiled}")
        print("Fitting")
        self._model.fit(
            batch_generator,
            epochs=self.num_epochs,
            steps_per_epoch=steps,
            batch_size=self.batch_size,
            shuffle=False,
            callbacks=[tensorboard_callback],
            verbose=1,
        )
        return self

    def _create_model(self):
        self._model = keras.Sequential()
        self._model.add(keras.layers.Conv2D(3, (1, 1), padding="same", input_shape=(255, 1000, 5)))
        self._model.add(keras.applications.ResNet50V2(include_top=False, weights="imagenet", pooling="avg"))
        self._model.add(keras.layers.Flatten())
        self._model.add(keras.layers.Dense(1000))
        self._model.compile(optimizer="adam", loss="mse")
        return self
