import os
from datetime import datetime
from typing import Optional, Tuple
import tensorflow as tf

import keras
import numpy as np
from tpcp import Algorithm, OptimizableParameter


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
        batch_size: int = 128,
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

    def batch_generator(self, base_path):
        subjects = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        i = 0
        batch_size = 32
        while True:
            if i == len(subjects):
                i = 0  # reset the counter if we've gone through all subjects

            subject_path = os.path.join(base_path, subjects[i])
            input_path = os.path.join(subject_path, "inputs")
            label_path = os.path.join(subject_path, "labels")

            input_paths = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(".npy")]
            label_paths = [os.path.join(label_path, file) for file in os.listdir(label_path) if file.endswith(".npy")]

            # Order the paths
            input_paths.sort()
            label_paths.sort()

            # zip the paths
            paths = zip(input_paths, label_paths)

            for element in paths:
                # Load inputs
                inputs = np.load(element[0])
                # Load labels
                labels = np.load(element[1])
                # Yield batches
                for j in range(0, len(inputs), batch_size):
                    yield inputs[j : j + batch_size], labels[j : j + batch_size]
            i += 1

    def self_optimize(self, training_data_path: str, label_path: str):
        """Use the training data and the corresponding labels to train the model with the hyperparameters passed in the init

        Args:
            training_data (list): training data, multiple inputs
            labels (np.ndarray): corresponding labels
        """

        if self._model is None:
            self._create_model()

        log_dir = "~/Runs/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        batch_generator = self.batch_generator(training_data_path, label_path, self.batch_size)

        # assert (
        #     self._model.layers[0].input_shape[1] == training_data.shape[1]
        # ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"
        # assert (
        #     self._model.layers[0].input_shape[2] == training_data.shape[2]
        # ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"

        self._model.fit(
            batch_generator,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=True,
            callbacks=[tensorboard_callback],
        )
        return self

    def _create_model(self):
        self._model = keras.Sequential()
        self._model.add(keras.applications.ResNet50V2(include_top=False, input_shape=(255, 1000, 5)))
        self._model.add(keras.layers.Dense(1))
        self._model.compile(optimizer="adam", loss="mse")
        return self
