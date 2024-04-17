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

    def self_optimize(self, training_data: list, labels: np.ndarray):
        """Use the training data and the corresponding labels to train the model with the hyperparameters passed in the init

        Args:
            training_data (list): training data, multiple inputs
            labels (np.ndarray): corresponding labels
        """

        if self._model is None:
            self._create_model()

        log_dir = "~/Runs/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # assert (
        #     self._model.layers[0].input_shape[1] == training_data.shape[1]
        # ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"
        # assert (
        #     self._model.layers[0].input_shape[2] == training_data.shape[2]
        # ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"

        self._model.fit(
            training_data,
            labels,
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
