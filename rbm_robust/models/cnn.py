from typing import Optional, Tuple

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

    # Model
    _model = Optional[keras.Sequential]

    # Results
    predictions_: np.ndarray

    def __init__(
        self,
        kernel_size: int = 3,
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

    def self_optimize(self, training_data: np.ndarray, labels: np.ndarray):
        """Use the training data and the corresponding labels to train the model with the hyperparameters passed in the init

        Args:
            training_data (np.ndarray): training data, multiple inputs
            labels (np.ndarray): corresponding labels
        """
        if self._model is None:
            self._create_model(training_data.shape[1], training_data.shape[2])

        assert (
            self._model.layers[0].input_shape[1] == training_data.shape[1]
        ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"
        assert (
            self._model.layers[0].input_shape[2] == training_data.shape[2]
        ), f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"

        self._model.fit(
            training_data,
            labels,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=False,
        )
        return self

    def _create_model(self, input_shape: int, output_shape: int):
        self._model = keras.Sequential()
        self._model.add(
            keras.layers.Conv2D(
                filters=output_shape,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                groups=self.groups,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                input_shape=(input_shape, input_shape, 3),
            )
        )
        self._model.add(keras.layers.MaxPool3D(pool_size=(2, 2)))
        self._model.add(keras.layers.Flatten())
        self._model.compile(keras.optimizers.Adam(self.learning_rate), loss="mse")
        return self
