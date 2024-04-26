import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
import keras
import numpy as np
from tpcp import Algorithm, OptimizableParameter
from keras.preprocessing.image import load_img, img_to_array
from itertools import groupby
from tensorflow.keras.callbacks import Callback
import gc


class PrintShapeCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            print(f"Input shape of layer {layer.name}: {layer.input_shape}")
            print(f"Output shape of layer {layer.name}: {layer.output_shape}")


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

    def batch_generator(self, base_path):
        base_path = Path(base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        while True:
            for subject_id in subjects:
                subject_path = base_path / subject_id
                phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
                for phase in phases:
                    phase_path = subject_path / phase
                    input_path = phase_path / "inputs"
                    label_path = phase_path / "labels"
                    if not input_path.exists() or not label_path.exists():
                        continue
                    input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                    grouped_inputs = {k: list(g) for k, g in groupby(input_names, key=lambda s: s.split("_")[0])}
                    for key, group in grouped_inputs.items():
                        label = np.load(label_path / f"{key}.npy")
                        inputs = [self._load_input(input_path / name) for name in group]
                        inputs = np.stack(inputs, axis=0)
                        inputs = np.transpose(inputs)
                        inputs = np.array([inputs])
                        yield inputs, label
                        del inputs, label
                        gc.collect()

    def validation_generator(self, base_path):
        base_path = Path(base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                grouped_inputs = {k: list(g) for k, g in groupby(input_names, key=lambda s: s.split("_")[0])}
                for key, group in grouped_inputs.items():
                    label = np.load(label_path / f"{key}.npy")
                    inputs = [self._load_input(input_path / name) for name in group]
                    inputs = np.stack(inputs, axis=0)
                    inputs = np.transpose(inputs)
                    inputs = np.array([inputs])
                    yield inputs, label

    def _load_input(self, path):
        if path.suffix == ".png":
            return img_to_array(load_img(path, target_size=(224, 224)))
        elif path.suffix == ".npy":
            return np.load(path)

    def get_steps_per_epoch(self, base_path):
        base_path = Path(base_path)
        steps = 0
        for subject_path in base_path.iterdir():
            if not subject_path.is_dir():
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
        return steps

    def predict(self, data_path: str):
        data_path = Path(data_path)
        for subject_path in data_path.iterdir():
            if not subject_path.is_dir():
                continue
            for phase_path in subject_path.iterdir():
                if not phase_path.is_dir():
                    continue
                input_path = phase_path / "inputs"
                prediction_path = phase_path / "predictions"
                prediction_path.mkdir(exist_ok=True)
                input_files = sorted(input_path.glob("*.npy"))
                grouped_inputs = {k: list(g) for k, g in groupby(input_files, key=lambda s: s.stem.split("_")[0])}
                for key, group in grouped_inputs.items():
                    inputs = [np.load(file) for file in group]
                    pred = self._model.predict(inputs)
                    np.save(prediction_path / f"{key}.npy", pred)
                    print(f"Predictions for {inputs.shape} are {pred.shape} shape")
        return self

    def self_optimize(self, base_path: str = "Data", image_based: bool = False):
        if not image_based:
            self._create_model()
        else:
            self._image_model()

        log_dir = "Runs/logs/fit/"
        log_dir += datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Commented out since the tensorboard callback leads to too much necessary memory
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # print_shape_callback = PrintShapeCallback()

        batch_generator = self.batch_generator(base_path)
        validation_generator = self.batch_generator("Validation")
        steps = self.get_steps_per_epoch(base_path)

        print("Fitting")
        self._model.fit(
            batch_generator,
            epochs=self.num_epochs,
            steps_per_epoch=steps,
            batch_size=self.batch_size,
            shuffle=False,
            validation_data=validation_generator,
            # callbacks=[tensorboard_callback, print_shape_callback],
            verbose=1,
        )
        return self

    def _image_model(self):
        input_layer = keras.layers.Input(shape=(5, 224, 224, 3))

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
        self._model = keras.Sequential()
        self._model.add(
            keras.layers.Conv2D(3, (1, 1), padding="same", input_shape=(1000, 255, 5), batch_size=self.batch_size)
        )
        self._model.add(keras.applications.ResNet50V2(include_top=False, weights="Weights/resNet50V2.h5"))
        self._model.add(keras.layers.TimeDistributed(keras.layers.Dense(1000)))
        self._model.add(keras.layers.Flatten())
        self._model.add(keras.layers.Dense(1000))
        self._model.compile(optimizer="adam", loss="mse")
        return self
