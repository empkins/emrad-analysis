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
