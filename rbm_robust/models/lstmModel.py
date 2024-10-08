import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import keras
import numpy as np
import tensorflow as tf
from tpcp import Algorithm, OptimizableParameter


class LSTM(Algorithm):
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
        learning_rate: float = 0.001,
        training_ds: tf.data.Dataset = None,
        validation_ds: tf.data.Dataset = None,
        model_name: str = None,
        epochs: int = 12,
        training_steps: int = 0,
        validation_steps: int = 0,
        batch_size: int = 8,
        loss: str = "bce",
        dual_channel: bool = False,
        model_path: str = None,
    ):
        self.second_dropout_rate = 0.6
        self.mono_lstm_units = 256
        self.bi_lstm_units = 256
        self.first_dropout_rate = 0.6
        self.learning_rate = learning_rate
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.model_name = model_name
        self.epochs = epochs
        self.training_steps = training_steps
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self.loss = loss
        self.dual_channel = dual_channel
        self.model_path = model_path

    def self_optimize(
        self,
    ):
        self._create_model()
        if os.getenv("HPCVAULT") is None:
            log_dir = "~/Runs/logs/fit/"
        else:
            log_dir = os.getenv("HPCVAULT") + f"/Runs/logs/fit/{self.model_name}"

        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir += dt

        if not os.path.exists(log_dir):
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, update_freq="epoch")
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
            callbacks=[tensorboard_callback],
        )

        if os.getenv("HPCVAULT") is None:
            history_path = "~/Runs/History/"
        else:
            history_path = os.getenv("HPCVAULT") + "/Runs/History/"
        if not os.path.exists(history_path):
            Path(history_path).mkdir(parents=True, exist_ok=True)
        history_path += self.model_name + dt + "_history.pkl"
        pickle.dump(history.history, open(history_path, "wb"))
        self.save_model()
        return self

    def _create_model(self):
        self._model = keras.Sequential()
        self._model.add(tf.keras.layers.Input(shape=(5, 1000), dtype=tf.float64))
        self._model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.bi_lstm_units, return_sequences=True)))
        self._model.add(tf.keras.layers.Dropout(self.first_dropout_rate))
        self._model.add(tf.keras.layers.LSTM(self.mono_lstm_units))
        self._model.add(tf.keras.layers.Dropout(self.second_dropout_rate))
        self._model.add(tf.keras.layers.Dense(1000))
        loss_func = keras.losses.BinaryCrossentropy(from_logits=False, reduction="sum_over_batch_size")
        self._model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss=loss_func)
        return self

    def save_model(self):
        name = self.model_name + datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists("Models"):
            os.makedirs("Models")
        self._model.save("Models/" + name + ".keras")
        with open("Models/" + name + "_history.pkl", "wb") as f:
            pickle.dump(self._model.history, f)
        return self

    def _get_input_array(self, input_file):
        return np.load(input_file)
