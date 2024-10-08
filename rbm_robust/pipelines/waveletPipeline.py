import os

import pytz
from typing_extensions import Self

import numpy as np
from tpcp import cf, OptimizablePipeline
from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.waveletModel import UNetWaveletTF


class WaveletPipeline(OptimizablePipeline):
    wavelet_model: UNetWaveletTF

    result_ = np.ndarray

    def __init__(
        self,
        wavelet_model: UNetWaveletTF = cf(UNetWaveletTF()),
    ):
        self.wavelet_model = wavelet_model

    def self_optimize(
        self,
        dataset: D02Dataset,
        validation: D02Dataset,
        path: str = "/home/woody/iwso/iwso116h/Data",
        model_path: str = None,
    ) -> Self:
        self.wavelet_model = self.wavelet_model.clone()
        print("Optimizing Wavelet Model")
        training_subjects = dataset.subjects
        validation_subjects = validation.subjects
        self.wavelet_model.self_optimize(
            base_path=path,
            training_subjects=training_subjects,
            validation_subjects=validation_subjects,
        )
        return self

    def run(self, testing_subjects: list = None, path: str = "/home/woody/iwso/iwso116h/TestData") -> Self:
        print("Run")
        self.wavelet_model.predict(path, testing_subjects)
        return self
