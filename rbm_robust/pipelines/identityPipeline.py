import os

import pytz
from typing_extensions import Self

import numpy as np
from tpcp import cf, OptimizablePipeline
from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.identityModel import IdentityModel


class IdentityPipeline(OptimizablePipeline):
    identity_model: IdentityModel

    result_ = np.ndarray

    def __init__(
        self,
        identity_model: IdentityModel = cf(IdentityModel()),
    ):
        self.identity_model = identity_model

    def self_optimize(
        self,
        dataset: D02Dataset,
        validation: D02Dataset,
        path: str = "/home/woody/iwso/iwso116h/Data",
    ) -> Self:
        self.identity_model = self.identity_model.clone()
        print("Optimizing Identity Model")
        training_subjects = dataset.subjects
        validation_subjects = validation.subjects
        self.identity_model.self_optimize(path, training_subjects, validation_subjects)
        return self

    def run(self, testing_subjects: list = None, path: str = "/home/woody/iwso/iwso116h/Data") -> Self:
        print("Run")
        self.identity_model.predict(path, testing_subjects)
        return self
