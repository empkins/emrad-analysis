from typing_extensions import Self

import numpy as np
from tpcp import Algorithm, make_action_safe, cf, OptimizablePipeline, OptimizableParameter
from tpcp._dataset import DatasetT

from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.label_generation.label_generation_algorithm import ComputeEcgBlips
from rbm_robust.models.cnn import CNN
from rbm_robust.preprocessing.preprocessing import (
    ButterBandpassFilter,
    Downsampling,
    EmpiricalModeDecomposer,
    WaveletTransformer,
)


class PreProcessor(Algorithm):
    _action_methods = "preprocess"

    bandpass_filter: ButterBandpassFilter
    downsampling: Downsampling
    emd: EmpiricalModeDecomposer
    wavelet_transformer: WaveletTransformer
    downsample_factor: int = 200

    # Results
    preprocessed_signal_: np.array

    def __init__(
        self,
        bandpass_filter: ButterBandpassFilter,
        downsampling: Downsampling,
        emd: EmpiricalModeDecomposer,
        wavelet_transform: WaveletTransformer,
    ):
        self.bandpass_filter = bandpass_filter
        self.downsampling = downsampling
        self.emd = emd
        self.wavelet_transform = wavelet_transform

    @make_action_safe
    def preprocess(self, raw_radar: np.array, sampling_rate: float):
        """Preprocess the input signal using a bandpass filter

        Args:
            signal (np.array): Input signal to be preprocessed
            sampling_freq (float): Sampling frequency of the input signal

        Returns:
            np.array: Preprocessed signal
        """
        self.bandpass_filter = self.bandpass_filter.clone()
        self.preprocessed_signal_ = self.bandpass_filter.filter(raw_radar, sampling_rate)
        # Downsample the signal
        self.downsampling = self.downsampling.clone()
        # TODO: Check if the downsample method is correct and check if it works
        self.preprocessed_signal_ = self.downsampling.downsample(self.preprocessed_signal_, sampling_rate)

        # Empirical Mode Decomposition
        self.emd = self.emd.clone()
        # TODO: Calculate the magnidute of the signal and use that as input
        self.preprocessed_signal_ = self.emd.decompose(self.preprocessed_signal_)

        # Wavelet Transform
        self.wavelet_transform = self.wavelet_transform.clone()
        self.preprocessed_signal_ = self.wavelet_transform.transform(self.preprocessed_signal_)

        return self


class InputAndLabelGenerator(Algorithm):
    """Class generating the Input and Label matrices for the BiLSTM model.

    Results:
        self.input_data
        self.input_labels
    """

    _action_methods = ("generate_training_input", "generate_training_labels")

    # PreProcessing
    pre_processor: PreProcessor

    # Label Generation
    blip_algo: ComputeEcgBlips

    # Input & Label parameters
    segment_size_in_seconds: int
    overlap: float

    # Results
    input_data_: np.ndarray
    input_labels_: np.ndarray

    def __init__(
        self,
        pre_processor: PreProcessor = cf(PreProcessor()),
        blip_algo: ComputeEcgBlips = cf(ComputeEcgBlips()),
        segment_size_in_seconds: int = 5,
        overlap: float = 0.8,
    ):
        self.pre_processor = pre_processor
        self.blip_algo = blip_algo
        self.segment_size_in_seconds = segment_size_in_seconds
        self.overlap = overlap

    @make_action_safe
    def generate_training_input(self, dataset: D02Dataset):
        """Generate the input data for the BiLSTM model

        Args:
            raw_radar (np.array): Raw radar signal
            sampling_rate (float): Sampling frequency of the radar signal

        Returns:
            np.ndarray: Input data for the BiLSTM model
        """
        self.pre_processor = self.pre_processor.clone()

        for i in range(len(dataset.subjects)):
            data = dataset[i].synced_radar
            self.input_data_ = self.pre_processor.preprocess(data)
            # TODO: Segmentation

        # TODO: Set Input Data
        return self

    @make_action_safe
    def generate_training_labels(self, dataset: D02Dataset):
        """Generate the input labels for the BiLSTM model

        Args:
            raw_radar (np.array): Raw radar signal
            sampling_rate (float): Sampling frequency of the radar signal

        Returns:
            np.ndarray: Input labels for the BiLSTM model
        """
        self.blip_algo = self.blip_algo.clone()

        for i in range(len(dataset.subjects)):
            data = dataset[i].synced_ecg
            # TODO: ecg_data also has to be downsampled
            self.input_labels_ = self.blip_algo.compute(data)

        return self


class CnnPipeline(OptimizablePipeline):
    feature_extractor: InputAndLabelGenerator
    cnn: CNN
    cnn_model: OptimizableParameter

    result_ = np.ndarray

    def __init__(
        self,
        feature_extractor: InputAndLabelGenerator,
        cnn: CNN,
    ):
        self.feature_extractor = feature_extractor
        self.cnn = cnn

    def self_optimize(self, dataset: D02Dataset, **kwargs) -> Self:
        self.feature_extractor = self.feature_extractor.clone()
        self.cnn = self.cnn.clone()

        self.feature_extractor.generate_training_input(dataset)
        self.feature_extractor.generate_training_labels(dataset)

        self.cnn.self_optimize(self.feature_extractor.input_data_, self.feature_extractor.input_labels_)

        return self

    def run(self, datapoint: D02Dataset) -> Self:
        # Get data from dataset
        input_data = self.feature_extractor.generate_training_input(datapoint).input_data_

        # model predict
        cnn_copy = self.cnn.clone()
        cnn_copy = cnn_copy.predict(input_data)
        self.result_ = cnn_copy.predictions_

        return self
