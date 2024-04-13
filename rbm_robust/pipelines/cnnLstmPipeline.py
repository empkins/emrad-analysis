from typing_extensions import Self

import numpy as np
from tpcp import Algorithm, make_action_safe, cf, OptimizablePipeline, OptimizableParameter
from tpcp._dataset import DatasetT
import pandas as pd
from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.label_generation.label_generation_algorithm import ComputeEcgBlips
from rbm_robust.models.cnn import CNN
from rbm_robust.preprocessing.preprocessing import (
    ButterBandpassFilter,
    Downsampling,
    EmpiricalModeDecomposer,
    WaveletTransformer,
)
from emrad_toolbox.radar_preprocessing.radar import RadarPreprocessor


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
        bandpass_filter: ButterBandpassFilter = cf(ButterBandpassFilter()),
        downsampling: Downsampling = cf(Downsampling()),
        emd: EmpiricalModeDecomposer = cf(EmpiricalModeDecomposer()),
        wavelet_transform: WaveletTransformer = cf(WaveletTransformer()),
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

        # Initializing
        bandpass_filter_clone = self.bandpass_filter.clone()
        emd_clone = self.emd.clone()
        wavelet_transform_clone = self.wavelet_transform.clone()
        downsampling_clone = self.downsampling.clone()

        # Calculate Power
        radar_mag = RadarPreprocessor().calculate_power(i=raw_radar["I"], q=raw_radar["Q"])

        # Bandpass Filter
        self.preprocessed_signal_ = bandpass_filter_clone.filter(radar_mag, sampling_rate).filtered_signal_

        # Downsampling
        self.preprocessed_signal_ = downsampling_clone.downsample(
            self.preprocessed_signal_, 200, sampling_rate
        ).downsampled_signal_

        # Empirical Mode Decomposition
        self.preprocessed_signal_ = emd_clone.decompose(self.preprocessed_signal_).imfs_

        # Wavelet Transform
        self.preprocessed_signal_ = wavelet_transform_clone.transform(self.preprocessed_signal_).transformed_signal_

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
    downsampled_hz: int

    # Results
    input_data_: np.ndarray
    input_labels_: np.ndarray

    def __init__(
        self,
        pre_processor: PreProcessor = cf(PreProcessor()),
        blip_algo: ComputeEcgBlips = cf(ComputeEcgBlips()),
        segment_size_in_seconds: int = 5,
        overlap: float = 0.8,
        downsampled_hz: int = 200,
    ):
        self.pre_processor = pre_processor
        self.blip_algo = blip_algo
        self.segment_size_in_seconds = segment_size_in_seconds
        self.overlap = overlap
        self.downsampled_hz = downsampled_hz

    @make_action_safe
    def generate_training_input(self, dataset: D02Dataset):
        """Generate the input data for the BiLSTM model

        Args:
            raw_radar (np.array): Raw radar signal
            sampling_rate (float): Sampling frequency of the radar signal

        Returns:
            np.ndarray: Input data for the BiLSTM model
        """
        pre_processor_clone = self.pre_processor.clone()
        res = []
        for i in range(len(dataset.subjects)):
            radar_data = dataset[i].synced_radar
            phases = dataset[i].phases
            sampling_rate = dataset[i].SAMPLING_RATE_DOWNSAMPLED
            phase_res = []
            for phase in phases.keys():
                # Get the data for the current phase
                phase_radar_data = radar_data[phases[phase]["start"] : phases[phase]["end"]]
                time_step = phase_radar_data.index[1] - phase_radar_data.index[0]
                # Segmentation
                step_size = int(self.segment_size_in_seconds * self.overlap)
                total_seconds = (phase_radar_data.index.max() - phase_radar_data.index.min()).total_seconds()
                step_count = int((total_seconds // step_size) - 1)
                start_time = phase_radar_data.index[0]
                for j in range(1, step_count):
                    end = start_time + pd.Timedelta(seconds=step_size * j)
                    # Preprocess the data
                    data_segment = phase_radar_data[start_time:end]
                    time_diff = data_segment.index[-1] - data_segment.index[0]
                    # Zero padding
                    if len(data_segment) < self.segment_size_in_seconds * sampling_rate:
                        rows_needed = int((pd.Timedelta(seconds=4) - time_diff) / time_step)
                        zero_df = pd.DataFrame(
                            np.zeros((rows_needed, len(data_segment.columns))), columns=data_segment.columns
                        )
                        data_segment = data_segment.append(zero_df, ignore_index=True)
                        data_segment.index = pd.date_range(
                            start=data_segment.index[0], periods=len(data_segment), freq=time_step
                        )
                    pre_processor_clone = self.pre_processor.clone()
                    segment = pre_processor_clone.preprocess(
                        data_segment, dataset[i].SAMPLING_RATE_DOWNSAMPLED
                    ).preprocessed_signal_
                    start_time = end
                    input_data_segment = [imf["coefficients"] for imf in segment]
                    phase_res.append(input_data_segment)
            res.append(phase_res)
        self.input_data_ = np.array(res)
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
        blip_algo_clone = self.blip_algo.clone()
        res = []
        for i in range(len(dataset.subjects)):
            data = dataset[i].synced_ecg
            phases = dataset[i].phases
            sampling_rate = dataset[i].SAMPLING_RATE_DOWNSAMPLED
            time_step = data.index[1] - data.index[0]
            phase_res = []
            for phase in phases.keys():
                # Get the data for the current phase
                phase_data = data[phases[phase]["start"] : phases[phase]["end"]]
                # Generate the labels
                labels = blip_algo_clone.compute(phase_data).blips_
                # Segmentation
                step_size = int(self.segment_size_in_seconds * self.overlap)
                total_seconds = (phase_data.index.max() - phase_data.index.min()).total_seconds()
                step_count = (total_seconds // step_size) - 1
                start_time = phase_data.index[0]
                for i in range(step_count):
                    end = start_time + pd.Timedelta(seconds=step_size * i)
                    segment = labels[start_time:end]
                    time_diff = segment.index[-1] - segment.index[0]
                    # Zero padding
                    if len(segment) < self.segment_size_in_seconds * sampling_rate:
                        rows_needed = int((pd.Timedelta(seconds=4) - time_diff) / time_step)
                        zero_df = pd.DataFrame(np.zeros((rows_needed, len(segment.columns))), columns=segment.columns)
                        segment = segment.append(zero_df, ignore_index=True)
                        segment.index = pd.date_range(start=segment.index[0], periods=len(segment), freq=time_step)
                    start_time = end
                    phase_res.append(segment)
            res.append(phase_res)
        self.input_labels_ = np.array(res)
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
