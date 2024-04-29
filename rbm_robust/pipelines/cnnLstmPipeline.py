import os
import pickle

import pytz
from typing_extensions import Self

import numpy as np
from tpcp import Algorithm, make_action_safe, cf, OptimizablePipeline, OptimizableParameter
from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.label_generation.label_generation_algorithm import ComputeEcgBlips
from rbm_robust.models.cnn import CNN
from rbm_robust.preprocessing.preprocessing import (
    ButterBandpassFilter,
    Downsampling,
    EmpiricalModeDecomposer,
    WaveletTransformer,
    Segmentation,
    Normalizer,
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
    def preprocess(
        self,
        raw_radar: np.array,
        sampling_rate: float,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "/home/woody/iwso/iwso116h/Data",
        image_based: bool = False,
    ):
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
        self.preprocessed_signal_ = wavelet_transform_clone.transform(
            self.preprocessed_signal_, subject_id, phase, segment, base_path, image_based
        ).transformed_signal_

        return self


class LabelProcessor(Algorithm):
    _action_methods = "label_generation"

    blip_algo: ComputeEcgBlips
    downsampling: Downsampling
    normalizer: Normalizer

    labels_: np.array

    def __init__(
        self,
        blip_algo: ComputeEcgBlips = cf(ComputeEcgBlips()),
        downsampling: Downsampling = cf(Downsampling()),
        normalizer: Normalizer = cf(Normalizer()),
    ):
        self.blip_algo = blip_algo
        self.downsampling = downsampling
        self.normalizer = normalizer

    @make_action_safe
    def label_generation(
        self,
        raw_ecg: np.array,
        sampling_rate: float,
        subject_id: str,
        phase: str,
        segment: int,
        downsample_hz: int = 200,
        base_path: str = "Data",
    ):
        blip_algo_clone = self.blip_algo.clone()
        downsampling_clone = self.downsampling.clone()
        normalization_clone = self.normalizer.clone()
        # Compute the blips
        processed_ecg = blip_algo_clone.compute(raw_ecg).blips_
        # Downsample the segment
        processed_ecg = downsampling_clone.downsample(processed_ecg, downsample_hz, sampling_rate).downsampled_signal_
        # Normalize the segment
        processed_ecg = normalization_clone.normalize(processed_ecg).normalized_signal_

        # Save the labels
        path = self.get_path(subject_id, phase, base_path) + f"/{segment}.npy"
        np.save(path, processed_ecg)
        self.labels_ = processed_ecg
        return self

    def get_path(self, subject_id: str, phase: str, base_path: str = "Data"):
        path = f"{base_path}/{subject_id}/{phase}/labels"
        if not os.path.exists(path):
            os.makedirs(path)
        return path


class InputAndLabelGenerator(Algorithm):
    """Class generating the Input and Label matrices for the BiLSTM model.

    Results:
        self.input_data
        self.input_labels
    """

    _action_methods = ("generate_training_input", "generate_training_labels", "generate_training_inputs_and_labels")

    # PreProcessing
    pre_processor: PreProcessor

    # Downsampling
    downsampling: Downsampling

    # Label Generation
    blip_algo: ComputeEcgBlips

    # Segmentation
    segmentation: Segmentation

    # Normalization
    normalizer: Normalizer

    labelProcessor: LabelProcessor

    # Input & Label parameters
    segment_size_in_seconds: int
    overlap: float
    downsampled_hz: int

    # Results
    input_data_path_: str
    input_labels_path_: str
    test_label_dict_: dict

    def __init__(
        self,
        pre_processor: PreProcessor = cf(PreProcessor()),
        blip_algo: ComputeEcgBlips = cf(ComputeEcgBlips()),
        downsampling: Downsampling = cf(Downsampling()),
        segmentation: Segmentation = cf(Segmentation()),
        normalizer: Normalizer = cf(Normalizer()),
        labelProcessor: LabelProcessor = cf(LabelProcessor()),
        segment_size_in_seconds: int = 5,
        overlap: float = 0.4,
        downsampled_hz: int = 200,
    ):
        self.pre_processor = pre_processor
        self.blip_algo = blip_algo
        self.segment_size_in_seconds = segment_size_in_seconds
        self.overlap = overlap
        self.downsampled_hz = downsampled_hz
        self.downsampling = downsampling
        self.segmentation = segmentation
        self.normalizer = normalizer
        self.labelProcessor = labelProcessor

    @make_action_safe
    def generate_training_input(self, dataset: D02Dataset, base_path: str = "Data"):
        """Generate the input data for the BiLSTM model

        Args:
            raw_radar (np.array): Raw radar signal
            sampling_rate (float): Sampling frequency of the radar signal

        Returns:
            np.ndarray: Input data for the BiLSTM model
        """
        segmentation_clone = self.segmentation.clone()
        pre_processor_clone = self.pre_processor.clone()
        base_path = base_path
        for i in range(len(dataset.subjects)):
            subject = dataset.get_subset(participant=dataset.subjects[i])
            subject_path = base_path + f"/{subject.subjects[0]}"
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)
            print(f"Subject {subject.subjects[0]}")
            radar_data = subject.synced_radar
            ecg_data = subject.synced_ecg
            phases = subject.phases
            sampling_rate = subject.SAMPLING_RATE_DOWNSAMPLED
            for phase in phases.keys():
                if "ei" not in phase:
                    continue
                print(f"Starting phase {phase}")
                # Get the data for the current phase
                timezone = pytz.timezone("Europe/Berlin")
                phase_start = timezone.localize(phases[phase]["start"])
                phase_end = timezone.localize(phases[phase]["end"])
                phase_radar_data = radar_data[phase_start:phase_end]
                phase_ecg_data = ecg_data[phase_start:phase_end]
                # Segmentation
                if len(phase_radar_data) == 0 or len(phase_ecg_data) == 0:
                    continue
                # Create Dir
                phase_path = subject_path + f"/{phase}/inputs"
                if not os.path.exists(phase_path):
                    os.makedirs(phase_path)
                segments = segmentation_clone.segment(phase_radar_data, sampling_rate).segmented_signal_
                for j in range(len(segments)):
                    pre_processor_clone.preprocess(
                        segments[j], subject.SAMPLING_RATE_DOWNSAMPLED, subject.subjects[0], phase, j
                    )
        self.input_data_path_ = base_path
        return self

    @make_action_safe
    def generate_training_inputs_and_labels(
        self, dataset: D02Dataset, base_path: str = "Data", image_based: bool = False
    ):
        # Init Clones
        pre_processor_clone = self.pre_processor.clone()
        label_processor_clone = self.labelProcessor.clone()
        segmentation_clone = self.segmentation.clone()
        already_done = [
            "004",
            "008",
            "038",
            "077",
            "139",
            "142",
            "143",
            "146",
            "160",
            "162",
            "199",
            "213",
            "230",
            "249",
            "254",
            "284",
            "338",
            "416",
            "471",
            "482",
            "559",
        ]
        for i in range(len(dataset.subjects)):
            subject = dataset.get_subset(participant=dataset.subjects[i])
            if subject.subjects[0] in already_done:
                continue
            print(f"Subject {subject.subjects[0]}")
            try:
                radar_data = subject.synced_radar
                ecg_data = subject.synced_ecg
            except Exception:
                print(f"Exclude Subject {subject}")
                continue
            phases = subject.phases
            sampling_rate = subject.SAMPLING_RATE_DOWNSAMPLED
            for phase in phases.keys():
                print(f"Starting phase {phase}")
                timezone = pytz.timezone("Europe/Berlin")
                phase_start = timezone.localize(phases[phase]["start"])
                phase_end = timezone.localize(phases[phase]["end"])
                phase_radar_data = radar_data[phase_start:phase_end]
                phase_ecg_data = ecg_data[phase_start:phase_end]
                # Segmentation
                if len(phase_radar_data) == 0 or len(phase_ecg_data) == 0:
                    continue
                segments_radar = segmentation_clone.segment(phase_radar_data, sampling_rate).segmented_signal_
                segments_ecg = segmentation_clone.segment(phase_ecg_data, sampling_rate).segmented_signal_
                if len(segments_radar) != len(segments_ecg):
                    # print(
                    #     f"Length of radar and ecg segments do not match for {subject} in pahse {phase} the radar_length is {len(segments_radar)} and the ecg_length is {len(segments_ecg)}"
                    # )
                    continue
                # Create Inputs
                length = min(len(segments_radar), len(segments_ecg))
                for j in range(length):
                    pre_processor_clone.preprocess(
                        segments_radar[j],
                        subject.SAMPLING_RATE_DOWNSAMPLED,
                        subject.subjects[0],
                        phase,
                        j,
                        base_path,
                        image_based,
                    )
                    label_processor_clone.label_generation(
                        segments_ecg[j],
                        subject.SAMPLING_RATE_DOWNSAMPLED,
                        subject.subjects[0],
                        phase,
                        j,
                        self.downsampled_hz,
                        base_path,
                    )
        self.input_data_path_ = base_path
        return self

    @make_action_safe
    def generate_training_labels(self, dataset: D02Dataset, base_path: str = "Data"):
        """Generate the input labels for the BiLSTM model

        Args:
            raw_radar (np.array): Raw radar signal
            sampling_rate (float): Sampling frequency of the radar signal

        Returns:
            np.ndarray: Input labels for the BiLSTM model
        """
        print("Labels")
        pre_processor_clone = self.pre_processor.clone()
        label_processor_clone = self.labelProcessor.clone()
        segmentation_clone = self.segmentation.clone()
        for i in range(len(dataset.subjects)):
            subject = dataset.get_subset(participant=dataset.subjects[i])
            radar_data = subject.synced_radar
            ecg_data = subject.synced_ecg
            phases = subject.phases
            sampling_rate = subject.SAMPLING_RATE_DOWNSAMPLED
            for phase in phases.keys():
                if "ei" not in phase:
                    continue
                print(f"Starting phase {phase}")
                timezone = pytz.timezone("Europe/Berlin")
                phase_start = timezone.localize(phases[phase]["start"])
                phase_end = timezone.localize(phases[phase]["end"])
                phase_radar_data = radar_data[phase_start:phase_end]
                phase_ecg_data = ecg_data[phase_start:phase_end]
                # Segmentation
                if len(phase_radar_data) == 0 or len(phase_ecg_data) == 0:
                    continue
                segments_radar = segmentation_clone.segment(phase_radar_data, sampling_rate).segmented_signal_
                segments_ecg = segmentation_clone.segment(phase_ecg_data, sampling_rate).segmented_signal_
                if len(segments_radar) != len(segments_ecg):
                    print("Length of radar and ecg segments do not match")
                    continue
                # Create Inputs
                for j in range(len(segments_ecg)):
                    label_processor_clone.label_generation(
                        segments_ecg[j],
                        subject.SAMPLING_RATE_DOWNSAMPLED,
                        subject.subjects[0],
                        phase,
                        j,
                        self.downsampled_hz,
                    )
        self.input_data_path_ = base_path
        return self

    @make_action_safe
    def generate_label_dict(self, dataset: D02Dataset):
        blip_algo_clone = self.blip_algo.clone()
        downsampling_clone = self.downsampling.clone()
        segmentation_clone = self.segmentation.clone()
        normalization_clone = self.normalizer.clone()
        label_dict = {}
        for i in range(len(dataset.subjects)):
            subject_dict = {}
            subject = dataset.get_subset(participant=dataset.subjects[i])
            ecg_data = subject.synced_ecg
            radar_data = subject.synced_radar
            phases = subject.phases
            sampling_rate = subject.SAMPLING_RATE_DOWNSAMPLED
            for phase in phases.keys():
                if "ei" not in phase:
                    continue
                phase_data = ecg_data[phases[phase]["start"] : phases[phase]["end"]]
                phase_radar = radar_data[phases[phase]["start"] : phases[phase]["end"]]
                if len(phase_data) == 0 or len(phase_radar) == 0:
                    continue
                segments = segmentation_clone.segment(phase_data, sampling_rate).segmented_signal_
                phase_list = []
                for segment in segments:
                    # Compute the blips
                    segment = blip_algo_clone.compute(segment).blips_
                    # Downsample the segment
                    segment = downsampling_clone.downsample(
                        segment, self.downsampled_hz, sampling_rate
                    ).downsampled_signal_
                    # Normalize the segment
                    segment = normalization_clone.normalize(segment).normalized_signal_
                    phase_list.append(segment)
                subject_dict[phase] = phase_list
            label_dict[subject.subjects[0]] = subject_dict
        self.test_label_dict_ = label_dict
        return self


class CnnPipeline(OptimizablePipeline):
    feature_extractor: InputAndLabelGenerator
    cnn: CNN
    cnn_model: OptimizableParameter

    result_ = np.ndarray

    def __init__(
        self,
        feature_extractor: InputAndLabelGenerator = cf(InputAndLabelGenerator()),
        cnn: CNN = cf(CNN()),
    ):
        self.feature_extractor = feature_extractor
        self.cnn = cnn

    def prepare_data(
        self,
        training_data: D02Dataset,
        validation_data: D02Dataset,
        testing_data: D02Dataset,
        path: str = "/home/woody/iwso/iwso116h/Data",
        image_based: bool = False,
    ):
        print("Extracting features and Labels")
        self.feature_extractor.generate_training_inputs_and_labels(training_data, path, image_based)
        print("Generating Validation Set")
        self.feature_extractor.generate_training_inputs_and_labels(
            validation_data, "/home/woody/iwso/iwso116h/Validation", image_based
        )
        print("Generating Testing Set")
        self.feature_extractor.generate_training_inputs_and_labels(
            testing_data, "/home/woody/iwso/iwso116h/Testing", image_based
        )
        return self

    def self_optimize(
        self,
        dataset: D02Dataset,
        validation: D02Dataset,
        path: str = "/home/woody/iwso/iwso116h/Data",
        image_based: bool = False,
    ) -> Self:
        self.feature_extractor = self.feature_extractor.clone()
        self.cnn = self.cnn.clone()
        print("Optimizing CNN")
        self.cnn.self_optimize(path, image_based)
        return self

    def run(self, datapoint: D02Dataset) -> Self:
        print("Run")
        self.cnn.predict("/home/woody/iwso/iwso116h/Testing")
        return self
