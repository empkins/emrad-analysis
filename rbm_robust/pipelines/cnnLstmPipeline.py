import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
import pytz
from tpcp._dataset import DatasetT
from typing_extensions import Self
import tensorflow as tf

import numpy as np
from tpcp import Algorithm, make_action_safe, cf, OptimizablePipeline

from emrad_analysis.feature_extraction.feature_generation_algorithms import ComputeEnvelopeSignal
from emrad_analysis.preprocessing.pre_processing_algorithms import ButterHighpassFilter, ComputeDecimateSignal
from rbm_robust.data_loading.datasets import D02Dataset, RadarCardiaStudyDataset
from rbm_robust.data_loading.tf_datasets import DatasetFactory
from rbm_robust.label_generation.label_generation_algorithm import ComputeEcgBlips, ComputeEcgPeakGaussians
from rbm_robust.models.waveletModel import UNetWaveletTF
from rbm_robust.preprocessing.preprocessing import (
    ButterBandpassFilter,
    Downsampling,
    EmpiricalModeDecomposer,
    WaveletTransformer,
    Segmentation,
    Normalizer,
)
from emrad_toolbox.radar_preprocessing.radar import RadarPreprocessor

from rbm_robust.validation.RPeakF1Score import RPeakF1Score
from rbm_robust.validation.instantenous_heart_rate import ScoreCalculator


def _get_dataset(
    data_path,
    subjects,
    batch_size: int = 8,
    wavelet_type: str = "morl",
    ecg_labels: bool = False,
    log_transform: bool = False,
    single_channel: bool = True,
    diff: bool = False,
    image_based: bool = False,
) -> (tf.data.Dataset, int):
    ds_factory = DatasetFactory()
    if single_channel:
        return ds_factory.get_single_channel_wavelet_dataset_for_subjects(
            base_path=data_path,
            training_subjects=subjects,
            batch_size=batch_size,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            image_based=image_based,
        )
    else:
        return ds_factory.get_dual_channel_wavelet_dataset_for_subjects(
            base_path=data_path,
            training_subjects=subjects,
            batch_size=batch_size,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
        )


class PreProcessor(Algorithm):
    _action_methods = ("preprocess", "preprocess_mag")

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
        highpass_filter: ButterHighpassFilter = cf(ButterHighpassFilter()),
        envelope_algo: ComputeEnvelopeSignal = cf(ComputeEnvelopeSignal()),
        decimation_algo: ComputeDecimateSignal = cf(ComputeDecimateSignal(downsampling_factor=10)),
    ):
        self.highpass_filter = highpass_filter
        self.bandpass_filter = bandpass_filter
        self.downsampling = downsampling
        self.emd = emd
        self.wavelet_transform = wavelet_transform
        self.envelope_algo = envelope_algo
        self.decimation_algo = decimation_algo

    @make_action_safe
    def preprocess_mag(
        self,
        raw_radar: np.array,
        sampling_rate: float,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "/home/woody/iwso/iwso116h/Data",
        image_based: bool = False,
        diff: bool = False,
    ):
        # Initializing
        bandpass_filter_clone = self.bandpass_filter.clone()
        wavelet_transform_clone = self.wavelet_transform.clone()
        downsampling_clone = self.downsampling.clone()
        highpass_filter_clone = self.highpass_filter.clone()
        envelope_algo_clone = self.envelope_algo.clone()
        decimation_algo_clone = self.decimation_algo.clone()

        # I, Q, angle, power, envelope

        # Highpass Filter
        highpassed_radi = highpass_filter_clone.filter(raw_radar["I"], sample_frequency_hz=1000).filtered_signal_
        highpassed_radq = highpass_filter_clone.filter(raw_radar["Q"], sample_frequency_hz=1000).filtered_signal_

        radar_I = downsampling_clone.downsample(highpassed_radi, 200, sampling_rate).downsampled_signal_
        radar_Q = downsampling_clone.downsample(highpassed_radq, 200, sampling_rate).downsampled_signal_

        zeroes_angle = np.zeros_like(radar_I)
        angle = np.diff(np.unwrap(np.arctan2(radar_I, radar_Q)), axis=0)
        zeroes_angle[: len(angle)] = angle[:]
        angle = zeroes_angle

        # Compute the radar power from I and Q
        rad_power = np.sqrt(np.square(radar_I) + np.square(radar_Q))

        # Extract heart sound band and compute the hilbert envelope
        heart_sound_radar = bandpass_filter_clone.filter(rad_power, 200).filtered_signal_
        heart_sound_radar_envelope = envelope_algo_clone.compute(heart_sound_radar).envelope_signal_

        # Get collected Array
        all_inputs = self.collect_and_standardize_array(radar_I, radar_Q, angle, rad_power, heart_sound_radar_envelope)

        # Save the inputs
        path = self.get_filtered_radar_path(subject_id, phase, base_path) + f"/{segment}.npy"
        np.save(path, all_inputs)

        self.preprocessed_signal_ = all_inputs
        return self

    def collect_and_standardize_array(self, I, Q, angle, power, envelope):
        I_norm = self.safe_transform_and_normalize(I)
        Q_norm = self.safe_transform_and_normalize(Q)
        angle_norm = self.safe_transform_and_normalize(angle)
        power_norm = self.safe_transform_and_normalize(power)
        envelope_norm = self.safe_transform_and_normalize(envelope)
        return np.array([I_norm, Q_norm, angle_norm, power_norm, envelope_norm])

    def safe_transform_and_normalize(self, array):
        # array = self.normalize_safely(array)
        return self.safe_range_normalize(array)

    def normalize_safely(self, array):
        std = np.std(array)
        if std == 0:
            return np.zeros_like(array)
        return (array - np.mean(array)) / np.std(array)

    def safe_range_normalize(self, data):
        data_min = np.min(data, axis=0, keepdims=True)
        data_max = np.max(data, axis=0, keepdims=True)
        range_ = data_max - data_min
        range_[range_ == 0] = 1
        range_normalized = (data - data_min) / range_
        return range_normalized

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
        diff: bool = False,
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
        if not isinstance(raw_radar, pd.Series) and raw_radar.shape[1] == 2:
            i_diff = np.diff(raw_radar["I"])
            q_diff = np.diff(raw_radar["Q"])
            radar_mag = RadarPreprocessor().calculate_power(i=raw_radar["I"], q=raw_radar["Q"])
        else:
            radar_mag = raw_radar
        # Bandpass Filter
        self.preprocessed_signal_ = bandpass_filter_clone.filter(radar_mag, sampling_rate).filtered_signal_

        # Downsampling
        self.preprocessed_signal_ = downsampling_clone.downsample(
            self.preprocessed_signal_, 200, sampling_rate
        ).downsampled_signal_

        # Empirical Mode Decomposition
        # self.preprocessed_signal_ = emd_clone.decompose(self.preprocessed_signal_).imfs_

        # Wavelet Transform
        self.preprocessed_signal_ = wavelet_transform_clone.transform(
            self.preprocessed_signal_, subject_id, phase, segment, base_path, image_based, single_signal=True
        ).transformed_signal_

        return self

    def get_filtered_radar_path(
        self, subject_id: str, phase: str, base_path: str = "/home/woody/iwso/iwso116h/RadarData"
    ):
        path = f"{base_path}/{subject_id}/{phase}/filtered_radar"
        if not os.path.exists(path):
            os.makedirs(path)
        return path


class LabelProcessor(Algorithm):
    _action_methods = "label_generation"

    blip_algo: ComputeEcgBlips
    downsampling: Downsampling
    normalizer: Normalizer
    gaussian: ComputeEcgPeakGaussians
    wavelet_transform: WaveletTransformer

    labels_: np.array

    def __init__(
        self,
        blip_algo: ComputeEcgBlips = cf(ComputeEcgBlips()),
        downsampling: Downsampling = cf(Downsampling()),
        normalizer: Normalizer = cf(Normalizer()),
        gaussian: ComputeEcgPeakGaussians = cf(ComputeEcgPeakGaussians()),
        wavelet_transform: WaveletTransformer = cf(WaveletTransformer()),
    ):
        self.blip_algo = blip_algo
        self.downsampling = downsampling
        self.normalizer = normalizer
        self.gaussian = gaussian
        self.wavelet_transform = wavelet_transform

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
        gaussian_clone = self.gaussian.clone()
        wavelet_transform_clone = self.wavelet_transform.clone()

        if isinstance(raw_ecg, pd.DataFrame):
            raw_ecg = raw_ecg["ecg"]

        processed_ecg = raw_ecg
        # Downsample the segment
        processed_ecg = downsampling_clone.downsample(processed_ecg, downsample_hz, sampling_rate).downsampled_signal_
        # Normalize the segment
        processed_ecg = normalization_clone.normalize(processed_ecg).normalized_signal_

        # Save normalized signal
        # path = self.get_ecg_path(subject_id, phase, base_path) + f"/{segment}.npy"
        # np.save(path, processed_ecg)

        # Compute the gaussian
        processed_ecg = gaussian_clone.compute(processed_ecg, downsample_hz).peak_gaussians_

        # # Compute the blips
        # processed_ecg = blip_algo_clone.compute(raw_ecg).blips_

        # Wavelet Transform Label
        # wavelet_transform_clone.transform(
        #     processed_ecg, subject_id, phase, segment, base_path, single_signal=True, identity=True
        # ).transformed_signal_

        # Save the labels
        path = self.get_path(subject_id, phase, base_path) + f"/{segment}.npy"
        np.save(path, processed_ecg)
        self.labels_ = processed_ecg
        return self

    def get_ecg_path(self, subject_id: str, phase: str, base_path: str = "Data"):
        path = f"{base_path}/{subject_id}/{phase}/labels_ecg"
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_path(self, subject_id: str, phase: str, base_path: str = "Data"):
        if phase is None:
            path = f"{base_path}/{subject_id}/labels_gaussian"
        else:
            path = f"{base_path}/{subject_id}/{phase}/labels_gaussian"
        if not os.path.exists(path):
            os.makedirs(path)
        return path


class InputAndLabelGenerator(Algorithm):
    """Class generating the Input and Label matrices for the BiLSTM model.

    Results:
        self.input_data
        self.input_labels
    """

    _action_methods = (
        "generate_training_inputs",
        "generate_training_labels",
        "generate_training_inputs_and_labels",
        "generate_training_inputs_and_labels_radarcadia",
        "generate_training_inputs_and_labels_mag",
        "generate_training_inputs_and_labels_radarcadia_mag",
    )

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
        for i in range(len(dataset.subjects)):
            subject = dataset.get_subset(participant=dataset.subjects[i])
            print(f"Subject {subject.subjects[0]}")
            try:
                radar_data = subject.synced_radar
                ecg_data = subject.synced_ecg
            except Exception as e:
                print(f"Exclude Subject {subject} due to error {e}")
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
                combined_df = pd.concat([phase_radar_data, phase_ecg_data], axis=1, join="outer").fillna(0)
                combined_df_segmented = segmentation_clone.segment(combined_df, sampling_rate).segmented_signal_
                # Create Inputs
                for j in range(len(combined_df_segmented)):
                    pre_processor_clone.preprocess(
                        combined_df_segmented[j][list(phase_radar_data.columns)],
                        subject.SAMPLING_RATE_DOWNSAMPLED,
                        subject.subjects[0],
                        phase,
                        j,
                        base_path,
                        image_based,
                    )
                    label_processor_clone.label_generation(
                        combined_df_segmented[j][list(phase_ecg_data.columns)],
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
    def generate_training_inputs_and_labels_mag(
        self, dataset: D02Dataset, base_path: str = "Data", image_based: bool = False
    ):
        # Init Clones
        pre_processor_clone = self.pre_processor.clone()
        label_processor_clone = self.labelProcessor.clone()
        segmentation_clone = self.segmentation.clone()
        for i in range(len(dataset.subjects)):
            subject = dataset.get_subset(participant=dataset.subjects[i])
            print(f"Subject {subject.subjects[0]}")
            try:
                radar_data = subject.synced_radar
                ecg_data = subject.synced_ecg
            except Exception as e:
                print(f"Exclude Subject {subject} due to error {e}")
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
                combined_df = pd.concat([phase_radar_data, phase_ecg_data], axis=1, join="outer").fillna(0)
                combined_df_segmented = segmentation_clone.segment(combined_df, sampling_rate).segmented_signal_
                for j in range(len(combined_df_segmented)):
                    pre_processor_clone.preprocess_mag(
                        combined_df_segmented[j][list(phase_radar_data.columns)],
                        subject.SAMPLING_RATE_DOWNSAMPLED,
                        subject.subjects[0],
                        phase,
                        j,
                        base_path,
                        image_based,
                    )
                    label_processor_clone.label_generation(
                        combined_df_segmented[j][list(phase_ecg_data.columns)],
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
    def generate_training_inputs_and_labels_radarcadia_mag(
        self, dataset: RadarCardiaStudyDataset, base_path: str = "Data", image_based: bool = False
    ):
        # Init Clones
        pre_processor_clone = self.pre_processor.clone()
        label_processor_clone = self.labelProcessor.clone()
        segmentation_clone = self.segmentation.clone()
        subjects = list(set(dataset.index["subject"]))
        for i in range(len(subjects)):
            print(f"Subject {subjects[i]}")
            # iterate over different locations
            for location in ["carotis", "aorta_prox", "aorta_med", "aorta_dist"]:
                # Check which breathing types are avaialable
                subject = dataset.get_subset(
                    subject=subjects[i],
                    location=location,
                )
                breathing_types = list(set(subject.index["breathing"]))
                for breath in breathing_types:
                    subject = dataset.get_subset(
                        subject=subjects[i],
                        location=location,
                        breathing=breath,
                    )
                    try:
                        radar_data, radar_sampling_rate = subject.emrad_data
                        ecg_data = subject.load_data_from_location("biopac_data_preprocessed")["ecg"]
                    except Exception as e:
                        print(f"Exclude Subject {subject} due to error {e}")
                        continue
                    sampling_rate = subject.sampling_rates["resampled"]
                    segments_radar = segmentation_clone.segment(radar_data, sampling_rate).segmented_signal_
                    segments_ecg = segmentation_clone.segment(ecg_data, sampling_rate).segmented_signal_
                    if len(segments_radar) != len(segments_ecg):
                        continue
                    print(f"Location {location}")
                    # Create Inputs
                    for j in range(len(segments_radar)):
                        pre_processor_clone.preprocess_mag(
                            segments_radar[j],
                            radar_sampling_rate,
                            subjects[0],
                            location + "_" + breath,
                            j,
                            base_path,
                            image_based,
                        )
                        label_processor_clone.label_generation(
                            segments_ecg[j],
                            sampling_rate,
                            subjects[0],
                            location + "_" + breath,
                            j,
                            self.downsampled_hz,
                            base_path,
                        )
        self.input_data_path_ = base_path
        return self

    @make_action_safe
    def generate_training_inputs_and_labels_radarcadia(
        self, dataset: RadarCardiaStudyDataset, base_path: str = "Data", image_based: bool = False
    ):
        # Init Clones
        pre_processor_clone = self.pre_processor.clone()
        label_processor_clone = self.labelProcessor.clone()
        segmentation_clone = self.segmentation.clone()
        subjects = list(set(dataset.index["subject"]))
        for i in range(len(subjects)):
            # subject = dataset.get_subset(subject=subjects[i])
            print(f"Subject {subjects[i]}")
            # iterate over different locations
            for location in ["carotis", "aorta_prox", "aorta_med", "aorta_dist"]:
                # Check which breathing types are avaialable
                subject = dataset.get_subset(
                    subject=subjects[i],
                    location=location,
                )
                breathing_types = list(set(subject.index["breathing"]))
                for breath in breathing_types:
                    subject = dataset.get_subset(
                        subject=subjects[i],
                        location=location,
                        breathing=breath,
                    )
                    try:
                        radar_data = subject.load_data_from_location("emrad_data_preprocessed")["hs_Will_2018"]
                        ecg_data = subject.load_data_from_location("biopac_data_preprocessed")["ecg"]
                    except Exception as e:
                        print(f"Exclude Subject {subject} due to error {e}")
                        continue
                    sampling_rate = subject.sampling_rates["resampled"]
                    segments_radar = segmentation_clone.segment(radar_data, sampling_rate).segmented_signal_
                    segments_ecg = segmentation_clone.segment(ecg_data, sampling_rate).segmented_signal_
                    if len(segments_radar) != len(segments_ecg):
                        continue
                    print(f"Location {location}")
                    # Create Inputs
                    for j in range(len(segments_radar)):
                        pre_processor_clone.preprocess(
                            segments_radar[j],
                            sampling_rate,
                            subjects[0],
                            location + "_" + breath,
                            j,
                            base_path,
                            image_based,
                        )
                        label_processor_clone.label_generation(
                            segments_ecg[j],
                            sampling_rate,
                            subjects[0],
                            location + "_" + breath,
                            j,
                            self.downsampled_hz,
                            base_path,
                        )
        self.input_data_path_ = base_path
        return self

    @make_action_safe
    def generate_training_inputs(self, dataset: D02Dataset, base_path: str = "Data", image_based: bool = False):
        # Init Clones
        pre_processor_clone = self.pre_processor.clone()
        label_processor_clone = self.labelProcessor.clone()
        segmentation_clone = self.segmentation.clone()
        for i in range(len(dataset.subjects)):
            subject = dataset.get_subset(participant=dataset.subjects[i])
            print(f"Subject {subject.subjects[0]}")
            try:
                radar_data = subject.synced_radar
                ecg_data = subject.synced_ecg
            except Exception as e:
                print(f"Exclude Subject {subject} due to error {e}")
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
                    )
        self.input_data_path_ = base_path
        return self

    @make_action_safe
    def generate_training_labels(self, dataset: D02Dataset, base_path: str = "Data"):
        label_processor_clone = self.labelProcessor.clone()
        segmentation_clone = self.segmentation.clone()
        for i in range(len(dataset.subjects)):
            subject = dataset.get_subset(participant=dataset.subjects[i])
            print(f"Subject {subject.subjects[0]}")
            try:
                radar_data = subject.synced_radar
                ecg_data = subject.synced_ecg
            except Exception as e:
                print(f"Exclude Subject {subject} due to error {e}")
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
                    continue
                # Create Inputs
                length = min(len(segments_radar), len(segments_ecg))
                for j in range(length):
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


class PreTrainedPipeline(OptimizablePipeline):
    wavelet_model: UNetWaveletTF
    result_ = np.ndarray
    testing_subjects: list
    testing_path: Path
    ecg_labels: bool
    log_transform: bool
    wavelet_type: str
    batch_size: int
    image_based: bool
    prediction_folder_name: str
    dual_channel: bool

    def __init__(
        self,
        wavelet_type: str = "morl",
        ecg_labels: bool = False,
        log_transform: bool = False,
        batch_size: int = 8,
        image_based: bool = False,
        dual_channel: bool = False,
        model_path: str = None,
        testing_subjects: list = None,
        testing_path: Path = Path("/home/woody/iwso/iwso116h/TestData"),
    ):
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size
        self.image_based = image_based
        self.dual_channel = dual_channel
        self.model_path = model_path
        self.testing_subjects = testing_subjects
        self.testing_path = testing_path
        model_name = Path(model_path).stem

        self.prediction_folder_name = f"predictions_pretrained_{model_name}"

        # Initialize the model
        self.wavelet_model = UNetWaveletTF(
            model_name=model_name,
            batch_size=batch_size,
            image_based=image_based,
            dual_channel=dual_channel,
            model_path=model_path,
        )

    def run(self, path_to_save_predictions: str):
        input_folder_name = f"inputs_wavelet_array_{self.wavelet_type}"
        if self.log_transform and not self.dual_channel:
            input_folder_name += "_log"
        if self.image_based:
            input_folder_name = input_folder_name.replace("array", "image")

        self.wavelet_model.predict(
            testing_subjects=self.testing_subjects,
            data_path=self.testing_path,
            input_folder_name=input_folder_name,
            prediction_folder_name=self.prediction_folder_name,
        )
        return self

    def score(self, datapoint: DatasetT) -> Union[float, dict[str, float]]:
        test_data_folder_name = Path(self.testing_path).name
        label_folder_name = "labels_gaussian" if not self.ecg_labels else "labels_ecg"
        test_path = Path(self.testing_path)

        label_path = test_path
        prediction_path = Path(
            str(label_path).replace(test_data_folder_name, f"Predictions/{self.prediction_folder_name}")
        )

        prominences = [round(i, 2) for i in np.arange(0.05, 0.36, 0.05)]
        for prominence in prominences:
            score_calculator = ScoreCalculator(
                prediction_path=prediction_path,
                label_path=label_path,
                overlap=int(0.4),
                fs=200,
                label_suffix=label_folder_name,
                prominence=prominence,
            )

            if os.getenv("WORK") is None:
                save_path = Path("/Users/simonmeske/Desktop/Masterarbeit")
            else:
                save_path = Path(os.getenv("WORK"))

            scores = score_calculator.calculate_scores()
            # Save the scores as a csv file
            score_path = save_path / "Scores"
            if not score_path.exists():
                score_path.mkdir(parents=True)
            scores.to_csv(score_path / f"scores_{self.prediction_folder_name}_{prominence}.csv")

        # Tar the predictions
        self.tar_predictions(prediction_path)

        # Delete the prediction Directory
        shutil.rmtree(prediction_path)

        return scores

    def tar_predictions(self, prediction_path):
        output_filename = str(prediction_path) + ".tar"
        with tarfile.open(output_filename, "w") as tar:
            tar.add(prediction_path, arcname=os.path.basename(prediction_path))


class D02PipelineImproved(OptimizablePipeline):
    wavelet_model: UNetWaveletTF
    result_ = np.ndarray
    training_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset
    testing_subjects: list
    epochs: int
    learning_rate: float
    data_path: str
    testing_path: Path
    ecg_labels: bool
    log_transform: bool
    breathing_type: str
    training_subjects: list
    validation_subjects: list
    wavelet_type: str
    batch_size: int
    image_based: bool
    prediction_folder_name: str
    dual_channel: bool
    identity: bool
    loss: str
    diff: bool

    def __init__(
        self,
        learning_rate: float = 0.0001,
        data_path: str = "/home/woody/iwso/iwso116h/Data",
        testing_path: Path = Path("/home/woody/iwso/iwso116h/TestData"),
        epochs: int = 50,
        training_subjects: list = None,
        validation_subjects: list = None,
        testing_subjects: list = None,
        wavelet_type: str = "morl",
        ecg_labels: bool = False,
        log_transform: bool = False,
        batch_size: int = 8,
        image_based: bool = False,
        dual_channel: bool = False,
        identity: bool = False,
        loss: str = "bce",
        diff: bool = False,
    ):
        # Set the different fields
        self.diff = diff
        self.loss = loss
        self.identity = identity
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.data_path = data_path
        self.testing_path = testing_path
        self.testing_subjects = testing_subjects
        self.image_based = image_based
        self.dual_channel = dual_channel
        self.training_ds, self.training_steps = _get_dataset(
            data_path=data_path,
            subjects=training_subjects,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            single_channel=not self.dual_channel,
            diff=diff,
            image_based=self.image_based,
        )

        self.validation_ds, self.validation_steps = _get_dataset(
            data_path=data_path,
            subjects=validation_subjects,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            single_channel=not self.dual_channel,
            diff=diff,
            image_based=self.image_based,
        )
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size

        learning_rate_txt = str(learning_rate).replace(".", "_")
        model_name = f"d02_{wavelet_type}_{epochs}_{learning_rate_txt}_{loss}"
        if ecg_labels:
            model_name += "_ecg"
        if log_transform:
            model_name += "_log"
        if image_based:
            model_name += "_image"
        if dual_channel:
            model_name += "_dual"
        if identity:
            model_name += "_identity"
        if diff:
            model_name += "_diff"

        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prediction_folder_name = f"predictions_{model_name}_{time}"

        # Initialize the model
        self.wavelet_model = UNetWaveletTF(
            learning_rate=learning_rate,
            epochs=epochs,
            model_name=model_name,
            training_steps=self.training_steps,
            validation_steps=self.validation_steps,
            training_ds=self.training_ds,
            validation_ds=self.validation_ds,
            batch_size=self.batch_size,
            image_based=image_based,
            loss=loss,
            dual_channel=dual_channel,
        )

    def self_optimize(self):
        self.wavelet_model.self_optimize()
        return self

    def run(self, path_to_save_predictions: str, image_based: bool = False, identity: bool = False):
        input_folder_name = f"inputs_wavelet_array_{self.wavelet_type}"
        if self.log_transform and not self.dual_channel:
            input_folder_name += "_log"
        if self.image_based:
            input_folder_name = input_folder_name.replace("array", "image")
        if identity:
            input_folder_name = input_folder_name.replace("wavelet", "identity")

        if self.diff:
            input_folder_name = "inputs_wavelet_array_diff"

        self.wavelet_model.predict(
            testing_subjects=self.testing_subjects,
            data_path=self.testing_path,
            input_folder_name=input_folder_name,
            prediction_folder_name=self.prediction_folder_name,
        )
        return self

    def score(self, datapoint: DatasetT) -> Union[float, dict[str, float]]:
        test_data_folder_name = Path(self.testing_path).name
        label_folder_name = "labels_gaussian" if not self.ecg_labels else "labels_ecg"
        test_path = Path(self.testing_path)

        label_path = test_path
        prediction_path = Path(
            str(label_path).replace(test_data_folder_name, f"Predictions/{self.prediction_folder_name}")
        )

        score_calculator = ScoreCalculator(
            prediction_path=prediction_path,
            label_path=label_path,
            overlap=0.4,
            fs=200,
            label_suffix=label_folder_name,
        )

        if os.getenv("WORK") is None:
            save_path = Path("/Users/simonmeske/Desktop/Masterarbeit")
        else:
            save_path = Path(os.getenv("WORK"))

        scores = score_calculator.calculate_scores()
        # Save the scores as a csv file
        score_path = save_path / "Scores"
        if not score_path.exists():
            score_path.mkdir(parents=True)
        scores.to_csv(score_path / f"scores_{self.prediction_folder_name}.csv")

        # Tar the predictions
        self.tar_predictions(prediction_path)

        # Delete the prediction Directory
        shutil.rmtree(prediction_path)

        print(f"Scores: {scores}")
        return scores

    def tar_predictions(self, prediction_path):
        output_filename = str(prediction_path) + ".tar"
        with tarfile.open(output_filename, "w") as tar:
            tar.add(prediction_path, arcname=os.path.basename(prediction_path))


class D02Pipeline(OptimizablePipeline):
    model: UNetWaveletTF
    result_ = np.ndarray
    training_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset
    testing_subjects: list
    epochs: int
    learning_rate: float
    data_path: str
    testing_path: str
    ecg_labels: bool
    log_transform: bool
    training_subjects: list
    validation_subjects: list
    wavelet_type: str
    batch_size: int
    image_based: bool
    single_channel: bool
    result_ = np.ndarray

    def __init__(
        self,
        learning_rate: float = 0.0001,
        data_path: str = "/home/woody/iwso/iwso116h/Data",
        testing_path: str = "/home/woody/iwso/iwso116h/TestData",
        epochs: int = 50,
        training_subjects: list = None,
        validation_subjects: list = None,
        testing_subjects: list = None,
        wavelet_type: str = "morl",
        ecg_labels: bool = False,
        log_transform: bool = False,
        batch_size: int = 8,
        image_based: bool = False,
        single_channel: bool = True,
    ):
        # Set the different fields
        self.single_channel = single_channel
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.data_path = data_path
        self.testing_path = testing_path
        self.testing_subjects = testing_subjects
        self.image_based = image_based
        self.training_ds, self.training_steps = _get_dataset(
            data_path=data_path,
            subjects=training_subjects,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            image_based=image_based,
            single_channel=single_channel,
        )
        self.validation_ds, self.validation_steps = _get_dataset(
            data_path=data_path,
            subjects=validation_subjects,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            batch_size=batch_size,
            image_based=image_based,
            single_channel=single_channel,
        )
        self.ecg_labels = ecg_labels
        self.log_transform = log_transform
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects
        self.wavelet_type = wavelet_type
        self.batch_size = batch_size

        model_name = f"d02_{wavelet_type}_{epochs}"
        if image_based:
            model_name += "_image_based"
        if single_channel:
            model_name += "_single_channel"
        if ecg_labels:
            model_name += "_ecg"
        if log_transform and single_channel:
            model_name += "_log"

        # Initialize the model
        self.model = UNetWaveletTF(
            learning_rate=learning_rate,
            epochs=epochs,
            model_name=model_name,
            training_steps=self.training_steps,
            validation_steps=self.validation_steps,
            training_ds=self.training_ds,
            validation_ds=self.validation_ds,
            batch_size=self.batch_size,
            image_based=image_based,
        )

    def self_optimize(self):
        self.wavelet_model.self_optimize()
        return self

    def run(self, testing_subjects: list = None, path: str = "/home/woody/iwso/iwso116h/TestDataRef") -> Self:
        input_folder_name = f"inputs_wavelet_array_{self.wavelet_type}"
        if self.log_transform:
            input_folder_name += "_log"
        if self.image_based:
            input_folder_name = input_folder_name.replace("array", "image")

        self.model.predict(
            testing_subjects=self.testing_subjects,
            data_path=self.testing_path,
            input_folder_name=input_folder_name,
        )
        return self

    def score(self, testing_path: str):
        true_positives = 0
        total_gt_peaks = 0
        total_pred_peaks = 0

        for subject in testing_path.iterdir():
            if not subject.is_dir():
                continue
            if subject.name not in self.testing_subjects:
                continue
            print(f"subject {subject}")
            for phase in subject.iterdir():
                if not phase.is_dir():
                    continue
                if phase.name == "logs" or phase.name == "raw":
                    continue
                print(f"phase {phase}")
                prediction_path = phase
                prediction_path = Path(
                    str(prediction_path).replace("TestDataRef", f"Predictions/{self.prediction_folder_name}")
                )
                label_path = phase / "labels_gaussian"
                prediction_files = sorted(path.name for path in prediction_path.iterdir() if path.is_file())
                f1RPeakScore = RPeakF1Score(max_deviation_ms=100)
                for prediction_file in prediction_files:
                    prediction = np.load(prediction_path / prediction_file)
                    label = np.load(label_path / prediction_file)
                    f1RPeakScore.compute_predictions(prediction, label)
                    true_positives += f1RPeakScore.tp_
                    total_gt_peaks += f1RPeakScore.total_peaks_
                    total_pred_peaks += f1RPeakScore.pred_peaks_

        # Save the Model
        self.model.save_model()

        if total_pred_peaks == 0:
            print("No Peaks detected")
            return {
                "abs_hr_error": 0,
                "mean_instantaneous_error": 0,
                "f1_score": 0,
                "mean_relative_error_hr": 0,
                "mean_absolute_error": 0,
            }

        precision = true_positives / total_pred_peaks
        recall = true_positives / total_gt_peaks
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"f1 Score {f1_score}")
