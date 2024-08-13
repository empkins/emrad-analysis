import os
import pathlib
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pytz
from emrad_toolbox import RadarPreprocessor
from tpcp import Algorithm, cf, make_action_safe

from emrad_analysis.feature_extraction.feature_generation_algorithms import ComputeEnvelopeSignal
from emrad_analysis.preprocessing.pre_processing_algorithms import ButterHighpassFilter, ComputeDecimateSignal
from rbm_robust.data_loading.datasets import D02Dataset, RadarCardiaStudyDataset
from rbm_robust.label_generation.label_generation_algorithm import ComputeEcgBlips, ComputeEcgPeakGaussians
from rbm_robust.preprocessing.preprocessing import (
    ButterBandpassFilter,
    Downsampling,
    EmpiricalModeDecomposer,
    WaveletTransformer,
    Normalizer,
    Segmentation,
)


def run_d02(
    data_path: str,
    target_path: str,
):
    _run_dataset(
        D02Dataset,
        data_path,
        target_path,
        process_function=process_d02_subset,
        num_processes=4,
    )


def run_d02_Mag(
    data_path: str,
    target_path: str,
):
    _run_dataset(
        D02Dataset,
        data_path,
        target_path,
        process_function=process_d02_mag_subset,
        num_processes=4,
    )


def run_d05(data_path: str, target_path: str):
    _run_dataset(
        RadarCardiaStudyDataset,
        data_path,
        target_path,
        process_function=process_radarcadia_subset,
        num_processes=4,
    )


def run_d05_Mag(data_path: str, target_path: str):
    _run_dataset(
        RadarCardiaStudyDataset,
        data_path,
        target_path,
        process_function=process_radarcadia_mag_subset,
        num_processes=4,
    )


def _run_dataset(
    dataset_class,
    data_path,
    target_path,
    process_function,
    num_processes,
):
    available_cores = cpu_count()
    if num_processes > available_cores:
        num_processes = max(1, available_cores - 1)

    subjects_dataset = dataset_class(pathlib.Path(data_path))
    subsets = []
    if isinstance(subjects_dataset, D02Dataset):
        subjects = list(subjects_dataset.subjects)
        subsets = [subjects_dataset.get_subset(participant=subject) for subject in subjects]
    process_function(subjects_dataset, target_path)
    # with Pool(num_processes) as p:
    #     p.starmap(
    #         process_function,
    #         [(subset, target_path) for subset in subsets],
    #     )


def process_d02_subset(
    data_set: D02Dataset,
    target_path: str,
):
    _process_subset(
        data_set,
        target_path,
        generator_method="generate_training_inputs_and_labels",
    )


def process_d02_mag_subset(
    data_set: D02Dataset,
    target_path: str,
):
    _process_subset(
        data_set,
        target_path,
        generator_method="generate_training_inputs_and_labels_mag",
    )


def process_radarcadia_subset(data_set: RadarCardiaStudyDataset, target_path: str):
    _process_subset(
        data_set,
        target_path,
        generator_method="generate_training_inputs_and_labels_radarcadia",
    )


def process_radarcadia_mag_subset(data_set: RadarCardiaStudyDataset, target_path: str):
    _process_subset(
        data_set,
        target_path,
        generator_method="generate_training_inputs_and_labels_radarcadia_mag",
    )


def _process_subset(data_set, target_path, generator_method):
    generator = InputAndLabelGenerator()
    try:
        getattr(generator, generator_method)(data_set, target_path)
    except Exception as e:
        print(f"Error in processing with error {e}")


class PreProcessor(Algorithm):
    _action_methods = ("preprocess_d02", "preprocess_mag", "preprocess_d05")

    bandpass_filter: ButterBandpassFilter
    downsampling: Downsampling
    emd: EmpiricalModeDecomposer
    wavelet_transformer: WaveletTransformer
    downsample_factor: int = 200

    # Results
    preprocessed_signal_: np.ndarray

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
        raw_radar: np.ndarray,
        sampling_rate: float,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "/home/woody/iwso/iwso116h/Data",
    ) -> np.ndarray:
        """
        Preprocess the radar signal for magnitude.

        Args:
            raw_radar (np.ndarray): Raw radar data.
            sampling_rate (float): Sampling rate of the radar data.
            subject_id (str): Subject identifier.
            phase (str): Phase of the data.
            segment (int): Segment number.
            base_path (str): Base path to save the preprocessed data.

        Returns:
            np.ndarray: Preprocessed radar signal.
        """
        radar_I, radar_Q = self._highpass_and_downsample(raw_radar, sampling_rate)
        angle = self._compute_angle(radar_I, radar_Q)
        rad_power = self._compute_power(radar_I, radar_Q)
        heart_sound_radar_envelope = self._compute_heart_sound_envelope(rad_power)
        all_inputs = self._collect_and_standardize_array(radar_I, radar_Q, angle, rad_power, heart_sound_radar_envelope)
        self._save_preprocessed_signal(all_inputs, subject_id, phase, segment, base_path, "filtered_radar")
        self.preprocessed_signal_ = all_inputs
        return self

    def _highpass_and_downsample(self, raw_radar: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
        highpassed_radi = self.highpass_filter.clone().filter(raw_radar["I"], sample_frequency_hz=1000).filtered_signal_
        highpassed_radq = self.highpass_filter.clone().filter(raw_radar["Q"], sample_frequency_hz=1000).filtered_signal_
        radar_I = self.downsampling.clone().downsample(highpassed_radi, 200, sampling_rate).downsampled_signal_
        radar_Q = self.downsampling.clone().downsample(highpassed_radq, 200, sampling_rate).downsampled_signal_
        return radar_I, radar_Q

    def _compute_angle(self, radar_I: np.ndarray, radar_Q: np.ndarray) -> np.ndarray:
        zeroes_angle = np.zeros_like(radar_I)
        angle = np.diff(np.unwrap(np.arctan2(radar_I, radar_Q)), axis=0)
        zeroes_angle[: len(angle)] = angle[:]
        return zeroes_angle

    def _compute_power(self, radar_I: np.ndarray, radar_Q: np.ndarray) -> np.ndarray:
        return np.sqrt(np.square(radar_I) + np.square(radar_Q))

    def _compute_heart_sound_envelope(self, rad_power: np.ndarray) -> np.ndarray:
        heart_sound_radar = self.bandpass_filter.clone().filter(rad_power, 200).filtered_signal_
        return self.envelope_algo.clone().compute(heart_sound_radar).envelope_signal_

    def _collect_and_standardize_array(
        self, I: np.ndarray, Q: np.ndarray, angle: np.ndarray, power: np.ndarray, envelope: np.ndarray
    ) -> np.ndarray:
        I_norm = self._normalize_safely(I)
        Q_norm = self._normalize_safely(Q)
        angle_norm = self._normalize_safely(angle)
        power_norm = self._normalize_safely(power)
        envelope_norm = self._normalize_safely(envelope)
        return np.array([I_norm, Q_norm, angle_norm, power_norm, envelope_norm])

    def _normalize_safely(self, array: np.ndarray) -> np.ndarray:
        std = np.std(array)
        if std == 0:
            return np.zeros_like(array)
        return (array - np.mean(array)) / std

    def _save_preprocessed_signal(
        self, signal: np.ndarray, subject_id: str, phase: str, segment: int, base_path: str, folder_name: str
    ):
        path = Path(base_path) / subject_id / phase / folder_name / f"{segment}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, signal)

    @make_action_safe
    def preprocess_d02(
        self,
        raw_radar: np.ndarray,
        sampling_rate: float,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "/home/woody/iwso/iwso116h/Data",
    ) -> np.ndarray:
        """
        Preprocess the input signal using a bandpass filter.

        Args:
            raw_radar (np.ndarray): Input signal to be preprocessed.
            sampling_rate (float): Sampling frequency of the input signal.
            subject_id (str): Subject identifier.
            phase (str): Phase of the data.
            segment (int): Segment number.
            base_path (str): Base path to save the preprocessed data.
            diff (bool): Whether to use differential data.

        Returns:
            np.ndarray: Preprocessed signal.
        """
        radar_mag = self._calculate_radar_magnitude(raw_radar)
        filtered_signal = self.bandpass_filter.clone().filter(radar_mag, sampling_rate).filtered_signal_
        downsampled_signal = (
            self.downsampling.clone().downsample(filtered_signal, 200, sampling_rate).downsampled_signal_
        )
        transformed_signal = (
            self.wavelet_transform.clone()
            .transform(downsampled_signal, subject_id, phase, segment, base_path)
            .transformed_signal_
        )
        self.preprocessed_signal_ = transformed_signal
        return self

    @staticmethod
    def _calculate_radar_magnitude(raw_radar: np.ndarray) -> np.ndarray:
        if not isinstance(raw_radar, pd.Series) and raw_radar.shape[1] == 2:
            return RadarPreprocessor().calculate_power(i=raw_radar["I"], q=raw_radar["Q"])
        elif isinstance(raw_radar, pd.DataFrame) and "I" in raw_radar.columns and "Q" in raw_radar.columns:
            return RadarPreprocessor().calculate_power(i=raw_radar["I"], q=raw_radar["Q"])
        return raw_radar


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
        label_type: str = "gaussian",
    ):
        blip_algo_clone = self.blip_algo.clone()
        downsampling_clone = self.downsampling.clone()
        normalization_clone = self.normalizer.clone()
        gaussian_clone = self.gaussian.clone()

        if isinstance(raw_ecg, pd.DataFrame):
            raw_ecg = raw_ecg["ecg"]

        processed_ecg = raw_ecg
        # Downsample the segment
        processed_ecg = downsampling_clone.downsample(processed_ecg, downsample_hz, sampling_rate).downsampled_signal_
        # Normalize the segment
        processed_ecg = normalization_clone.normalize(processed_ecg).normalized_signal_

        # Compute the gaussian
        if label_type == "gaussian":
            processed_ecg = gaussian_clone.compute(processed_ecg, downsample_hz).peak_gaussians_

        # # Compute the blips
        # processed_ecg = blip_algo_clone.compute(raw_ecg).blips_

        # Save the labels
        path = self.get_path(subject_id, phase, base_path) + f"/{segment}.npy"
        np.save(path, processed_ecg)
        self.labels_ = processed_ecg
        return self

    def get_path(self, subject_id: str, phase: str, base_path: str = "Data", label_type: str = "gaussian"):
        if label_type == "gaussian":
            label_folder_name = "labels_gaussian"
        elif label_type == "ecg":
            label_folder_name = "labels_ecg"
        else:
            raise ValueError(f"Label type {label_type} not supported.")
        path = f"{base_path}/{subject_id}/{phase}/{label_folder_name}"
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
                    pre_processor_clone.preprocess_d02(
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
                    pre_processor_clone.preprocess_d02(
                        combined_df_segmented[j][list(phase_radar_data.columns)],
                        subject.SAMPLING_RATE_DOWNSAMPLED,
                        subject.subjects[0],
                        phase,
                        j,
                        base_path,
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
                    self.pre_processor.clone().preprocess_mag(
                        combined_df_segmented[j][list(phase_radar_data.columns)],
                        subject.SAMPLING_RATE_DOWNSAMPLED,
                        subject.subjects[0],
                        phase,
                        j,
                        base_path,
                    )
                    self.labelProcessor.clone().label_generation(
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
                        self.pre_processor.clone().preprocess_mag(
                            segments_radar[j],
                            radar_sampling_rate,
                            subjects[i],
                            location + "_" + breath,
                            j,
                            base_path,
                        )
                        self.labelProcessor.clone().label_generation(
                            segments_ecg[j],
                            sampling_rate,
                            subjects[i],
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
                        # radar_data = subject.load_data_from_location("emrad_data_preprocessed")["hs_Will_2018"]
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
                        pre_processor_clone.preprocess_d02(
                            segments_radar[j], sampling_rate, subjects[i], location + "_" + breath, j, base_path
                        )
                        label_processor_clone.label_generation(
                            segments_ecg[j],
                            sampling_rate,
                            subjects[i],
                            location + "_" + breath,
                            j,
                            self.downsampled_hz,
                            base_path,
                        )
        self.input_data_path_ = base_path
        return self

    @make_action_safe
    def generate_training_inputs(self, dataset: D02Dataset, base_path: str = "Data"):
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
                    pre_processor_clone.preprocess_d02(
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
