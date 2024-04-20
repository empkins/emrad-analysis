from typing import Tuple

import numpy as np
import pywt
import resampy
import scipy
from PyEMD import EMD
from scipy.signal import filtfilt, butter
from tpcp import Algorithm, Parameter, make_action_safe

import pandas as pd


class ButterBandpassFilter(Algorithm):
    _action_methods = "filter"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float]
    low_pass_filter_cutoff_hz: Parameter[float]
    band_pass_filter_order: Parameter[int]

    # Results
    filtered_signal_: pd.Series

    def __init__(
        self,
        high_pass_filter_cutoff_hz: float = 80,
        low_pass_filter_cutoff_hz: float = 18,
        band_pass_filter_order: int = 4,
    ):
        self.low_pass_filter_cutoff_hz = low_pass_filter_cutoff_hz
        self.high_pass_filter_cutoff_hz = high_pass_filter_cutoff_hz
        self.band_pass_filter_order = band_pass_filter_order

    @make_action_safe
    def filter(self, radar_data: pd.Series, sample_frequency_hz: float):
        """Bandpass filter, filtering the power signal of the radar

        Args:
            radar_data (pd.Series): rad (magnitude/power of complex radar signal)
            sample_frequency_hz (float): For Radar: When aligned already with biopac is 1000Hz, raw data is 1953.125Hz.

        Returns:
            _type_: bandpass-filterd signal
        """
        radar = radar_data.to_numpy().flatten()

        nyq = 0.5 * sample_frequency_hz
        low = self.low_pass_filter_cutoff_hz / nyq
        high = self.high_pass_filter_cutoff_hz / nyq
        b, a = butter(self.band_pass_filter_order, [low, high], btype="band", analog=False)
        res = filtfilt(b, a, radar, axis=0)
        self.filtered_signal_ = pd.Series(res)
        return self


class Downsampling(Algorithm):
    _action_methods = "downsample"

    # Input Parameters
    target_sample_frequency_hz: Parameter[float]

    # Results
    downsampled_signal_: np.ndarray

    def __init__(self, target_sample_frequency_hz: float = 200):
        self.target_sample_frequency_hz = target_sample_frequency_hz

    @make_action_safe
    def downsample(self, signal: pd.Series, sample_frequency_hz: float, fs_in: float = 1000):
        """Downsample the input signal to the target frequency

        Args:
            signal (pd.Series): Input signal to be downsampled
            sample_frequency_hz (float): Sampling frequency of the input signal
            fs_in (float): Input sampling frequency
        Returns:
            _type_: Downsampled signal
        """
        if fs_in <= self.target_sample_frequency_hz:
            self.downsampled_signal_ = signal.to_numpy()
            return self
        if isinstance(signal, pd.Series):
            signal = signal.to_numpy()
        self.downsampled_signal_ = resampy.resample(
            signal, sr_orig=fs_in, sr_new=self.target_sample_frequency_hz, axis=0, parallel=True
        )
        return self


class EmpiricalModeDecomposer(Algorithm):
    _action_methods = "decompose"

    sampling_rate: Parameter[float]

    # Input Parameters
    n_imfs: Parameter[int]

    # Results
    imfs_: np.ndarray

    def __init__(self, n_imfs: int = 4, sampling_rate: float = 200):
        self.n_imfs = n_imfs
        self.sampling_rate = sampling_rate

    @make_action_safe
    def decompose(self, signal: np.array):
        """Decompose the input signal into IMFs

        Args:
            signal (pd.Series): Input signal to be decomposed

        Returns:
            _type_: Decomposed signal
        """
        # Hier kann manchmal ein leeres array zurückgegeben werden
        emd = EMD()
        self.imfs_ = emd.emd(signal, np.arange(len(signal)) / self.sampling_rate, max_imf=self.n_imfs)
        if len(self.imfs_) == 0:
            self.imfs_ = np.zeros((5, 1000))
        return self


class WaveletTransformer(Algorithm):
    _action_methods = "transform"

    # Input Parameters
    wavelet_coefficients: Parameter[Tuple[int, int]]
    wavelet_type: Parameter[str]
    sampling_rate: Parameter[float]

    # Results
    transformed_signal_: np.array

    def __init__(
        self, wavelet_coefficients: Tuple[int, int] = (1, 256), wavelet_type="morl", sampling_rate: float = 200
    ):
        self.wavelet_coefficients = wavelet_coefficients
        self.wavelet_type = wavelet_type
        self.sampling_rate = sampling_rate

    @make_action_safe
    def transform(self, signal: np.array):
        """Transform the input signal using a wavelet transform

        Args:
            signal (pd.Series): Input signal to be transformed

        Returns:
            _type_: Transformed signal
        """
        transformed = []
        for i in range(len(signal)):
            imf = signal[i]
            scales = range(self.wavelet_coefficients[0], self.wavelet_coefficients[1])
            coefficients, frequencies = pywt.cwt(imf, scales, self.wavelet_type)

            # Normalize Coefficients
            if np.sum(coefficients) != 0:
                coefficients = (coefficients - np.mean(coefficients)) / np.std(coefficients)
            transformed.append(coefficients)
        if len(transformed) == 0:
            self.transformed_signal_ = np.zeros(1)
        self.transformed_signal_ = np.stack(transformed, axis=2)
        return self


class Segmentation(Algorithm):
    _action_methods = "segment"

    # Input Parameters
    window_size_in_seconds: Parameter[int]
    overlap: Parameter[float]

    # Results
    segmented_signal_: list

    def __init__(self, window_size_in_seconds: int = 5, overlap: float = 0.8):
        self.window_size_in_seconds = window_size_in_seconds
        self.overlap = overlap

    @make_action_safe
    def segment(self, signal: pd.Series, sampling_rate: float):
        step_size = int(self.window_size_in_seconds - self.window_size_in_seconds * self.overlap)
        total_seconds = (signal.index.max() - signal.index.min()).total_seconds()
        step_count = int((total_seconds // step_size) - 1)
        start_time = signal.index[0]
        time_step = signal.index[1] - signal.index[0]
        segments = []
        for j in range(1, step_count):
            end = start_time + pd.Timedelta(seconds=self.window_size_in_seconds)
            # Preprocess the data
            data_segment = signal[start_time:end]
            if len(data_segment) == 0:
                time_diff = pd.Timedelta(seconds=0)
            else:
                time_diff = data_segment.index[-1] - data_segment.index[0]
            # Zero padding
            if len(data_segment) < self.window_size_in_seconds * sampling_rate:
                rows_needed = int((pd.Timedelta(seconds=self.window_size_in_seconds) - time_diff) / time_step)
                # rows_needed = int((pd.Timedelta(seconds=4) - time_diff) / time_step)
                zero_df = pd.DataFrame(np.zeros((rows_needed, len(data_segment.columns))), columns=data_segment.columns)
                data_segment = data_segment.append(zero_df, ignore_index=True)
                data_segment.index = pd.date_range(
                    start=data_segment.index[0], periods=len(data_segment), freq=time_step
                )
            start_time = start_time + pd.Timedelta(seconds=step_size)
            if isinstance(data_segment, pd.DataFrame) and "ecg" in data_segment.columns:
                data_segment = data_segment["ecg"]
            segments.append(data_segment)
        self.segmented_signal_ = segments
        return self


class Normalizer(Algorithm):
    _action_methods = "normalize"

    # Input Parameters
    method: Parameter[str]

    # Results
    normalized_signal_: pd.Series

    def __init__(self, method: str = "zscore"):
        self.method = method

    @make_action_safe
    def normalize(self, signal: pd.Series):
        """Normalize the input signal

        Args:
            signal (pd.Series): Input signal to be normalized

        Returns:
            _type_: Normalized signal
        """
        if self.method == "zscore":
            self.normalized_signal_ = (signal - signal.mean()) / signal.std()
        elif self.method == "minmax":
            self.normalized_signal_ = (signal - signal.min()) / (signal.max() - signal.min())
        return self
