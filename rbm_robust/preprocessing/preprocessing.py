from typing import Tuple

import numpy as np
import pywt
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
    downsampled_signal_: pd.Series

    def __init__(self, target_sample_frequency_hz: float = 1000):
        self.target_sample_frequency_hz = target_sample_frequency_hz

    @make_action_safe
    def downsample(self, signal: pd.Series, sample_frequency_hz: float):
        """Downsample the input signal to the target frequency

        Args:
            signal (pd.Series): Input signal to be downsampled
            sample_frequency_hz (float): Sampling frequency of the input signal

        Returns:
            _type_: Downsampled signal
        """
        if sample_frequency_hz <= self.target_sample_frequency_hz:
            self.downsampled_signal_ = signal
            return self
        total_seconds = (signal.index.max() - signal.index.min()).total_seconds()
        samps = total_seconds * self.target_sample_frequency_hz
        self.downsampled_signal_ = scipy.signal.resample(signal, samps)
        return self


class EmpiricalModeDecomposer(Algorithm):
    _action_methods = "decompose"

    sampling_rate: Parameter[float]

    # Input Parameters
    n_imfs: Parameter[int]

    # Results
    imfs_: np.ndarray

    def __init__(self, n_imfs: int = 4, sampling_rate: float = 100):
        self.n_imfs = n_imfs
        self.sampling_rate = sampling_rate

    @make_action_safe
    def decompose(self, signal: pd.Series):
        """Decompose the input signal into IMFs

        Args:
            signal (pd.Series): Input signal to be decomposed

        Returns:
            _type_: Decomposed signal
        """

        emd = EMD()
        self.imfs_ = emd.emd(signal.to_numpy(), np.arange(len(signal)) / self.sampling_rate, max_imf=self.n_imfs)
        return self


class WaveletTransformer(Algorithm):
    _action_methods = "transform"

    # Input Parameters
    wavelet_coefficients: Parameter[Tuple[int, int]]
    wavelet_type: Parameter[str]
    sampling_rate: Parameter[float]

    # Results
    transformed_signal_: np.ndarray

    def __init__(
        self, wavelet_coefficients: Tuple[int, int] = (1, 256), wavelet_type="morl", sampling_rate: float = 200
    ):
        self.wavelet_coefficients = wavelet_coefficients
        self.wavelet_type = wavelet_type
        self.sampling_rate = sampling_rate

    @make_action_safe
    def transform(self, signal: pd.Series):
        """Transform the input signal using a wavelet transform

        Args:
            signal (pd.Series): Input signal to be transformed

        Returns:
            _type_: Transformed signal
        """
        scales = np.arange(self.wavelet_coefficients[0], self.wavelet_coefficients[1])
        coefficients, frequencies = pywt.cwt(signal, scales, self.wavelet_type)
        time = np.arange(0, len(signal) / self.sampling_rate, 1 / self.sampling_rate)
        self.transformed_signal_ = np.ndarray[coefficients, frequencies, time]
        return self
