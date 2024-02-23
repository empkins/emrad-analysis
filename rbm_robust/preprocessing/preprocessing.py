from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import filtfilt, butter
from tpcp import Algorithm, Parameter, make_action_safe
from PyEMD import EMD
import pandas as pd

from rbm_robust.preprocessing.preprocessing_errors import WrongInputFormat, WaveletCoefficientsNotProvidedError, \
    WaveletScalesNotWellFormed


class EmpiricalModeDecomposition(Algorithm):

    _action_methods = "decompose"
    emd = EMD()

    # Input Parameters
    max_modes: Parameter[int]
    # Results
    decomposed_signals_: np.array

    def __init__(self, max_modes: int = 4):
        self.max_modes = max_modes

    @make_action_safe
    def decompose(self, radar_data: np.array):

        # Check if signal is complex and if so, take the magnitude
        if np.any(np.iscomplex(radar_data)):
            radar_data = np.abs(radar_data)

        # Check if signal is 1D and if not raise an error
        if radar_data.ndim != 1:
            raise WrongInputFormat()

        # Decompose the signal
        self.decomposed_signals_ = self.emd(radar_data, max_imf=self.max_modes)
        return self


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
            low_pass_filter_cutoff_hz: float = 15,
            band_pass_filter_order: int = 5
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
        b, a = butter(self.band_pass_filter_order, [low, high], btype='band', analog=False)
        res = filtfilt(b, a, radar, axis=0)
        self.filtered_signal_ = pd.Series(res)
        return self


class WaveletTransformation(Algorithm):

    _action_methods = "transform"

    # Input Parameters
    wavelet: Parameter[str]
    wavelet_scales: Parameter[Tuple[int]]
    log_scale: Parameter[bool]

    # Results
    wavelet_frequencies_: np.array
    wavelet_coefficients_: np.array
    times_: np.array

    def __init__(
        self,
        wavelet: str = "morl",
        wavelet_scales: Tuple[int] = (40, 200),
        log_scale: bool = True
    ):
        if len(wavelet_scales) != 2:
            raise WaveletScalesNotWellFormed()
        self.wavelet = wavelet
        self.wavelet_scales = wavelet_scales
        self.log_scale = log_scale

    @make_action_safe
    def transform(self, radar_data, sampling_rate: float):

        if self.wavelet_scales is None:
            raise WaveletCoefficientsNotProvidedError()

        # Compute the wavelet transform
        coefficients, frequencies = pywt.cwt(radar_data, self.wavelet_scales, self.wavelet)
        time = np.arange(0, len(radar_data) / sampling_rate, 1 / sampling_rate)

        # Store the results
        self.wavelet_coefficients_ = coefficients
        self.wavelet_frequencies_ = frequencies
        self.times_ = time
        return self


class WaveletGraph(Algorithm):

    _action_methods = "plot"

    # Input Parameters
    log_scale: Parameter[bool]

    # Results
    wavelet_plot: plt.Figure

    def __init__(
        self,
        log_scale: bool = True
    ):

        self.log_scale = log_scale

    @make_action_safe
    def plot(self, wavelet_coefficients, times, frequencies):
        # Set the DPI and size in inches
        dpi = 80.0
        width, height = 800, 600  # width and height in pixels
        figsize = width / dpi, height / dpi  # size in inches

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(np.abs(wavelet_coefficients), extent=[times.min(), times.max(), frequencies.min(), frequencies.max()],
                  aspect='auto', cmap='jet')
        if self.log_scale:
            ax.set_yscale('log')
        ax.axis('off')
        self.wavelet_plot = fig
        return self
