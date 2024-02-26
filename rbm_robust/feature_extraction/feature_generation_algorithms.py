import io
from typing import Tuple

import numpy as np
import pywt
from PIL import Image
from PyEMD import EMD
from matplotlib import pyplot as plt
from tpcp import Algorithm, make_action_safe, Parameter
from keras.preprocessing import image as keras_image


class ComputeEmd(Algorithm):

    _action_methods = "compute"

    # Input Parameters
    average_length: int
    max_modes: int

    # Results
    emd_wavelet_transformation_: np.array

    def __init__(
        self,
        max_modes: int = 4,
    ):
        self.max_modes = max_modes
        self._emd = EMD()

    @make_action_safe
    def compute(self, radar_data: np.array):
        """Compute the envelope of an underlying signal using same-length convolution with a normalized impulse train

        Args:
            radar_data (np.array): rad (magnitude/power of complex radar signal usually already bandpass filtered to
        """

        self.emd_wavelet_transformation_ = self._emd(radar_data, max_imf=self.max_modes)
        return self


class ComputeWaveletTransform(Algorithm):

        _action_methods = "compute"

        # Input Parameters
        wavelet: str
        wavelet_scales: Tuple[int]
        log_scale: bool
        sampling_rate: float

        # Results
        wavelet_coefficients_: np.array
        wavelet_frequencies_: np.array
        times_: np.array

        def __init__(
            self,
            wavelet: str = "morl",
            wavelet_scales: Tuple[int] = (40, 200),
            log_scale: bool = True,
        ):
            self.wavelet = wavelet
            self.wavelet_scales = wavelet_scales
            self.log_scale = log_scale

        @make_action_safe
        def compute(self, radar_data: np.array, sampling_rate: float = 1953.125):
            """Compute the envelope of an underlying signal using same-length convolution with a normalized impulse train

            Args:
                radar_data (np. array): rad (magnitude/power of complex radar signal usually already bandpass filtered to
            """
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
    wavelet_plot: np.array

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
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_array = keras_image.img_to_array(img)
        self.wavelet_plot = img_array
        return self