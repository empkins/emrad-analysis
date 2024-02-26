import numpy as np
from tpcp import Algorithm, make_action_safe

from rbm_robust.preprocessing.preprocessing import ButterBandpassFilter


class PreProcessor(Algorithm):


    _action_methods = "preprocess"

    bandpass_filter: ButterBandpassFilter

    #Results
    preprocessed_signal_: np.array

    def __init__(self, bandpass_filter: ButterBandpassFilter):
        self.bandpass_filter = bandpass_filter

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

        return self
