from tpcp import Algorithm, Parameter, make_action_safe

import pandas as pd
import numpy as np
from neurokit2 import ecg_process, ecg_peaks
from scipy.signal.windows import gaussian
from scipy import signal


class ComputeEcgPeakGaussians(Algorithm):
    _action_methods = "compute"

    # Input Parameters
    method: Parameter[str]
    gaussian_length: Parameter[int]
    gaussian_std: Parameter[float]

    # Results
    peak_gaussians_: pd.Series

    def __init__(self, gaussian_length: int = 400, gaussian_std: float = 8):
        self.gaussian_length = gaussian_length
        self.gaussian_std = gaussian_std

    @make_action_safe
    def compute(self, ecg_signal: pd.Series, sampling_freq: float):
        """Compute the target signal with gaussians positioned at the ecg's R-peaks

        Args:
            ecg_signal (pd.Series): ECG-signal being ground-truth, containing the peaks.
            sampling_freq (float): Sampling frequency of the ECG signal.

        Returns:
            pd.Series: Signal with Gaussians located at the R-peaks of the ECG signal.
        """
        try:
            signal, info = ecg_process(ecg_signal, sampling_freq)
            self.peak_gaussians_ = np.convolve(
                signal["ECG_R_Peaks"], gaussian(self.gaussian_length, self.gaussian_std), mode="same"
            )
            return self
        except Exception as e:
            print(f"Error in ecg_process: {e}")
            self.peak_gaussians_ = np.zeros(len(ecg_signal))
            return self


class ComputeEcgBlips(Algorithm):
    _action_methods = "compute"

    # Input Parameters
    alpha: Parameter[float]

    # Results
    blips_: pd.Series

    def __init__(self, alpha: float = 0.002):
        self.alpha = alpha

    @make_action_safe
    def compute(self, ecg_signal: pd.Series):
        """Compute the target signal with gaussians positioned at the ecg's R-peaks

        Args:
            ecg_signal (pd.Series): ECG-signal being ground-truth, containing the peaks.

        Returns:
            pd.Series: Signal with Gaussian's located at the R-peaks of the ECG signal.
        """
        ecg_abs = np.absolute(ecg_signal.copy())
        integrated_signal = ecg_abs.copy()
        for i in range(1, len(ecg_abs)):
            integrated_signal[i] = self.alpha * ecg_abs[i] + (1 - self.alpha) * integrated_signal[i - 1]
        self.blips_ = integrated_signal
        return self


class ComputeTriangularWavesRPeaks(Algorithm):
    _action_methods = "compute"

    # Results
    triangular_waves_: pd.Series

    @make_action_safe
    def compute(self, ecg_signal: pd.Series, sampling_rate: float):
        """Compute the target signal with triangular waves

        Args:
            sampling_rate (float): Sampling frequency of the ECG signal.

        Returns:
            pd.Series: Signal with triangular waves.
        """
        _, rpeaks = ecg_peaks(ecg_signal, sampling_rate=int(sampling_rate))
        triangular_waves = np.zeros_like(ecg_signal)
        for i in range(len(rpeaks["ECG_R_Peaks"]) - 1):
            start = rpeaks["ECG_R_Peaks"][i]
            end = rpeaks["ECG_R_Peaks"][i + 1]
            length = end - start
            t = np.linspace(0, np.pi, length)
            triangular_wave = -1 * signal.sawtooth(t + np.pi / 2, 0.5)
            triangular_waves[start:end] = triangular_wave
        self.triangular_waves_ = pd.Series(triangular_waves)
        return self


class ComputeTriangularWavesSPeaks(Algorithm):
    _action_methods = "compute"

    # Results
    triangular_waves_: pd.Series

    @make_action_safe
    def compute(self, ecg_signal: pd.Series, sampling_rate: float):
        """Compute the target signal with triangular waves

        Args:
            ecg_signal (pd.Series): ECG signal.
            sampling_rate (float): Sampling frequency of the ECG signal.

        Returns:
            pd.Series: Signal with triangular waves.
        """
        _, rpeaks = ecg_peaks(ecg_signal, sampling_rate=int(sampling_rate))
        triangular_waves = np.zeros_like(ecg_signal)
        speaks = [
            np.argmin(ecg_signal[rpeaks["ECG_R_Peaks"][i] : rpeaks["ECG_R_Peaks"][i + 1]]) + rpeaks["ECG_R_Peaks"][i]
            for i in range(len(rpeaks["ECG_R_Peaks"]) - 1)
        ]
        for i in range(len(speaks) - 1):
            start = speaks[i]
            end = speaks[i + 1]
            length = end - start
            t = np.linspace(0, np.pi, length)
            triangular_wave = signal.sawtooth(t + np.pi / 2, 0.5)
            triangular_waves[start:end] = triangular_wave
        self.triangular_waves_ = pd.Series(triangular_waves)

        return self
