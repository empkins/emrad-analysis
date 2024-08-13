from pathlib import Path
from typing import Tuple, List, Union
import numpy
import numpy as np
import pywt
import resampy
from PyEMD import EMD
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter
from tpcp import Algorithm, Parameter, make_action_safe, cf

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
        if isinstance(radar_data, pd.Series):
            radar = radar_data.to_numpy().flatten()
        else:
            radar = radar_data.flatten()

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
    downsampled_signal_: numpy.ndarray

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
    imfs_: numpy.ndarray

    def __init__(self, n_imfs: int = 4, sampling_rate: float = 200):
        self.n_imfs = n_imfs
        self.sampling_rate = sampling_rate

    @make_action_safe
    def decompose(self, signal: numpy.array):
        """Decompose the input signal into IMFs

        Args:
            signal (pd.Series): Input signal to be decomposed

        Returns:
            _type_: Decomposed signal
        """
        # Hier kann manchmal ein leeres array zurückgegeben werden
        emd = EMD()
        self.imfs_ = emd.emd(signal, numpy.arange(len(signal)) / self.sampling_rate, max_imf=self.n_imfs)
        self.imfs_ = self.process_array(self.imfs_)
        return self

    def process_array(self, input_array):
        output_array = np.zeros((5, 1000))
        if input_array.size > 0:
            n = input_array.shape[0]
            for i in range(n - 1):
                output_array[i, :] = input_array[i, :]
            output_array[4, :] = input_array[n - 1, :]
        return output_array


class Segmentation(Algorithm):
    _action_methods = "segment"

    # Input Parameters
    window_size_in_seconds: Parameter[int]
    overlap: Parameter[float]

    # Results
    segmented_signal_: List[pd.Series]

    def __init__(self, window_size_in_seconds: int = 5, overlap: float = 0.4):
        self.window_size_in_seconds = window_size_in_seconds
        self.overlap = overlap

    @make_action_safe
    def segment(self, signal: pd.Series, sampling_rate: float):
        """
        Segment the input signal into overlapping windows.

        Args:
            signal (pd.Series): Input signal to be segmented.
            sampling_rate (float): Sampling rate of the input signal.

        Returns:
            Segmentation: The instance with segmented signals.
        """
        step_size = int(self.window_size_in_seconds - self.window_size_in_seconds * self.overlap)
        total_seconds = (signal.index.max() - signal.index.min()).total_seconds()
        step_count = int((total_seconds // step_size) - 1)
        start_time = signal.index[0]
        time_step = signal.index[1] - signal.index[0]
        segments = []

        for j in range(step_count):
            end = start_time + pd.Timedelta(seconds=self.window_size_in_seconds)
            data_segment = signal[start_time:end]
            start_time = start_time + pd.Timedelta(seconds=step_size)
            data_segment = self._zero_pad(data_segment, sampling_rate, time_step)
            if data_segment is not None:
                segments.append(data_segment)

        self.segmented_signal_ = segments
        return self

    def _zero_pad(self, data_segment: Union[pd.Series, pd.DataFrame], sampling_rate: float, time_step: pd.Timedelta):
        """
        Zero pad the data segment to the window size.

        Args:
            data_segment (Union[pd.Series, pd.DataFrame]): Data segment to be zero padded.
            sampling_rate (float): Sampling rate of the input signal.
            time_step (pd.Timedelta): Time step between samples.

        Returns:
            Zero padded data segment or None if the segment is empty.
        """
        if len(data_segment) < self.window_size_in_seconds * sampling_rate:
            if isinstance(data_segment, pd.Series):
                data_segment = self._zero_pad_series(data_segment)
            elif isinstance(data_segment, pd.DataFrame) and len(data_segment) > 0:
                data_segment = self._zero_pad_df(data_segment)
            elif len(data_segment) == 0:
                print(f"Data segment is empty: {data_segment}")
                return None
            data_segment.index = pd.date_range(start=data_segment.index[0], periods=len(data_segment), freq=time_step)
        return data_segment

    def _zero_pad_series(self, data_segment: pd.Series):
        """
        Zero pad a pandas Series to the window size.

        Args:
            data_segment (pd.Series): Data segment to be zero padded.

        Returns:
            Zero padded data segment.
        """
        start_time = data_segment.index[0]
        end_time = start_time + pd.Timedelta(seconds=self.window_size_in_seconds)
        index = pd.date_range(start=start_time, end=end_time, freq="1ms")
        df = pd.Series(0, index=index)
        df.update(data_segment)
        return df

    def _zero_pad_df(self, data_segment: pd.DataFrame):
        """
        Zero pad a pandas DataFrame to the window size.

        Args:
            data_segment (pd.DataFrame): Data segment to be zero padded.

        Returns:
            pd.DataFrame: Zero padded data segment.
        """
        start_time = data_segment.index[0]
        end_time = start_time + pd.Timedelta(seconds=self.window_size_in_seconds)
        index = pd.date_range(start=start_time, end=end_time, freq="1ms")
        df = pd.DataFrame(0, index=index, columns=data_segment.columns)
        df.update(data_segment)
        return df


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
        """
        Normalize the input signal.

        Args:
            signal (pd.Series): Input signal to be normalized.

        Returns:
            Normalizer: The instance with the normalized signal.
        """
        try:
            if self.method == "zscore":
                self.normalized_signal_ = self._z_score_normalize(signal)
            elif self.method == "minmax":
                self.normalized_signal_ = self._min_max_normalize(signal)
            else:
                raise ValueError(f"Normalization method '{self.method}' is not supported.")
        except Exception as e:
            print(f"Normalization failed: {e}")
            self.normalized_signal_ = pd.Series(0, index=signal.index)
        return self

    def _z_score_normalize(self, data: pd.Series) -> pd.Series:
        """
        Apply Z-score normalization to the data.

        Args:
            data (pd.Series): Data to be normalized.

        Returns:
            pd.Series: Z-score normalized data.
        """
        mean = data.mean()
        std = data.std()
        if std == 0:
            raise ValueError("Standard deviation is zero, cannot perform Z-score normalization.")
        return (data - mean) / std

    def _min_max_normalize(self, data: pd.Series) -> pd.Series:
        """
        Apply Min-Max normalization to the data.

        Args:
            data (pd.Series): Data to be normalized.

        Returns:
            pd.Series: Min-Max normalized data.
        """
        min_val = data.min()
        max_val = data.max()
        if max_val == min_val:
            raise ValueError("Min and Max values are the same, cannot perform Min-Max normalization.")
        return (data - min_val) / (max_val - min_val)


class WaveletTransformer(Algorithm):
    _action_methods = ("transform", "transform_diff")

    # Input Parameters
    wavelet_coefficients: Parameter[Tuple[int, int]]
    wavelet_type: Parameter[str]
    sampling_rate: Parameter[float]
    normalizer: Normalizer
    window_size: Parameter[int]
    num_imfs: Parameter[int]
    normalize: Parameter[bool]

    # Results
    transformed_signal_: np.ndarray

    def __init__(
        self,
        wavelet_coefficients: Tuple[int, int] = (1, 257),
        wavelet_type: str = "morl",
        sampling_rate: float = 200,
        window_size: int = 5,
        normalizer: Normalizer = cf(Normalizer()),
        num_imfs: int = 5,
        normalize: bool = True,
    ):
        self.wavelet_coefficients = wavelet_coefficients
        self.wavelet_type = wavelet_type
        self.sampling_rate = sampling_rate
        self.normalizer = normalizer
        self.window_size = window_size
        self.num_imfs = num_imfs
        self.normalize = normalize

    @make_action_safe
    def transform_diff(
        self,
        signal: np.ndarray,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "Data",
        identity: bool = False,
    ):
        path = self._get_path(base_path, subject_id, phase, identity, diff=True)
        scales = np.geomspace(
            self.wavelet_coefficients[0],
            self.wavelet_coefficients[1],
            num=self.wavelet_coefficients[1] - self.wavelet_coefficients[0],
        )
        coefficients, _ = pywt.cwt(signal, scales, self.wavelet_type, sampling_period=1 / self.sampling_rate)
        np.save(path / f"{segment}.npy", coefficients)
        self.transformed_signal_ = []
        return self

    @make_action_safe
    def transform(
        self,
        signal: np.ndarray,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "Data",
    ):
        self._calculate_signal(signal, segment, base_path, subject_id, phase)
        self.transformed_signal_ = []
        return self

    def _normalize_and_fix_shape(self, coefficients: np.ndarray) -> np.ndarray:
        normalizer_clone = self.normalizer.clone()
        if np.iscomplexobj(coefficients):
            coefficients = np.abs(coefficients)
        coefficients_normalized = normalizer_clone.normalize(pd.Series(coefficients.flatten())).normalized_signal_
        return coefficients_normalized.to_numpy().reshape(coefficients.shape)

    def _calculate_signal(
        self,
        signal: np.ndarray,
        segment: int,
        base_path: str,
        subject_id: str,
        phase: str,
    ):
        wavelet_types = ["mexh", "shan1-1", "shan1-1", "morl"]
        for wavelet_type in wavelet_types:
            path = self._get_path(base_path, subject_id, phase, wavelet_type)
            scales = np.geomspace(
                self.wavelet_coefficients[0],
                self.wavelet_coefficients[1],
                num=self.wavelet_coefficients[1] - self.wavelet_coefficients[0],
            )
            coefficients, _ = pywt.cwt(signal, scales, wavelet_type, sampling_period=1 / self.sampling_rate)
            coefficients_normalized = self._normalize_and_fix_shape(coefficients)
            np.save(path / f"{segment}.npy", coefficients_normalized)

    def _get_path(
        self,
        base_path: str,
        subject_id: str,
        phase: str,
        wavelet_type: str = None,
        diff: bool = False,
        create_dir: bool = True,
    ) -> Path:
        path = Path(base_path) / subject_id
        if phase:
            path /= phase
        if wavelet_type:
            path /= wavelet_type
        if diff:
            path /= "diff"
        if create_dir and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_array(
        self,
        coefficients: np.ndarray,
        segment_nr: int,
        imf_nr: int,
        normalize: bool,
        path: Path,
    ):
        coefficients = self._normalize_and_fix_shape(coefficients) if normalize else coefficients
        np.save(path / f"{segment_nr}_{imf_nr}.npy", coefficients)

    def _save_image(
        self,
        coefficients: np.ndarray,
        frequencies: np.ndarray,
        segment_nr: int,
        imf_nr: int,
        path: Path,
    ):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        time = np.arange(0, len(coefficients) / self.sampling_rate, 1 / self.sampling_rate)
        ax.imshow(
            np.abs(coefficients),
            aspect="auto",
            cmap="jet",
            extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(path / f"{segment_nr}_{imf_nr}.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
