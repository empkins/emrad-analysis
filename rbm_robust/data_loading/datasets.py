import os
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Any, Dict, Optional, Sequence, Union
import numpy as np
from biopsykit.io.biopac import BiopacDataset
from biopsykit.io.psg import PSGDataset
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sync import SyncedDataset
from empkins_io.utils._types import path_t
from emrad_toolbox.radar_preprocessing.radar import RadarPreprocessor
from scipy.signal import periodogram, find_peaks
from rbm_robust.data_loading.base.dataset import BaseDataset
from tpcp import Dataset
import neurokit2 as nk
from functools import lru_cache
import pytz
import warnings
import glob
import itertools
from datetime import datetime, timedelta
from itertools import product


class D02Dataset(Dataset):
    """
    D02Dataset is a subclass of the Dataset class that is specifically designed to handle the D02 dataset.
    It provides methods for loading ECG and radar data, synchronizing the data, and creating an index of participants.
    """

    SAMPLING_RATE_ACQ = 2000
    SAMPLING_RATE_RADAR = 1953.125
    SAMPLING_RATE_DOWNSAMPLED = 1000
    SAMPLING_RATE_DOWNSAMPLED_ML = 200
    _CHANNEL_MAPPING = {
        "ECG": "ecg",
        "SyncSignal": "Sync_Out",
    }
    _PHASE_MAPPING = {
        "lat": {"start": "lat_rating_n", "end": "lat_end"},
        "ei": {"start": "ei_01", "end": "ei_end"},
        "coping": {"start": "coping_trial_1", "end": "coping_end"},
        "training": {
            "start": "training_base-rating_start",
            "end": "training_end",
        },
    }
    EXCLUDE_SUBJECTS = (
        "005",
        "074",
        "044",
        "115",
        "Radar_DB",
        "418",
        "569",
        "535",
        "552",
        "161",
        "537",
        "350",
        "125",  # No Sync Out channel
        "572",  # No acq
        "303",  # Too large DS
        "597",  # No acq file
        "430",  # No acq file
        "512",  # Too large DS
        "158",  # No acq file
        "567",  # No acq file
        "324",  # No acq file
        "131",  # No raw folder
        "202",  # No raw folder
        "540",  # No acq file
        "325",  # Empty
        "565",  # No acq file
        "148",  # No acq file
        "545",  # No acq file
        "568",  # No acq file
        "556",  # No acq file
        "594",  # No acq file
        "502",  # No acq file
        "566",  # No acq file
        "403",
        "516",
        "132",
        "093",
        "123",
        "124",
        "574",
        "536",
        "453",
        "459",
        "318",
        "507",
        "469",
        "460",
        "306",
        "386",
        "276",
        "469",
        "518",
        "467",
        "321",
        "538",
        "438",
        "389",
        "511",
        "553",
    )

    def __init__(
        self,
        data_path: Path,
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize a D02Dataset instance.

        :param data_path: Path to the directory containing the D02 dataset.
        :param groupby_cols: Columns to group by when creating the index. Default is None.
        :param subset_index: Subset of the index to use. Default is None.
        """
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        self.data_path = data_path
        self.radar_dataset = None
        self.ecg_dataset = None
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        """
        Create an index of participants in the D02 dataset.

        :return: DataFrame containing the index.
        """
        participant_ids = [item.name for item in Path(self.data_path).iterdir() if item.is_dir()]
        participant_ids = [pid for pid in participant_ids if pid not in self.EXCLUDE_SUBJECTS]
        df = pd.DataFrame({"participant": participant_ids})
        if df.empty:
            raise ValueError(
                f"The dataset is empty. Are you sure you selected the correct folder? "
                f"Current folder is: {self.data_path}"
            )
        return df

    @property
    def sampling_rate_radar(self):
        """
        Get the sampling rate of the radar data.

        :return: Sampling rate of the radar data.
        """
        return self.SAMPLING_RATE_RADAR

    @property
    def sampling_rate_ecg(self):
        """
        Get the sampling rate of the ECG data.

        :return: Sampling rate of the ECG data.
        """
        return self.SAMPLING_RATE_ACQ

    @property
    def subjects(self) -> list[str]:
        """
        Get a list of unique subjects in the D02 dataset.

        :return: List of unique subjects.
        """
        return self.index["participant"].unique().tolist()

    @property
    def ecg(self) -> pd.DataFrame:
        """
        Load the ECG data for the first subject in the D02 dataset.

        :return: DataFrame containing the ECG data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        subject_id = self.subjects[0]
        return self._load_ecg(subject_id)

    @property
    def ecg_clean(self) -> pd.DataFrame:
        """
        Load the ECG data for the first subject in the D02 dataset.

        :return: DataFrame containing the ECG data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        ecg_signal = self._load_ecg(subject_id)
        ecg_clean = nk.ecg_clean(
            ecg_signal=ecg_signal["ecg"], sampling_rate=int(self.SAMPLING_RATE_ACQ), method="neurokit"
        )
        return pd.DataFrame(ecg_clean, columns=["ecg_clean"], index=ecg_signal.index)

    @property
    def synced_ecg(self) -> pd.DataFrame:
        """
        Load the synchronized ECG data for the first subject in the D02 dataset.

        :return: DataFrame containing the synchronized ECG data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        ecg_signal = self.synced_data[["ecg"]]
        ecg_signal["ecg"] = nk.ecg_clean(
            ecg_signal=ecg_signal["ecg"], sampling_rate=int(self.SAMPLING_RATE_ACQ), method="neurokit"
        )
        return ecg_signal[["ecg"]]

    @property
    def filtered_radar(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        radar_df = self._load_radar(subject_id)
        filtered_dict = RadarPreprocessor.butterworth_band_pass_filter(
            radar_df["I"], radar_df["Q"], filter_cutoff=(18, 80)
        )
        for key, value in filtered_dict.items():
            radar_df = pd.concat([radar_df, value], axis=1, keys=[key])
        return radar_df

    @property
    def filtered_synced_radar(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        radar_df = self.synced_data[["radar_I", "radar_Q"]]
        filtered_dict = RadarPreprocessor.butterworth_band_pass_filter(
            radar_df["radar_I"], radar_df["radar_Q"], filter_cutoff=(18, 80)
        )
        for key, value in filtered_dict.items():
            radar_df = pd.concat([radar_df, value], axis=1, keys=[key])
        return radar_df

    @lru_cache(maxsize=1)
    def _load_ecg(self, subject_id: str) -> pd.DataFrame:
        """
        Load the ECG data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the ECG data.
        """
        subject_path = self.data_path.joinpath(subject_id, "raw")
        acq_path = self._get_only_matching_file_path(subject_path, "acq")
        return BiopacDataset.from_acq_file(acq_path, channel_mapping=self._CHANNEL_MAPPING).data_as_df(
            index="local_datetime"
        )

    @property
    @lru_cache(maxsize=1)
    def clean_phase_df(self):
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        df = self._log_df(subject_id)
        df = self._clean_phase_df(df)
        return df

    @property
    def phases(self):
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        return self._clean_phase_df(self._log_df(subject_id))

    @property
    @lru_cache(maxsize=1)
    def log(self):
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        return self._log_df(subject_id)

    @property
    def synced_radar(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        return self._load_synced_radar(subject_id)

    @property
    def radar(self) -> pd.DataFrame:
        """
        Load the radar data for the first subject in the D02 dataset.

        :return: DataFrame containing the radar data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        subject_id = self.subjects[0]
        return self._load_radar(subject_id, add_sync_out=True, add_sync_in=True)

    @lru_cache(maxsize=1)
    def _load_radar(self, subject_id: str, add_sync_in: bool = False, add_sync_out: bool = False) -> pd.DataFrame:
        """
        Load the radar data for a specific subject.

        :param subject_id: ID of the subject.
        :param add_sync_in: Whether to add the Sync_In channel to the data. Default is False.
        :param add_sync_out: Whether to add the Sync_Out channel to the data. Default is False.
        :return: DataFrame containing the radar data.
        """
        subject_path = self.data_path.joinpath(subject_id, "raw")
        h5_path = self._get_only_matching_file_path(subject_path, "h5")
        dataset = EmradDataset.from_hd5_file(h5_path, sampling_rate_hz=self.SAMPLING_RATE_RADAR)
        df = dataset.data_as_df(index="local_datetime", add_sync_in=add_sync_in, add_sync_out=add_sync_out)
        df.columns = [val[1] for val in df.columns]
        return df

    @staticmethod
    def _get_only_matching_file_path(path, file_type: str) -> str:
        """
        Get the only file of a specific type in a directory.

        :param path: Path to the directory.
        :param file_type: Type of the file to get.
        :return: Name of the file.
        """
        matching_files = list(path.glob(f"*.{file_type}"))
        if len(matching_files) == 0:
            raise ValueError(f"No {file_type} files found in {path}. Are you sure you selected the correct folder?")
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple {file_type} files found in {path}. Are you sure you selected the correct folder?"
            )
        return matching_files[0]

    @property
    def synced_data(self) -> pd.DataFrame:
        """
        Load the synchronized ECG and radar data for the first subject in the D02 dataset.

        :return: DataFrame containing the synchronized data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        subject_id = self.subjects[0]
        return self._load_synced_data(subject_id)

    @lru_cache(maxsize=1)
    def _load_synced_data(self, subject_id: str) -> pd.DataFrame:
        """
        Load the synchronized ECG and radar data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the synchronized data.
        """

        # Load the ECG data
        ecg_df = self._load_ecg(subject_id)

        # Load the radar data
        radar_df = self._load_radar(subject_id, add_sync_in=True, add_sync_out=False)
        radar_df.rename(columns={"Sync_In": "Sync_Out"}, inplace=True)

        # Synchronize the data
        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset(
            "radar", data=radar_df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "ecg", data=ecg_df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_ACQ
        )
        synced_dataset.resample_datasets(fs_out=self.SAMPLING_RATE_DOWNSAMPLED, method="dynamic", wave_frequency=0.2)
        synced_dataset.align_and_cut_m_sequence(
            primary="ecg",
            reset_time_axis=True,
            cut_to_shortest=True,
            sync_params={"sync_region_samples": (0, 1000000)},
        )
        dict_shift = synced_dataset._find_shift(
            primary="ecg_aligned_", sync_params={"sync_region_samples": (-1000000, -1)}
        )
        synced_dataset.resample_sample_wise(primary="ecg_aligned_", dict_sample_shift=dict_shift)
        df_dict = synced_dataset.datasets_aligned
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        print(result_df.columns)
        result_df.columns = [
            "".join(col).replace("aligned_", "") if col[1] != "ecg" else "ecg" for col in result_df.columns.values
        ]
        return result_df

    @lru_cache(maxsize=1)
    def _load_synced_radar(self, subject_id):
        """
        Load the synchronized radar data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the synchronized radar data.
        """
        synced_data = self._load_synced_data(subject_id).copy()
        df = synced_data.filter(regex="^radar").copy()
        df.columns = [col.replace("radar_", "") for col in df.columns]
        if "Sync_Out" in df.columns:
            df.drop(columns=["Sync_Out"], inplace=True)
        if "Sync_In" in df.columns:
            df.drop(columns=["Sync_In"], inplace=True)
        return df

    @lru_cache(maxsize=1)
    def _load_synced_ecg(self, subject_id):
        """
        Load the synchronized ecg data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the synchronized ecg data.
        """
        synced_data = self._load_synced_data(subject_id).copy()
        df = synced_data.filter(regex="^ecg")
        df.drop(columns=["ecg_Sync_Out"], inplace=True)
        return df

    def _log_df(self, subject_id: str) -> pd.DataFrame:
        """
        Load the log file for a specific subject. The log contains the start and end times for the different phases, e.g. coping, ei, etc.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the log data.
        """
        subject_path = self.data_path.joinpath(subject_id, "logs")
        csv_files = glob.glob(os.path.join(subject_path, "*.csv"))
        df = pd.DataFrame()
        for file in csv_files:
            log_df = pd.read_csv(file)
            log_df["timestamp"] = log_df["timestamp"].str.replace("\n", "")
            log_df = log_df[log_df["timestamp"].str.contains("\d", regex=True)]
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
            log_df.set_index(log_df["timestamp"], drop=True, inplace=True)
            log_df.drop(["timestamp", "content"], axis=1, inplace=True)
            log_df.index.names = ["index"]
            df = pd.concat([df, log_df])
        df = df.sort_index().reset_index().drop_duplicates().set_index("index")
        return df

    def _clean_phase_df(self, df) -> Dict:
        """
        Cleans the phase data for a specific subject.

        :param df: DataFrame containing the log data.
        :return: Dictionary with phase names as keys and lists of (start, end) tuples as values.
        """
        phases = {}
        for phase, times in self._PHASE_MAPPING.items():
            start_time, end_time = times["start"], times["end"]
            phase_times = self._get_phase_times(start_time, end_time, df, phase)
            for index, (start, end) in enumerate(phase_times, start=1):
                phase_name = f"{phase}_{index}" if len(phase_times) > 1 else phase
                phases[phase_name] = {"start": start, "end": end}
        return phases

    def _get_phase_times(
        self, start_label: str, end_label: str, df: pd.DataFrame, phase: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Get the start and end times for a specific phase.

        :param start_label: Label for the start time.
        :param end_label: Label for the end time.
        :param df: DataFrame containing the log data.
        :param phase: Name of the phase.
        :return: List of tuples containing the start and end times.
        """
        start_times = df[df["label"] == start_label].index.tolist()
        start_codes = [phase["start"] for phase in self._PHASE_MAPPING.values()]
        end_times = df[df["label"] == end_label].index.tolist()
        alternative_ends = ["ei", "training", "coping", "lat"]
        if len(start_times) > len(end_times):
            for start in start_times:
                if len(start_times) == len(end_times):
                    break
                fitting_alternative_ends = [end for end in alternative_ends if end != phase]
                filtered_df = df[df.index > start]
                mask = filtered_df["label"].apply(
                    lambda x: any(str(x).startswith(value) for value in fitting_alternative_ends)
                )
                mask_start = filtered_df["label"].apply(lambda x: any(str(x) == value for value in start_codes))
                mask = mask | mask_start
                filtered_df = filtered_df[mask]
                end_to_add = (
                    df.index[df.index.get_loc(filtered_df.index[0]) - 1] if len(filtered_df) > 0 else df.index[-1]
                )
                end_times.append(end_to_add)
        elif len(start_times) < len(end_times):
            raise ValueError(f"Phase {phase} has more end times than start times.")

        if len(start_times) != len(end_times):
            raise ValueError(f"Phase {phase} has more end times than start times.")

        start_times.sort()
        end_times.sort()
        return list(zip(start_times, end_times))


class RadarCardiaStudyDataset(BaseDataset):
    exclude_ecg_seg_failed: bool
    exclude_ecg_corrupted: bool

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ecg": "ecg",
        "icg": "icg_der",
        "ppg": "ppg",
        "pcg": "pcg",
        "sync": "sync",
    }

    SUBJECTS_ECG_SEG_FAILED: Tuple[str] = ("VP_04",)

    SUBJECT_ECG_CORRUPTED: Tuple[Tuple] = (
        ("VP_17", "baseline", "normal"),
        ("VP_17", "temporalis", "normal"),
    )

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: Optional[bool] = False,
        calc_biopac_timelog_shift: Optional[bool] = True,
        trigger_data_extraction: Optional[bool] = False,
        exclude_ecg_seg_failed: Optional[bool] = True,
        exclude_ecg_corrupted: Optional[bool] = True,
    ):
        self.exclude_ecg_seg_failed = exclude_ecg_seg_failed
        self.exclude_ecg_corrupted = exclude_ecg_corrupted

        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            calc_biopac_timelog_shift=calc_biopac_timelog_shift,
            trigger_data_extraction=trigger_data_extraction,
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        if self.exclude_ecg_seg_failed:
            for subject_id in self.SUBJECTS_ECG_SEG_FAILED:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        breathing = ["normal", "hold"]
        body_parts_1 = [
            "baseline",
            "temporalis",
            "carotis",
            "brachialis",
            "radialis_prox",
            "radialis_med",
            "radialis_dist",
            "dorsalis_pedis",
            "poplitea",
        ]
        body_parts_2 = ["aorta_prox", "aorta_med", "aorta_dist"]

        index = list(product(body_parts_1, ["normal"]))
        index.extend(list(product(body_parts_2, breathing)))
        index = [(subject, *i) for subject, i in product(subject_ids, index)]

        if self.exclude_ecg_corrupted:
            for subject_id, body_part, breathing in self.SUBJECT_ECG_CORRUPTED:
                if (subject_id, body_part, breathing) in index:
                    index.remove((subject_id, body_part, breathing))

        index = pd.DataFrame(index, columns=["subject", "location", "breathing"])

        return index
