from pathlib import Path
from typing import Optional, Union
from biopsykit.io.biopac import BiopacDataset
import pandas as pd
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sync import SyncedDataset
from tpcp import Dataset
import neurokit2 as nk


class D02Dataset(Dataset):
    """
    D02Dataset is a subclass of the Dataset class that is specifically designed to handle the D02 dataset.
    It provides methods for loading ECG and radar data, synchronizing the data, and creating an index of participants.
    """

    SAMPLING_RATE_ACQ = 2000
    SAMPLING_RATE_RADAR = 1953.125
    _CHANNEL_MAPPING = {
        "ECG": "ecg",
        "SyncSignal": "sync",
    }

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
        subject_id = self.subjects[0]
        return self._load_ecg(subject_id)["ecg"]

    @property
    def ecg_clean(self) -> pd.DataFrame:
        """
        Load the ECG data for the first subject in the D02 dataset.

        :return: DataFrame containing the ECG data.
        """
        subject_id = self.subjects[0]
        ecg_signal = self._load_ecg(subject_id)
        ecg_clean = nk.ecg_clean(
            ecg_signal=ecg_signal["ecg"], sampling_rate=int(self.SAMPLING_RATE_ACQ), method="neurokit"
        )
        return pd.DataFrame(ecg_clean, columns=["ecg_clean"], index=ecg_signal.index)

    def _load_ecg(self, subject_id: str) -> pd.DataFrame:
        """
        Load the ECG data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the ECG data.
        """
        subject_path = self.data_path.joinpath(subject_id, "raw")
        acq_path = self._get_only_matching_file_path(subject_path, "acq")
        # Load the ECG data  , channel_mapping=self._CHANNEL_MAPPING
        return BiopacDataset.from_acq_file(acq_path).data_as_df(index="local_datetime")

    @property
    def radar(self) -> pd.DataFrame:
        """
        Load the radar data for the first subject in the D02 dataset.

        :return: DataFrame containing the radar data.
        """
        subject_id = self.subjects[0]
        return self._load_radar(subject_id)

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
        dataset = EmradDataset.from_hd5_file(h5_path)
        df = dataset.data_as_df(index="local_datetime", add_sync_in=add_sync_in, add_sync_out=add_sync_out)
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
        subject_id = self.subjects[0]
        return self._load_synced_data(subject_id)

    def _load_synced_data(self, subject_id: str) -> pd.DataFrame:
        """
        Load the synchronized ECG and radar data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the synchronized data.
        """

        # Load the ECG data
        ecg_df = self._load_ecg(subject_id)

        # Load the radar data
        radar_df = self._load_radar(subject_id, add_sync_in=True, add_sync_out=True)

        # Synchronize the data
        synced_dataset = SyncedDataset(sync_type="m-sequence")
        radars = [x for x in radar_df.columns if "rad" in x]
        for radar in radars:
            synced_dataset.add_dataset(
                radar, radar_df[radar], sync_channel_name="Sync_In", sampling_rate=int(self.SAMPLING_RATE_RADAR)
            )
        synced_dataset.add_dataset("acq", ecg_df, sync_channel_name="sync", sampling_rate=self.SAMPLING_RATE_ACQ)
        synced_dataset.resample_datasets(fs_out=self.SAMPLING_RATE_ACQ, method="static")
        synced_dataset.align_and_cut_m_sequence(
            primary="acq", reset_time_axis=False, cut_to_shortest=True, sync_params={"sync_region_samples": (0, 1000)}
        )
        df_dict = synced_dataset.datasets_aligned
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        multiindex = result_df.columns
        keep_columns = [x for x in multiindex if "sync" not in x]
        result_df = result_df.loc[:, keep_columns]
        return result_df
