import os
from collections import defaultdict
from json.decoder import JSONDecodeError
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.data_ingestion.constants import Constants
from src.data_ingestion.utils import (fetch_ids_of_reviewed_services,
                                      is_document_in_specified_language,
                                      prepare_data_chunk, time_progress_bar)


class DataIngestion:
    def __init__(self, service_ids_path: str):
        """
        Initializes the DataIngestion class.

        :param str service_ids_path: Path to the file containing service IDs.
        """
        self.service_ids_path = service_ids_path
        self.service_ids = self._load_service_ids()
        self.data = defaultdict(list)

    def _load_service_ids(self) -> List[int]:
        """
        Loads service IDs from the specified file or fetches them if the file does not exist.

        :return List[int]: List of service IDs.
        """
        if not os.path.exists(self.service_ids_path):
            service_ids = fetch_ids_of_reviewed_services()
            self._save_service_ids(service_ids)
        else:
            with open(self.service_ids_path, 'r', encoding='utf-8') as file:
                service_ids = [int(id_.strip()) for id_ in file.readlines()]
        return service_ids

    def _save_service_ids(self, service_ids: List[int]) -> None:
        """
        Saves service IDs to the specified file.

        :param List[int] service_ids: List of service IDs to save.
        """
        with open(self.service_ids_path, 'w', encoding='utf-8') as file:
            for id_ in sorted(service_ids):
                file.write(f'{id_}\n')

    def fetch_data(self, start_index: Optional[int] = 0) -> None:
        """
        Fetches data for model training.

        :param Optional[int] start_index: Starting index for fetching data. Default is 0.
        """
        for service_id in tqdm(self.service_ids[start_index:], desc="Fetching data"):
            try:
                response = requests.get(f'{Constants.service_api_url}?id={service_id}').json()
                self.data = prepare_data_chunk(response, self.data)
            except KeyError:
                time_progress_bar(60, "API is exhausted. Sleeping for 60 seconds.")
            except JSONDecodeError:
                time_progress_bar(300, "Error parsing JSON. API might be exhausted. Sleeping for 5 minutes.")

    def filter_data_by_language(self, language: str) -> List[Dict[str, str]]:
        """
        Filters data that is in a specific language.

        :param str language: Language code (e.g., 'en' for English).
        :return List[Dict[str, str]]: Filtered data.
        """
        filtered_data = []
        for category, sources in self.data.items():
            unique_sources = set(sources)
            for source in unique_sources:
                if is_document_in_specified_language(source, language):
                    filtered_data.append({"category": category, "source": source})
        return filtered_data

    def create_pandas_dataframe(self, data: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Creates a Pandas DataFrame from a list of data.

        :param List[Dict[str, str]] data: Data to convert to a DataFrame.
        :return pd.DataFrame: Pandas DataFrame.
        """
        return pd.DataFrame(data)

    def save_pandas_dataframe(self, dataframe: pd.DataFrame, file_output_path: str) -> None:
        """
        Saves a Pandas DataFrame to a CSV file.

        :param pd.DataFrame dataframe: DataFrame to save.
        :param str file_output_path: Path to save the CSV file.
        """
        os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
        dataframe.to_csv(file_output_path, index=False)
