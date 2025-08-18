import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

from database.postgresql import engine

from .config.category_maps import CATEGORY_MAPS
from .config.classification_levels import (HIGH_CRITICALITY, LOW_CRITICALITY,
                                           MEDIUM_CRITICALITY)
from .utils import is_document_in_specified_language


class Preprocessor:
    def __init__(self):
        self.dataframe = pd.read_sql("SELECT * FROM policies", engine)
        self.dataframe = self.dataframe.drop_duplicates(subset=["source"])
        self.categories = HIGH_CRITICALITY + MEDIUM_CRITICALITY + LOW_CRITICALITY


    def filter_by_language(self, language_code) -> None:
        """
        Filters the dataframe to only include documents in the specified language.

        :param language_code: The language code to filter by.
        """
        self.dataframe["is_specified_language"] = \
        self.dataframe["source"].progress_apply(lambda x : is_document_in_specified_language(x, language_code))

        self.dataframe = self.dataframe[self.dataframe["is_specified_language"] == True].copy()
    
    
    def standardize_categories(self) -> None:
        """
        Standardizes the categories in the dataframe.
        """
        self.dataframe["category"] = self.dataframe["category"].replace(CATEGORY_MAPS)
        self.dataframe = self.dataframe[self.dataframe["category"].isin(self.categories)].copy()


    def store_changes(self) -> None:
        """
        Stores the changes made to the dataframe back to the database.
        """
        self.dataframe.to_sql("policies_processed", engine, if_exists="replace", index=False, method="multi")
