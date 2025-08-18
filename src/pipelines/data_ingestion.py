import asyncio
from src.components.data_ingestion.data_loader import Loader
from src.components.data_ingestion.preprocessor import Preprocessor


async def load():
    data_loader = Loader()
    await data_loader.store_page_services()
    await data_loader.store_service_data()

    rows = data_loader.services_collection.find().to_list()
    await data_loader.store_processed_data(rows)


def process():
    preprocessor = Preprocessor()
    preprocessor.filter_by_language("en")
    preprocessor.standardize_categories()
    preprocessor.store_changes()


if __name__ == "__main__":
    # asyncio.run(load())
    process()
