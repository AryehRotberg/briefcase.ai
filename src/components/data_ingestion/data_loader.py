import asyncio

import httpx
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from database.models.policy import Policy
from database.mongodb import database
from database.postgresql import get_database_session
from .config.api_settings import SERVICE_API_URL
from .utils import fetch_services, prepare_data_chunk

load_dotenv()


class Loader:
    def __init__(self):
        """
        Initializes the Loader class.
        """
        self.rate_limiter = AsyncLimiter(max_rate=1, time_period=5)
        self.pages = self._load_pages_list()
        self.service_pages_collection = database.get_collection("tosdr_service_pages")
        self.services_collection = database.get_collection("tosdr_services")


    async def store_page_services(self) -> None:
        """
        Stores the fetched page services into the specified MongoDB collection.
        """
        urls = [f"{SERVICE_API_URL}?page={page}" for page in self.pages]

        async with httpx.AsyncClient() as client:
            page_services = await fetch_services(self.rate_limiter, client, urls)

        docs = []

        for page_number, data in zip(self.pages, page_services):
            if data is not None:
                docs.append({"_id": page_number, **data})

        self.service_pages_collection.insert_many(docs, ordered=False)


    async def store_service_data(self) -> None:
        """
        Stores the fetched service data into the specified MongoDB collection.
        """
        service_ids = self._load_reviewed_service_ids()
        urls = [f"{SERVICE_API_URL}?id={service_id}" for service_id in service_ids]

        async with httpx.AsyncClient() as client:
            service_data = await fetch_services(self.rate_limiter, client, urls)

        docs = []

        for service_id, data in zip(service_ids, service_data):
            if data is not None:
                docs.append({"_id": service_id, **data})

        self.services_collection.insert_many(docs, ordered=False)


    async def store_processed_data(self, rows: list[str]) -> None:
        """
        Stores the processed data into the specified PostgreSQL database.

        :param rows: A list of row data to process and store.
        """
        semaphore = asyncio.Semaphore(10)

        async def process_with_semaphore(row):
            async with semaphore:
                return await prepare_data_chunk(row)

        tasks = [process_with_semaphore(row) for row in rows]
        results = await tqdm.gather(*tasks, desc="Processing rows")

        data = []

        for result in results:
            data.extend(result)

        with get_database_session() as session:
            session.bulk_insert_mappings(Policy, data)
            session.commit()


    def _load_reviewed_service_ids(self) -> list[int]:
        """
        Loads the IDs of services that have been comprehensively reviewed.
        """
        rows = self.service_pages_collection.find().to_list()

        if not rows:
            raise ValueError("No service pages found.")
        
        service_ids = []
        for row in rows:
            for service in row["services"]:
                if service["is_comprehensively_reviewed"]:
                    service_ids.append(service["id"])

        return sorted(service_ids)


    def _load_pages_list(self) -> list[int]:
        """
        Fetches the list of pages from the service API.

        :return: A list of page numbers.
        """
        response = httpx.get(SERVICE_API_URL)
        response.raise_for_status()
        response = response.json()

        first_page = response["page"]["start"]
        last_page = response["page"]["end"]

        pages = list(range(first_page, last_page + 1))
        return pages
