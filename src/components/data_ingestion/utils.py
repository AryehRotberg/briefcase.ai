import asyncio
import re
from typing import DefaultDict, Dict, List, Optional

import aiohttp
import httpx
import spacy
import spacy_fastlang
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

from .config.api_settings import POINTS_API_URL

print("Loading SpaCy language detector model...")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")


def is_document_in_specified_language(text: str, language: str) -> bool:
    """
    Checks whether the text is in the specified language.

    :param text: Plain text or document
    :param language: Language code (e.g., 'en' for English)
    :return: True if the text is in the specified language, False otherwise
    """
    return nlp(text)._.language == language


def is_point_suitable(point: Dict) -> bool:
    """
    Checks whether the point is suitable for model training by verifying its source and status.

    :param point: A dictionary containing information about a point
    :return: True if the point is suitable, False otherwise
    """
    return point.get("source") and point.get("status") == "approved"


def clean_source_text(text: str) -> str:
    """
    Cleans source text by removing HTML tags and unnecessary spaces.

    :param text: Raw text with potential HTML content
    :return: Cleaned text
    """
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("\n", " ").replace("</", "")
    return re.sub(r"\s{2,}", " ", text).strip()


async def fetch_annotated_source(session: aiohttp.ClientSession, point_id: int) -> Optional[str]:
    """
    Fetches the annotated source for a given point ID.

    :param point_id: Point ID from the TOS;DR.org website
    :return: Cleaned annotated source text, or None if not found
    """
    try:
        async with session.get(f"{POINTS_API_URL}{point_id}") as response:
            response.raise_for_status()
            content = await response.text()
            soup = BeautifulSoup(content, "html.parser")

            html = soup.find("blockquote") or soup.find("div", {"class": "col-sm-10 col-sm-offset-1 p30 bgw"})
            if not html:
                print(f"Please review information about point {point_id}.")
                return None

            if html.footer:
                html.footer.decompose()

            return clean_source_text(html.get_text(strip=True))
    
    except httpx.HTTPError as e:
        print(f"Error fetching point {point_id}: {e}")
        return None


async def prepare_data_chunk(response: Dict) -> List[Dict[str, str]]:
    """
    Async version that fetches all sources concurrently.

    :param response: The API response containing points
    :return: A list of dictionaries with point IDs and their annotated sources
    """
    suitable_points = [point for point in response.get("points", []) if is_point_suitable(point)]

    if not suitable_points:
        return []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        tasks = [fetch_annotated_source(session, point["id"]) for point in suitable_points]
        annotated_sources = await asyncio.gather(*tasks, return_exceptions=True)

    data_chunk = []
    for point, source in zip(suitable_points, annotated_sources):
        if isinstance(source, str) and source:
            data_chunk.append({
                "point_id": point["id"],
                "category": point["case"]["title"].strip(),
                "source": source,
                "service_id": response["id"],
                "service_name": response["name"]
            })
    
    return data_chunk


async def fetch_service(rate_limiter: AsyncLimiter, client: httpx.AsyncClient, url: str) -> dict:
    """
    Fetch a service from the API.

    :param rate_limiter: The rate limiter to use for the request.
    :param client: The HTTP client to use for the request.
    :param url: The URL of the service to fetch.
    :return: The JSON response from the API.
    """
    async with rate_limiter:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def fetch_services(rate_limiter: AsyncLimiter, client: httpx.AsyncClient, urls: list[str]):
    """
    Fetch multiple services from the API.

    :param rate_limiter: The rate limiter to use for the requests.
    :param client: The HTTP client to use for the requests.
    :param urls: The list of URLs of the services to fetch.
    :return: A list of JSON responses from the API.
    """
    tasks = [fetch_service(rate_limiter, client, url) for url in urls]
    results = await tqdm_asyncio.gather(*tasks, desc="Fetching services", unit="request")
    return results
