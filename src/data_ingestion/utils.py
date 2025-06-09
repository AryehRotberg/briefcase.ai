import re
import time
from typing import DefaultDict, Dict, List, Optional

import requests
import spacy
import spacy_fastlang
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.data_ingestion.constants import Constants

# Initialize SpaCy language detector model
print('Loading SpaCy language detector model...')
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('language_detector')


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


def fetch_annotated_source(point_id: int) -> Optional[str]:
    """
    Fetches the annotated source for a given point ID.

    :param point_id: Point ID from the TOS;DR.org website
    :return: Cleaned annotated source text, or None if not found
    """
    try:
        response = requests.get(f"{Constants.points_api_url}{point_id}")
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        html = soup.find("blockquote") or soup.find("div", {"class": "col-sm-10 col-sm-offset-1 p30 bgw"})
        if not html:
            print(f"Please review information about point {point_id}.")
            return None

        if html.footer:
            html.footer.decompose()

        return clean_source_text(html.get_text(strip=True))
    
    except requests.RequestException as e:
        print(f"Error fetching point {point_id}: {e}")
        return None


def prepare_data_chunk(response: Dict, data_chunk: DefaultDict[str, List[str]]) -> DefaultDict[str, List[str]]:
    """
    Creates a data chunk from an HTTP response containing annotated sources and their categories.

    :param response: HTTP response in JSON format
    :param data_chunk: Existing data chunk to append to
    :return: Updated data chunk
    """
    for point in response.get("parameters", {}).get("points", []):
        if is_point_suitable(point):
            case = point["case"]["title"].strip()
            point_id = point["id"]
            annotated_source = fetch_annotated_source(point_id)

            if annotated_source:
                data_chunk[case].append(annotated_source)

    return data_chunk


def time_progress_bar(seconds: int, description: str) -> None:
    """
    Displays a tqdm progress bar for a specified duration.

    :param seconds: Duration in seconds
    :param description: Description for the progress bar
    """
    for _ in tqdm(range(seconds), desc=description, leave=False):
        time.sleep(1)


def fetch_ids_of_reviewed_services() -> List[int]:
    """
    Fetches a list of service IDs for reviewed services.

    :return: List of service IDs
    """
    service_ids = []

    try:
        response = requests.get(Constants.service_api_url).json()
        first_page = response['parameters']['_page']['start']
        last_page = response['parameters']['_page']['end']

        for page in tqdm(range(first_page, last_page + 1), desc="Fetching reviewed services"):
            time.sleep(5)

            try:
                response = requests.get(f"{Constants.service_api_url}?page={page}").json()
                services = response.get('parameters', {}).get('services', [])

                for service in services:
                    service_ids.append(service['id'])

            except KeyError:
                print("\nAPI is exhausted. Sleeping for 30 seconds.")
                time_progress_bar(30, "API cooldown")

    except requests.RequestException as e:
        print(f"Error fetching reviewed services: {e}")

    return service_ids
