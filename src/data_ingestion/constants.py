from dataclasses import dataclass

@dataclass(frozen=True)
class Constants:
    """A class to hold constant values for the application."""
    service_api_url: str = 'https://api.tosdr.org/service/v2/'
    points_api_url: str = 'https://edit.tosdr.org/points/'
