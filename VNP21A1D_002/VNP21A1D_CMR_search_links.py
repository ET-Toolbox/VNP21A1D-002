from typing import List, Union
import requests
from datetime import date
from dateutil import parser

from modland import parsehv, generate_modland_grid

from .constants import * 

TILE_SIZE = 1200
ALLOWED_EXTENSIONS = (".h5")

def VNP21A1D_CMR_search_links(
        tile: str, 
        start_date: Union[date, str], 
        end_date: Union[date, str] = None,
        tile_size: int = TILE_SIZE,
        allowed_extensions: List[str] = ALLOWED_EXTENSIONS,
        concept_ID: str = VNP21A1D_002_CONCEPT_ID,
        CMR_search_URL: str = CMR_SEARCH_URL) -> List[str]:
    """
    Search for VNP21A1D granule links from the CMR API.

    Parameters:
    tile (str): The MODIS tile identifier (e.g., 'h18v04').
    start_date (Union[date, str]): The start date for the search (ISO 8601 format if str).
    end_date (Union[date, str], optional): The end date for the search (ISO 8601 format if str). Defaults to start_date.
    tile_size (int, optional): The size of the tile in meters. Defaults to TILE_SIZE.
    allowed_extensions (List[str], optional): List of allowed file extensions. Defaults to ALLOWED_EXTENSIONS.
    concept_ID (str, optional): The concept ID for the VNP21A1D product. Defaults to VNP21A1D_002_CONCEPT_ID.
    CMR_search_URL (str, optional): The base URL for the CMR search API. Defaults to CMR_SEARCH_URL.

    Returns:
    List[str]: A list of URLs for the VNP21A1D granules.
    """
    # Parse start_date and end_date if they are strings
    if isinstance(start_date, str):
        start_date = parser.parse(start_date).date()

    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, str):
        end_date = parser.parse(end_date).date()

    # Parse the tile identifier to get horizontal and vertical indices
    h, v = parsehv(tile)

    # Get the centroid coordinates of the MODIS tile
    geometry = generate_modland_grid(
        h=h, 
        v=v, 
        tile_size=tile_size
    )

    centroid = geometry.centroid_latlon
    lon, lat = centroid.x, centroid.y

    # Construct the CMR API request URL and parameters
    granule_search_URL = f"{CMR_search_URL}granules.json"
    params = {
        "concept_id": concept_ID,
        "bounding_box": f"{lon},{lat},{lon},{lat}",  # Point search using the tile's centroid
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",  # ISO 8601 format for date/time
        "page_size": 2000  # Retrieve up to 2000 granules per page
    }

    # Send the request to the CMR API
    response = requests.get(granule_search_URL, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 Not Found)

    # Extract URLs from the JSON response
    data = response.json()
    URLs = []
    for entry in data.get('feed', {}).get('entry', []):  # Safely navigate the JSON structure
        for link in entry.get('links', []):
            URL = link.get('href')
            # Filter for URLs that start with "https" and end with specific file extensions
            if URL and URL.startswith("https") and URL.endswith(allowed_extensions):  
                URLs.append(URL)

    return URLs