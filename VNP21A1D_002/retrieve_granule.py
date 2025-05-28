from typing import Union
from datetime import date
import earthaccess

import VIIRS_tiled_granules

from .constants import VNP21A1D_002_CONCEPT_ID
from .VNP21A1D_granule import VNP21A1DGranule

class GranuleNotFoundError(Exception):
    pass

def retrieve_granule(
        remote_granule: earthaccess.search.DataGranule = None,
        date_UTC: Union[date, str] = None,
        tile: str = None,
        download_directory: str = ".",
        parent_directory: str = None,
        concept_ID: str = VNP21A1D_002_CONCEPT_ID) -> VNP21A1DGranule:
    if remote_granule is None:
        remote_granules = VIIRS_tiled_granules.search_granules(
            concept_ID=concept_ID,
            date_UTC=date_UTC,
            tile=tile,
            tile_size=1200
        )

        if len(remote_granules) == 0:
            raise GranuleNotFoundError(f"no VNP21A1D.002 granules found at tile {tile} on date {date_UTC}")

        remote_granule = remote_granules[0]

    granule = VIIRS_tiled_granules.retrieve_granule(
        remote_granule, 
        download_directory=download_directory, 
        parent_directory=parent_directory
    )

    granule = VNP21A1DGranule(granule)

    return granule
