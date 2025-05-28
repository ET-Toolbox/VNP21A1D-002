from typing import Union, List
from datetime import date, datetime

import earthaccess

import rasters as rt
from rasters import SpatialGeometry, RasterGeometry, Raster

from .constants import *
from .search_granules import search_granules
from .retrieve_granule import retrieve_granule
from .VNP21A1D_granule import VNP21A1DGranule

class VNP21A1D:
    def __init__(
            self,
            download_directory: str = DOWNLOAD_DIRECTORY):
        self.download_directory = download_directory
    
    def search(
            self,
            date_UTC: Union[date, datetime, str] = None,
            start_date: Union[date, datetime, str] = None,
            end_date: Union[date, datetime, str] = None,
            target_geometry: SpatialGeometry = None,
            tile: str = None) -> List[earthaccess.search.DataGranule]:
        return search_granules(
            date_UTC=date_UTC,
            start_date=start_date,
            end_date=end_date,
            target_geometry=target_geometry,
            tile=tile
        )
    
    def granule(
            self,
            date_UTC: Union[date, str] = None,
            tile: str = None,
            download_directory: str = DOWNLOAD_DIRECTORY) -> VNP21A1DGranule:
        return retrieve_granule(
            date_UTC=date_UTC,
            tile=tile,
            download_directory=download_directory
        )
    
    def ST_K(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        remote_granules = self.search(
            date_UTC=date_UTC,
            target_geometry=geometry
        )

        granules = [
            retrieve_granule(remote_granule)
            for remote_granule 
            in remote_granules
        ]

        images = [
            granule.ST_K
            for granule 
            in granules
        ]

        # TODO rasters mosaic needs to accept resampling method
        mosaic = rt.mosaic(
            images=images,
            geometry=geometry,
            resampling=resampling
        )

        return mosaic

    def ST_C(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        ST_K = self.ST_K(
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )

        ST_C = ST_K - 273.15

        return ST_C
