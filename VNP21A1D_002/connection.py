from typing import Union, List
from datetime import date, datetime

import earthaccess

import rasters as rt
from rasters import SpatialGeometry, RasterGeometry, Raster

from VIIRS_tiled_granules import VIIRSTiledProductConnection

from .constants import *
from .search_granules import search_granules
from .retrieve_granule import retrieve_granule
from .VNP21A1D_granule import VNP21A1DGranule

class VNP21A1D(VIIRSTiledProductConnection):
    GranuleClass = VNP21A1DGranule

    def __init__(
            self,
            download_directory: str = DOWNLOAD_DIRECTORY):
        super().__init__(
            concept_ID=VNP21A1D_002_CONCEPT_ID,
            download_directory=download_directory
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
    
    def variable(
            self,
            variable: str,
            date_UTC: Union[date, str],
            geometry: RasterGeometry = None,
            tile: str = None,
            tile_size: int = 1200,
            filename: str = None,
            resampling: str = None) -> Raster:
        if geometry is None and tile_size is None:
            raise ValueError("neither geometry nor tile size given")

        if geometry is None:
            geometry = generate_modland_grid(tile=tile, tile_size=tile_size)

        remote_granules = self.search(
            date_UTC=date_UTC,
            geometry=geometry,
            tile=tile,
            tile_size=tile_size
        )

        granules = [
            retrieve_granule(remote_granule)
            for remote_granule 
            in remote_granules
        ]

        images = [
            granule.variable(variable)
            for granule 
            in granules
        ]

        mosaic = rt.mosaic(
            images=images,
            geometry=geometry,
            resampling=resampling
        )

        return mosaic
    
    def ST_K(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="ST_K",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )
    
    def ST_C(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="ST_C",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )
    
    def QC(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="QC",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )
    
    def cloud(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="cloud",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )
    
    def Emis_14(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="Emis_14",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )
    
    def Emis_15(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="Emis_15",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )
    
    def Emis_16(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="Emis_16",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )
    
    def view_angle(
            self,
            date_UTC: Union[date, str],
            geometry: RasterGeometry,
            filename: str = None,
            resampling: str = None) -> Raster:
        return self.variable(
            variable="View_Angle",
            date_UTC=date_UTC,
            geometry=geometry,
            filename=filename,
            resampling=resampling
        )