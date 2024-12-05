from typing import Union, List
from datetime import datetime, date
from dateutil import parser
import logging

import earthaccess

import colored_logging as cl
from rasters import Point, Polygon, RasterGeometry

__author__ = "Gregory H. Halverson, Evan Davis"

logger = logging.getLogger(__name__)

class CMRServerUnreachable(Exception):
    pass

def earliest_datetime(date_in: Union[date, str]) -> datetime:
    if isinstance(date_in, str):
        datetime_in = parser.parse(date_in)
    else:
        datetime_in = date_in

    date_string = datetime_in.strftime("%Y-%m-%d")
    return parser.parse(f"{date_string}T00:00:00Z")


def latest_datetime(date_in: Union[date, str]) -> datetime:
    if isinstance(date_in, str):
        datetime_in = parser.parse(date_in)
    else:
        datetime_in = date_in

    date_string = datetime_in.strftime("%Y-%m-%d")
    return parser.parse(f"{date_string}T23:59:59Z")

def VIIRS_CMR_query(
        concept_ID: str,
        start_date: Union[date, str],
        end_date: Union[date, str],
        target_geometry: Union[Point, Polygon, RasterGeometry] = None,
        tile: str = None) -> List[earthaccess.search.DataGranule]:
    """function to search for VIIRS at tile in date range"""
    query = earthaccess.granule_query() \
        .concept_id(concept_ID) \
        .temporal(earliest_datetime(start_date), latest_datetime(end_date))

    if isinstance(target_geometry, Point):
        query = query.point(target_geometry.x, target_geometry.y)
    if isinstance(target_geometry, Polygon):
        ring = target_geometry.exterior
        if not ring.is_ccw:
            ring = ring.reverse()
        coordinates = ring.coords
        query = query.polygon(coordinates)
    if isinstance(target_geometry, RasterGeometry):
        ring = target_geometry.corner_polygon_latlon.exterior
        if not ring.is_ccw:
            ring = ring.reverse()
        coordinates = ring.coords
        query = query.polygon(coordinates)
    if tile is not None:
        query = query.readable_granule_name(f"*.{tile}.*")

    granules: List[earthaccess.search.DataGranule]
    try:
        granules = query.get()
    except Exception as e:
        raise CMRServerUnreachable(e)
    granules = sorted(granules, key=lambda granule: granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])

    logger.info("Found the following granules for VIIRS 2 using the CMR search:")
    for granule in granules:
        logger.info("  " + cl.file(granule["meta"]["native-id"]))
    logger.info(f"Number of VIIRS 2 granules found using CMR search: {len(granules)}")

    return granules
