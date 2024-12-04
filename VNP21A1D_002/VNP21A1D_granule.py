from typing import Union, List
import logging
from datetime import datetime
from glob import glob
from os import makedirs
from os.path import exists, join, abspath, expanduser, basename, splitext
import json
import h5py
import numpy as np
import pandas as pd
from dateutil import parser
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, Polygon
from skimage.transform import resize

import colored_logging
import rasters
import rasters as rt
from modland import parsehv, generate_modland_grid

from rasters import Raster, RasterGrid, RasterGeometry

from .VIIRS_filename_parsing import *

# Define colormaps for NDVI and Albedo
NDVI_COLORMAP = LinearSegmentedColormap.from_list(
    name="NDVI",
    colors=[
        "#0000ff",
        "#000000",
        "#745d1a",
        "#e1dea2",
        "#45ff01",
        "#325e32"
    ]
)

ALBEDO_COLORMAP = "gray"

DEFAULT_WORKING_DIRECTORY = "."

logger = logging.getLogger(__name__)

class VIIRSGranule:
    """
    Class representing a VIIRS Granule.
    """
    CLOUD_DATASET_NAME = "HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SurfReflect_QF1_1"

    def __init__(self, filename: str):
        """
        Initialize the VIIRSGranule object.

        :param filename: Path to the VIIRS granule file.
        """
        self._filename = filename
        self._cloud_mask = None

    def __repr__(self):
        """
        Return a string representation of the VIIRSGranule object.
        """
        display_dict = {
            "filename": self.filename
        }

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    @property
    def filename(self):
        """
        Return the filename of the granule.
        """
        return self._filename
    
    @property
    def filename_absolute(self):
        """
        Return the absolute path of the filename.
        """
        return abspath(expanduser(self.filename))

    @property
    def filename_base(self):
        """
        Return the base name of the filename.
        """
        return basename(self.filename)

    @property
    def filename_stem(self) -> str:
        """
        Return the stem of the filename.
        """
        return splitext(self.filename_base)[0]

    @property
    def tile(self):
        """
        Return the tile information from the filename.
        """
        return parse_VIIRS_tile(self.filename)

    @property
    def hv(self):
        """
        Return the horizontal and vertical tile indices.
        """
        return parsehv(self.tile)

    @property
    def h(self):
        """
        Return the horizontal tile index.
        """
        return self.hv[0]

    @property
    def v(self):
        """
        Return the vertical tile index.
        """
        return self.hv[1]

    @property
    def date_UTC(self):
        """
        Return the date in UTC from the filename.
        """
        return datetime.strptime(self.filename_base.split(".")[1][1:], "%Y%j")

    @property
    def grids(self) -> List[str]:
        """
        Return the list of grids in the HDF5 file.
        """
        with h5py.File(self.filename_absolute, "r") as file:
            return list(file["HDFEOS/GRIDS/"].keys())

    def variables(self, grid: str) -> List[str]:
        """
        Return the list of variables in a specific grid.

        :param grid: The grid name.
        """
        with h5py.File(self.filename_absolute, "r") as file:
            return list(file[f"HDFEOS/GRIDS/{grid}/Data Fields/"].keys())

class VNP21A1DGranule(VIIRSGranule):
    """
    Class representing a VNP21A1D Granule, inheriting from VIIRSGranule.
    """
    CLOUD_DATASET_NAME = "HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SurfReflect_QF1_1"

    def get_QC(self, geometry: RasterGeometry = None, resampling: str = "nearest") -> Raster:
        """
        Get the Quality Control (QC) data as a Raster object.

        :param geometry: The target geometry for resampling.
        :param resampling: The resampling method.
        """
        with h5py.File(self.filename_absolute, "r") as f:
            dataset_name = "HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/QC"
            QC = np.array(f[dataset_name])
            h, v = self.hv
            grid = generate_modland_grid(h, v, QC.shape[0])

            logger.info("opening VIIRS file: " + colored_logging.file(self.filename))

            logger.info(
                f"loading {colored_logging.val(dataset_name)} " +
                "at " + colored_logging.val(f"{grid.cell_size:0.2f} m") + " resolution"
            )

            QC = Raster(QC, geometry=grid)

        if geometry is not None:
            QC = QC.to_geometry(geometry, resampling=resampling)

        return QC

    QC = property(get_QC)

    def get_cloud_mask(self, target_shape: tuple = None) -> Raster:
        """
        Get the cloud mask as a Raster object.

        :param target_shape: The target shape for resizing.
        """
        h, v = self.hv

        if self._cloud_mask is None:
            QC = self.QC
            cloud_mask = ((QC >> 4) & 3) > 0
            self._cloud_mask = cloud_mask
        else:
            cloud_mask = self._cloud_mask

        if target_shape is not None:
            cloud_mask = resize(cloud_mask, target_shape, order=0).astype(bool)
            shape = target_shape
        else:
            shape = cloud_mask.shape

        geometry = generate_modland_grid(h, v, shape[0])
        cloud_mask = Raster(cloud_mask, geometry=geometry)

        return cloud_mask

    cloud_mask = property(get_cloud_mask)

    def dataset(
            self,
            filename: str,
            dataset_name: str,
            scale_factor: float = 1,
            offset: float = 0,
            fill: float = None,
            lower_range: float = None,
            upper_range: float = None,
            cloud_mask: Raster = None,
            apply_cloud_mask: bool = True,
            geometry: RasterGeometry = None,
            resampling: str = None) -> Raster:
        """
        Get a dataset as a Raster object.

        :param filename: The filename of the dataset.
        :param dataset_name: The name of the dataset.
        :param scale_factor: The scale factor to apply.
        :param offset: The offset to apply.
        :param fill: The fill value to replace with NaN.
        :param lower_range: The lower range for valid data.
        :param upper_range: The upper range for valid data.
        :param cloud_mask: The cloud mask to apply.
        :param apply_cloud_mask: Whether to apply the cloud mask.
        :param geometry: The target geometry for resampling.
        :param resampling: The resampling method.
        """
        filename = abspath(expanduser(filename))

        with h5py.File(filename, "r") as f:
            DN = np.array(f[dataset_name])
            h, v = self.hv
            grid = generate_modland_grid(h, v, DN.shape[0])

            logger.info("opening VIIRS file: " + colored_logging.file(self.filename))

            logger.info(
                f"loading {colored_logging.val(dataset_name)} " +
                "at " + colored_logging.val(f"{grid.cell_size:0.2f} m") + " resolution"
            )

            DN = Raster(DN, geometry=grid)

        data = DN

        if fill is not None:
            data = np.where(data == fill, np.nan, data)

        if lower_range is not None:
            data = np.where(data < lower_range, np.nan, data)

        if upper_range is not None:
            data = np.where(data > upper_range, np.nan, data)

        data = data * scale_factor + offset

        if apply_cloud_mask:
            if cloud_mask is None:
                cloud_mask = self.get_cloud_mask(target_shape=data.shape)

            data = rt.where(cloud_mask, np.nan, data)

        if geometry is not None:
            data = data.to_geometry(geometry, resampling=resampling)

        return data

    @property
    def geometry(self) -> RasterGrid:
        """
        Return the geometry of the granule.
        """
        return generate_modland_grid(*self.hv, 1200)

    def get_Emis_14(self, geometry: RasterGeometry = None) -> Raster:
        """
        Get the Emissivity Band 14 data as a Raster object.

        :param geometry: The target geometry for resampling.
        """
        image = self.dataset(
            self.filename_absolute,
            f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/Emis_14",
            scale_factor=0.002,
            offset=0.49,
            cloud_mask=None,
            apply_cloud_mask=False
        )

        if np.all(np.isnan(image)):
            raise ValueError("blank emissivity band 14 image")
        
        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    Emis_14 = property(get_Emis_14)

    def get_Emis_15(self, geometry: RasterGeometry = None) -> Raster:
        """
        Get the Emissivity Band 15 data as a Raster object.

        :param geometry: The target geometry for resampling.
        """
        image = self.dataset(
            self.filename_absolute,
            f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/Emis_15",
            scale_factor=0.002,
            offset=0.49,
            cloud_mask=None,
            apply_cloud_mask=False
        )

        if np.all(np.isnan(image)):
            raise ValueError("blank emissivity band 15 image")

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    Emis_15 = property(get_Emis_15)

    def get_Emis_16(self, geometry: RasterGeometry = None) -> Raster:
        """
        Get the Emissivity Band 16 data as a Raster object.

        :param geometry: The target geometry for resampling.
        """
        image = self.dataset(
            self.filename_absolute,
            f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/Emis_16",
            scale_factor=0.002,
            offset=0.49,
            cloud_mask=None,
            apply_cloud_mask=False
        )

        if np.all(np.isnan(image)):
            raise ValueError("blank emissivity band 16 image")

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    Emis_16 = property(get_Emis_16)

    def get_LST_1KM(self, geometry: RasterGeometry = None) -> Raster:
        """
        Get the Land Surface Temperature (LST) 1KM data as a Raster object.

        :param geometry: The target geometry for resampling.
        """
        image = self.dataset(
            self.filename_absolute,
            f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/LST_1KM",
            scale_factor=0.02,
            offset=0.0,
            fill=0,
            lower_range=7500,
            upper_range=65535,
            cloud_mask=None,
            apply_cloud_mask=True
        )

        if np.all(np.isnan(image)):
            raise ValueError("blank LST 1km image")

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    LST_1KM = property(get_LST_1KM)

    ST_K = LST_1KM

    @property
    def ST_C(self):
        """
        Return the Land Surface Temperature in Celsius.
        """
        return self.ST_K - 273.15

    def get_View_Angle(self, geometry: RasterGeometry = None) -> Raster:
        """
        Get the View Angle data as a Raster object.

        :param geometry: The target geometry for resampling.
        """
        image = self.dataset(
            self.filename_absolute,
            f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/View_Angle",
            scale_factor=1.0,
            offset=-65.0,
            cloud_mask=None,
            apply_cloud_mask=False
        )

        if np.all(np.isnan(image)):
            raise ValueError("blank view angle image")

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    View_Angle = property(get_View_Angle)