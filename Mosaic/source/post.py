# Imports
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict, replace
from enum import Enum, unique, auto
import json
import logging
import math
import multiprocessing
import os
import psutil
import re
import time
import subprocess
from typing import Tuple, Dict, List, Iterable

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import fitting
from astropy.modeling.functional_models import Gaussian2D
from astropy import modeling
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.table import Table, Column, vstack, join
import astropy.units as u
from astropy.wcs import WCS
import ccdproc
import numpy as np
from photutils.detection import StarFinderBase, DAOStarFinder, IRAFStarFinder
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, EllipticalAperture, \
    SkyCircularAperture
from scipy import optimize
from scipy.special import ellipe, roots_legendre

from skimage.transform import downscale_local_mean
from PIL import Image, ImageOps

from cramer_rao import get_psf_centroid_location_precision, get_psf_circular_mask

# Constants
VERSION = 1.0
"""The module version"""

FITS_FILE_EXTENSION = '.fits'
"""The standard fits file name extension. Used in batch processing."""

PNG_FILE_EXTENSION = '.png'
"""The standard PNG file name extension. Used in batch processing."""

MOSAIC_FILE_SUFFIX = '_mosaic'
"""The suffix to be added before the extension to image quality diagnostic source table files."""


def colorise(image, black, white, mid=None, blackpoint=1, whitepoint=255, midpoint=127):
    """
    Colorize grayscale image.
    This function calculates a color wedge which maps all black pixels in
    the source image to the first color and all white pixels to the
    second color. If ``mid`` is specified, it uses three-color mapping.
    The ``black`` and ``white`` arguments should be RGB tuples or color names;
    optionally you can use three-color mapping by also specifying ``mid``.
    Mapping positions for any of the colors can be specified
    (e.g. ``blackpoint``), where these parameters are the integer
    value corresponding to where the corresponding color should be mapped.
    These parameters must have logical order, such that
    ``blackpoint <= midpoint <= whitepoint`` (if ``mid`` is specified).

    :param image: The image to colorize.
    :param black: The color to use for black input pixels.
    :param white: The color to use for white input pixels.
    :param mid: The color to use for midtone input pixels.
    :param blackpoint: an int value [0, 255] for the black mapping.
    :param whitepoint: an int value [0, 255] for the white mapping.
    :param midpoint: an int value [0, 255] for the midtone mapping.
    :return: An image.
    """

    # Initial asserts
    assert image.mode == "L"
    if mid is None:
        assert 1 <= blackpoint <= whitepoint <= 255
    else:
        assert 1 <= blackpoint <= midpoint <= whitepoint <= 255

    # Define colors from arguments
    black = ImageOps._color(black, "RGB")
    white = ImageOps._color(white, "RGB")
    if mid is not None:
        mid = ImageOps._color(mid, "RGB")

    # Empty lists for the mapping
    red = []
    green = []
    blue = []

    # Force black at LUT(0)
    red.append(0)
    green.append(0)
    blue.append(0)

    for i in range(1, blackpoint):
        red.append(black[0])
        green.append(black[1])
        blue.append(black[2])

    # Create the mapping (2-color)
    if mid is None:
        range_map = range(0, whitepoint - blackpoint)

        for i in range_map:
            red.append(black[0] + i * (white[0] - black[0]) // len(range_map))
            green.append(black[1] + i * (white[1] - black[1]) // len(range_map))
            blue.append(black[2] + i * (white[2] - black[2]) // len(range_map))

    # Create the mapping (3-color)
    else:
        range_map1 = range(0, midpoint - blackpoint)
        range_map2 = range(0, whitepoint - midpoint)

        for i in range_map1:
            red.append(black[0] + i * (mid[0] - black[0]) // len(range_map1))
            green.append(black[1] + i * (mid[1] - black[1]) // len(range_map1))
            blue.append(black[2] + i * (mid[2] - black[2]) // len(range_map1))
        for i in range_map2:
            red.append(mid[0] + i * (white[0] - mid[0]) // len(range_map2))
            green.append(mid[1] + i * (white[1] - mid[1]) // len(range_map2))
            blue.append(mid[2] + i * (white[2] - mid[2]) // len(range_map2))

    # Create the high-end values
    for i in range(0, 256 - whitepoint):
        red.append(white[0])
        green.append(white[1])
        blue.append(white[2])

    # Return converted image
    image = image.convert("RGB")
    return ImageOps._lut(image, red + green + blue)




@dataclass
class PostProcessingConfig:
    """Post processing configuration"""
    max_threads: int
    reduction_factor: int
    low_color: tuple(int)
    mid_color: tuple(int)
    high_color: tuple(int)
    file_name: str = None

    def to_json(self):
        """JSON dump. Quadrant enum => name. QuadrantData => Dict.
            :return a JSON dump of the input object.
        """
        return json.dumps(asdict(self), indent=4)

    def to_json_file(self, file_name: str):
        """
        Dump to json file
        :param file_name: The output JSON file
        """
        json_str = replace(self, file_name=None).to_json()
        with open(file_name, 'w') as output_file:
            output_file.write(json_str)

    @classmethod
    def from_json(cls, json_document: str, file_name: str = None):
        """JSON deserialization.
            :param json_document: document
            :param file_name: The input JSON file name, if applicable
            :return: the decoded PostProcessingConfig
        """
        decoded = cls(**json.loads(json_document))
        decoded.file_name = file_name
        return decoded

    @classmethod
    def from_json_file(cls, config_file_name: str):
        """
        JSON file deserialization
        :param config_file_name: The input JSON file path
        :return: the decoded PostProcessingConfig
        """
        if isinstance(config_file_name, str):
            file_name = os.path.basename(config_file_name)
        else:
            file_name = None
        with open(config_file_name, 'r') as config_file:
            post_processing_config = cls.from_json(config_file.read(), file_name)
        return post_processing_config


class HDUList:
    """
    Handles a fits HDUList, 
    Class extension did not fully work: write_to had to be reimplemented
    Input data are deep copied.
    """
    hdu_list: fits.HDUList
    """The handled HDUList"""
    file_name: str
    """The I/O file_name, when applicable"""

    def __init__(self, hdu_list: fits.HDUList, file_name: str = None):
        """
        :param hdu_list: the input hdu_list. It will be deep copied.
        :param file_name: the I/O file name, when applicable
        """
        self.hdu_list = fits.HDUList([hdu.copy() for hdu in hdu_list])
        self.file_name = file_name

    @classmethod
    def from_fits(cls, fits_file):
        """
        Instantiate from an existing fits file. Note all HDUs must be read and copied.
        This can represent an overhead with respect to a simpler HDUList class.
        However, it may be convenient for frequently accessed frames (e.g. master dark or flat)
        whose quick accessibility in RAM must be guaranteed.
        :param fits_file: input fits file
        :return: the instantiated VisHDUList object
        """
        if isinstance(fits_file, str):
            file_name = os.path.basename(fits_file)
        else:
            file_name = None
        with fits.open(fits_file) as hdu_list:
            return cls(hdu_list, file_name)


def thumb_hdu_list(hdu_list: HDUList, post_processing_config: PostProcessingConfig) -> list[Image.Image]:
    """
    Generate a thumbnail
    :param hdu_list: the input hdu_list. It will not be modified.
    """

    imgs = []

    for idx in range(1, len(hdu_list.hdu_list)):
        logging.info(f'Processing extension {idx}')
        data = downscale_local_mean(hdu_list.hdu_list[idx].data, 
                                    (post_processing_config.reduction_factor, 
                                    post_processing_config.reduction_factor))
        img = Image.fromarray(data).convert("L")
        img = colorise(img, 
                    post_processing_config.low_color, 
                    post_processing_config.mid_color, 
                    post_processing_config.high_color)
        imgs.append(img.transpose(Image.FLIP_TOP_BOTTOM))

    return imgs


@unique
class FileProcessingStep(Enum):
    """ Defines the file processing steps and the associated file suffixes, when appropriate."""
    # TODO Use or remove values?
    THUMB = auto()


@dataclass()
class PostProcessor:
    """
    File-driven Post processor.
    It is based on a pair of I/O folders and a bunch of auxiliary variables.
    Output might not be written to disk if needed.
    """
    input_folder: str
    """The input folder with Level 1 processed frames (or their averages)"""
    output_folder: str
    """The output folder where all other files are stored"""
    prefix: str = None
    """The prefix common for all fits files associated with a given Level 1 frame"""
    hdu_list: HDUList = None
    """
    Processing buffer for different processing modules. A level 1 frame, an average of them or a CCD processed output
    """
    post_processing_config: PostProcessingConfig = None
    write_output_files: bool = True
    """If False, do not write files to disk"""

    def _hdu_list_from_fits(self, input_folder: str = None, suffix: str = '') -> int:
        """
        Load VisHduList. Default place: input folder. Default type: Level 1 frame
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        if input_folder is None:
            input_folder = self.input_folder
        input_file = os.path.join(input_folder, f'{self.prefix}{suffix}{FITS_FILE_EXTENSION}')
        self.vis_hdu_list = HDUList.from_fits(input_file)
        return time.time_ns() - now

    def load_files(self, file_processing_step: FileProcessingStep):
        """
        Load default input files needed for a given file processing steps.
        :param file_processing_step: File processing step
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        if file_processing_step is FileProcessingStep.THUMB:
            self._hdu_list_from_fits(self.output_folder, MOSAIC_FILE_SUFFIX)
        else:
            raise TypeError(f'Unsupported file processing step: {file_processing_step}')
        return time.time_ns() - now

    def _thumb(self):
        output_images = thumb_hdu_list(self.vis_hdu_list, self.post_processing_config)
        self.vis_stars = None
        self.vis_cutouts = None

        if self.write_output_files:
            output_file = os.path.join(
                self.output_folder, f'{self.prefix}{MOSAIC_FILE_SUFFIX}{PNG_FILE_EXTENSION}')
            output_images[0].save(output_file, overwrite=True)

    def process(self, file_processing_step: FileProcessingStep) -> int:
        """
        Apply one processing step. Internal buffers will be overwritten. Output files might be generated.
        :param file_processing_step: File processing step
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        if file_processing_step is FileProcessingStep.THUMB:
            self._thumb()
        else:
            raise TypeError(f'Unsupported file processing step: {file_processing_step}')
        return time.time_ns() - now


def post_process(input_folder, output_folder, prefixes: Iterable[str],
                      file_processing_steps: Iterable[str], log_file, post_processing_config_file: str,
                      write_output_files: bool = True):
    """
    Process all VIS level 1 frames in a given folder. Useful for batch processing
    :param input_folder: The input folder. All "*.fits" files within will be processed
    :param output_folder: The output folder. Will be created, if needed.
                            Pre-existing files with the same name will be overwritten
    :param prefixes: The file prefixes to process
    :param file_processing_steps: The processing steps to apply. All of them if none
    :param log_file: The output log file. Entries will be appended
    :param post_processing_config_file: The VIS processing configuration file
    :param astrometric_indices_config_file: The astrometric indices config file
    :param master_dark: The master dark fits stack
    :param master_flat: The master flat PRNU fits stack
    :param write_output_files: If False, do all calculations in memory and store nothing
    """

    # Initialise logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Make unique and sort file processing steps
    file_processing_steps = sorted({FileProcessingStep[step] for step in file_processing_steps},
                                   key=(lambda x: x.value))
    logging.info(f'File processing steps to apply: {file_processing_steps}')

    # Instantiate file processor. Instantiate VisCcdProcessor only if CCD processing is requested
    logging.info('Instantiating Post processor')
    post_processor = PostProcessor(
        input_folder, output_folder, write_output_files=write_output_files)
    logging.info('Reading VIS processing configuration file')
    post_processor.post_processing_config = PostProcessingConfig.from_json_file(post_processing_config_file)

    # Create output folder, if needed
    os.makedirs(output_folder, exist_ok=True)

    # Loop on prefixes and processing steps
    n_prefixes = len(prefixes)
    for i, prefix in enumerate(prefixes):
        post_processor.prefix = prefix
        logging.info(f'Processing file name prefix {i + 1} out of {n_prefixes}: {post_processor.prefix}')
        for j, file_processing_step in enumerate(file_processing_steps):
            if j == 0:
                logging.info(f'Loading files for initial processing step: {file_processing_step.name}')
                elapsed_ns = post_processor.load_files(file_processing_step)
                logging.info(f'Elapsed ms: {elapsed_ns // 1000000}')
            logging.info(f'Processing.step: {file_processing_step.name}')
            elapsed_ns = post_processor.process(file_processing_step)
            logging.info(f'Elapsed ms: {elapsed_ns // 1000000}')

    logging.info('End. All input file prefixes processed')


