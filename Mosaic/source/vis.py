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
VERSION = 1.2
"""The module version"""

NUM_CORES = psutil.cpu_count(False)
"""The number of physical cores in the machine executing the code. If not available, number of logical cores."""
if NUM_CORES is None:
    NUM_CORES = psutil.cpu_count()

FITS_EXTENSION_NAME_KEY = 'EXTNAME' #TODO remove. Use "name" HDU attribute when possible
"""The header keyword holding the name for a given fits extension."""

QUADRANT_NAXIS_1 = 2128
"""Number of samples per quadrant in X serial   direction."""

QUADRANT_NAXIS_2 = 2086
"""Number of samples per quadrant in Y parallel direction."""

MOSAIC_NAXIS1 = 25186 
"""Number of pixels in mosaic X direction"""

MOSAIC_NAXIS2 = 27966    
"""Number of pixels in mosaic Y direction"""

MOSAIC_NAXIS2_MSTP = 28192 
"""Number of pixels in mosaic Y direction for MSTP mode"""

PRESCAN_X = 51
"""Number of serial   prescan  pixels."""

OVERSCAN_X = 29
"""Number of serial   overscan samples."""

OVERSCAN_Y = 20
"""Number of parallel overscan samples."""

QUADRANT_NAXIS_1_TRIMMED: int = QUADRANT_NAXIS_1 - PRESCAN_X - OVERSCAN_X
"""Number of image pixels per quadrant in X serial direction."""

QUADRANT_NAXIS_2_TRIMMED: int = QUADRANT_NAXIS_2 - OVERSCAN_Y
"""Number of image pixels per quadrant in Y parallel direction."""

CHARGE_INJECTION_LINES = 4
"""Number of charge injection lines, split between the two top and bottom quadrants."""

CCD_SATURATION_LEVEL = 190000
"""Saturation level (e-). Same value used for all CCDs and pixels."""

BIAS_OPTIMUM_PRESCAN_X_RANGE = (10, 50)
""""Pre-scan column numpy index range in a level-1 frame to be used for optimum bias subtraction.
    Zero index is first pre-scan sample, independent of quadrant location within CCD and FPA."""

BIAS_OPTIMUM_POSTSCAN_DISCARDED_ROWS = 1
""""Post-scan rows to discard in optimum bias subtraction. Begins with first post-scan row."""

MASK_HDU_NAME_SUFFIX = '_MASK'
"""The name suffix to bad/cosmic pixel masks associated to a give data fits HDU."""

CUTOUT_SIZE = 41
"""Star image cutout size (pixels)."""

ASTROMETRY_QUADRANT_COLUMN = 'quadrant'
"""Astrometry.net column storing the original quadrant for each star."""

ASTROMETRY_X_CENTROID_CCD_COLUMN = 'x_centroid_ccd'
"""Astrometry.net column storing the x_centroid for each star in the CCD reference frame."""

ASTROMETRY_Y_CENTROID_CCD_COLUMN = 'y_centroid_ccd'
"""Astrometry.net column storing the y_centroid for each star in the CCD reference frame."""

ASTROMETRY_FLUX_COLUMN = 'flux'
"""Astrometry.net column storing the flux for each star in the CCD reference frame."""

FITS_FILE_REGEX = re.compile('.[fF][iI][tT][sS]\\b')
"""The fits file matching regular expression"""

FITS_FILE_EXTENSION = '.fits'
"""The standard fits file name extension. Used in batch processing."""

ASTROMETRY_NET_XMATCH_FITS_FILE_EXTENSION = '.corr'
"""The standard fits file name extension. Used in batch processing."""

ASTROMETRY_NET_WCS_FITS_FILE_EXTENSION = '.wcs'
"""The standard fits file name extension. Used in batch processing."""

VIS_STARS_CCD_FILE_NAME_FORMAT = '{}_ccd_{}{}'
"""VIS star data per CCD file name format"""

ASTROMETRY_NET_XMATCH_TABLE_UNITS: Dict[str, u.Unit] = {
    'field_x': u.pix,
    'field_y': u.pix,
    'field_ra': u.deg,
    'field_dec': u.deg,
    'index_x': u.pix,
    'index_y': u.pix,
    'index_ra': u.deg,
    'index_dec': u.deg,
    'index_id': None,
    'field_id': None,
    'match_weight': None,
}
"""The astrometry.net cross-match table column units."""

PLATE_SCALE_MAJOR_COLUMN = 'plate_scale_major'
"""Image quality metrics table column with major plate scale (arcsec)."""

PLATE_SCALE_MINOR_COLUMN = 'plate_scale_minor'
"""Image quality metrics table column with minor plate scale (arcsec)."""

PLATE_SCALE_ANGLE_COLUMN = 'plate_scale_angle'
"""Image quality metrics table column with major plate scale position angle (arcsec)."""

BACKGROUND_ANNULUS_INNER_RADIUS = 12
"""Cutout inner radius for background estimation (pix)."""

BACKGROUND_ANNULUS_OUTER_RADIUS = 15
"""Cutout outer radius for background estimation (pix)."""

APERTURE_PHOTOMETRY_PIXEL_RADIUS = 2.0
"""Cutout aperture photometry radius (pix)."""

APERTURE_PHOTOMETRY_PIXEL_SKY = 0.2 * u.arcsec
"""Cutout aperture photometry radius (arcsec)."""

APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN = 'aperture_sum'
"""The aperture photometry output column containing the total flux."""

CRAMER_RAO_CUTOUT_SIZE = 14
"""Star image cutout size for Cramer-Rao diagnostic (pixels)"""

FWHM_DETERMINATION_MAX_QUADRANTS = 15
"""Maximum number of quadrants for FWHM re-determination."""

X_CENTROID_COLUMN = 'xcentroid'
"""Column storing the x-axis centroid data in tables obtained from source detection."""

Y_CENTROID_COLUMN = 'ycentroid'
"""Column storing the y-axis centroid data in tables obtained from source detection."""

X_CENTER_DESCRIPTOR = 'X_CENTER'
"""Descriptor in image cutout with the x-axis center for its object."""

Y_CENTER_DESCRIPTOR = 'Y_CENTER'
"""Descriptor in image cutout with the x-axis center for its object."""

X_MIN_CUTOUT_DESCRIPTOR = 'X_MIN'
"""Descriptor in image cutout with the x-axis lower bound in the parent image."""

Y_MIN_CUTOUT_DESCRIPTOR = 'Y_MIN'
"""Descriptor in image cutout with the y-axis lower bound in the parent image."""

X_MAX_CUTOUT_DESCRIPTOR = 'X_MAX'
"""Descriptor in image cutout with the x-axis upper bound in the parent image."""

Y_MAX_CUTOUT_DESCRIPTOR = 'Y_MAX'
"""Descriptor in image cutout with the y-axis upper bound in the parent image."""

EXPOSURE_TIME_DESCRIPTOR = 'EXPTIME'
"""Descriptor in fits primary header with the exposure time (s)."""

READ_OUT_NOISE_DESCRIPTOR = 'RON'
"""Descriptor in image cutout with the read-out noise (e-)."""

MAX_ADU = np.iinfo(np.uint16).max
"""The ADC 16 bit saturation value."""

HOT_PIXEL_MASK_VALUE = 1
"""The mask value corresponding to a hot pixel in the master dark image."""

DARK_PIXEL_MASK_VALUE = 2
"""The mask value corresponding to a dark pixel in the master flat-field image."""

CALIBRATION_NAN_PIXEL_MASK_VALUE = 4
"""The mask value corresponding to an invalid (NaN) value in either master dark or master flat-field images."""

ZERO_ADU_MASK_VALUE = 8
"""The mask value corresponding to a zero in the raw image."""

SATURATION_ADU_MASK_VALUE = 16
"""The mask value coresponding to a ADC saturation in the raw image."""

OUTSIDE_RANGE_MASK_VALUE = 32
"""The mask value for pixels with too much signal (e-)."""

OUTSIDE_RANGE_MIN_VALUE = -1000
"""The minimum valid electron count for any pixel. """

OUTSIDE_RANGE_MAX_VALUE = 250000
"""The maximum valid electron count for any pixel. """

COSMIC_RAY_CORRECTED_MASK_VALUE = 64
"""The mask value coresponding to pixels corrected by LA Cosmic, not necessarily hit by a cosmic ray."""

WCS_HEADER_KEYWORDS = ('CRPIX1', 'CRPIX1', 'CRVAL1', 'CRVAL2')
"""Some mandatory WCS header keyworkds. Useful to recognize whether a WCS can be reconstructed from a given header."""

WCS_X_REFERENCE_PIXEL_KEYWORD = 'CRPIX1'
"""The WCS x-axis reference pixel (1-based)."""

WCS_Y_REFERENCE_PIXEL_KEYWORD = 'CRPIX2'
"""The WCS y-axis reference pixel (1-based)."""

RA_COLUMN = 'ra'
"""Right ascension column"""

DEC_COLUMN = 'dec'
"""Declination column"""

R2_PIX_COLUMN = 'r2_pix'
"""Image quality metrics: r2 (pix)."""

ELLIPTICITY_PIX_COLUMN = 'e_pix'
"""Image quality metrics: ellipticity (pix)."""

X_MOMENT_PIX_COLUMN = 'x_moment_pix'
"""Image quality metrics: x-axis quadrupole moment (pix)."""

Y_MOMENT_PIX_COLUMN = 'y_moment_pix'
"""Image quality metrics: y-axis quadrupole moment (pix)."""

X_GAUSSIAN_PIX_COLUMN = 'x_gauss_pix'
"""Image quality metrics: Gaussian fit x-axis peak position (pix)."""

Y_GAUSSIAN_PIX_COLUMN = 'y_gauss_pix'
"""Image quality metrics: Gaussian fit y-axis peak position (pix)."""

FWHM_PIX_COLUMN = 'fwhm_pix'
"""Image quality metrics: Gaussian core full-width at half maximum (pix)."""

FWHM_PIXELATED_PIX_COLUMN = 'fwhm_pixelated_pix'
"""Image quality metrics: Gaussian core full-width at half maximum, fluxes integrated on pixel (pix)."""

EER100_PIX_FLUX_COLUMN = 'eer100_pix_flux'
"""Image quality metrics: Total flux (e-) for a given circular aperture (pix)."""

X_CRAMER_RAO_COLUMN = 'x_cramer_rao'
"""Image quality metrics: x-axis maximum astrometric precision (pix). Cramer-Rao lower bound estimated on samples."""

Y_CRAMER_RAO_COLUMN = 'y_cramer_rao'
"""Image quality metrics: y-axis maximum astrometric precision (pix). Cramer-Rao lower bound estimated on samples."""

X_CRAMER_RAO_NO_RON_COLUMN = 'x_cramer_rao_no_ron'
"""
Image quality metrics: x-axis maximum astrometric precision (pix). Cramer-Rao lower bound estimated on samples.
CCD read-out noise set to zero. Divergence between RON and no-RON metrics determine maximum usable magnitude.
"""

Y_CRAMER_RAO_NO_RON_COLUMN = 'y_cramer_rao_no_ron'
"""
Image quality metrics: y-axis maximum astrometric precision (pix). Cramer-Rao lower bound estimated on samples.
CCD read-out noise set to zero. Divergence between RON and no-RON metrics determine maximum usable magnitude.
"""

EER50_PIX_COLUMN = 'eer50_pix'
"""Image quality metrics: 50% encircled energy radius (compared to EER100) (pix)."""

EER80_PIX_COLUMN = 'eer80_pix'
"""Image quality metrics: 80% encircled energy radius (compared to EER100) (pix)."""

EER_PIX_XTOL = 0.001
"""Absolute tolerance for encircled energy radii calculations (pix)."""

AP_PHOTO_PIX_COLUMN = 'ap_photo_pix'
"""Image quality metrics: Aperture photometry flux (e-) for a set of circular apertures (pix)."""

R2_SKY_COLUMN = 'r2_sky'
"""Image quality metrics: r2 (arcsec)."""

ELLIPTICITY_SKY_COLUMN = 'e_sky'
"""Image quality metrics: ellipticity (sky)."""

FWHM_SKY_COLUMN = 'fwhm_sky'
"""Image quality metrics: Gaussian core full-width at half maximum (arcsec)."""

FWHM_PIXELATED_SKY_COLUMN = 'fwhm_pixelated_sky'
"""Image quality metrics: Gaussian core full-width at half maximum, fluxes integrated on pixel (arcsec)."""

EER50_SKY_COLUMN = 'eer50_sky'
"""Image quality metrics: 50% encircled energy radius (compared to EER100) (arcsec)."""

EER80_SKY_COLUMN = 'eer80_sky'
"""Image quality metrics: 80% encircled energy radius (compared to EER100) (arcsec)."""

EER_SKY_XTOL = 0.0001
"""Absolute tolerance for encircled energy radii calculations (arcsec)."""

AP_PHOTO_SKY_COLUMN = 'ap_photo_sky'
"""Image quality metrics: Aperture photometry flux (e-) for a set of circular apertures (sky)."""

GAUSSIAN_QUADRATURE_ORDER = 4
"""The Gaussian quadrature order for pixel integrations."""

_legendre_x, _legendre_w = roots_legendre(GAUSSIAN_QUADRATURE_ORDER)
_subpix_1d = 0.5 * _legendre_x
_weight_1d = 0.5 * _legendre_w

GAUSSIAN_QUADRATURE_SUBPIXEL_OFFSET_X = _subpix_1d[:, np.newaxis] * np.ones((1, GAUSSIAN_QUADRATURE_ORDER))
"""The 2D Gaussian quadrature x-axis offsets to apply for each subpixel evaluation."""

GAUSSIAN_QUADRATURE_SUBPIXEL_OFFSET_Y = _subpix_1d[np.newaxis, :] * np.ones((GAUSSIAN_QUADRATURE_ORDER, 1))
"""The 2D Gaussian quadrature y-axis offsets to apply for each subpixel evaluation."""

GAUSSIAN_QUADRATURE_SUBPIXEL_WEIGHT = _weight_1d[:, np.newaxis] * _weight_1d[np.newaxis, :]
"""The 2D Gaussian quadrature weight to apply for each subpixel evaluation."""

VIS_PSF_CORE_GAUSSIAN_SIGMA_GUESS_PIX = 1. / gaussian_sigma_to_fwhm
"""The intial sigma value guess for 2D Gaussian fit to the PSF core (pixels)."""

EER100_PIX_RADIUS = 5
"""The assumed angular distance containing 100% of the source flux (pixels)."""

EER100_SKY_RADIUS = 1.3
"""The assumed angular distance containing 100% of the source flux (arcsec)."""

QUADRUPOLE_MOMENT_SIGMA_PIX = 7.5
"""The Gaussian weight function sigma value for quadrupole moment estimation (pixels)."""

QUADRUPOLE_MOMENT_SIGMA_SKY = 0.75
"""The Gaussian weight function sigma value for quadrupole moment estimation (arcsec)."""

QUADRUPOLE_MOMENT_SIGMA_THRESHOLD = 1.0
"""The threshold in radial distance above which pixels are not considered in the moment calculation."""

R2_ELLIPCITICY_ITERATIONS = 4
"""The number of iterations on the R2 and ellipticity calculation needed for a good centroid determination."""

COSMIC_IMAGE_QUADRANT_LOCATION_DESCRIPTOR = 'QUADLOC'
"""The descriptor with the QuadrantLocation information on each cosmic rays image used to simulate VIS data."""

MIN_AVERAGE_FILE_SUFFIX = 'min'
"""The suffix to be added before the extension to minimum averaged files."""

CCD_PROCESSED_FILE_SUFFIX = '_processed'
"""The suffix to be added before the extension to CCD processed files."""

COSMICS_CLEANED_FILE_SUFFIX = '_cosmics'
"""The suffix to be added before the extension to cosmic rays cleaned files."""

SOURCES_FILE_SUFFIX = '_sources'
"""The suffix to be added before the extension to detected source table files."""

CUTOUT_FILE_SUFFIX = '_cutouts'
"""The suffix to be added before the extension to source image cutout files."""

ASTROMETRY_FILE_SUFFIX = '_astrometry'
"""The suffix to be added before the extension to astrometry reduced source table files."""

IMAGE_QUALITY_FILE_SUFFIX = '_image_quality'
"""The suffix to be added before the extension to image quality diagnostic source table files."""

MOSAIC_FILE_SUFFIX = '_mosaic'
"""The suffix to be added before the extension to image quality diagnostic source table files."""


@unique
class VisMode(str, Enum):
    NOMINAL_SHORT = 'NOMINAL/SHORT'
    BIAS = 'BIAS'
    BIAS_LIMITED = 'BIAS with Limited Scan'
    CHARGE_INJECTION = 'CHARGE INJECTION'
    DARK = 'DARK'
    FLAT_FIELD = 'FLAT-FIELD'
    LINEARITY = 'LINEARITY'
    NOMINAL_SHORT_LIMITED = 'NOMINAL/SHORT with Limited Scan'
    PARALLEL_TRAP_PUMPING = 'VERTICAL TRAP PUMPING'
    MULTI_SERIAL_PARALLEL_TRAP_PUMPING = 'MULTI SERIAL TRAP PUMPING'

def VisModeFromHdu(hdu) -> VisMode:
    return VisMode(hdu.header['SEQID'])


@dataclass(frozen=True)
class QuadrantLocationData:
    """For a given quadrant, which FPA axes been inverted with respect to read-out order.
     For a given quadrant and CCD, the pixel offsets with respect to the bottom left quadrant"""
    invert_x: bool
    invert_y: bool
    x_offset: int
    y_offset: int


class QuadrantLocation(Enum):
    """ Bottom/Top: low/high Y_fpa. Left/Right: low/high X_fpa. See VIS Data ICD Fig. 7-3 """
    BOTTOM_LEFT  = QuadrantLocationData(False, False, 0, 0) # Quadrant E in ROEs 1-6, G in ROEs 7-12
    BOTTOM_RIGHT = QuadrantLocationData(True,  False, QUADRANT_NAXIS_1_TRIMMED, 0)
                    # Quadrant F in ROEs 1-6, H in ROEs 7-12
    TOP_RIGHT    = QuadrantLocationData(True,  True,
                                        QUADRANT_NAXIS_1_TRIMMED, QUADRANT_NAXIS_2_TRIMMED + CHARGE_INJECTION_LINES)
                    # Quadrant G in ROEs 1-6, E in ROEs 7-12
    TOP_LEFT     = QuadrantLocationData(False, True, 0, QUADRANT_NAXIS_2_TRIMMED + CHARGE_INJECTION_LINES)
                    # Quadrant H in ROEs 1-6, F in ROEs 7-12

    def get_numpy_trim_range(self) -> Tuple[int, int, int, int]:
        """
        :return: the numpy trim range: y_min, y_max, x_min, x_max (y/x: first/second index)
        """

        y_min: int
        y_max: int
        x_min: int
        x_max: int

        # x-axis range
        if not self.value.invert_x:
            x_min = PRESCAN_X
            x_max = QUADRANT_NAXIS_1 - OVERSCAN_X
        else:
            x_min = OVERSCAN_X
            x_max = QUADRANT_NAXIS_1 - PRESCAN_X

        # y-axis range
        if not self.value.invert_y:
            y_min = 0
            y_max = QUADRANT_NAXIS_2 - OVERSCAN_Y
        else:
            y_min = OVERSCAN_Y
            y_max = QUADRANT_NAXIS_2

        return y_min, y_max, x_min, x_max

    def get_numpy_bias_optimum_prescan_range(self):
        """
        :return: the x-axis numpy range to estimate the optimum per-column bias value
        """
        if not self.value.invert_x:
            x_min = BIAS_OPTIMUM_PRESCAN_X_RANGE[0]
            x_max = BIAS_OPTIMUM_PRESCAN_X_RANGE[1]
        else:
            x_min = QUADRANT_NAXIS_1 - BIAS_OPTIMUM_PRESCAN_X_RANGE[1]
            x_max = QUADRANT_NAXIS_1 - BIAS_OPTIMUM_PRESCAN_X_RANGE[0]

        return x_min, x_max

    def get_numpy_bias_optimum_postcan_range(self):
        """
        :return: the numpy range to estimate the optimum post-scan global bias value
        """
        # x-axis range
        if not self.value.invert_x:
            x_min = QUADRANT_NAXIS_1 - OVERSCAN_X
            x_max = QUADRANT_NAXIS_1
        else:
            x_min = 0
            x_max = OVERSCAN_X

        # y-axis range
        if not self.value.invert_y:
            y_min = QUADRANT_NAXIS_2 - OVERSCAN_Y + BIAS_OPTIMUM_POSTSCAN_DISCARDED_ROWS
            y_max = QUADRANT_NAXIS_2
        else:
            y_min = 0
            y_max = OVERSCAN_Y - BIAS_OPTIMUM_POSTSCAN_DISCARDED_ROWS

        return y_min, y_max, x_min, x_max


@unique
class Quadrant(str, Enum):
    """ Keeps fixed data for all quadrants: fits EXTNAME """
    Q_1_1_E = '1-1.E'
    Q_1_1_F = '1-1.F'
    Q_1_1_G = '1-1.G'
    Q_1_1_H = '1-1.H'
    Q_1_2_E = '1-2.E'
    Q_1_2_F = '1-2.F'
    Q_1_2_G = '1-2.G'
    Q_1_2_H = '1-2.H'
    Q_1_3_E = '1-3.E'
    Q_1_3_F = '1-3.F'
    Q_1_3_G = '1-3.G'
    Q_1_3_H = '1-3.H'
    Q_1_4_E = '1-4.E'
    Q_1_4_F = '1-4.F'
    Q_1_4_G = '1-4.G'
    Q_1_4_H = '1-4.H'
    Q_1_5_E = '1-5.E'
    Q_1_5_F = '1-5.F'
    Q_1_5_G = '1-5.G'
    Q_1_5_H = '1-5.H'
    Q_1_6_E = '1-6.E'
    Q_1_6_F = '1-6.F'
    Q_1_6_G = '1-6.G'
    Q_1_6_H = '1-6.H'

    Q_2_1_E = '2-1.E'
    Q_2_1_F = '2-1.F'
    Q_2_1_G = '2-1.G'
    Q_2_1_H = '2-1.H'
    Q_2_2_E = '2-2.E'
    Q_2_2_F = '2-2.F'
    Q_2_2_G = '2-2.G'
    Q_2_2_H = '2-2.H'
    Q_2_3_E = '2-3.E'
    Q_2_3_F = '2-3.F'
    Q_2_3_G = '2-3.G'
    Q_2_3_H = '2-3.H'
    Q_2_4_E = '2-4.E'
    Q_2_4_F = '2-4.F'
    Q_2_4_G = '2-4.G'
    Q_2_4_H = '2-4.H'
    Q_2_5_E = '2-5.E'
    Q_2_5_F = '2-5.F'
    Q_2_5_G = '2-5.G'
    Q_2_5_H = '2-5.H'
    Q_2_6_E = '2-6.E'
    Q_2_6_F = '2-6.F'
    Q_2_6_G = '2-6.G'
    Q_2_6_H = '2-6.H'

    Q_3_1_E = '3-1.E'
    Q_3_1_F = '3-1.F'
    Q_3_1_G = '3-1.G'
    Q_3_1_H = '3-1.H'
    Q_3_2_E = '3-2.E'
    Q_3_2_F = '3-2.F'
    Q_3_2_G = '3-2.G'
    Q_3_2_H = '3-2.H'
    Q_3_3_E = '3-3.E'
    Q_3_3_F = '3-3.F'
    Q_3_3_G = '3-3.G'
    Q_3_3_H = '3-3.H'
    Q_3_4_E = '3-4.E'
    Q_3_4_F = '3-4.F'
    Q_3_4_G = '3-4.G'
    Q_3_4_H = '3-4.H'
    Q_3_5_E = '3-5.E'
    Q_3_5_F = '3-5.F'
    Q_3_5_G = '3-5.G'
    Q_3_5_H = '3-5.H'
    Q_3_6_E = '3-6.E'
    Q_3_6_F = '3-6.F'
    Q_3_6_G = '3-6.G'
    Q_3_6_H = '3-6.H'

    Q_4_1_E = '4-1.E'
    Q_4_1_F = '4-1.F'
    Q_4_1_G = '4-1.G'
    Q_4_1_H = '4-1.H'
    Q_4_2_E = '4-2.E'
    Q_4_2_F = '4-2.F'
    Q_4_2_G = '4-2.G'
    Q_4_2_H = '4-2.H'
    Q_4_3_E = '4-3.E'
    Q_4_3_F = '4-3.F'
    Q_4_3_G = '4-3.G'
    Q_4_3_H = '4-3.H'
    Q_4_4_E = '4-4.E'
    Q_4_4_F = '4-4.F'
    Q_4_4_G = '4-4.G'
    Q_4_4_H = '4-4.H'
    Q_4_5_E = '4-5.E'
    Q_4_5_F = '4-5.F'
    Q_4_5_G = '4-5.G'
    Q_4_5_H = '4-5.H'
    Q_4_6_E = '4-6.E'
    Q_4_6_F = '4-6.F'
    Q_4_6_G = '4-6.G'
    Q_4_6_H = '4-6.H'

    Q_5_1_E = '5-1.E'
    Q_5_1_F = '5-1.F'
    Q_5_1_G = '5-1.G'
    Q_5_1_H = '5-1.H'
    Q_5_2_E = '5-2.E'
    Q_5_2_F = '5-2.F'
    Q_5_2_G = '5-2.G'
    Q_5_2_H = '5-2.H'
    Q_5_3_E = '5-3.E'
    Q_5_3_F = '5-3.F'
    Q_5_3_G = '5-3.G'
    Q_5_3_H = '5-3.H'
    Q_5_4_E = '5-4.E'
    Q_5_4_F = '5-4.F'
    Q_5_4_G = '5-4.G'
    Q_5_4_H = '5-4.H'
    Q_5_5_E = '5-5.E'
    Q_5_5_F = '5-5.F'
    Q_5_5_G = '5-5.G'
    Q_5_5_H = '5-5.H'
    Q_5_6_E = '5-6.E'
    Q_5_6_F = '5-6.F'
    Q_5_6_G = '5-6.G'
    Q_5_6_H = '5-6.H'

    Q_6_1_E = '6-1.E'
    Q_6_1_F = '6-1.F'
    Q_6_1_G = '6-1.G'
    Q_6_1_H = '6-1.H'
    Q_6_2_E = '6-2.E'
    Q_6_2_F = '6-2.F'
    Q_6_2_G = '6-2.G'
    Q_6_2_H = '6-2.H'
    Q_6_3_E = '6-3.E'
    Q_6_3_F = '6-3.F'
    Q_6_3_G = '6-3.G'
    Q_6_3_H = '6-3.H'
    Q_6_4_E = '6-4.E'
    Q_6_4_F = '6-4.F'
    Q_6_4_G = '6-4.G'
    Q_6_4_H = '6-4.H'
    Q_6_5_E = '6-5.E'
    Q_6_5_F = '6-5.F'
    Q_6_5_G = '6-5.G'
    Q_6_5_H = '6-5.H'
    Q_6_6_E = '6-6.E'
    Q_6_6_F = '6-6.F'
    Q_6_6_G = '6-6.G'
    Q_6_6_H = '6-6.H'

    def mask_hdu_name(self) -> str:
        """HDU name for corresponding mask"""
        return self.value + MASK_HDU_NAME_SUFFIX


MASK_HDU_NAME_TO_QUADRANT = {quadrant.mask_hdu_name(): quadrant for quadrant in Quadrant}
"""Mask fits HDU extension name to quadrant dictionary."""


QuadrantMask = Enum('QuadrantMask', {quadrant.name: quadrant.value + MASK_HDU_NAME_SUFFIX for quadrant in Quadrant})
"""Enum with the names of the mask fits extension for each quadrant"""


QuadrantOrder = Enum('QuadrantOrder', [quadrant.name for quadrant in Quadrant])
"""The enum with the automatic order for all quadrants.
    It is needed because the Docker version of astrometry.net cannot read strings from fits files."""


@dataclass(frozen=True)
class QuadrantData:
    """ Class for storing per quadrant data."""
    gain:           float  # e-/ADU
    read_out_noise: float  # e-
    bias:           float  # Average bias in image area (ADU)
    ignore:         bool   # Will CCD reduction skip it?
    non_linearity:  float  # ADU non-linearity parameter: ADU
    qe_698nm:       float  # Quantum efficiency at 698 nm


@dataclass(frozen=True)
class CcdData:
    """ Class for storing the CCD fits CCDID key and the quadrants as projected on FPA coordinates."""
    ccdid:        str
    bottom_left:  Quadrant
    bottom_right: Quadrant
    top_right:    Quadrant
    top_left:     Quadrant


@unique
class Ccd(Enum):
    """ Defines the quadrant geometry per CCD in x,y FPA coordinates."""
    C_1_1 = CcdData('1_1', Quadrant.Q_1_1_E, Quadrant.Q_1_1_F, Quadrant.Q_1_1_G, Quadrant.Q_1_1_H)
    C_1_2 = CcdData('1_2', Quadrant.Q_1_2_E, Quadrant.Q_1_2_F, Quadrant.Q_1_2_G, Quadrant.Q_1_2_H)
    C_1_3 = CcdData('1_3', Quadrant.Q_1_3_E, Quadrant.Q_1_3_F, Quadrant.Q_1_3_G, Quadrant.Q_1_3_H)
    C_1_4 = CcdData('1_4', Quadrant.Q_1_4_G, Quadrant.Q_1_4_H, Quadrant.Q_1_4_E, Quadrant.Q_1_4_F)
    C_1_5 = CcdData('1_5', Quadrant.Q_1_5_G, Quadrant.Q_1_5_H, Quadrant.Q_1_5_E, Quadrant.Q_1_5_F)
    C_1_6 = CcdData('1_6', Quadrant.Q_1_6_G, Quadrant.Q_1_6_H, Quadrant.Q_1_6_E, Quadrant.Q_1_6_F)
    C_2_1 = CcdData('2_1', Quadrant.Q_2_1_E, Quadrant.Q_2_1_F, Quadrant.Q_2_1_G, Quadrant.Q_2_1_H)
    C_2_2 = CcdData('2_2', Quadrant.Q_2_2_E, Quadrant.Q_2_2_F, Quadrant.Q_2_2_G, Quadrant.Q_2_2_H)
    C_2_3 = CcdData('2_3', Quadrant.Q_2_3_E, Quadrant.Q_2_3_F, Quadrant.Q_2_3_G, Quadrant.Q_2_3_H)
    C_2_4 = CcdData('2_4', Quadrant.Q_2_4_G, Quadrant.Q_2_4_H, Quadrant.Q_2_4_E, Quadrant.Q_2_4_F)
    C_2_5 = CcdData('2_5', Quadrant.Q_2_5_G, Quadrant.Q_2_5_H, Quadrant.Q_2_5_E, Quadrant.Q_2_5_F)
    C_2_6 = CcdData('2_6', Quadrant.Q_2_6_G, Quadrant.Q_2_6_H, Quadrant.Q_2_6_E, Quadrant.Q_2_6_F)
    C_3_1 = CcdData('3_1', Quadrant.Q_3_1_E, Quadrant.Q_3_1_F, Quadrant.Q_3_1_G, Quadrant.Q_3_1_H)
    C_3_2 = CcdData('3_2', Quadrant.Q_3_2_E, Quadrant.Q_3_2_F, Quadrant.Q_3_2_G, Quadrant.Q_3_2_H)
    C_3_3 = CcdData('3_3', Quadrant.Q_3_3_E, Quadrant.Q_3_3_F, Quadrant.Q_3_3_G, Quadrant.Q_3_3_H)
    C_3_4 = CcdData('3_4', Quadrant.Q_3_4_G, Quadrant.Q_3_4_H, Quadrant.Q_3_4_E, Quadrant.Q_3_4_F)
    C_3_5 = CcdData('3_5', Quadrant.Q_3_5_G, Quadrant.Q_3_5_H, Quadrant.Q_3_5_E, Quadrant.Q_3_5_F)
    C_3_6 = CcdData('3_6', Quadrant.Q_3_6_G, Quadrant.Q_3_6_H, Quadrant.Q_3_6_E, Quadrant.Q_3_6_F)
    C_4_1 = CcdData('4_1', Quadrant.Q_4_1_E, Quadrant.Q_4_1_F, Quadrant.Q_4_1_G, Quadrant.Q_4_1_H)
    C_4_2 = CcdData('4_2', Quadrant.Q_4_2_E, Quadrant.Q_4_2_F, Quadrant.Q_4_2_G, Quadrant.Q_4_2_H)
    C_4_3 = CcdData('4_3', Quadrant.Q_4_3_E, Quadrant.Q_4_3_F, Quadrant.Q_4_3_G, Quadrant.Q_4_3_H)
    C_4_4 = CcdData('4_4', Quadrant.Q_4_4_G, Quadrant.Q_4_4_H, Quadrant.Q_4_4_E, Quadrant.Q_4_4_F)
    C_4_5 = CcdData('4_5', Quadrant.Q_4_5_G, Quadrant.Q_4_5_H, Quadrant.Q_4_5_E, Quadrant.Q_4_5_F)
    C_4_6 = CcdData('4_6', Quadrant.Q_4_6_G, Quadrant.Q_4_6_H, Quadrant.Q_4_6_E, Quadrant.Q_4_6_F)
    C_5_1 = CcdData('5_1', Quadrant.Q_5_1_E, Quadrant.Q_5_1_F, Quadrant.Q_5_1_G, Quadrant.Q_5_1_H)
    C_5_2 = CcdData('5_2', Quadrant.Q_5_2_E, Quadrant.Q_5_2_F, Quadrant.Q_5_2_G, Quadrant.Q_5_2_H)
    C_5_3 = CcdData('5_3', Quadrant.Q_5_3_E, Quadrant.Q_5_3_F, Quadrant.Q_5_3_G, Quadrant.Q_5_3_H)
    C_5_4 = CcdData('5_4', Quadrant.Q_5_4_G, Quadrant.Q_5_4_H, Quadrant.Q_5_4_E, Quadrant.Q_5_4_F)
    C_5_5 = CcdData('5_5', Quadrant.Q_5_5_G, Quadrant.Q_5_5_H, Quadrant.Q_5_5_E, Quadrant.Q_5_5_F)
    C_5_6 = CcdData('5_6', Quadrant.Q_5_6_G, Quadrant.Q_5_6_H, Quadrant.Q_5_6_E, Quadrant.Q_5_6_F)
    C_6_1 = CcdData('6_1', Quadrant.Q_6_1_E, Quadrant.Q_6_1_F, Quadrant.Q_6_1_G, Quadrant.Q_6_1_H)
    C_6_2 = CcdData('6_2', Quadrant.Q_6_2_E, Quadrant.Q_6_2_F, Quadrant.Q_6_2_G, Quadrant.Q_6_2_H)
    C_6_3 = CcdData('6_3', Quadrant.Q_6_3_E, Quadrant.Q_6_3_F, Quadrant.Q_6_3_G, Quadrant.Q_6_3_H)
    C_6_4 = CcdData('6_4', Quadrant.Q_6_4_G, Quadrant.Q_6_4_H, Quadrant.Q_6_4_E, Quadrant.Q_6_4_F)
    C_6_5 = CcdData('6_5', Quadrant.Q_6_5_G, Quadrant.Q_6_5_H, Quadrant.Q_6_5_E, Quadrant.Q_6_5_F)
    C_6_6 = CcdData('6_6', Quadrant.Q_6_6_G, Quadrant.Q_6_6_H, Quadrant.Q_6_6_E, Quadrant.Q_6_6_F)


CCD_ID_TO_CCD: Dict[str, Ccd] = {ccd.value.ccdid: ccd for ccd in Ccd}
"""Dictionary giving the Ccd object for each CCD ID."""


QUADRANT_TO_CCD: Dict[Quadrant, Ccd] = {
    **{ccd.value.bottom_left: ccd for ccd in Ccd},
    **{ccd.value.bottom_right: ccd for ccd in Ccd},
    **{ccd.value.top_right: ccd for ccd in Ccd},
    **{ccd.value.top_left: ccd for ccd in Ccd}}
"""Dictionary giving the parent CCD for each quadrant."""


QUADRANT_TO_QUADRANT_LOCATION: Dict[Quadrant, QuadrantLocation] = {
    **{ccd.value.bottom_left: QuadrantLocation.BOTTOM_LEFT for ccd in Ccd},
    **{ccd.value.bottom_right: QuadrantLocation.BOTTOM_RIGHT for ccd in Ccd},
    **{ccd.value.top_right: QuadrantLocation.TOP_RIGHT for ccd in Ccd},
    **{ccd.value.top_left: QuadrantLocation.TOP_LEFT for ccd in Ccd}}
"""Dictionary giving the quadrant location in FPA coordinates for each CCD quadrant."""


@dataclass
class VisProcessingConfig:
    """VIS FPA processing configuration"""
    max_threads: int
    cosmics_psf_fwhm: float
    cosmics_obj_lim: float
    daofind_fwhm: float
    daofind_threshold: float
    daofind_peak_max: float
    daofind_sharpness_min: float
    daofind_sharpness_max: float
    daofind_roundness_min: float
    daofind_roundness_max: float
    daofind_brightest: int
    compute_moments_pix: bool
    compute_fwhm: bool
    compute_fwhm_pixelated: bool
    compute_cramer_rao: bool
    compute_cramer_rao_no_ron: bool
    compute_eer_pix: bool
    compute_aperture_photometry_pix: bool
    compute_moments_sky: bool
    compute_eer_sky: bool
    compute_aperture_photometry_sky: bool
    quadrants: Dict[Quadrant, QuadrantData]
    determined_fwhm_scale_factor: float = 1.0
    quadrupole_moment_sigma_threshold: float = QUADRUPOLE_MOMENT_SIGMA_THRESHOLD
    eer_recenter: bool = False
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
        """JSON deserialization. Name => Quadrant enum. Dict => QuadrantData.
            :param json_document: document
            :param file_name: The input JSON file name, if applicable
            :return: the decoded VisProcessingConfig
        """
        decoded = cls(**json.loads(json_document))
        decoded.file_name = file_name
        decoded.quadrants = {Quadrant(name): QuadrantData(**values) for name, values in decoded.quadrants.items()}
        return decoded

    @classmethod
    def from_json_file(cls, config_file_name: str):
        """
        JSON file deserialization
        :param config_file_name: The input JSON file path
        :return: the decoded VisProcessingConfig
        """
        if isinstance(config_file_name, str):
            file_name = os.path.basename(config_file_name)
        else:
            file_name = None
        with open(config_file_name, 'r') as config_file:
            vis_processing_config = cls.from_json(config_file.read(), file_name)
        return vis_processing_config


class VisHDUList:
    """
    Handles a fits HDUList, including lookup dictionaries on secondary fits HDUs by quadrant (image and mask).
    Class extension did not fully work: write_to had to be reimplemented
    Input data are deep copied.
    """
    hdu_list: fits.HDUList
    """The handled HDUList"""
    images: Dict[Quadrant, fits.ImageHDU]
    """Dictionary indexing science image secondary fits extensions by quadrant"""
    masks: Dict[Quadrant, fits.ImageHDU]
    """Dictionary indexing mask secondary fits extensions by quadrant"""
    file_name: str
    """The I/O file_name, when applicable"""
    mode: VisMode
    """The VIS exposure mode"""
    
    def __init__(self, hdu_list: fits.HDUList, file_name: str = None):
        """
        :param hdu_list: the input hdu_list. It will be deep copied.
        :param file_name: the I/O file name, when applicable
        """
        self.hdu_list = fits.HDUList([hdu.copy() for hdu in hdu_list])
        self.images = {}
        self.masks = {}
        self.file_name = file_name
        self.mode = VisModeFromHdu(self.hdu_list[0])

        for hdu in self.hdu_list[1:]:
            # Add HDU to images dictionary, if appropriate
            try:
                quadrant = Quadrant(hdu.name)
                self.images[quadrant] = hdu
            except ValueError:
                # Add HDU to masks dictionary, if appropriate
                if hdu.name in MASK_HDU_NAME_TO_QUADRANT.keys():
                    self.masks[MASK_HDU_NAME_TO_QUADRANT[hdu.name]] = hdu

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

    def update_hdu_list(self):
        """
        Updates the HDU list with the current contents of the images and masks dictionaries
        """
        self.hdu_list = fits.HDUList([self.hdu_list[0], *self.images.values(), *self.masks.values()])

    def random_sample(self, max_quadrants):
        """
        Return a new VisHDUList composed of a random selection of this instance quadrants
        :param max_quadrants: The maximum number of quadrants to return.
        """
        if len(self.images) <= max_quadrants: #TODO revise condition?
            return VisHDUList(self.hdu_list)
        else:
            quadrants_selection_indices = np.random.default_rng().permutation(len(self.images))[:max_quadrants]
            quadrant_selection = [quadrant for i, quadrant in enumerate(self.images.keys())
                                  if i in quadrants_selection_indices]
            vis_hdu_list_selection = fits.HDUList(
                [self.hdu_list[0], *[self.images[quadrant] for quadrant in quadrant_selection],
                 *[self.masks[quadrant] for quadrant in quadrant_selection]])
            return VisHDUList(vis_hdu_list_selection)


def min_vis_hdu_list(vis_hdu_lists: Iterable[VisHDUList]) -> VisHDUList:
    """
    Generates a VisHDUList whose primary HDU, headers and extensions corresponds to the first input image
    and data to the minimum of all input VisHDUList. The computation is done for all VIS quadrants in the first input.
    Quadrants not present in the first input are ignored if present in others.
    An error is raised if quadrants in the first input are not present in others.
    The minimum will ignore NaN values (unless all are NaN).
    For each quadrant, masks are used if present in the first VisHDUList.
    The output mask value is the bitwise AND of all masks, to get a zero when at list a valid value exists.
    :param vis_hdu_lists: Iterator with the input VisHDUList
    :return: Output VisHDUList
    """
    # Create list from input iterable
    vis_hdu_lists = list(vis_hdu_lists)

    # Create output VisHDUList skeleton from first input deep copy
    minimum = VisHDUList(vis_hdu_lists[0].hdu_list)

    # Raise error if quadrants in the first input are not present in others
    minimum_image_quadrants = set(minimum.images.keys())
    minimum_mask_quadrants = set(minimum.masks.keys())
    for index, vis_hdu_list in enumerate(vis_hdu_lists[1:]):
        vis_hdu_list_image_quadrants = set(vis_hdu_list.images.keys())
        vis_hdu_list_mask_quadrants = set(vis_hdu_list.masks.keys())
        if not (minimum_image_quadrants.issubset(vis_hdu_list_image_quadrants)
                and minimum_mask_quadrants.issubset(vis_hdu_list_mask_quadrants)):
            raise ValueError(f'Input VisHDUList {index} has less image or mask quadrants than the first one')

    # Loop on output quadrant images
    for quadrant, hdu in minimum.images.items():
        # Data minimum computation
        quadrants_data = [vis_hdu_list.images[quadrant].data for vis_hdu_list in vis_hdu_lists]
        hdu.data = np.nanmin(quadrants_data, axis=0)
        # Mask
        if minimum.masks.get(quadrant) is not None:
            for input_vis_hdu_list in vis_hdu_lists:
                minimum.masks[quadrant].data = np.bitwise_and(minimum.masks[quadrant].data,
                                                              input_vis_hdu_list.masks[quadrant].data)

    # Add comment to primary header
    minimum.hdu_list[0].header['COMMENT'] = f'Minimum of {len(vis_hdu_lists)} input VIS fits files'

    return minimum


def subtract_bias_optimum_quadrant(quadrant: Quadrant, input_array: np.ndarray) -> np.ndarray:
    """
    Subtract bias from a given input array using the optimum algorithm in EUCL-ESAC-TN-3-002.
    A warning is raised in case of incorrect input geometry.
    :param quadrant: the VIS CCD quadrant
    :param input_array: input data array
    :return: the bias subtracted numpy array. dtype=np.float32
    """
    # Geometry check
    if input_array.shape[0] < QUADRANT_NAXIS_2 or  input_array.shape[1] < QUADRANT_NAXIS_1:
        ValueError(f'Incorrect input array shape: {input_array.shape}')
    elif input_array.shape[0] > QUADRANT_NAXIS_2 or input_array.shape[1] > QUADRANT_NAXIS_1:
        logging.warning(f'Incorrect input array shape: {input_array.shape}, maybe NOMINAL/SHORT with limited scan, data will be trimmed')

    # Subtract pre-scan optimum bias (individual value for each row)
    x_min, x_max = QUADRANT_TO_QUADRANT_LOCATION[quadrant].get_numpy_bias_optimum_prescan_range()
    prescan_area = input_array[:, x_min:x_max]
    prescan_values = np.mean(prescan_area, axis=1, dtype=np.float32)
    prescan_corrected = np.subtract(input_array.T, prescan_values, dtype=np.float32).T

    # Subtract post-scan optimum bias (single global value)
    y_min, y_max, x_min, x_max = QUADRANT_TO_QUADRANT_LOCATION[quadrant].get_numpy_bias_optimum_postcan_range()
    postscan_value = np.mean(prescan_corrected[y_min:y_max, x_min:x_max], dtype=np.float32)
    bias_corrected = np.subtract(prescan_corrected, postscan_value, dtype=np.float32)

    return bias_corrected


def subtract_bias_optimum(vis_hdu_list: VisHDUList):
    """
    Carries out optimum bias subtraction for a VIS HDU list
    :param vis_hdu_list: the input VIS hdu_list. It will modified.
    """
    # The data array of image quadrant-labeled HDUs is dark subtracted. Others are left intact
    # Masked values are set to NaN at this stage, when they are coded as np.float32
    for quadrant, hdu in vis_hdu_list.images.items():
        hdu.data = subtract_bias_optimum_quadrant(quadrant, hdu.data)
        hdu.data[vis_hdu_list.masks[quadrant].data > 0] = np.nan

    # Add comment to primary header
    vis_hdu_list.hdu_list[0].header['COMMENT'] = 'Optimum bias subtracted'


def get_raw_data_mask(input_array: np.ndarray) -> np.ndarray:
    """
    Get the mask corresponding to an input array containing raw ADU CCD values
    :param input_array: input data array
    :return: the output mask (uint8)
    """
    return np.add(ZERO_ADU_MASK_VALUE * (input_array == 0), SATURATION_ADU_MASK_VALUE * (input_array == MAX_ADU),
                  dtype=np.uint8, casting='unsafe')


def add_raw_data_mask(vis_hdu_list: VisHDUList):
    """
    Add a raw data mask to an input HDU list. Add comment header. Changes are done in place on input HDUs.
    :param vis_hdu_list: the input VIS hdu_list. It will modified.
    """
    # Raw data masks added for image quadrant-labeled HDUs is dark subtracted. Others are ignored
    for quadrant, hdu in vis_hdu_list.images.items():
        mask = get_raw_data_mask(hdu.data)
        vis_hdu_list.masks[quadrant] = fits.ImageHDU(mask, name=quadrant.mask_hdu_name())
    vis_hdu_list.update_hdu_list()

    # Add comment to primary header
    vis_hdu_list.hdu_list[0].header['COMMENT'] = 'Raw data ADU values mask added'


def trim_quadrant_data(quadrant: Quadrant, input_array: np.ndarray):
    """
    Trim a numpy array with level 1 processing geometry. Only the image area is kept.
    A warning is raised in case of incorrect input geometry.
    :param quadrant: the VIS CCD quadrant
    :param input_array: input data array
    :return: the trimmed quadrant numpy array. Note it is a view and not a copy
    """
    if input_array.shape != (QUADRANT_NAXIS_2, QUADRANT_NAXIS_1):
        logging.warning(f'Incorrect input array shape: {input_array.shape}, maybe NOMINAL/SHORT with limited scan?')
    y_min, y_max, x_min, x_max = QUADRANT_TO_QUADRANT_LOCATION[quadrant].get_numpy_trim_range()
    return input_array[y_min:y_max, x_min:x_max]


def trim_vis_hdu_list(vis_hdu_list: VisHDUList):
    """
    Trim all quadrants within an input HDU list. Add comment header. Changes are done in place on input HDUs.
    :param vis_hdu_list: the input VIS hdu_list. It will modified.
    """
    # The data array of image and mask quadrant-labeled HDUs is trimmed. Others are left intact
    for quadrant, hdu in vis_hdu_list.images.items():
        hdu.data = trim_quadrant_data(quadrant, hdu.data)
    for quadrant, hdu in vis_hdu_list.masks.items():
        hdu.data = trim_quadrant_data(quadrant, hdu.data)

    # Add comment to primary header
    vis_hdu_list.hdu_list[0].header['COMMENT'] = 'Quadrants trimmed to active pixels area'


def subtract_dark_current_quadrant_data(input_array: np.ndarray, dark_current: np.ndarray) -> np.ndarray:
    """
    Subtract master dark from a given quadrant data array. Computations are done in single precision
    :param input_array: input data array
    :param dark_current: dark current data array
    """
    return np.subtract(input_array, dark_current, dtype=np.float32)


class DarkCurrentSubtractor(VisHDUList):
    """This class stores a master dark frame and provides methods for subtracting it."""

    def subtract(self, vis_hdu_list: VisHDUList):
        """
        Subtract master dark from input fits stack. Changes are done in place on input HDUs.
        Computations are done in single precision
        :param vis_hdu_list: the input VIS hdu_list. It will modified.
        """
        # The data array of image quadrant-labeled HDUs is dark subtracted. Others are left intact
        for quadrant, hdu in vis_hdu_list.images.items():
            hdu.data = subtract_dark_current_quadrant_data(hdu.data, self.images[quadrant].data)
        # Combine masks
        for quadrant, mask in vis_hdu_list.masks.items():
            mask.data = mask.data | self.masks[quadrant].data

        # Add comment to primary header
        vis_hdu_list.hdu_list[0].header['COMMENT'] = 'Dark current subtracted'
        vis_hdu_list.hdu_list[0].header['DARK'] = (self.file_name, 'Dark current file')


def non_linearity_gain_qe_correct_quadrant(quadrant_data: np.ndarray, non_linearity_factor: float = 0,
                                           gain: float = 1, quantum_efficiency: float = 1) -> np.ndarray:
    """
    Apply non-linearity, gain and quantum efficiency corrections to a given VIS quadrant data.
    Computations are done in single precision
    :param quadrant_data: The VIS CCD quadrant data (bias corrected and dark subtracted, ADU)
    :param non_linearity_factor: The non-linearity factor
    :param gain: the CCD quadrant gain (e-/ADU)
    :param quantum_efficiency: the CCD quantum efficiency (adim)
    """
    return np.multiply(quadrant_data, gain / quantum_efficiency / (1 + non_linearity_factor * quadrant_data),
                       dtype=np.float32)


@dataclass
class NonLinearityGainQuantumEfficiencyCorrector:
    """
    Handles non-linearity, gain and quantum efficiency corrections applied to bias and dark subtracted images as follows:
    pixel_value_adu_corrected =
        pixel_value_adu * gain / quantum_efficiency / (1 + non_linearity_factor * pixel_value_adu)
    """

    non_linearity_factor: Dict[Quadrant, float]
    """Dictionary with non-linearity factors for each quadrant"""
    gain: Dict[Quadrant, float]
    """Dictionary with the CCD gain (e-/ADU) for each quadrant"""
    quantum_efficiency: Dict[Quadrant, float]
    """Dictionary with the CCD quantum efficiency (adim) for each quadrant"""

    @classmethod
    def from_vis_processing_config(cls, vis_processing_config: VisProcessingConfig):
        """
        Instantiate from an existing fits file. Note all HDUs must be read and copied.
        This can represent an overhead with respect to the parent HDUList class.
        However, it may be convenient for frequently accessed frames (e.g. master dark or flat)
        whose quick accessibiity in RAM must be guaranteed.
        :param fits_file: input fits file
        :param vis_processing_config: Object with key VIS processing configuration parameters
        :return: the instantiated VisHDUList object
        """
        non_linearity_factor = {quadrant: quadrant_data.non_linearity
                                for quadrant, quadrant_data in vis_processing_config.quadrants.items()}
        gain = {quadrant: quadrant_data.gain
                for quadrant, quadrant_data in vis_processing_config.quadrants.items()}
        quantum_efficiency = {quadrant: quadrant_data.qe_698nm
                              for quadrant, quadrant_data in vis_processing_config.quadrants.items()}
        return cls(non_linearity_factor, gain, quantum_efficiency)

    def correct(self, vis_hdu_list: VisHDUList):
        """
        Apply quadrant-differential non-linearity, gain and quantum efficiency corrections to an input fits stack.
        Changes are done in place on input HDUs. Computations are done in single precision
        :param vis_hdu_list: the input hdu_list. It will be modified.
        """

        # The data array of image quadrant-labeled HDUs is corrected. Others are left intact
        for quadrant, hdu in vis_hdu_list.images.items():
            hdu.data = non_linearity_gain_qe_correct_quadrant(
                hdu.data, self.non_linearity_factor[quadrant], self.gain[quadrant], self.quantum_efficiency[quadrant])

        # Add comment to primary header
        vis_hdu_list.hdu_list[0].header['COMMENT'] = \
            'Quadrant-differential non-linearity, gain and quantum efficiency corrections applied'


def correct_prnu_quadrant_data(input_array: np.ndarray, prnu: np.ndarray) -> np.ndarray:
    """
    Correct PRNU from a given quadrant data array. Computations are done in single precision
    :param input_array: input data array
    :param prnu: the flat-field data array
    """
    return np.divide(input_array, prnu, dtype=np.float32)


class PrnuCorrector(VisHDUList):
    """This class stores a master flat-field frame and provides methods for dividing by it."""

    def correct(self, vis_hdu_list: VisHDUList):
        """
        Correct PRNU from input fits stack. Changes are done in place on input HDUs.
        Computations are done in single precision
        :param vis_hdu_list: the input VIS hdu_list. It will be modified.
        """
        # The data array of image quadrant-labeled HDUs is dark subtracted. Others are left intact
        for quadrant in vis_hdu_list.images.keys():
            hdu = vis_hdu_list.images[quadrant]
            mask = vis_hdu_list.masks[quadrant]
            hdu.data = correct_prnu_quadrant_data(hdu.data, self.images[quadrant].data)
            # Min/max electron count mask
            outside_range_mask = ((hdu.data < OUTSIDE_RANGE_MIN_VALUE) | (hdu.data > OUTSIDE_RANGE_MAX_VALUE)).astype(
                np.uint8) * OUTSIDE_RANGE_MASK_VALUE  # Cast to uint8 only needed for old numpy versions
            # Combine masks
            mask.data = mask.data | self.masks[quadrant].data | outside_range_mask

        # Add comments to primary header
        vis_hdu_list.hdu_list[0].header['COMMENT'] = 'Flat-field PRNU correction applied'
        vis_hdu_list.hdu_list[0].header['FLAT'] = (self.file_name, 'Flat-field file')


@dataclass
class VisCcdProcessor:
    """
    VIS CCD processor. It follows the steps prescribed by Ralf Kohley
    """
    dark_current_subtractor: DarkCurrentSubtractor
    non_linearity_gain_qe_corrector: NonLinearityGainQuantumEfficiencyCorrector
    prnu_corrector: PrnuCorrector

    @classmethod
    def from_files(cls, master_dark, vis_processing_config_file, master_flat):
        """
        Instantiate a VIS CCD processor reading master frames and parameters from files
        :param master_dark: The master dark fits stack
        :param vis_processing_config_file: The VIS processing configuration file
        :param master_flat: The master flat PRNU fits stack
        """
        # Instantiate future class fields
        dark_current_subtractor = DarkCurrentSubtractor.from_fits(master_dark)
        vis_processing_config = VisProcessingConfig.from_json_file(vis_processing_config_file)
        non_linearity_gain_qe_corrector = \
            NonLinearityGainQuantumEfficiencyCorrector.from_vis_processing_config(vis_processing_config)
        prnu_corrector = PrnuCorrector.from_fits(master_flat)
        # Call constructor and return instance
        return cls(dark_current_subtractor, non_linearity_gain_qe_corrector, prnu_corrector)

    # TODO make module method
    def process_quadrant(self, quadrant: Quadrant, hdu: fits.ImageHDU):
        """
        Process a VIS quadrant. Computations are done in single precision
        :param quadrant: The VIS CCD quadrant
        :param hdu: the input fits ImageHDU. It will modified.
        """
        hdu.data = subtract_bias_optimum_quadrant(quadrant, hdu.data)
        hdu.data = trim_quadrant_data(quadrant, hdu.data)
        self.dark_current_subtractor.subtract_quadrant(quadrant, hdu)
        hdu.data = self.non_linearity_gain_qe_corrector.correct_quadrant_data(quadrant, hdu.data)

    def process(self, vis_hdu_list: VisHDUList):
        """
        Process a VIS HDU list. Single threaded. Computations are done in single precision
        :param vis_hdu_list: the input hdu_list. It will modified.
        """
        add_raw_data_mask(vis_hdu_list)
        subtract_bias_optimum(vis_hdu_list)
        trim_vis_hdu_list(vis_hdu_list)
        self.dark_current_subtractor.subtract(vis_hdu_list)
        self.non_linearity_gain_qe_corrector.correct(vis_hdu_list)
        self.prnu_corrector.correct(vis_hdu_list)

        # TODO do not process by quadrants
        # TODO add another method por parallel quadrant processing, if needed
        #for _ in map(self.process_quadrant, vis_hdu_list.images.keys(), vis_hdu_list.images.values()):
        #    continue
        # TODO Add comment methods

    def process_file(self, input_file: str, output_file: str) -> Tuple[int, int, int]:
        """
        Process a VIS input fits file and produces an output filts file.
        Single threaded. Computations are done in single precision
        :param input_file: the input fits file
        :param output_file: the output fits file
        :return: Timing information (ms): (input file read, CCD process, output file write)
        """
        now = time.time_ns()
        vis_hdu_list = VisHDUList.from_fits(input_file)
        read_ns = time.time_ns() - now
        now += read_ns
        self.process(vis_hdu_list)
        process_ns = time.time_ns() - now
        now += process_ns
        vis_hdu_list.hdu_list.writeto(output_file, overwrite=True)
        write_ns = time.time_ns() - now
        return read_ns, process_ns, write_ns


def process_vis_level1_frames(input_folder, output_folder, log_file,  # TODO remove?
                              master_dark, vis_processing_config_file, master_flat):
    """
    Process all VIS level 1 frames in a given folder. Useful for batch processing
    :param input_folder: The input folder. All "*.fits" files within will be processed
    :param output_folder: The output folder. Will be created, if needed.
                            Pre-existing files with the same name will be overwritten
    :param log_file: The output log file. Entries will be appended
    :param master_dark: The master dark fits stack
    :param vis_processing_config_file: The VIS processing configuration file
    :param master_flat: The master flat PRNU fits stack
    """

    # Initialise logger
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Create input fits file list. Return if empty
    input_files = [f for f in os.listdir(input_folder) if FITS_FILE_REGEX.search(f) is not None]
    if not input_files:
        logging.critical('Exiting. No input files found')
        return
    logging.info(f'Input files to process: {len(input_files)}')

    # Instantiate VIS CCD processor
    logging.info('Instantiating VIS CCD processor')
    now = time.time_ns()
    vis_ccd_processor = VisCcdProcessor.from_files(master_dark, vis_processing_config_file, master_flat)
    processor_init_ns = time.time_ns() - now
    logging.info(f'Instantiation time (ms): {processor_init_ns // 1000000}')

    # Create output folder, if needed
    os.makedirs(output_folder, exist_ok=True)

    # Loop on input fits files
    for i, input_file in enumerate(input_files):
        logging.info(f'Processing file {i + 1} out of {len(input_files)}: {input_file}')
        input_file_path = os.path.join(input_folder, input_file)
        output_file_path = os.path.join(
            output_folder, re.sub(FITS_FILE_REGEX, CCD_PROCESSED_FILE_SUFFIX + FITS_FILE_EXTENSION, input_file))
        read_ns, process_ns, write_ns = vis_ccd_processor.process_file(input_file_path, output_file_path)
        logging.info(f'Read time (ms): {read_ns // 1000000:6d} '
                     f'Process time (ms): {process_ns // 1000000:6d} '
                     f'Write time(ms): {write_ns // 1000000:6d}')

    logging.info('End. All input files processed')


def clean_cosmics_quadrant_data(
        input_array: np.ndarray, objlim: float = None, readnoise: float = None,
        satlevel: float = None, psffwhm: float = None,) -> Tuple[np.ndarray]:
    """
    Clean cosmic rays according to L.A. Cosmic algorithm.
    :param input_array: input data array in electrons. It can contain bad pixels coded as NaN values
    :param objlim: LA Cosmics parameter objlim
    :param readnoise: LA Cosmics parameter readnoise (e-)
    :param satlevel: LA Cosmics parameter satlevel (e-)
    :param psffwhm: LA Cosmics parameter psffwhm (pix)
    :return: output_array, output_mask
    """
    output_array, output_mask = ccdproc.cosmicray_lacosmic(
        input_array,
        objlim=objlim,
        readnoise=readnoise,
        satlevel=satlevel,
        psffwhm=psffwhm,
        gain_apply=False,
    )
    return output_array, output_mask


def clean_cosmics_vis_hdu_list(vis_hdu_list: VisHDUList, vis_processing_config: VisProcessingConfig):
    """
    Apply cosmic ray correction to all quadrants within an input HDU list. Add comment header.
     Changes are done in place on input HDUs.
    :param vis_hdu_list: the input VIS hdu_list. It will modified.
    :param vis_processing_config: the VIS processing configuration
    """
    # The data array of image and mask quadrant-labeled HDUs is modified. Others are left intact
    # A pre-existing mask is needed as a result of previous basic CCD processing

    quadrants = sorted(vis_hdu_list.images.keys())

    # Multiprocessing
    with multiprocessing.Pool(min(NUM_CORES, vis_processing_config.max_threads)) as pool:
        parameters = [[vis_hdu_list.images[quadrant].data,
                       vis_processing_config.cosmics_obj_lim,
                       vis_processing_config.quadrants[quadrant].read_out_noise,
                       CCD_SATURATION_LEVEL,
                       vis_processing_config.cosmics_psf_fwhm
                       ] for quadrant in quadrants]
        results = pool.starmap(clean_cosmics_quadrant_data, parameters)

    # VIS HDU list update
    for quadrant, (cosmics_cleaned_data, cosmics_cleaned_mask) in zip(quadrants, results):
        image = vis_hdu_list.images[quadrant]
        mask = vis_hdu_list.masks[quadrant]
        image.data = cosmics_cleaned_data
        mask.data = np.bitwise_or(mask.data, cosmics_cleaned_mask * COSMIC_RAY_CORRECTED_MASK_VALUE,
                                  dtype=np.uint8, casting='unsafe')

    # Add comment to primary header
    vis_hdu_list.hdu_list[0].header['COMMENT'] = 'Cosmic ray correction applied'

    # TODO CCD_SATURATION_LEVEL by configuration parameter?
    # TODO verify all mask values in CCD processing consistent with Ralf values

def mosaic_vis_hdu_list(vis_hdu_list: VisHDUList) -> fits.HDUList:
    """
    Generate a mosaic
    :param vis_hdu_list: the input VIS hdu_list. It will not be modified.
    """

    logging.info(f'Processing type {vis_hdu_list.mode}')

    if vis_hdu_list.mode == VisMode.MULTI_SERIAL_PARALLEL_TRAP_PUMPING:
        M_NAXIS1 = MOSAIC_NAXIS1
        M_NAXIS2 = MOSAIC_NAXIS2_MSTP
    else:
        M_NAXIS1 = MOSAIC_NAXIS1
        M_NAXIS2 = MOSAIC_NAXIS2

    mosaic_image = np.zeros((M_NAXIS2, MOSAIC_NAXIS1), dtype='uint16')

    for quadrant, hdu in vis_hdu_list.images.items():

        # Get the trim sizes
        y_min, _, x_min, _ = QUADRANT_TO_QUADRANT_LOCATION[quadrant].get_numpy_trim_range()

        target_y_0 = int(M_NAXIS2 / 2 - hdu.header['CRPIX2']) + y_min
        target_x_0 = int(M_NAXIS1 / 2 - hdu.header['CRPIX1']) + x_min

        target_y_1 = target_y_0 + len(hdu.data)
        target_x_1 = target_x_0 + len(hdu.data[0])

        logging.info(f'{target_y_0},{target_y_1}  {target_x_0},{target_x_1}')

        mosaic_image[target_y_0:target_y_1, target_x_0:target_x_1] = hdu.data

        hdu_list = fits.HDUList(fits.PrimaryHDU(header=vis_hdu_list.hdu_list[0].header))
        hdu_list.append(fits.ImageHDU(mosaic_image))
        hdu_list.update_extend()

    return hdu_list


def detect_stars_quadrant(input_array: np.ndarray, mask: np.ndarray = None,
                          starfinderbase: StarFinderBase = DAOStarFinder, starfinderbase_args: Dict = {}) -> Table:
    """
    Detect star-like objects for a given quadrant using a StarFinderBase (DAOPhot by defauld).
    Photometry is not further refined.
    This approach is suboptimal in crowded fields, but reasonable for low density areas like the self-calibration
    field. A combination of simple aperture photometry with a clean external catalogue free of crowded objects might
    be sufficient for the purposes of estimating image quality metrics.
    It should use a cosmics cleaned image (e.g. minimum, LACosmic), if available.
    :param input_array: input data array in electrons.
    :param mask: input data array boolean mask.
    :param starfinderbase: The StarFinderBase class to look for stars
    :param starfinderbase_args: The parameters to initialise the StarFinderBase (e.g.
     threshold, fwhm, sharplo, sharphi, roundlo, roundhi, brightest, peakmax, xycoords)
    :return: The StarFinderBase output table.
    """

    # Apply sigma-clipping before detection
    _, median, std = sigma_clipped_stats(input_array, sigma=3.0)
    # TODO is median sky subtraction needed?
    if starfinderbase is DAOStarFinder:
        starfinder = DAOStarFinder(**starfinderbase_args, exclude_border=True)
    elif starfinderbase is IRAFStarFinder:
        starfinder = IRAFStarFinder(**starfinderbase_args, exclude_border=True)
    else:
        raise ValueError(f'Unsupported StarFinderBase: {starfinderbase}')

    return starfinder(input_array - median, mask=mask)


def get_cutout_extname(quadrant: Quadrant, x_centroid: float, y_centroid: float) -> str:
    """
    Encodes several parameters into the HDU fits name for a given image cutout
    :param quadrant: CCD quadrant
    :param x_centroid: source x-axis centroid location
    :param y_centroid: source y-axis centroid location
    :return: The encoded output string
    """
    return f'{quadrant.value}_X_{int(round(x_centroid))}_Y_{int(round(y_centroid))}'


def get_cutout_mask_extname(quadrant: Quadrant, x_centroid: float, y_centroid: float) -> str:
    """
    Encodes several parameters into the HDU fits name for a given image cutout mask
    :param quadrant: CCD quadrant
    :param x_centroid: source x-axis centroid location
    :param y_centroid: source y-axis centroid location
    :return: The encoded output string
    """
    return f'{quadrant.value}{MASK_HDU_NAME_SUFFIX}_X_{int(round(x_centroid))}_Y_{int(round(y_centroid))}'


def parse_cutout_extame(cutout_extname: str) -> Tuple[Quadrant, int, int]:
    """
    Decodes an image cutout HDU fits extension name into its components
    :param cutout_extname: Input HDU fits extension name encoded string (str)
    :return: (Quadrant, int, int): CCD quadrant, x-axis rounded centroid, y-axis rounded centroid
    """
    quadrant_name, _, x_round_text, _, y_round_text = cutout_extname.split('_')
    return Quadrant(quadrant_name), int(x_round_text), int(y_round_text)


def parse_cutout_mask_extame(mask_extname: str) -> Tuple[Quadrant, int, int]:
    """
    Decodes an image cutout mask HDU fits extension name into its components
    :param mask_extname: Input HDU fits extension name encoded string (str)
    :return: (Quadrant, int, int): CCD quadrant, x-axis rounded centroid, y-axis rounded centroid
    """
    quadrant_name, _, _, x_round_text, _, y_round_text = mask_extname.split('_')
    return Quadrant(quadrant_name), int(x_round_text), int(y_round_text)


@dataclass()
class VisCutouts:
    """
    Collection of stellar image cutouts
    """
    primary_header: fits.Header
    """Fits primary HDU header"""
    cutouts: Dict[str, fits.ImageHDU]
    """
    Dictionary of cutouts. Key: cutout HDU extension name (see parse_cutout_extname). Value: HDU cutout
    """
    masks: Dict[str, fits.ImageHDU]
    """
    Dictionary of masks. Key: cutout mask HDU extension name (see parse_cutout_mask_extname). Value: HDU cutout mask
    """

    @classmethod
    def from_vis_hdu_list(
            cls, vis_hdu_list: VisHDUList, vis_stars: VisStars, vis_processing_config: VisProcessingConfig):
        """
        Generate image cutouts for a given VIS observation
        :param vis_hdu_list: The (preferably) cosmic rays cleaned VIS observation
        :param vis_stars: The detected stars
        :param vis_processing_config: The VIS processing configuration.
        """

        # Create cutouts dictionaries
        cutouts = {}
        masks = {}

        # Loop on quadrants and stars
        for quadrant, stars in vis_stars.stars.items():
            for star in stars:
                # Trim range
                y_min, y_max, x_min, x_max = get_cutout_numpy_trim_range(
                    star[X_CENTROID_COLUMN], star[Y_CENTROID_COLUMN],
                    vis_hdu_list.images[quadrant].data.shape[1], vis_hdu_list.images[quadrant].data.shape[0],
                    CUTOUT_SIZE, CUTOUT_SIZE)

                # Cutout
                cutout_name = get_cutout_extname(quadrant, star[X_CENTROID_COLUMN], star[Y_CENTROID_COLUMN])
                cutout = fits.ImageHDU(vis_hdu_list.images[quadrant].data[y_min:y_max, x_min:x_max].copy(),  # TODO copy needed?
                                       name=cutout_name)
                cutouts[cutout_name] = cutout
                cutout.header['COMMENT'] = 'Euclid VIS image cutout'
                cutout.header[X_CENTER_DESCRIPTOR] = star[X_CENTROID_COLUMN]
                cutout.header[Y_CENTER_DESCRIPTOR] = star[Y_CENTROID_COLUMN]
                cutout.header[X_MIN_CUTOUT_DESCRIPTOR] = x_min
                cutout.header[X_MAX_CUTOUT_DESCRIPTOR] = x_max
                cutout.header[Y_MIN_CUTOUT_DESCRIPTOR] = y_min
                cutout.header[Y_MAX_CUTOUT_DESCRIPTOR] = y_max
                cutout.header[READ_OUT_NOISE_DESCRIPTOR] = vis_processing_config.quadrants[quadrant].read_out_noise
                cutout.header[EXPOSURE_TIME_DESCRIPTOR] = vis_stars.primary_header[EXPOSURE_TIME_DESCRIPTOR]

                # Mask
                mask_name = get_cutout_mask_extname(quadrant, star[X_CENTROID_COLUMN], star[Y_CENTROID_COLUMN])
                mask = fits.ImageHDU(vis_hdu_list.masks[quadrant].data[y_min:y_max, x_min:x_max].copy(), name=mask_name)  # TODO copy needed?
                masks[mask_name] = mask
                mask.header['COMMENT'] = 'Euclid VIS image cutout mask'

        header = vis_stars.primary_header.copy()
        header["COMMENT"] = "Euclid VIS image cutouts. Header copied from parent CCD processed frame"

        return VisCutouts(header, cutouts, masks)

    @classmethod
    def from_fits(cls, fits_file):
        """
        Instantiate from an existing fits file. Note all HDUs must be read and copied.
        This can represent an overhead with respect to a simpler HDUList class.
        However, it may be convenient for frequently accessed frames (e.g. master dark or flat)
        whose quick accessibility in RAM must be guaranteed.
        :param fits_file: input fits file
        :return: the instantiated VisCutouts object
        """
        cutouts = {}
        """Keys: quadrant, round(x_centroid), round(y_centroid). Value: ImageHDU"""
        masks = {}
        """Keys: quadrant, round(x_centroid), round(y_centroid). Value: ImageHDU"""
        with fits.open(fits_file) as hdu_list:
            primary_header = hdu_list[0].header
            for hdu in hdu_list[1:]:
                if MASK_HDU_NAME_SUFFIX not in hdu.name:
                    cutouts[hdu.name] = hdu.copy()
                else:
                    masks[hdu.name] = hdu.copy()
        return cls(primary_header, cutouts, masks)

    def to_fits(self, file_name: str, overwrite: bool = True):
        """
        Write object to fits. Sub-HDUs will be sorted by quadrant order
        :param file_name: The output fits file name
        :param overwrite: Overwrite output fits file flag
        """
        hdu_list = fits.HDUList(fits.PrimaryHDU(header=self.primary_header))
        hdu_list += self.cutouts.values()
        hdu_list += self.masks.values()
        hdu_list.update_extend()
        hdu_list.writeto(file_name, overwrite=overwrite)


def get_vis_stars_ccd_file_name(prefix: str, ccd: Ccd, extension: str = FITS_FILE_EXTENSION) -> str:
    """
    Get the file name for stars detected in a CCD encoding a prefix and CCD ID.
    :param prefix: the file name prefix
    :param ccd: The CCD
    :param extension: The output CCD fits file extension (e.g. .fits or .corr)
    :return: The file name
    """
    return VIS_STARS_CCD_FILE_NAME_FORMAT.format(prefix, ccd.value.ccdid, extension)


def get_vis_stars_ccd_file_names(prefix: str, extension: str = FITS_FILE_EXTENSION) -> List[str]:
    """
    Get all possible file names for stars detected in a CCD encoding a prefix and CCD ID
    :param prefix: the file name prefix
    :param extension: The output CCD fits file extension (e.g. .fits or .corr)
    :return: all possible valid file names
    """
    return [get_vis_stars_ccd_file_name(prefix, ccd, extension) for ccd in Ccd]


def decode_vis_stars_ccd_file_name(file_name: str, extension: str = FITS_FILE_EXTENSION) -> Tuple[str, Ccd]:
    """
    Decode a file name for stars detected in a CCD into a prefix and CCD ID.
    :param file_name: The file name
    :param extension: The output CCD fits file extension (e.g. .fits or .corr)
    :return: The prefix and CCD
    """
    tokens = file_name.replace(extension, '').split('_')
    ccd_id = tokens[-2] + '_' + tokens[-1]
    prefix = file_name.replace(VIS_STARS_CCD_FILE_NAME_FORMAT.format('', ccd_id, extension), '')
    return prefix, CCD_ID_TO_CCD[ccd_id]


@dataclass()
class VisStars:
    """
    Collection of stars detected in a given VisHduList. It contains a table per quadrant.
    It might be generated at three different times in the data processing: object detection, astrometric solution and
    image quality metrics.
    The number of columns and descriptors is thus variable (might include astrometry or image quality metrics)
    and can be different for each quadrant.
    It can be written to or loaded from a fits file. Astrometric information is stored as header information.
    """
    primary_header: fits.Header
    """The fits primary HDU header"""
    stars: Dict[Quadrant, Table]
    """The star tables, indexed by quadrant"""

    @classmethod
    def from_starfinderbase(cls, vis_hdu_list: VisHDUList, vis_processing_config: VisProcessingConfig,
                            stars_centroid_guess: Dict[Quadrant, Table] = None,
                            starfinderbase: StarFinderBase = DAOStarFinder):
        """
        :param vis_hdu_list: the input cosmics cleaned vis_hdu_list.
        :param vis_processing_config: The VIS processing configuration
        :param stars_centroid_guess: VisStars.stars from another object used as centroids first guess.
          Useful for e.g. use the minimum of several frames to define the good objects and initial centroids.
        :param starfinderbase: The StarFinderBase class to look for stars
        """
        # Copy and update primary header from vis_hdu_list
        primary_header = vis_hdu_list.hdu_list[0].header.copy()

        # Multiprocessing
        quadrants = sorted(vis_hdu_list.images.keys())
        xycoords_all_quadrants: Dict[Quadrant: np.ndarray]
        if stars_centroid_guess is not None:
            xycoords_all_quadrants = {
                quadrant: np.array((table[X_CENTROID_COLUMN], table[Y_CENTROID_COLUMN])).transpose()
                for quadrant, table in stars_centroid_guess.items()}
        else:
            xycoords_all_quadrants = {quadrant: None for quadrant in quadrants}
        with multiprocessing.Pool(min(NUM_CORES, vis_processing_config.max_threads)) as pool:
            if starfinderbase is DAOStarFinder:
                parameters = [[vis_hdu_list.images[quadrant].data,
                               vis_hdu_list.masks[quadrant].data & ~COSMIC_RAY_CORRECTED_MASK_VALUE,  # Cosmics corrected are OK
                               starfinderbase,
                               {'fwhm': vis_processing_config.daofind_fwhm,
                                'sharplo': vis_processing_config.daofind_sharpness_min,
                                'sharphi': vis_processing_config.daofind_sharpness_max,
                                'roundlo': vis_processing_config.daofind_roundness_min,
                                'roundhi': vis_processing_config.daofind_roundness_max,
                                'threshold': vis_processing_config.daofind_threshold,
                                'peakmax': vis_processing_config.daofind_peak_max,
                                'brightest': vis_processing_config.daofind_brightest,
                                'xycoords': xycoords_all_quadrants[quadrant],
                                },
                               ]for quadrant in quadrants]
            elif starfinderbase is IRAFStarFinder:
                parameters = [[vis_hdu_list.images[quadrant].data,
                               vis_hdu_list.masks[quadrant].data & ~COSMIC_RAY_CORRECTED_MASK_VALUE,  # Cosmics corrected are OK
                               starfinderbase,
                               {'fwhm': vis_processing_config.daofind_fwhm,
                                'threshold': vis_processing_config.daofind_threshold,
                                'peakmax': vis_processing_config.daofind_peak_max,
                                'brightest': vis_processing_config.daofind_brightest,
                                'xycoords': xycoords_all_quadrants[quadrant],
                                },
                               ] for quadrant in quadrants]
            else:
                raise ValueError(f'Unsupported StarFinderBase: {starfinderbase}')
            tables = pool.starmap(detect_stars_quadrant, parameters)
        stars = {quadrant: table for quadrant, table in zip(quadrants, tables) if table is not None}
        for quadrant, table in stars.items():
            table.meta[FITS_EXTENSION_NAME_KEY] = quadrant.value

        return cls(primary_header, stars)

    @classmethod
    def from_fits(cls, fits_file):
        """
        Instantiate from an existing fits file. Note all HDUs must be read and copied.
        This can represent an overhead with respect to a simpler HDUList class.
        However, it may be convenient for frequently accessed frames (e.g. master dark or flat)
        whose quick accessibility in RAM must be guaranteed.
        :param fits_file: input fits file
        :return: the instantiated VisStars object
        """
        with fits.open(fits_file) as hdu_list:
            primary_header = hdu_list[0].header.copy()
            tables = {Quadrant(hdu.name): Table.read(hdu) for hdu in hdu_list[1:]}
            return cls(primary_header, tables)

    def to_fits(self, file_name: str, overwrite: bool = True):
        """
        Write object to fits. Tables will be sorted by quadrant order.
        If present, astrometry will be stored as image headers.
        :param file_name: The output fits file name
        :param overwrite: Overwrite output fits file flag
        """
        hdu_list = fits.HDUList(fits.PrimaryHDU(header=self.primary_header))
        hdu_list.extend([fits.BinTableHDU(self.stars[quadrant]) for quadrant in self.stars.keys()])
        hdu_list.update_extend()
        hdu_list.writeto(file_name, overwrite=overwrite)

    def to_fits_per_ccd(self, prefix: str):
        """
        Save data to different files for each CCD.
        The primary header will be stored in all children
        CCD pixel coordinates are generated for each star.
        Useful for an astrometry.net reduction, which works much better on a per-CCD level
        :param prefix: The output CCD fits files prefix
        """

        # Group input tables by CCD and quadrant location keys: CCD, quadrant, value: table
        # CCDs are ordered by enum position
        ccd_quadrants: Dict[Ccd, Dict[QuadrantLocation, Table]] = defaultdict(dict)
        for quadrant, table in self.stars.items():
            ccd_quadrants[QUADRANT_TO_CCD[quadrant]][quadrant] = table

        # Clean temporary astrometry.net files TODO necessary?
        #for file_name in os.listdir():
        #    if file_name.startswith(prefix):
        #        os.remove(file_name)

        # Loop on CCDs
        for ccd, quadrants in ccd_quadrants.items():

            # Create astropy table merging data for all CCDs and save to fits
            edited_tables = []
            for quadrant, table in quadrants.items():
                edited_table = table.copy(copy_data=True)
                edited_tables.append(edited_table)
                # edited_table[quadrant_column] = quadrant.value TODO better code, but incompatible with Docker astrometry.net
                edited_table[ASTROMETRY_QUADRANT_COLUMN] = QuadrantOrder[quadrant.name].value
                quadrant_location = QUADRANT_TO_QUADRANT_LOCATION[quadrant]
                edited_table[ASTROMETRY_X_CENTROID_CCD_COLUMN] = table[X_CENTROID_COLUMN] + quadrant_location.value.x_offset
                edited_table[ASTROMETRY_Y_CENTROID_CCD_COLUMN] = table[Y_CENTROID_COLUMN] + quadrant_location.value.y_offset

            ccd_sources = vstack(edited_tables, metadata_conflicts='silent')
            ccd_sources.meta[FITS_EXTENSION_NAME_KEY] = ccd.value.ccdid
            primary_header = self.primary_header.copy()
            primary_header['COMMENT'] = 'Input data for astrometry.net astrometric reduction'  # TODO OK?
            hdu_list = fits.HDUList([fits.PrimaryHDU(header=primary_header), fits.table_to_hdu(ccd_sources)])
            hdu_list.writeto(get_vis_stars_ccd_file_name(prefix, ccd), overwrite=True)

    @classmethod
    def from_fits_per_ccd(cls, prefix: str, extension: str = FITS_FILE_EXTENSION):
        """
        Load data stored in different files for each CCD produced for or by an astrometry.net run.
        The primary header will be taken from the first table (it should be identical for all of them).
        CCD pixel coordinates are converted into quadrant coordinates for each star.
        Useful to load the output of an astrometry.net reduction, which works much better on a per-CCD level
        :param prefix: The output CCD fits files prefix
        :param extension: The output CCD fits file extension (e.g. .fits or .corr)
        """
        primary_header: fits.Header = None
        stars: Dict[Quadrant, Table] = {}

        # Loop on input files               ]
        for input_file in get_vis_stars_ccd_file_names(prefix, extension):
            if os.path.exists(input_file):

                # Read fits table
                ccd_table = Table.read(input_file)

                # Use first available primary header (all of them should be equal)
                if primary_header is None:
                    with fits.open(input_file) as hdu_list:
                        primary_header = hdu_list[0].header

                # Add units to astrometry.net columns, if available
                for column, unit in ASTROMETRY_NET_XMATCH_TABLE_UNITS.items():
                    if column in ccd_table.columns:
                        ccd_table[column].unit = unit

                # Distribute table rows into quadrants
                tables_by_quadrant = ccd_table.group_by(ASTROMETRY_QUADRANT_COLUMN)
                for quadrant_name, sources_per_quadrant \
                        in zip(tables_by_quadrant.groups.keys, tables_by_quadrant.groups):
                    quadrant = Quadrant[QuadrantOrder(quadrant_name[ASTROMETRY_QUADRANT_COLUMN]).name]
                    stars[quadrant] = sources_per_quadrant
                    sources_per_quadrant.meta[FITS_EXTENSION_NAME_KEY] = quadrant.value
                    sources_per_quadrant.remove_column(ASTROMETRY_QUADRANT_COLUMN)

        return cls(primary_header, stars)

    @classmethod
    def from_inserts_file(cls, inserts_file_name: str):
        """
        Use inserts to analyze the ideal case
        :param inserts_file_name: The json file with all the inserts.
        """
        # Decode json file with inserts
        with open(inserts_file_name, 'r') as inserts_file:
            image_inserts = inserts_file.read()
        json_list = json.loads(image_inserts)
        inserts = [ImageInsert(**dict) for dict in json_list]

        # Map inserts to dictionary. Key: quadrant, Value: inserts list
        inserts_map = defaultdict(list)
        for insert in inserts:
            inserts_map[Quadrant[insert.quadrant]].append(insert)

        # Create stars map
        stars: Dict[Quadrant: Table] = {}
        for quadrant, quadrant_inserts in inserts_map.items():
            x_centroids = []
            y_centroids = []
            fluxes = []
            ids = []
            longitudes = []
            latitudes = []
            for insert in quadrant_inserts:
                x_centroids.append(insert.x_insert + insert.x_trim / 2)
                y_centroids.append(insert.y_insert + insert.y_trim / 2)
                fluxes.append(math.pow(10, (19 - insert.magnitude) / 2.5))
                ids.append(insert.id)
                longitudes.append(insert.longitude)
                latitudes.append(insert.latitude)
            table = Table({
                X_CENTROID_COLUMN: x_centroids,
                Y_CENTROID_COLUMN: y_centroids,
                ASTROMETRY_FLUX_COLUMN: fluxes,
                'id': ids,
                'longitude': longitudes,
                'latitude': latitudes,
            })
            stars[quadrant] = table
        return cls(fits.Header(), stars)

    def left_join(self, right: VisStars):
        """
        Per quadrant table left join to another VisStars.
        The join table is converted to BinTable and back to Table again to convert mixin columns into regular Column.
        :param right: The donor VisStars instance from where additional data will be taken
        """
        for quadrant in set(self.stars.keys()).intersection(right.stars.keys()):
            left_table = self.stars[quadrant]
            right_table = right.stars[quadrant]
            join_table = join(left_table, right_table, join_type='left')
            join_table = Table.read(fits.BinTableHDU(join_table))
            self.stars[quadrant] = join_table

    def compute_image_quality_metrics_full_fpa(
            self, vis_cutouts: VisCutouts, vis_processing_config: VisProcessingConfig):
        """
        Compute the image quality metrics for each star in all quadrants.
        VisProcessingConfig specifies which diagnostics are run.
        Pixel based diagnostics are always evaluated.
        Sky based diagnostics are only evaluated if local plate scale info is available.
        :param vis_cutouts: The VIS cutouts.
        :param vis_processing_config: The VIS processing configuration.
        """
        # Distribute vis_cutouts per quadrant to make multithread pickle easier
        vis_cutout_images_per_quadrant = defaultdict(dict)
        for key, cutout in vis_cutouts.cutouts.items():
            quadrant, _, _ = parse_cutout_extame(key)
            vis_cutout_images_per_quadrant[quadrant][key] = cutout
        vis_cutout_masks_per_quadrant = defaultdict(dict)
        for key, mask in vis_cutouts.masks.items():
            quadrant, _, _ = parse_cutout_mask_extame(key)
            vis_cutout_masks_per_quadrant[quadrant][key] = mask
        vis_cutouts_per_quadrant = {quadrant: VisCutouts(vis_cutouts.primary_header, this_cutouts,
                                                         vis_cutout_masks_per_quadrant[quadrant])
                                    for quadrant, this_cutouts in vis_cutout_images_per_quadrant.items()}

        with multiprocessing.Pool(min(NUM_CORES, vis_processing_config.max_threads)) as pool:
            parameters = [[quadrant, self.stars[quadrant], vis_cutouts_per_quadrant[quadrant], vis_processing_config]
                          for quadrant in self.stars.keys()]
            tables = pool.starmap(compute_image_quality_metrics_quadrant, parameters)
        self.stars.update({quadrant: table for quadrant, table in zip(self.stars.keys(), tables)})

    def astrometric_reduction(self, path_prefix: str, astrometric_indices_config_file: str,
                              max_threads: int = NUM_CORES) -> Iterable[int]:
        """
        Carry out a full astrometric reduction. Intermediate astrometry.net files will be created.
        When a solution is available, WCS headers will be created, plate scales estimated and a left join with
        the astrometry.net cross-match corr table be done.
        :param path_prefix: Combination of folder plus file name prefix for astrometry.net intermediate files
        :param astrometric_indices_config_file: the file with the indices to use,
         typically Gaia EDR3 in the self-calibration field or FGS
        :param max_threads: The maximum number of parallel threads.
        :return: the success flags from the execution of astrometric_reduction_one_ccd
        """
        # Write per-CCD files and carry out astrometric reduction
        self.to_fits_per_ccd(path_prefix)
        status = astrometric_reduction_full_fpa(path_prefix, astrometric_indices_config_file, max_threads)

        # If at least one CCD produced astrometry
        if len(status) > np.count_nonzero(status):

            # Load WCS astrometry copy to header and compute plate scales
            vis_astrometry = VisAstrometry.from_fits_per_ccd(path_prefix)
            vis_astrometry.to_vis_stars(self)
            vis_astrometry.vis_stars_compute_sky_coordinates(self)
            vis_astrometry.vis_stars_compute_plate_scales(self)

            # Load CCD coordinates and left join with astrometry.net cross-match tables
            vis_astrometry_stars = VisStars.from_fits_per_ccd(path_prefix)
            self.left_join(vis_astrometry_stars)
            vis_astrometry_corr = VisStars.from_fits_per_ccd(path_prefix, ASTROMETRY_NET_XMATCH_FITS_FILE_EXTENSION)
            self.left_join(vis_astrometry_corr)

        # Return astrometry.net fitting status flags
        return status


def compute_image_quality_metrics_quadrant(quadrant: Quadrant, table: Table, vis_cutouts: VisCutouts,
                                           vis_processing_config: VisProcessingConfig) -> Table:
    """
    Compute the image quality metrics for each star in a given quadrant.
    VisProcessingConfig specifies which diagnostics are run.
    Pixel based diagnostics are always evaluated.
    Most sky based diagnostics are only evaluated if local plate scale info is available.
    On-sky aperture photometry needs valid WCS headers in the corresponding stars table.
    :param quadrant: The VIS CCD quadrant.
    :param table: The sources table. It may have local plate scales and WCS astrometry metadata.
    :param vis_cutouts: The VIS cutouts.
    :param vis_processing_config: The VIS processing configuration.
    """
    # TODO distinguish between 0- and 1-based x coordinates: e.g, astrometry.net (1) and cutouts (0)

    # Have local plate scales been estimated (needed for most sky diagnostics)?
    has_plate_scales = set((PLATE_SCALE_MAJOR_COLUMN, PLATE_SCALE_MINOR_COLUMN, PLATE_SCALE_ANGLE_COLUMN)
                           ).issubset(table.keys())

    # Are there WCS entries in the table header?
    # TODO can on-sky aperture photometry be done using local plate scales?
    has_celestial_wcs = set(WCS_HEADER_KEYWORDS).issubset(table.meta.keys())

    # Crate new columns in summary table
    if vis_processing_config.compute_moments_pix:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=R2_PIX_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=ELLIPTICITY_PIX_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=X_MOMENT_PIX_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=Y_MOMENT_PIX_COLUMN),
        ])
    if vis_processing_config.compute_fwhm:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=FWHM_PIX_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=X_GAUSSIAN_PIX_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=Y_GAUSSIAN_PIX_COLUMN),
        ])
    if vis_processing_config.compute_fwhm_pixelated:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=FWHM_PIXELATED_PIX_COLUMN),
        ])
    if vis_processing_config.compute_cramer_rao or vis_processing_config.compute_cramer_rao_no_ron:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=EER100_PIX_FLUX_COLUMN),
        ])
    if vis_processing_config.compute_cramer_rao:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=X_CRAMER_RAO_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=Y_CRAMER_RAO_COLUMN),
        ])
    if vis_processing_config.compute_cramer_rao_no_ron:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=X_CRAMER_RAO_NO_RON_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=Y_CRAMER_RAO_NO_RON_COLUMN),
        ])
    if vis_processing_config.compute_eer_pix:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=EER50_PIX_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=EER80_PIX_COLUMN),
        ])
    if vis_processing_config.compute_aperture_photometry_pix:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=AP_PHOTO_PIX_COLUMN),
        ])
    if vis_processing_config.compute_moments_sky and vis_processing_config.compute_moments_pix \
            and has_plate_scales:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=R2_SKY_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=ELLIPTICITY_SKY_COLUMN),
        ])
    if vis_processing_config.compute_fwhm and has_plate_scales:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=FWHM_SKY_COLUMN),
        ])
    if vis_processing_config.compute_fwhm_pixelated and has_plate_scales:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=FWHM_PIXELATED_SKY_COLUMN),
        ])
    if vis_processing_config.compute_eer_sky and has_plate_scales:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=EER50_SKY_COLUMN),
            Column(np.zeros(len(table), dtype=np.float32), name=EER80_SKY_COLUMN),
        ])
    if vis_processing_config.compute_aperture_photometry_sky and has_celestial_wcs:
        table.add_columns([
            Column(np.zeros(len(table), dtype=np.float32), name=AP_PHOTO_SKY_COLUMN),
        ])

    # Loop on sources
    for source in table:

        # TODO optimize fitting thresholds and maximum number of iterations

        # Get cutout
        cutout_name = get_cutout_extname(quadrant, source[X_CENTROID_COLUMN], source[Y_CENTROID_COLUMN])
        cutout_hdu = vis_cutouts.cutouts[cutout_name]

        # Subtract background from sky annulus in pixels
        annulus_aperture = CircularAnnulus(
            [cutout_hdu.header[X_CENTER_DESCRIPTOR] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
             cutout_hdu.header[Y_CENTER_DESCRIPTOR] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR], ],
            r_in=BACKGROUND_ANNULUS_INNER_RADIUS, r_out=BACKGROUND_ANNULUS_OUTER_RADIUS)
        annulus_mask = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_mask.multiply(cutout_hdu.data)
        annulus_data_1d = annulus_data[annulus_mask.data > 0]
        mean_sigclip, _, _ = sigma_clipped_stats(annulus_data_1d)
        cutout_hdu.data -= mean_sigclip

        # Pixel-based image quality metrics
        if vis_processing_config.compute_moments_pix:
            source[R2_PIX_COLUMN], source[ELLIPTICITY_PIX_COLUMN], x_moment_cutout_pix, y_moment_cutout_pix \
                = get_r2_ellipticity(
                cutout_hdu.data,
                cutout_hdu.header[X_CENTER_DESCRIPTOR] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                cutout_hdu.header[Y_CENTER_DESCRIPTOR] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR],
                vis_processing_config.quadrupole_moment_sigma_threshold)
            source[X_MOMENT_PIX_COLUMN] = x_moment_cutout_pix + cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR]
            source[Y_MOMENT_PIX_COLUMN] = y_moment_cutout_pix + cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR]
        if vis_processing_config.compute_fwhm:
            _, x_gauss, y_gauss, a_regular, b_regular, position_angle_regular = get_gaussian_fit(
                    cutout_hdu.data,
                    cutout_hdu.header[X_CENTER_DESCRIPTOR] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                    cutout_hdu.header[Y_CENTER_DESCRIPTOR] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR])
            source[X_GAUSSIAN_PIX_COLUMN] = x_gauss + cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR]
            source[Y_GAUSSIAN_PIX_COLUMN] = y_gauss + cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR]
            source[FWHM_PIX_COLUMN] = get_gaussian_fwhm_2d_average(a_regular, b_regular)
        if vis_processing_config.compute_fwhm_pixelated:
            _, _, _, a_pixelated, b_pixelated, position_angle_pixelated = \
                get_gaussian_fit_pixelated(
                    cutout_hdu.data,
                    cutout_hdu.header[X_CENTER_DESCRIPTOR] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                    cutout_hdu.header[Y_CENTER_DESCRIPTOR] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR])
            source[FWHM_PIXELATED_PIX_COLUMN] = get_gaussian_fwhm_2d_average(a_pixelated, b_pixelated)
        if vis_processing_config.compute_cramer_rao or vis_processing_config.compute_cramer_rao_no_ron:
            source[EER100_PIX_FLUX_COLUMN] = get_eer100_pix_flux(
                cutout_hdu.data,
                source[X_CENTROID_COLUMN] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],  # weak: X_MOMENT_PIX_COLUMN
                source[Y_CENTROID_COLUMN] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR])  # weak: Y_MOMENT_PIX_COLUMN
        if vis_processing_config.compute_cramer_rao:
            source[X_CRAMER_RAO_COLUMN], source[Y_CRAMER_RAO_COLUMN], _ \
                = get_cutout_pix_metrics_cramer_rao(cutout_hdu, source[EER100_PIX_FLUX_COLUMN])
        if vis_processing_config.compute_cramer_rao_no_ron:
            source[X_CRAMER_RAO_NO_RON_COLUMN], source[Y_CRAMER_RAO_NO_RON_COLUMN], _ \
                = get_cutout_pix_metrics_cramer_rao_no_ron(cutout_hdu, source[EER100_PIX_FLUX_COLUMN])
        if vis_processing_config.compute_eer_pix:
            if vis_processing_config.compute_fwhm:
                x_center = source[X_GAUSSIAN_PIX_COLUMN]
                y_center = source[Y_GAUSSIAN_PIX_COLUMN]
            else:
                x_center = source[X_CENTROID_COLUMN]
                y_center = source[Y_CENTROID_COLUMN]
            source[EER50_PIX_COLUMN] = get_eer_pix(cutout_hdu.data,
                                                   x_center - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                                                   y_center - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR],
                                                   0.5, EER100_PIX_RADIUS, vis_processing_config.eer_recenter)
            source[EER80_PIX_COLUMN] = get_eer_pix(cutout_hdu.data,
                                                   x_center - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                                                   y_center - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR],
                                                   0.8, EER100_PIX_RADIUS, vis_processing_config.eer_recenter)
        if vis_processing_config.compute_aperture_photometry_pix:
            source[AP_PHOTO_PIX_COLUMN] = get_cutout_pix_metrics_aperture_photometry(cutout_hdu)

        # Sky-based image quality metrics
        if vis_processing_config.compute_moments_sky and vis_processing_config.compute_moments_pix \
                and has_plate_scales:
            source[R2_SKY_COLUMN], source[ELLIPTICITY_SKY_COLUMN], _, _ \
                = get_r2_ellipticity(
                cutout_hdu.data,
                source[X_MOMENT_PIX_COLUMN] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                source[Y_MOMENT_PIX_COLUMN] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR],
                source[PLATE_SCALE_MAJOR_COLUMN], source[PLATE_SCALE_MINOR_COLUMN],
                source[PLATE_SCALE_ANGLE_COLUMN], QUADRUPOLE_MOMENT_SIGMA_SKY,
                vis_processing_config.quadrupole_moment_sigma_threshold)
        if vis_processing_config.compute_fwhm and has_plate_scales:
            source[FWHM_SKY_COLUMN] = get_sky_ellipse_fwhm_2d_average(
                a_regular, b_regular, source[PLATE_SCALE_MAJOR_COLUMN], source[PLATE_SCALE_MINOR_COLUMN],
                source[PLATE_SCALE_ANGLE_COLUMN] - position_angle_regular)
        if vis_processing_config.compute_fwhm_pixelated and has_plate_scales:
            source[FWHM_PIXELATED_SKY_COLUMN] = get_sky_ellipse_fwhm_2d_average(
                a_pixelated, b_pixelated, source[PLATE_SCALE_MAJOR_COLUMN], source[PLATE_SCALE_MINOR_COLUMN],
                source[PLATE_SCALE_ANGLE_COLUMN] - position_angle_pixelated)
        if vis_processing_config.compute_eer_sky and has_plate_scales:
            source[EER50_SKY_COLUMN] = get_eer_sky(
                cutout_hdu.data,
                source[X_CENTROID_COLUMN] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                source[Y_CENTROID_COLUMN] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR],
                source[PLATE_SCALE_MAJOR_COLUMN], source[PLATE_SCALE_MINOR_COLUMN],
                source[PLATE_SCALE_ANGLE_COLUMN], 0.5, EER100_SKY_RADIUS, vis_processing_config.eer_recenter)
            source[EER80_SKY_COLUMN] = get_eer_sky(
                cutout_hdu.data,
                source[X_CENTROID_COLUMN] - cutout_hdu.header[X_MIN_CUTOUT_DESCRIPTOR],
                source[Y_CENTROID_COLUMN] - cutout_hdu.header[Y_MIN_CUTOUT_DESCRIPTOR],
                source[PLATE_SCALE_MAJOR_COLUMN], source[PLATE_SCALE_MINOR_COLUMN],
                source[PLATE_SCALE_ANGLE_COLUMN], 0.8, EER100_SKY_RADIUS, vis_processing_config.eer_recenter)
        if vis_processing_config.compute_aperture_photometry_sky and has_celestial_wcs:
            source[AP_PHOTO_SKY_COLUMN] = get_cutout_sky_metrics_aperture_photometry(cutout_hdu, table.meta)
    return table


def get_fwhm(vis_hdu_list: VisHDUList, vis_processing_config: VisProcessingConfig) -> float:
    """
    Redetermine the FWHM more appropriate for a given VIS image
    :param vis_hdu_list: The input VIS image
    :param vis_processing_config_file: The VIS processing configuration file
    :return: The refined FWHM (pix) and the number of quadrants used
    """
    vis_hdu_list_reduced = vis_hdu_list.random_sample(FWHM_DETERMINATION_MAX_QUADRANTS)
    vis_stars_reduced = VisStars.from_starfinderbase(
        vis_hdu_list_reduced, vis_processing_config, starfinderbase=DAOStarFinder)
    vis_processing_config_reduced = replace(
        vis_processing_config,
        compute_moments_pix=False,
        compute_fwhm=True,
        compute_fwhm_pixelated=False,
        compute_cramer_rao=False,
        compute_cramer_rao_no_ron=False,
        compute_eer_pix=False,
        compute_aperture_photometry_pix=False,
        compute_moments_sky=False,
        compute_eer_sky=False,
        compute_aperture_photometry_sky=False
    )
    vis_cutouts_reduced = VisCutouts.from_vis_hdu_list(
        vis_hdu_list_reduced, vis_stars_reduced, vis_processing_config_reduced)
    vis_stars_reduced.compute_image_quality_metrics_full_fpa(vis_cutouts_reduced, vis_processing_config_reduced)
    fwhms = [np.nanmedian(table[FWHM_PIX_COLUMN]) for table in vis_stars_reduced.stars.values()]
    fwhm = np.median(fwhms)
    return fwhm, len(fwhms)


@dataclass()
class VisAstrometry:
    """
    Stores the VIS astrometric solution for all quadrants where it is available
    """
    wcs: Dict[Quadrant, WCS]

    @classmethod
    def from_fits_per_ccd(cls, prefix: str, extension: str = ASTROMETRY_NET_WCS_FITS_FILE_EXTENSION):
        """
        Load WCS stored in different files for each CCD produced for or by an astrometry.net run
        Non-celestial WCS are silently discarded (e.g. pixel based).
        Different CCD pixel offests are applied for each quadrant.
        :param prefix: The output CCD fits files prefix
        :param extension: The output CCD fits file extension (e.g. .wcs)
        """
        wcs_per_quadrant = {}
        for input_file in get_vis_stars_ccd_file_names(prefix, extension):
            if os.path.exists(input_file):
                _, ccd = decode_vis_stars_ccd_file_name(input_file, extension)
                wcs_ccd = WCS(input_file)
                if wcs_ccd.has_celestial is False:
                    continue
                for quadrant in [ccd.value.bottom_left, ccd.value.bottom_right,
                                 ccd.value.top_right, ccd.value.top_left]:
                    wcs_header = wcs_ccd.to_header(relax=True).copy()
                    wcs_header[WCS_X_REFERENCE_PIXEL_KEYWORD] -= QUADRANT_TO_QUADRANT_LOCATION[quadrant].value.x_offset
                    wcs_header[WCS_Y_REFERENCE_PIXEL_KEYWORD] -= QUADRANT_TO_QUADRANT_LOCATION[quadrant].value.y_offset
                    wcs_quadrant = WCS(wcs_header)
                    wcs_per_quadrant[quadrant] = wcs_quadrant
        return cls(wcs_per_quadrant)

    @classmethod
    def from_vis_stars(cls, vis_stars: VisStars):
        """
        Load WCS information stored in a VisStars objects.
        Non-celestial WCS are silently discarded (e.g. pixel based).
        :param vis_stars: The VisStars object
        """
        wcs_per_quadrant = {}
        for quadrant, table in vis_stars.stars.items():
            wcs_quadrant = WCS(table.meta.copy())
            if wcs_quadrant.has_celestial is False:
                continue
            wcs_per_quadrant[quadrant] = wcs_quadrant
        return cls(wcs_per_quadrant)

    def to_vis_stars(self, vis_stars: VisStars):
        """
        Copy astrometric solution to a VisStars table metadata.
        Only quadrant matches from input VisAstrometry to output VisStars are considered.
        :param vis_stars: The VisStars objects to be upadated with WCS info.
        """
        quadrants = set(self.wcs.keys()).intersection(vis_stars.stars.keys())
        for quadrant in quadrants:
            vis_stars.stars[quadrant].meta.update(self.wcs[quadrant].to_header(relax=True))

    def vis_stars_compute_sky_coordinates(self, vis_stars: VisStars):
        """
        Compute sky coordinates for local plate scales for all sources in a VisStar object
        Only quadrant matches from input VisAstrometry to output VisStars are considered.
        :param vis_stars: The VisStars objects to be updated with sky coordinates.
        """
        quadrants = set(self.wcs.keys()).intersection(vis_stars.stars.keys())
        for quadrant in quadrants:

            # Get stars and WCS astrometry
            stars = vis_stars.stars[quadrant]
            wcs = self.wcs[quadrant]

            ra, dec = wcs.all_pix2world(stars[X_CENTROID_COLUMN], stars[Y_CENTROID_COLUMN], 1)
            stars[RA_COLUMN] = ra
            stars[DEC_COLUMN] = dec

    def vis_stars_compute_plate_scales(self, vis_stars: VisStars):
        """
        Compute local plate scales for all sources in a VisStar object
        Only quadrant matches from input VisAstrometry to output VisStars are considered.
        :param vis_stars: The VisStars objects to be updated with local plate scales.
        """
        quadrants = set(self.wcs.keys()).intersection(vis_stars.stars.keys())
        for quadrant in quadrants:

            # Get stars and WCS astrometry
            stars = vis_stars.stars[quadrant]
            wcs = self.wcs[quadrant]

            # Determine and store plate scale for each object
            stars[PLATE_SCALE_MAJOR_COLUMN] = math.nan
            stars[PLATE_SCALE_MINOR_COLUMN] = math.nan
            stars[PLATE_SCALE_ANGLE_COLUMN] = math.nan
            for row in range(len(stars)):
                stars[PLATE_SCALE_MAJOR_COLUMN][row], \
                stars[PLATE_SCALE_MINOR_COLUMN][row], \
                stars[PLATE_SCALE_ANGLE_COLUMN][row] \
                    = get_pixel_to_sky_plate_scale_ellipse(
                    wcs, stars[X_CENTROID_COLUMN][row], stars[Y_CENTROID_COLUMN][row])


def astrometric_reduction_one_ccd(input_file: str, astrometric_indices_config_file: str) -> int:
    """
    This method determines an astrometric calibration to the stack of sources detected by VisProcessQuadrants
    a standalone version of astrometry.net is used. The solutions are computed at the CCD level, joining the data
    for each four quadrants to improve robustness.
     astrometric solution
    :param input_file: The input file. It must end with the ".fits" extension
    :param astrometric_indices_config_file: the file with the indices to use,
     typically Gaia EDR3 in the self-calibration field or FGS
    :return success flag: 0 for a successful astrometric reduction, 1 otherwise
    """

    # Call solve-field astrometry.net external shell commmand
    arguments = ['--backend-config', astrometric_indices_config_file,
                 '--width', str(2 * QUADRANT_NAXIS_1_TRIMMED),
                 '--height', str(2 * QUADRANT_NAXIS_2_TRIMMED + CHARGE_INJECTION_LINES),
                 '--x-column', str(ASTROMETRY_X_CENTROID_CCD_COLUMN),
                 '--y-column', str(ASTROMETRY_Y_CENTROID_CCD_COLUMN),
                 '-s', str(ASTROMETRY_FLUX_COLUMN),
                 '--scale-units', 'arcsecperpix',
                 '--scale-low', str(0.09),
                 '--scale-high', str(0.11),
                 '--overwrite',
                 '--tag-all',
                 '--no-plots',
                 f'{input_file}']
    process = subprocess.run(['solve-field', *arguments],
                             stdout=subprocess.PIPE,
                             universal_newlines=True,
                             check=True)

    # Return success flag
    if os.path.isfile(input_file.replace('.fits', '') + '.solved'):
        return 0
    else:
        return 1


def astrometric_reduction_full_fpa(
        prefix: str, astrometric_indices_config_file: str, max_threads: int = NUM_CORES) -> Iterable[int]:
    """
    Parallel astrometric reduction of a bunch of CCDs
    File I/O is used because astrometry.net does not currently have a native Python wrapper.
    :param prefix: The input CCD fits files prefix
    :param astrometric_indices_config_file: the file with the indices to use,
     typically Gaia EDR3 in the self-calibration field or FGS
    :param max_threads: The maximum number of parallel threads.
    :return: the success flags from the execution of astrometric_reduction_one_ccd
    """
    input_files = [input_file for input_file in get_vis_stars_ccd_file_names(prefix) if os.path.exists(input_file)]
    parameters = [[input_file, astrometric_indices_config_file] for input_file in input_files]
    with multiprocessing.Pool(min(NUM_CORES, max_threads)) as pool:
        return pool.starmap(astrometric_reduction_one_ccd, parameters)


@unique
class FileProcessingStep(Enum):
    """ Defines the file processing steps and the associated file suffixes, when appropriate."""
    # TODO Use or remove values?
    CCD_PROCESSING = auto()
    COSMICS_CLEANING = auto()
    FWHM_DETERMINATION = auto()
    SOURCES_DETECTION = auto()
    ASTROMETRY = auto()
    IMAGE_QUALITY = auto()
    MOSAIC = auto()
    MOSAIC_CLEAN = auto()


@dataclass()
class VisFileProcessor:
    """
    File-driven VIS images processor.
    It is based on a pair of I/O folders and a bunch of auxiliary variables.
    Output might not be written to disk if needed.
    """
    input_folder: str
    """The input folder with Level 1 processed frames (or their averages)"""
    output_folder: str
    """The output folder where all other files are stored"""
    prefix: str = None
    """The prefix common for all fits files associated with a given Level 1 frame"""
    astrometric_indices_config_file: str = None
    """The config file pointing to astrometry.net catalogues"""
    vis_hdu_list: VisHDUList = None
    """
    Processing buffer for different processing modules. A level 1 frame, an average of them or a CCD processed output
    """
    vis_stars: VisStars = None
    """Processing buffer for different processing modules"""
    vis_cutouts: VisCutouts = None
    """Processing buffer for different processing modules"""
    astrometric_processing_flags = None
    """Per-quadrant astrometry.net processing flags"""
    vis_processing_config: VisProcessingConfig = None
    vis_ccd_processor: VisCcdProcessor = None
    write_output_files: bool = True
    """If False, do not write files to disk"""

    def _vis_hdu_list_from_fits(self, input_folder: str = None, suffix: str = '') -> int:
        """
        Load VisHduList. Default place: input folder. Default type: Level 1 frame
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        if input_folder is None:
            input_folder = self.input_folder
        input_file = os.path.join(input_folder, f'{self.prefix}{suffix}{FITS_FILE_EXTENSION}')
        self.vis_hdu_list = VisHDUList.from_fits(input_file)
        return time.time_ns() - now

    def _vis_stars_from_fits(self, suffix: str = SOURCES_FILE_SUFFIX) -> int:
        """
        Load VisStars. Default type: detected sources
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        input_file = os.path.join(self.output_folder, f'{self.prefix}{suffix}{FITS_FILE_EXTENSION}')
        self.vis_stars = VisStars.from_fits(input_file)
        return time.time_ns() - now

    def _vis_cutouts_from_fits(self) -> int:
        """
        Load VisCutouts
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        self.vis_cutouts = VisCutouts.from_fits(os.path.join(
            self.output_folder, f'{self.prefix}{CUTOUT_FILE_SUFFIX}{FITS_FILE_EXTENSION}'))
        return time.time_ns() - now

    def load_files(self, file_processing_step: FileProcessingStep):
        """
        Load default input files needed for a given file processing steps.
        :param file_processing_step: File processing step
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        if file_processing_step is FileProcessingStep.CCD_PROCESSING:
            self._vis_hdu_list_from_fits()
            self.vis_stars = None
            self.vis_cutouts = None
        elif file_processing_step is FileProcessingStep.COSMICS_CLEANING:
            self._vis_hdu_list_from_fits(self.output_folder, CCD_PROCESSED_FILE_SUFFIX)
            self.vis_stars = None
            self.vis_cutouts = None
        elif file_processing_step is FileProcessingStep.FWHM_DETERMINATION:
            self._vis_hdu_list_from_fits(self.output_folder, COSMICS_CLEANED_FILE_SUFFIX)
            self.vis_stars = None
            self.vis_cutouts = None
        elif file_processing_step is FileProcessingStep.SOURCES_DETECTION:
            self._vis_hdu_list_from_fits(self.output_folder, COSMICS_CLEANED_FILE_SUFFIX)
            self.vis_stars = None
            self.vis_cutouts = None
        elif file_processing_step is FileProcessingStep.ASTROMETRY:
            self.vis_hdu_list = None
            self._vis_stars_from_fits()
            self._vis_cutouts_from_fits()
        elif file_processing_step is FileProcessingStep.IMAGE_QUALITY:
            self.vis_hdu_list = None
            self._vis_stars_from_fits(ASTROMETRY_FILE_SUFFIX)
            self._vis_cutouts_from_fits()
        elif file_processing_step is FileProcessingStep.MOSAIC:
            self._vis_hdu_list_from_fits(self.output_folder, CCD_PROCESSED_FILE_SUFFIX)
            self.vis_stars = None
            self.vis_cutouts = None
        elif file_processing_step is FileProcessingStep.MOSAIC_CLEAN:
            self._vis_hdu_list_from_fits(self.output_folder, COSMICS_CLEANED_FILE_SUFFIX)
            self.vis_stars = None
            self.vis_cutouts = None
        else:
            raise TypeError(f'Unsupported file processing step: {file_processing_step}')
        return time.time_ns() - now

    def _ccd_processing(self):
        self.vis_ccd_processor.process(self.vis_hdu_list)
        self.vis_stars = None
        self.vis_cutouts = None
        self.vis_hdu_list.hdu_list[0].header['V_CCD'] = (
            VERSION, f'Software version for: {FileProcessingStep.CCD_PROCESSING.name}')
        self.vis_hdu_list.hdu_list[0].header['C_CCD'] = (
            self.vis_processing_config.file_name, f'Configuration file for: {FileProcessingStep.CCD_PROCESSING.name}')
        if self.write_output_files:
            output_file = os.path.join(
                self.output_folder, f'{self.prefix}{CCD_PROCESSED_FILE_SUFFIX}{FITS_FILE_EXTENSION}')
            self.vis_hdu_list.hdu_list.writeto(output_file, overwrite=True)

    def _cosmics_cleaning(self):
        clean_cosmics_vis_hdu_list(self.vis_hdu_list, self.vis_processing_config)
        self.vis_stars = None
        self.vis_cutouts = None
        self.vis_hdu_list.hdu_list[0].header['V_COSMIC'] = (
            VERSION, f'Software version for: {FileProcessingStep.COSMICS_CLEANING.name}')
        self.vis_hdu_list.hdu_list[0].header['C_COSMIC'] = (
            self.vis_processing_config.file_name, f'Configuration file for: {FileProcessingStep.COSMICS_CLEANING.name}')
        if self.write_output_files:
            output_file = os.path.join(
                self.output_folder, f'{self.prefix}{COSMICS_CLEANED_FILE_SUFFIX}{FITS_FILE_EXTENSION}')
            self.vis_hdu_list.hdu_list.writeto(output_file, overwrite=True)

    def _fwhm_determination(self):
        fwhm, n_fwhm = get_fwhm(self.vis_hdu_list, self.vis_processing_config)
        fwhm *= self.vis_processing_config.determined_fwhm_scale_factor
        logging.info(f'Refined and scaled FWHM: {fwhm:.3f} quadrants: {n_fwhm}')
        self.vis_processing_config.daofind_fwhm = fwhm
        self.vis_hdu_list.hdu_list[0].header['FWHM'] = (fwhm, 'Refined and scaled median FWHM (pix)')
        self.vis_hdu_list.hdu_list[0].header['N_FWHM'] = (n_fwhm, 'Quadrants used to refine the FWHM')
        self.vis_hdu_list.hdu_list[0].header['V_FWHM'] = (
            VERSION, f'Software version for: {FileProcessingStep.FWHM_DETERMINATION.name}')
        self.vis_hdu_list.hdu_list[0].header['C_SRCS'] = (
            self.vis_processing_config.file_name,
            f'Configuration file for: {FileProcessingStep.FWHM_DETERMINATION.name}')

    def _sources_detection(self):
        #self.vis_hdu_list.update_hdu_list()
        # with io.BytesIO() as filelike:
        #     self.vis_hdu_list.hdu_list.writeto(filelike)
        #     filelike.seek(0)
        #     self.vis_hdu_list = VisHDUList.from_fits(filelike)
        self.vis_stars = VisStars.from_starfinderbase(self.vis_hdu_list, self.vis_processing_config)
        self.vis_cutouts = VisCutouts.from_vis_hdu_list(self.vis_hdu_list, self.vis_stars, self.vis_processing_config)
        self.vis_stars.primary_header['V_SRCS'] = (
            VERSION, f'Software version for: {FileProcessingStep.SOURCES_DETECTION.name}')
        self.vis_cutouts.primary_header['V_SRCS'] = (
            VERSION, f'Software version for: {FileProcessingStep.SOURCES_DETECTION.name}')
        self.vis_stars.primary_header['C_SRCS'] = (
            self.vis_processing_config.file_name, f'Configuration file for: {FileProcessingStep.SOURCES_DETECTION.name}')
        self.vis_cutouts.primary_header['C_SRCS'] = (
            self.vis_processing_config.file_name, f'Configuration file for: {FileProcessingStep.SOURCES_DETECTION.name}')
        if self.write_output_files:
            output_file = os.path.join(self.output_folder, f'{self.prefix}{SOURCES_FILE_SUFFIX}{FITS_FILE_EXTENSION}')
            self.vis_stars.to_fits(output_file)
            output_file = os.path.join(self.output_folder, f'{self.prefix}{CUTOUT_FILE_SUFFIX}{FITS_FILE_EXTENSION}')
            self.vis_cutouts.to_fits(output_file)

    def _astrometric_reduction(self):
        self.astrometric_processing_flags = self.vis_stars.astrometric_reduction(
            os.path.join(self.output_folder, self.prefix), self.astrometric_indices_config_file,
            self.vis_processing_config.max_threads)
        self.vis_stars.primary_header['V_ASTROM'] = (
            VERSION, f'Software version for: {FileProcessingStep.ASTROMETRY.name}')
        self.vis_stars.primary_header['C_ASTROM'] = (
            self.vis_processing_config.file_name, f'Configuration file for: {FileProcessingStep.ASTROMETRY.name}')
        self.vis_stars.primary_header['I_ASTROM'] = (
            os.path.basename(self.astrometric_indices_config_file), f'Astrometry.net backend file')
        if self.write_output_files:
            output_file = os.path.join(
                self.output_folder, f'{self.prefix}{ASTROMETRY_FILE_SUFFIX}{FITS_FILE_EXTENSION}')
            self.vis_stars.to_fits(output_file)

    def _image_quality_diagnostics(self):
        self.vis_stars.compute_image_quality_metrics_full_fpa(self.vis_cutouts, self.vis_processing_config)
        self.vis_stars.primary_header['V_IQ'] = (
            VERSION, f'Software version for: {FileProcessingStep.IMAGE_QUALITY.name}')
        self.vis_stars.primary_header['C_IQ'] = (
            self.vis_processing_config.file_name, f'Configuration file for: {FileProcessingStep.IMAGE_QUALITY.name}')
        if self.write_output_files:
            output_file = os.path.join(
                self.output_folder, f'{self.prefix}{IMAGE_QUALITY_FILE_SUFFIX}{FITS_FILE_EXTENSION}')
            self.vis_stars.to_fits(output_file)

    def _mosaic(self):
        output_hdu = mosaic_vis_hdu_list(self.vis_hdu_list)
        self.vis_stars = None
        self.vis_cutouts = None

        if self.write_output_files:
            output_file = os.path.join(
                self.output_folder, f'{self.prefix}{MOSAIC_FILE_SUFFIX}{FITS_FILE_EXTENSION}')
            output_hdu.writeto(output_file, overwrite=True)


    def process(self, file_processing_step: FileProcessingStep) -> int:
        """
        Apply one processing step. Internal buffers will be overwritten. Output files might be generated.
        :param file_processing_step: File processing step
        :return: Elapsed time (ns)
        """
        now = time.time_ns()
        if file_processing_step is FileProcessingStep.CCD_PROCESSING:
            self._ccd_processing()
        elif file_processing_step is FileProcessingStep.COSMICS_CLEANING:
            self._cosmics_cleaning()
        elif file_processing_step is FileProcessingStep.FWHM_DETERMINATION:
            self._fwhm_determination()
        elif file_processing_step is FileProcessingStep.SOURCES_DETECTION:
            self._sources_detection()
        elif file_processing_step is FileProcessingStep.ASTROMETRY:
            self._astrometric_reduction()
        elif file_processing_step is FileProcessingStep.IMAGE_QUALITY:
            self._image_quality_diagnostics()
        elif file_processing_step is FileProcessingStep.MOSAIC or file_processing_step is FileProcessingStep.MOSAIC_CLEAN:
            self._mosaic()
        else:
            raise TypeError(f'Unsupported file processing step: {file_processing_step}')
        return time.time_ns() - now


def process_vis_files(input_folder, output_folder, prefixes: Iterable[str],
                      file_processing_steps: Iterable[str], log_file, vis_processing_config_file: str,
                      astrometric_indices_config_file: str, master_dark=None, master_flat=None,
                      write_output_files: bool = True):
    """
    Process all VIS level 1 frames in a given folder. Useful for batch processing
    :param input_folder: The input folder. All "*.fits" files within will be processed
    :param output_folder: The output folder. Will be created, if needed.
                            Pre-existing files with the same name will be overwritten
    :param prefixes: The file prefixes to process
    :param file_processing_steps: The processing steps to apply. All of them if none
    :param log_file: The output log file. Entries will be appended
    :param vis_processing_config_file: The VIS processing configuration file
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
    logging.info('Instantiating VIS file processor')
    vis_file_processor = VisFileProcessor(
        input_folder, output_folder, astrometric_indices_config_file=astrometric_indices_config_file,
        write_output_files=write_output_files)
    logging.info('Reading VIS processing configuration file')
    vis_file_processor.vis_processing_config = VisProcessingConfig.from_json_file(vis_processing_config_file)
    if FileProcessingStep.CCD_PROCESSING in file_processing_steps:
        logging.info('Instantiating VIS CCD processor')
        now = time.time_ns()
        vis_file_processor.vis_ccd_processor = \
            VisCcdProcessor.from_files(master_dark, vis_processing_config_file, master_flat)
        processor_init_ns = time.time_ns() - now
        logging.info(f'VIS CCD processor instantiation time (ms): {processor_init_ns // 1000000}')
    else:
        logging.info('VIS CCD processor not instantiated')

    # Create output folder, if needed
    os.makedirs(output_folder, exist_ok=True)

    # Loop on prefixes and processing steps
    n_prefixes = len(prefixes)
    for i, prefix in enumerate(prefixes):
        vis_file_processor.prefix = prefix
        logging.info(f'Processing file name prefix {i + 1} out of {n_prefixes}: {vis_file_processor.prefix}')
        for j, file_processing_step in enumerate(file_processing_steps):
            if j == 0:
                logging.info(f'Loading files for initial processing step: {file_processing_step.name}')
                elapsed_ns = vis_file_processor.load_files(file_processing_step)
                logging.info(f'Elapsed ms: {elapsed_ns // 1000000}')
            logging.info(f'Processing.step: {file_processing_step.name}')
            elapsed_ns = vis_file_processor.process(file_processing_step)
            logging.info(f'Elapsed ms: {elapsed_ns // 1000000}')

    logging.info('End. All input file prefixes processed')


# TODO per-CCD files cleaning method? where?
#if False:  # Set to True to erase output files TODO use name getter to do better
#    for file_name in glob.glob(prefix + "*"):
#        if file_name.startswith(prefix):
#            print(f'Erasing: {file_name}')
#            os.remove(file_name)


def get_eer100_pix_flux(image: np.ndarray, x_center: float, y_center: float) -> float:
    """
    Estimates the total flux of an object for a circular aperture in pixel space with EER100_PIX_RADIUS radius.
    :param image: the background subctracted input image
    :param x_center: the x-axis center
    :param y_center: the x-axis center
    :return: 
    """
    if np.isnan(x_center) or np.isnan(y_center):
        return np.nan
    outer_aperture = CircularAperture((x_center, y_center), r=EER100_PIX_RADIUS)
    outer_photometry = aperture_photometry(image, outer_aperture)
    return outer_photometry[APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN][0]


def get_cutout_pix_metrics_cramer_rao(cutout: fits.ImageHDU, total_flux: float)\
        -> Tuple[float, float, float]:
    """
    One line encapsulation of Cramer-Rao lower bound astrometric centroding precision normalised by flux.
    A circular mask will be used, more adapted to the Euclid circular pupil. It assumes non-zero CCD read-out noise
    :param cutout: background subtracted cutout
    :param total_flux: the total stellar flux (e-)
    :return: (x_cramer_rao, y_cramer_rao, Cramer-Rao cutout electron count),
     Normalised astrometric precision in x, y axes (unit: pix) considering detector read-out noise
     and total electron count within the window (not used for the normalised estimate).
    """
    if total_flux <= 0:
        return np.nan, np.nan, np.nan
    y_min, y_max, x_min, x_max = get_cutout_numpy_trim_range(
        cutout.header[X_CENTER_DESCRIPTOR] - cutout.header[X_MIN_CUTOUT_DESCRIPTOR],
        cutout.header[Y_CENTER_DESCRIPTOR] - cutout.header[Y_MIN_CUTOUT_DESCRIPTOR],
        cutout.header['NAXIS1'], cutout.header['NAXIS2'],
        CRAMER_RAO_CUTOUT_SIZE, CRAMER_RAO_CUTOUT_SIZE)
    cramer_rao_cutout = cutout.data[y_min:y_max, x_min:x_max]
    mask = get_psf_circular_mask(*cramer_rao_cutout.shape)
    y_cramer_rao, x_cramer_rao = get_psf_centroid_location_precision(
        cramer_rao_cutout, cutout.header[READ_OUT_NOISE_DESCRIPTOR], 0, None, False, mask=mask)

    return x_cramer_rao * np.sqrt(total_flux), y_cramer_rao * np.sqrt(total_flux),\
           np.ma.array(cramer_rao_cutout, mask=mask, keep_mask=False).sum()


def get_cutout_pix_metrics_cramer_rao_no_ron(cutout: fits.ImageHDU, total_flux: float)\
        -> Tuple[float, float, float]:
    """
    One line encapsulation of Cramer-Rao lower bound astrometric centroding precision normalised by flux.
    A circular mask will be used, more adapted to the Euclid circular pupil. It assumes zero CCD read-out noise
    :param cutout: background subtracted cutout
    :param total_flux: the total stellar flux (e-)
    :return: (x_cramer_rao_no_ron, y_cramer_rao_no_ron, Cramer-Rao cutout electron count),
     Normalised astrometric precision in x, y axes (unit: pix) without considering detector read-out noise
     but total electron count within the window (not used for the normalised estimate).
    """
    y_min, y_max, x_min, x_max = get_cutout_numpy_trim_range(
        cutout.header[X_CENTER_DESCRIPTOR] - cutout.header[X_MIN_CUTOUT_DESCRIPTOR],
        cutout.header[Y_CENTER_DESCRIPTOR] - cutout.header[Y_MIN_CUTOUT_DESCRIPTOR],
        cutout.header['NAXIS1'], cutout.header['NAXIS2'],
        CRAMER_RAO_CUTOUT_SIZE, CRAMER_RAO_CUTOUT_SIZE)
    cramer_rao_cutout = cutout.data[y_min:y_max, x_min:x_max]
    mask = get_psf_circular_mask(*cramer_rao_cutout.shape)
    y_cramer_rao_no_ron, x_cramer_rao_no_ron = get_psf_centroid_location_precision(
        cramer_rao_cutout, 0, 0, None, False, mask=mask)

    return x_cramer_rao_no_ron * np.sqrt(total_flux), y_cramer_rao_no_ron * np.sqrt(total_flux), \
           np.ma.array(cramer_rao_cutout, mask=mask, keep_mask=False).sum()


def get_cutout_pix_metrics_aperture_photometry(cutout: fits.ImageHDU) -> float:
    """
    One line encapsulation of circular aperture photometry on a predefined pixel radius
    :param cutout: background subtracted cutout
    :return: instrumental magnitude: -2.5 * log10(total_electrons / exposure time)
    """
    aperture = CircularAperture([cutout.header[X_CENTER_DESCRIPTOR] - cutout.header[X_MIN_CUTOUT_DESCRIPTOR],
                                 cutout.header[Y_CENTER_DESCRIPTOR] - cutout.header[Y_MIN_CUTOUT_DESCRIPTOR],],
                                r=APERTURE_PHOTOMETRY_PIXEL_RADIUS)
    photometry = aperture_photometry(cutout.data, aperture)
    return -2.5 * np.log10(photometry[APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN][0] / cutout.header[EXPOSURE_TIME_DESCRIPTOR])


def get_cutout_sky_metrics_aperture_photometry(cutout: fits.ImageHDU, wcs_header: fits.Header) -> float:
    """
    One line encapsulation of circular aperture photometry on a predefined pixel radius
    :param cutout: background subtracted cutout
    :param wcs_header: the fits header with the World Coordinate System descriptors for the parent image
    :return: instrumental magnitude: -2.5 * log10(total_electrons / exposure time)
    """
    # TODO 0 or 1 based pixels?
    # Apply pixel offset and reconstruct WCS from header copy
    wcs_header_offset = wcs_header.copy()
    wcs_header_offset[WCS_X_REFERENCE_PIXEL_KEYWORD] -= cutout.header[X_MIN_CUTOUT_DESCRIPTOR]
    wcs_header_offset[WCS_Y_REFERENCE_PIXEL_KEYWORD] -= cutout.header[Y_MIN_CUTOUT_DESCRIPTOR]
    wcs = WCS(wcs_header_offset, relax=True)

    # Define sky aperture and do photometry
    ra, dec = wcs.all_pix2world([
        [cutout.header[X_CENTER_DESCRIPTOR] - cutout.header[X_MIN_CUTOUT_DESCRIPTOR],
         cutout.header[Y_CENTER_DESCRIPTOR] - cutout.header[Y_MIN_CUTOUT_DESCRIPTOR]]], 1)[0]
    sky_aperture = SkyCircularAperture(SkyCoord(ra, dec, unit=u.deg), APERTURE_PHOTOMETRY_PIXEL_SKY)
    photometry = aperture_photometry(cutout.data, sky_aperture, wcs=wcs)
    return -2.5 * np.log10(photometry[APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN][0]
                             / cutout.header[EXPOSURE_TIME_DESCRIPTOR])


def get_cutout_numpy_trim_range(x_center: float, y_center: float,
                                x_parent_size: int, y_parent_size: int,
                                x_cutout_size: int, y_cutout_size:int) -> Tuple[int, int, int, int]:
    """
    Determines the numpy array trim range for a cutout of an input CCDData around a given centroid.
    :param input_image: The input CCDData, typically a CCD quadrant
    :param x_center: x-axis object location (second array index)
    :param y_center: y-axis object location (first  array index)
    :param x_parent_size: x-axis parent image size (second array index)
    :param y_parent_size: y-axis parent image size (first  array index)
    :param x_cutout_size: x-axis cutout size (second array index)
    :param y_cutout_size: y-axis cutout size (first  array index)
    :return: (y_min, y_max, x_min, x_max) the numpy image cutout range
    """

    # Define cutout slice range
    x_min = int(round(x_center - x_cutout_size / 2.))
    y_min = int(round(y_center - y_cutout_size / 2.))
    x_max = x_min + x_cutout_size
    y_max = y_min + y_cutout_size
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, x_parent_size)
    y_max = min(y_max, y_parent_size)

    # Return trim range
    return y_min, y_max, x_min, x_max


def trim_image(input_image: np.ndarray, x_size: int, y_size: int) -> np.ndarray:
    """
    Trims an input image array, keeping only the central x_size x y_size elements.
    Fits and matplotlib convention used: x_axis: second array index, y_axis: first array index.
    :param input_image: the input image array
    :param x_size: central elements to keep in x-axis (second array index)
    :param y_size: central elements to keep in y-axis (first  array index)
    :return: ndarray, the trimmed image array
    """
    if len(input_image.shape) != 2:
        raise ValueError('2D input Numpy arrays needed')
    x_min = int((input_image.shape[1] - x_size) / 2)
    y_min = int((input_image.shape[0] - y_size) / 2)
    x_max = int(x_min + x_size)
    y_max = int(y_min + y_size)
    return input_image[y_min:y_max, x_min:x_max]


def scale_image(unscaled: np.ndarray, mag_zero_point: float, mag_object: float, exp_time: float) -> np.ndarray:
    """
    Scales an input image so that the total flux conforms to a given instrument zero point, magnitude and exposure time.
    :param unscaled:       the input unscaled image array
    :param mag_zero_point: magnitude producing 1e-/s
    :param mag_object:     the object intended magnitude
    :param exp_time:       the image exposure time (s)
    :return: ndarray, scaled input image array (e-)
    """
    electrons = exp_time * 10 ** ((mag_zero_point - mag_object) / 2.5)
    return unscaled * electrons / unscaled.sum()


def saturate_and_quantize_image(
        input_image: np.ndarray, saturation_limit: float, gain: float) -> np.ndarray:
    """
    Saturates (in electrons) and quantizes (in ADU) an input image.
    The output is stored in signed 32 bit integers.
    :param input_image: ndarray (e-)
    :param saturation_limit: float (e-)
    :param gain: float (electrons/ADU)
    :return: ndarray, the saturated and quantized image (ADU, np.uint16)
    """
    saturated = np.minimum(input_image, saturation_limit)
    return (saturated / gain).astype(dtype=np.int32)


def add_subimage_adu(parent_image: np.ndarray, subimage: np.ndarray, x_insert: int, y_insert: int) -> np.ndarray:
    """
    Adds ADUs from a subimage to a parent image. Input images are not modified.
    Subimage pixels outside the boundaries of the parent image are silently discarded.
    Internal calculations are done in np.int32. Output is capped to np.uint36.
    16 bit ADC numerical saturation is thus applied.
    Fits and matplotlib convention used: x_axis: second array index, y_axis: first array index.
    :param parent_image: ndarray (ADU)
    :param subimage: ndarray (ADU)
    :param x_insert: int, x-axis insertion point (second array index)
    :param y_insert: int, y-axis insertion point (first array index)
    :return: ndarray (ADU, np.uint16), the output array
    """

    # Verify input arrays are 2D
    if len(parent_image.shape) != 2 or len(subimage.shape) != 2:
        raise ValueError('2D input Numpy arrays needed')

    # Copy input image. Use np.int32 data type
    intermediate = parent_image.astype(dtype=np.int32)

    # Get image dimentsions
    x_length_parent = intermediate.shape[1]
    y_length_parent = intermediate.shape[0]
    x_length_subimage = subimage.shape[1]
    y_length_subimage = subimage.shape[0]

    # Silently reject subimages fully placed outside parent image
    if (x_insert < x_length_parent) and (y_insert < y_length_parent)\
            and (x_insert > -x_length_subimage) and (y_insert > -y_length_subimage):

        # Insert subimage, discarding pixels out of parent boundaries
        x_min_parent = max(x_insert, 0)
        y_min_parent = max(y_insert, 0)
        x_max_parent = min(x_insert + x_length_subimage, x_length_parent)
        y_max_parent = min(y_insert + y_length_subimage, y_length_parent)
        x_min_subimage = max(-x_insert, 0)
        y_min_subimage = max(-y_insert, 0)
        x_max_subimage = min(x_length_parent - x_insert, x_length_subimage)
        y_max_subimage = min(y_length_parent - y_insert, y_length_subimage)
        intermediate[y_min_parent:y_max_parent, x_min_parent:x_max_parent] = \
            np.add(intermediate[y_min_parent:y_max_parent, x_min_parent:x_max_parent],
                   subimage[y_min_subimage: y_max_subimage, x_min_subimage:x_max_subimage], dtype=np.int32)

    # Cap and cast output
    # TODO replace iinfo with module constants?
    capped = np.maximum(np.minimum(intermediate, np.iinfo(np.uint16).max), np.iinfo(np.uint16).min)
    output = capped.astype(np.uint16)
    return output


@dataclass(frozen=True)
class ImageInsert:
    """ Class used to define a given subimage and how is it going to be scaled and inserted into a parent image. """
    subimage:  str
    quadrant:  str
    x_trim:    int
    y_trim:    int
    x_insert:  int
    y_insert:  int
    magnitude: float
    id:        str = ""
    longitude: float = float('NaN')
    latitude:  float = float('NaN')


def insert_objects_in_hdu(input_array: np.ndarray, inserts: List[ImageInsert], subimages: Dict[str, np.ndarray],
                          quadrant_location: QuadrantLocation, gain: float, mag_zero_point: float,
                          exposure_time: float, seed: int = 0,
                          cosmics_array: np.ndarray = None, sky: float = 0
                          ) -> fits.ImageHDU:
    """
    Insert objects in a given fits image HDU. Thread safe
    :param input_array: The input array
    :param inserts: list of ImageInserts for this quadrant.
    :param subimages: Dictionary cache of subimages (typically PSFs) to insert
    :param quadrant_location: The quadrant location
    :param gain: The CCD gain (e-/ADU)
    :param mag_zero_point: magnitude producing a total fluence of 1 e-/s in the focal plane.
    :param exposure_time: (s).
    :param seed: optional, random number generator seed
    :param cosmics_array: optional, will be added to output if not None
    :param sky: optional, additive constant sky background (e-)
    :return: The output array with the inserts included
    """
    # Copy input_array, add to list and identify quadrant
    inserts_array = input_array.copy()

    # Get quadrant trim range: minimum values will be added to FPA image area insertion points
    y_min, y_max, x_min, x_max = quadrant_location.get_numpy_trim_range()

    # Random number generator
    rng = np.random.default_rng(seed)

    # Add sky background, if positive
    if sky > 0:
        background = np.ones((y_max - y_min, x_max - x_min)) * sky
        noisy = rng.poisson(background)
        adus = saturate_and_quantize_image(noisy, CCD_SATURATION_LEVEL, gain)
        inserts_array[y_min:y_max, x_min:x_max] = add_subimage_adu(
            inserts_array[y_min:y_max, x_min:x_max], adus, 0, 0)

    # Add inserts for this quadrant. Pixels outside the image area will be discarded
    for insert in inserts:
        if np.isnan(insert.magnitude):
            continue
        subimage = subimages[insert.subimage]
        trimmed = trim_image(subimage, insert.x_trim, insert.y_trim)
        scaled = scale_image(trimmed, mag_zero_point, insert.magnitude, exposure_time)
        noisy = rng.poisson(scaled)
        adus = saturate_and_quantize_image(noisy, CCD_SATURATION_LEVEL, gain)
        inserts_array[y_min:y_max, x_min:x_max] = add_subimage_adu(
            inserts_array[y_min:y_max, x_min:x_max], adus, insert.x_insert, insert.y_insert)

    # Add cosmic rays image, if provided
    if cosmics_array is not None:
        inserts_array = add_subimage_adu(inserts_array, cosmics_array, x_min - 1, y_min - 1)

    return inserts_array


def insert_objects_in_vis_image_parallel(input_parent_fits: fits.HDUList, output_fits_file: str,
                                image_inserts: str, inserts_folder: str, vis_processing_config: VisProcessingConfig,
                                mag_zero_point: float, exposure_time: float, seed: int = 0, cosmics: str = None,
                                sky: float = 0):
    """
    Insert some objects into a parent VIS image stack. Inserts are defined in a configuration file.
    x, y insertion points are expressed in trimmed FPA image coordinates.
    Subimage values cannot be inserted outside the image area (i.e. in pre- or post-scan areas)
    :param input_parent_fits: (ADU, uint16). It is neither opened nor closed by this method.
    :param output_fits_file: (ADU, uint16)
    :param image_inserts: json string with all the inserts.
     Those not fitting any quadrant in the input parent fits will be silently discarded.
    :param inserts_folder: input folder with the image inserts.
    :param vis_processing_config: Object with key VIS processing configuration parameters
    :param mag_zero_point: magnitude producing a total fluence of 1 e-/s in the focal plane.
    :param exposure_time: (s).
    :param seed: optional, random number generator seed.
    :param cosmics: optional, fits file name with four pre-computed cosmic ray trimmed images (one per QuadrantLocation)
    :param sky: optional, additive constant sky background (e-)
    :return: None
    """

    # Decode json string with inserts
    json_list = json.loads(image_inserts)
    inserts = [ImageInsert(**dic) for dic in json_list]

    # Map inserts to dictionary. Key: quadrant, Value: inserts list
    inserts_map = defaultdict(list)
    for insert in inserts:
        inserts_map[Quadrant[insert.quadrant]].append(insert)

    # Subimages data cache dictionary. Key: subimage file name. Value: data numpy array
    subimages = {insert.subimage: None for insert in inserts}
    for subimage in subimages:
        with fits.open(inserts_folder + '/' + subimage) as subimage_hdu_list:
            subimages[subimage] = subimage_hdu_list[0].data
            subimage_hdu_list.close()

    # Read trimmed cosmic ray images, if provided, and store in dictionary.
    # Key: QuadrantLocation. Value: sample data (ADU)
    if cosmics is not None:
        with fits.open(cosmics, memmap=False, lazy_load_hdus=False) as cosmic_hdus:
            cosmics_per_quadrant_location = {}
            for hdu in cosmic_hdus[1:]:
                quadrant_location = QuadrantLocation[hdu.header[COSMIC_IMAGE_QUADRANT_LOCATION_DESCRIPTOR]]
                cosmics_per_quadrant_location[quadrant_location] = hdu.data
            cosmic_hdus.close()

    # Multiprocessing
    parameters = []
    for hdu in input_parent_fits[1:]:
        quadrant = Quadrant(hdu.name)
        inserts = inserts_map[quadrant]
        subimages_subset = {subimage_name: subimages[subimage_name] for subimage_name
                            in {insert.subimage for insert in inserts}}
        if cosmics is not None:
            cosmics_array = cosmics_per_quadrant_location[QUADRANT_TO_QUADRANT_LOCATION[quadrant]]
        else:
            cosmics_array = None
        parameters.append([hdu.data, inserts, subimages_subset, QUADRANT_TO_QUADRANT_LOCATION[quadrant],
                           vis_processing_config.quadrants[quadrant].gain, mag_zero_point, exposure_time,
                           seed, cosmics_array, sky])
    with multiprocessing.Pool(min(NUM_CORES, vis_processing_config.max_threads)) as pool:
        results = pool.starmap(insert_objects_in_hdu, parameters)

    # Create output primary HDU using input plus some comments
    output_primary_hdu = input_parent_fits[0].copy()
    output_primary_hdu.header['COMMENT'] = 'Synthetic objects added on top of original image'

    # Create secondary HDUs with input headers plus data with inserts
    secondary_hdus = [fits.ImageHDU(header=input_hdu.header, data= output_data) for input_hdu, output_data
                      in zip(input_parent_fits[1:], results)]

    # Save output HDUs to fits file
    with fits.HDUList([output_primary_hdu, *secondary_hdus]) as output_hdu_list:
        output_hdu_list.writeto(output_fits_file, overwrite=True)
        output_hdu_list.close()


def insert_objects_in_vis_image(input_parent_fits: fits.HDUList, output_fits_file: str,
                                image_inserts: str, inserts_folder: str, vis_processing_config: VisProcessingConfig,
                                mag_zero_point: float, exposure_time: float, seed: int = 0, cosmics: str = None,
                                sky: float = 0):
    """
    Insert some objects into a parent VIS image stack. Inserts are defined in a configuration file.
    x, y insertion points are expressed in trimmed FPA image coordinates.
    Subimage values cannot be inserted outside the image area (i.e. in pre- or post-scan areas)
    :param input_parent_fits: (ADU, uint16). It is neither opened nor closed by this method.
    :param output_fits_file: (ADU, uint16)
    :param image_inserts: json string with all the inserts.
     Those not fitting any quadrant in the input parent fits will be silently discarded.
    :param inserts_folder: input folder with the image inserts.
    :param vis_processing_config: Object with key VIS processing configuration parameters
    :param mag_zero_point: magnitude producing a total fluence of 1 e-/s in the focal plane.
    :param exposure_time: (s).
    :param seed: optional, random number generator seed.
    :param cosmics: optional, fits file name with four pre-computed cosmic ray trimmed images (one per QuadrantLocation)
    :param sky: optional, additive constant sky background (e-)
    :return: None
    """

    # Decode json string with inserts
    json_list = json.loads(image_inserts)
    inserts = [ImageInsert(**dic) for dic in json_list]

    # Map inserts to dictionary. Key: quadrant, Value: inserts list
    inserts_map = defaultdict(list)
    for insert in inserts:
        inserts_map[Quadrant[insert.quadrant]].append(insert)

    # Subimages data cache dictionary. Key: subimage file name. Value: data numpy array
    subimages = {insert.subimage: None for insert in inserts}
    for subimage in subimages:
        with fits.open(inserts_folder + '/' + subimage) as subimage_hdu_list:
            subimages[subimage] = subimage_hdu_list[0].data
            subimage_hdu_list.close()

    # Decode json string with FPA quadrants configuration
    quadrant_data = vis_processing_config.quadrants

    # Random number generator
    rng = np.random.default_rng(seed)

    # Create new HDU list, copy primary HDU into it and add some information to header
    output_hdus = [input_parent_fits[0].copy()]
    output_hdus[0].header['COMMENT'] = 'Synthetic objects added on top of original image'

    # Read trimmed cosmic ray images, if provided, and store in dictionary.
    # Key: QuadrantLocation. Value: sample data (ADU)
    if cosmics is not None:
        with fits.open(cosmics, memmap=False, lazy_load_hdus=False) as cosmic_hdus:
            cosmics_per_quadrant_location = {}
            for hdu in cosmic_hdus[1:]:
                quadrant_location = QuadrantLocation[hdu.header[COSMIC_IMAGE_QUADRANT_LOCATION_DESCRIPTOR]]
                cosmics_per_quadrant_location[quadrant_location] = hdu.data
            cosmic_hdus.close()

    # Loop on HDU extensions. They are all assumed to be images
    for hdu in input_parent_fits[1:]:

        # Copy HDU, add to list and identify quadrant
        hdu_inserts = hdu.copy()
        output_hdus.append(hdu_inserts)
        quadrant = Quadrant(hdu_inserts.name)

        # Get quadrant trim range: minimum values will be added to FPA image area insertion points
        y_min, y_max, x_min, x_max = QUADRANT_TO_QUADRANT_LOCATION[quadrant].get_numpy_trim_range()

        # Add sky background, if positive
        if sky > 0:
            background = np.ones((y_max - y_min, x_max - x_min)) * sky
            noisy = rng.poisson(background)
            adus = saturate_and_quantize_image(noisy, CCD_SATURATION_LEVEL, quadrant_data[quadrant].gain)
            hdu_inserts.data[y_min:y_max, x_min:x_max] = add_subimage_adu(
                hdu_inserts.data[y_min:y_max, x_min:x_max], adus, 0, 0)

        # Add inserts for this quadrant. Pixels outside the image area will be discarded
        for insert in inserts_map.get(quadrant, []):
            if np.isnan(insert.magnitude):
                continue
            subimage = subimages[insert.subimage]
            trimmed = trim_image(subimage, insert.x_trim, insert.y_trim)
            scaled = scale_image(trimmed, mag_zero_point, insert.magnitude, exposure_time)
            noisy = rng.poisson(scaled)
            adus = saturate_and_quantize_image(noisy, CCD_SATURATION_LEVEL, quadrant_data[quadrant].gain)
            hdu_inserts.data[y_min:y_max, x_min:x_max] = add_subimage_adu(
                hdu_inserts.data[y_min:y_max, x_min:x_max], adus, insert.x_insert, insert.y_insert)

        # Add cosmic rays image, if provided
        if cosmics is not None:
            quadrant_location = QUADRANT_TO_QUADRANT_LOCATION[quadrant]
            hdu_inserts.data = add_subimage_adu(
                hdu_inserts.data, cosmics_per_quadrant_location[quadrant_location], x_min - 1, y_min - 1)

    # Save output HDUs to fits file
    with fits.HDUList(output_hdus) as output_hdu_list:
        output_hdu_list.writeto(output_fits_file, overwrite=True)
        output_hdu_list.close()


def get_sky_to_pixel_derivatives(wcs: WCS, x: float, y: float, angle: float)\
        -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Returns the longitude and latitude derivatives along great circles with respect
     to pixel coordinates rotated by a given position angle (that is, they are corrected by cos(dec)).
     They are estimated from finite differences of one pixel.
    :param wcs: The input WCS
    :param x: The input x-axis pixel coordinate (1-based)
    :param y: The input y-axis pixel coordinate (1-based)
    :param angle: The position angle (deg), so the derivative is computed in terms of xp, yp:
     dx = cos(angle) * dxp - sin(angle) * dyp,
     dy = sin(angle) * dxp + cos(angle) * dyp
    :return: Sky derivatives with respect to the pixel coordinates (arcsec/pix):
     ((dra_dxp, ddec_dxp), (dra_dyp, ddec_dyp))
    """
    # Get plate scale derivatives. Units: arcsec in sky great circles
    radec_ref, radec_dx, radec_dy = wcs.all_pix2world(
        [[x, y],
         [x + math.cos(math.radians(angle)), y + math.sin(math.radians(angle))],
         [x - math.sin(math.radians(angle)), y + math.cos(math.radians(angle))]], 1)
    dra_dxp = (radec_dx[0] - radec_ref[0]) * math.cos(math.radians(radec_ref[1])) * 3600
    dra_dyp = (radec_dy[0] - radec_ref[0]) * math.cos(math.radians(radec_ref[1])) * 3600
    ddec_dxp = (radec_dx[1] - radec_ref[1]) * 3600
    ddec_dyp = (radec_dy[1] - radec_ref[1]) * 3600
    derivatives = ((dra_dxp, ddec_dxp), (dra_dyp, ddec_dyp))
    return derivatives


def get_pixel_to_sky_plate_scale_ellipse(wcs: WCS, x: float, y: float, angle: float = 0) -> Tuple[float, float, float]:
    """
    Returns the principal axes of the plate scale ellipse corresponding to a given WCS and pixel coordinates.
    Plate scales are estimated from finite differences of one pixel.
    :param wcs: The input WCS
    :param x: The input x-axis pixel coordinate (1-based)
    :param y: The input y-axis pixel coordinate (1-based)
    :param angle: The position angle (deg), so the derivative is computed in terms of xp, yp:
     dx = cos(angle) * dxp - sin(angle) * dyp,
     dy = sin(angle) * dxp + cos(angle) * dyp
    :return: major plate scale (arcsec/pix), minor plate scale (arcsec/pix), position angle (deg)
     of the closest semi-major axis with respect to xp. It is obtained as atan(u[1, 0] / u[0, 0])),
     where u is the one of the unitary matrices obtained from the singular value decomposition of
     the plate scale derivatives matrix with respect to (xp, yp)
    """
    # Get plate scale derivatives. Units: arcsec in sky great circles
    derivatives = get_sky_to_pixel_derivatives(wcs, x, y, angle)
    u_matrix, s_values, _ = np.linalg.svd(derivatives)
    return s_values[0], s_values[1],  math.degrees(math.atan(u_matrix[0, 1] / u_matrix[0, 0]))


def get_eer_pix(image: np.ndarray, x_center_guess: float, y_center_guess: float, fraction: float, eer100_pix: float,
                recenter: bool = False) -> float:
    """
    Determine the encircled energy radius (pixels) for a given input image, centroid and energy fraction.
    The image has been previously background subtracted.
    Fits and matplotlib convention used: x_axis: second array index, y_axis: first array index.
    :param image: The input image.
    :param x_center_guess: The x-axis central coordinate guess (0-based)
    :param y_center_guess: The y-axis central coordinate guess (0-based)
    :param fraction: The energy fraction, in interval (0, 1)
    :param eer100_pix: The pixel radius assumed to collect all flux from the object (pixels)
    :param recenter: If True, some recenter iterations are carried out to minimize EER
    :return: The encircled energy radius (pixels)
    """
    if np.isnan(x_center_guess) or np.isnan(y_center_guess):
        return np.nan
    # Determine the outer radius total flux
    outer_aperture = CircularAperture((x_center_guess, y_center_guess), r=eer100_pix)
    outer_photometry = aperture_photometry(image, outer_aperture)
    total_flux = outer_photometry[APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN][0]

    def ee_pix(radius: float, x_center: float, y_center: float) -> float:
        """
        Determine the encircled energy for a given radius and center
        :param radius: The radius (pixels)
        :param x_center: The x-axis central coordinate (0-based)
        :param y_center: The y-axis central coordinate (0-based)
        :return: The encircled energy
        """
        aperture = CircularAperture((x_center, y_center), r=radius)
        photometry = aperture_photometry(image, aperture)
        return photometry[APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN][0] / total_flux

    def eer_pix(x_center: float, y_center: float) -> float:
        """
        Determine the encircled energy radius for a given center
        :param x_center: The x-axis central coordinate (0-based)
        :param y_center: The y-axis central coordinate (0-based)
        :return: The encircled energy radius (pixels)
        """
        try:
            return optimize.brenth(lambda x: ee_pix(x, x_center, y_center) - fraction,
                                   a=0.1, b=eer100_pix, xtol=EER_PIX_XTOL)
        except ValueError:
            return np.nan

    if recenter:
        result = optimize.minimize(lambda x: eer_pix(x[0], x[1]), [x_center_guess, y_center_guess],
                                   method='BFGS', jac='2-point', options={'xrtol': EER_PIX_XTOL},  # maxiter: 2
                                   bounds=[(x_center_guess - 1, x_center_guess + 1),
                                           (y_center_guess - 1, y_center_guess + 1)])
        return result['fun']
    else:
        try:
            result = optimize.brenth(lambda x: ee_pix(x, x_center_guess, y_center_guess) - fraction,
                                     a=0.1, b=eer100_pix, xtol=EER_PIX_XTOL)
            return result
        except ValueError:
            return np.nan


def get_eer_sky(image: np.ndarray, x_center_guess: float, y_center_guess: float,
                major_plate_scale: float, minor_plate_scale: float, position_angle: float,
                fraction: float, eer100_sky: float,
                recenter: bool = False) -> float:
    """
    Determine the encircled energy radius (arcsec) for a given input image, centroid and energy fraction.
    The image has been previously background subtracted.
    Fits and matplotlib convention used: x_axis: second array index, y_axis: first array index.
    :param image: The input image.
    :param x_center_guess: The x-axis central coordinate guess (0-based)
    :param y_center_guess: The y-axis central coordinate guess (0-based)
    :param major_plate_scale: The major plate scale (arcsec/pix)
    :param minor_plate_scale: The minor plate scale (arcsec/pix)
    :param position_angle: The position angle (deg) of the closest semi-major axis with respect to x
    :param fraction: The energy fraction, in interval (0, 1)
    :param eer100_sky: The sky radius assumed to collect all flux from the object (arcsec)
    :param recenter: If True, some recenter iterations are carried out to minimize EER
    :return: The encircled energy radius (arcsec)
    """
    if np.isnan(x_center_guess) or np.isnan(y_center_guess):
        return np.nan
    # Determine the outer radius total flux
    outer_aperture = EllipticalAperture((x_center_guess, y_center_guess),
                                        a=eer100_sky / major_plate_scale,
                                        b=eer100_sky / minor_plate_scale,
                                        theta=math.radians(position_angle))
    outer_photometry = aperture_photometry(image, outer_aperture)
    total_flux = outer_photometry[APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN][0]

    def ee_sky(radius: float, x_center: float, y_center: float) -> float:
        """
        Determine the encircled energy for a given radius and center
        :param radius: The radius (arcsec)
        :param x_center: The x-axis central coordinate (0-based)
        :param y_center: The y-axis central coordinate (0-based)
        :return: The encircled energy
        """
        aperture = EllipticalAperture((x_center, y_center),
                                      a=radius / major_plate_scale,
                                      b=radius / minor_plate_scale,
                                      theta=math.radians(position_angle))
        photometry = aperture_photometry(image, aperture)
        return photometry[APERTURE_PHOTOMETRY_TOTAL_FLUX_COLUMN][0] / total_flux

    def eer_sky(x_center: float, y_center: float) -> float:
        """
        Determine the encircled energy radius for a given center
        :param x_center: The x-axis central coordinate (0-based)
        :param y_center: The y-axis central coordinate (0-based)
        :return: The encircled energy radius (arcsec)
        """
        return optimize.brenth(lambda x: ee_sky(x, x_center, y_center) - fraction,
                               a=0.01, b=eer100_sky, xtol=EER_SKY_XTOL)

    if recenter:
        result = optimize.minimize(lambda x: eer_sky(x[0], x[1]), [x_center_guess, y_center_guess],
                                   method='BFGS', jac='2-point', options={'xrtol': EER_SKY_XTOL},  # maxiter: 2
                                   bounds=[(x_center_guess - 1, x_center_guess + 1),
                                           (y_center_guess - 1, y_center_guess + 1)])
        return result['fun']
    else:
        return optimize.brenth(lambda x: ee_sky(x, x_center_guess, y_center_guess) - fraction,
                               a=0.01, b=eer100_sky, xtol=EER_SKY_XTOL)


def get_gaussian_fit(image: np.ndarray, x_center_guess: float, y_center_guess: float
                     ) -> Tuple[float, float, float, float, float, float]:
    """
    Determine some parameters for elliptical Gaussian fits to the PSF core in pixel space.
    Function used: standard Astropy Gaussian2D (regular)
    The image has been previously background subtracted.
    Fits and matplotlib convention used: x_axis: second array index, y_axis: first array index.
    :param image: The input image
    :param x_center_guess: The x-axis central coordinate guess (0-based)
    :param y_center_guess: The y-axis central coordinate guess (0-based)
    :return:
      amplitude peak value,
      x_mean peak position (pixels),
      y_mean peak position (pixels),
      regular semi-major axis (pixels),
      regular semi-minor axis (pixels),
      regular semi-major axis position angle (deg),
    """

    # Create fit models
    y_pixels, x_pixels = image.shape
    y, x = np.mgrid[:y_pixels, :x_pixels]
    finite_mask = np.isfinite(image)
    peak_guess = image[int(y_center_guess), int(x_center_guess)]
    model_regular = Gaussian2D(
        peak_guess, x_center_guess, y_center_guess,
        VIS_PSF_CORE_GAUSSIAN_SIGMA_GUESS_PIX, VIS_PSF_CORE_GAUSSIAN_SIGMA_GUESS_PIX, 0)

    # Fit to image data
    fitter = fitting.LevMarLSQFitter()
    try:
        results_regular = fitter(model_regular, x[finite_mask], y[finite_mask], image[finite_mask])
    except fitting.NonFiniteValueError:  #TODO Improve masking values instead?
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Compute and return major/minor axes sigma values
    a_regular = max(results_regular.x_stddev.value, results_regular.y_stddev.value)
    b_regular = min(results_regular.x_stddev.value, results_regular.y_stddev.value)
    position_angle_regular = math.degrees(results_regular.theta.value)\
                             + 90 * int(results_regular.y_stddev.value > results_regular.x_stddev.value)
    x_mean = results_regular.x_mean.value
    y_mean = results_regular.y_mean.value
    amplitude = results_regular.amplitude.value
    return amplitude, x_mean, y_mean, a_regular, b_regular, position_angle_regular


def get_gaussian_fit_pixelated(image: np.ndarray, x_center_guess: float, y_center_guess: float)\
        -> Tuple[float, float, float, float, float, float]:
    """
    Determine some parameters for elliptical Gaussian fits to the PSF core in pixel space.
    Function used: custom PixelatedEllipticalGaussian (pixelated).
    The image has been previously background subtracted.
    Fits and matplotlib convention used: x_axis: second array index, y_axis: first array index.
    :param image: The input image
    :param x_center_guess: The x-axis central coordinate guess (0-based)
    :param y_center_guess: The y-axis central coordinate guess (0-based)
    :return:
      amplitude peak value,
      x_mean peak position (pixels),
      y_mean peak position (pixels),
      pixelated semi-major axis (pixels),
      pixelated semi-minor axis (pixels),
      pixelated semi-major axis position angle (deg),
    """

    # Create fit models
    y_pixels, x_pixels = image.shape
    y, x = np.mgrid[:y_pixels, :x_pixels]
    peak_guess = image[int(y_center_guess), int(x_center_guess)]
    model_pixelated = PixelatedEllipticalGaussian(
        peak_guess, x_center_guess, y_center_guess,
        VIS_PSF_CORE_GAUSSIAN_SIGMA_GUESS_PIX, VIS_PSF_CORE_GAUSSIAN_SIGMA_GUESS_PIX, 0)

    # Fit to image data
    fitter = fitting.LevMarLSQFitter()
    try:
        results_pixelated = fitter(model_pixelated, x, y, image)
    except fitting.NonFiniteValueError:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Compute and return average FWHM values
    a_pixelated = max(results_pixelated.x_stddev.value, results_pixelated.y_stddev.value)
    b_pixelated = min(results_pixelated.x_stddev.value, results_pixelated.y_stddev.value)
    position_angle_pixelated = math.degrees(results_pixelated.theta.value)\
                               + 90 * int(results_pixelated.y_stddev.value > results_pixelated.x_stddev.value)
    x_mean = results_pixelated.x_mean.value
    y_mean = results_pixelated.y_mean.value
    amplitude = results_pixelated.amplitude.value
    return amplitude, x_mean, y_mean, a_pixelated, b_pixelated, position_angle_pixelated


class PixelatedEllipticalGaussian(modeling.Fittable2DModel):
    """
    This class provides a version of the EllipticalGaussian 2D model fitting function
    where the values and derivatives are integrated over whole pixels. They are centered around the (x,y) input pairs.
    """

    # Fitting parameters
    amplitude = modeling.Parameter(default=None)
    x_mean = modeling.Parameter(default=None)
    y_mean = modeling.Parameter(default=None)
    x_stddev = modeling.Parameter(default=None)
    y_stddev = modeling.Parameter(default=None)
    theta = modeling.Parameter(default=None)

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta) -> float:
        x_subpix = np.array(x)[..., np.newaxis, np.newaxis] + GAUSSIAN_QUADRATURE_SUBPIXEL_OFFSET_X
        y_subpix = np.array(y)[..., np.newaxis, np.newaxis] + GAUSSIAN_QUADRATURE_SUBPIXEL_OFFSET_Y
        evaluate_subpix = Gaussian2D.evaluate(x_subpix, y_subpix, amplitude,
                                                 x_mean, y_mean, x_stddev, y_stddev, theta)
        evaluate_integrated = np.sum(evaluate_subpix * GAUSSIAN_QUADRATURE_SUBPIXEL_WEIGHT, axis=(-1, -2))
        return evaluate_integrated

    @staticmethod
    def fit_deriv(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta) -> float:
        x_subpix = np.array(x)[..., np.newaxis, np.newaxis] + GAUSSIAN_QUADRATURE_SUBPIXEL_OFFSET_X
        y_subpix = np.array(y)[..., np.newaxis, np.newaxis] + GAUSSIAN_QUADRATURE_SUBPIXEL_OFFSET_Y
        fit_deriv_subpix = Gaussian2D.fit_deriv(x_subpix, y_subpix, amplitude,
                                                 x_mean, y_mean, x_stddev, y_stddev, theta)
        fit_deriv_integrated = [np.sum(x * GAUSSIAN_QUADRATURE_SUBPIXEL_WEIGHT, axis=(-1, -2))
                                for x in fit_deriv_subpix]
        return fit_deriv_integrated


def get_gaussian_fwhm_2d_average(semi_major_axis: float, semi_minor_axis: float) -> float:
    """
    Azimuthal Full Width at Half Maximum (FWHM) average of a 2D Gaussian
    :param semi_major_axis: The 2D Gaussian semi-major axis
    :param semi_minor_axis: The 2D Gaussian semi-minor axis
    :return: the azimuthally averaged FWHM
    """
    return 2 / math.pi * semi_major_axis * gaussian_sigma_to_fwhm\
           * ellipe((semi_major_axis - semi_minor_axis) / semi_major_axis)


def get_rotation_matrix(theta: float, radians: bool = True) -> np.ndarray:
    """
    Rotation angle for a 2D counterclock rotation: X' = M * X
    :param theta: rotation angle (radians)
    :return: the rotation matrix
    """
    return np.array(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta))))


def get_sky_ellipse_fwhm_2d_average(ellipse_a: float, ellipse_b: float,
                                    plate_scale_major: float, plate_scale_minor: float,
                                    position_angle_difference: float) -> float:
    """
    Determine the average FWHM of an ellipse defined in pixel coordinates in sky coordinates
    :param ellipse_a: The ellipse semi-major axis (pix)
    :param ellipse_b: The ellipse semi-minor axis (pix)
    :param plate_scale_major: major plate scale (arcsec/pix)
    :param plate_scale_minor: minor plate scale (arcsec/pix)
    :param position_angle_difference: Difference A-B between two counter-clockwise position angles:
     A: major plate scale axis versus pixel grid (deg). B: ellipse semi-major axis versus pixel grid.
    :return: The azimuthally averaged FWHM (arcsec)
    """
    # Return nan for nan input
    if np.isnan(ellipse_a) or np.isnan(ellipse_b):
        return np.nan

    # Determine ellipse semi-major axes from SVD of ellipse axes transformed into sky
    plate_scale = np.diag([plate_scale_major, plate_scale_minor])  # pix / arcsec
    rotation_matrix = get_rotation_matrix(math.radians(position_angle_difference))
    ellipse_axes_pix = np.diag([ellipse_a, ellipse_b])  # pix
    ellipse_axes_to_sky = plate_scale @ rotation_matrix @ ellipse_axes_pix 
    s = np.linalg.svd(ellipse_axes_to_sky, compute_uv=False)

    # Determine average FWHM
    return get_gaussian_fwhm_2d_average(s[0], s[1])


def get_unnormalized_moment(psf: np.ndarray, x: np.ndarray, y: np.ndarray, x_order: int, y_order: int, sigma: float,
                            quadrupole_moment_sigma_threshold: float = QUADRUPOLE_MOMENT_SIGMA_THRESHOLD)\
        -> float:
    """
    Estimate unnormalized moment for a given input PSF image:
     sum_i (x_i ** x_order * y_i ** y_order * exp(-0.5 * (x_i ** 2 + y_i ** 2) / sigma ** 2) * psf_i)
     for all "i" pixels within a given threshold in sigma.
     Fits and matplotlib convention used: x_axis: second array index, y_axis: first array index.
     It can be carried out both in pixel or sky coordinates, depending on the provided values
     NaN values will be masked
    :param psf: The input stellar image.
    :param x: The image containing x - x_centroid values for each pixel
    :param y: The image containing y - y_centroid values for each pixel
    :param x_order: The x-axis moment oder
    :param y_order: The y-axis moment oder
    :param sigma: The Gaussian weight function sigma value
    :param quadrupole_moment_sigma_threshold: The threshold in radial distance above which pixels are not considered in
     the moment calculation.
    :return: the unnormalized Q_x_order__y_order moment
    """
    mask = (x ** 2 + y ** 2 > (quadrupole_moment_sigma_threshold * sigma) ** 2) | np.isnan(psf)
    moment = np.ma.array(x ** x_order * y ** y_order * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2) * psf, mask=mask)\
        .sum()
    return moment


def get_r2_ellipticity(image: np.ndarray, x_center_guess: float, y_center_guess: float,
                       major_plate_scale: float = 1, minor_plate_scale: float = 1, position_angle: float = 0,
                       gaussian_weight_sigma: float = QUADRUPOLE_MOMENT_SIGMA_PIX,
                       quadrupole_moment_sigma_threshold: float = QUADRUPOLE_MOMENT_SIGMA_THRESHOLD,
                       ) -> Tuple[float, float, float, float]:
    """
    :param image: The input image.
    :param x_center_guess: The x-axis central coordinate guess (pix, 0-based)
    :param y_center_guess: The y-axis central coordinate guess (pix, 0-based)
    :param major_plate_scale: The major plate scale. Default: pixel scale. Can use arcsec/pix for on-sky calculation
    :param minor_plate_scale: The minor plate scale. Default: pixel scale. Can use arcsec/pix for on-sky calculation
    :param position_angle: The position angle (deg) of the semi-major axis with respect to x (deg).
     Default: pixel scale
    :param gaussian_weight_sigma: The Gaussian weight function sigma parameter (pixel or sky coordinates).
     Default: pixel scale
    :param quadrupole_moment_sigma_threshold: The threshold in radial distance above which pixels are not considered in
     the moment calculation.
    :return: the R2, ellipticity, x and y refined position values (pix or sky reference frames)
    """
    # Get pixel arrays
    y_pixels, x_pixels = image.shape
    y_pix, x_pix = np.mgrid[:y_pixels, :x_pixels]

    # Determine pixel to sky transform matrix and apply to pixel coordinates. Transform = unity for pixel calculation
    transform = np.diag([major_plate_scale, minor_plate_scale]) @ get_rotation_matrix(math.radians(position_angle))
    x_sky, y_sky = np.tensordot(transform, np.concatenate((x_pix[..., np.newaxis], y_pix[..., np.newaxis]), 2),
                                (1, 2))
    x_center_refined_sky, y_center_refined_sky = transform @ np.array([x_center_guess, y_center_guess])

    #Iterative loop: first order moments => centroid => first order moments
    for i in range(R2_ELLIPCITICY_ITERATIONS):
        m00_sky = get_unnormalized_moment(image, x_sky - x_center_refined_sky, y_sky - y_center_refined_sky, 0, 0,
                                          gaussian_weight_sigma, quadrupole_moment_sigma_threshold)
        m01_sky = get_unnormalized_moment(image, x_sky - x_center_refined_sky, y_sky - y_center_refined_sky, 0, 1,
                                          gaussian_weight_sigma, quadrupole_moment_sigma_threshold)
        m10_sky = get_unnormalized_moment(image, x_sky - x_center_refined_sky, y_sky - y_center_refined_sky, 1, 0,
                                          gaussian_weight_sigma, quadrupole_moment_sigma_threshold)
        x_center_refined_sky += m10_sky / m00_sky
        y_center_refined_sky += m01_sky / m00_sky

    # Second order moments
    m00_sky = get_unnormalized_moment(image, x_sky - x_center_refined_sky, y_sky - y_center_refined_sky, 0, 0,
                                      gaussian_weight_sigma, quadrupole_moment_sigma_threshold)
    m20_sky = get_unnormalized_moment(image, x_sky - x_center_refined_sky, y_sky - y_center_refined_sky, 2, 0,
                                      gaussian_weight_sigma, quadrupole_moment_sigma_threshold)
    m02_sky = get_unnormalized_moment(image, x_sky - x_center_refined_sky, y_sky - y_center_refined_sky, 0, 2,
                                      gaussian_weight_sigma, quadrupole_moment_sigma_threshold)
    m11_sky = get_unnormalized_moment(image, x_sky - x_center_refined_sky, y_sky - y_center_refined_sky, 1, 1,
                                      gaussian_weight_sigma, quadrupole_moment_sigma_threshold)

    # Determine and return R2 and ellipticity
    r2_sky = (m20_sky + m02_sky) / m00_sky
    e1_sky = (m20_sky - m02_sky) / (m20_sky + m02_sky)
    e2_sky = 2 * m11_sky / (m20_sky + m02_sky)
    e_sky = np.sqrt(e1_sky ** 2 + e2_sky ** 2)
    return r2_sky, e_sky, x_center_refined_sky, y_center_refined_sky
