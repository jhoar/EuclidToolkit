# This class provides static functions that do Cramer-Rao estimates on LSF and PSF numpy arrays
# It is based on a similar module in Java:  gaia.dpce.calteam.tools.vpuSimulator.CentroidLocationPrecision

from enum import Enum
from functools import lru_cache
import math
from typing import List, Tuple

import numpy as np

invalid_index_value = int(-99999)

class Derivative(Enum):
    """
    Finite difference derivative estimators
    """
    LEFT   = (-1, 0)
    CENTER = (-1, 1)
    RIGHT  = ( 0, 1)


@lru_cache()
def get_lsf_derivatives(n_samples: int, derivative: Derivative) -> List[Derivative]:
    """
    Get the finite difference derivative estimator for each sample in an unmasked LSF.
    Right/left derivatives are used at the interval boundaries as appropriate.
    :param n_samples: The number of LSF samples
    :param derivative: The desired finite difference derivative flavour. It will be used when possible
    :return: The list of finite difference derivatives per sample
    """
    derivatives = []
    for i in range(n_samples):
        if i == 0:
            derivatives.append(Derivative.RIGHT)
        elif i == n_samples - 1:
            derivatives.append(Derivative.LEFT)
        else:
            derivatives.append(derivative)
    return derivatives

@lru_cache()
def get_masked_lsf_derivatives(mask: Tuple[bool]) -> Tuple[Derivative]:
    """
    Get the finite difference derivative estimator for each sample in a masked LSF.
    Right/center/left derivatives are used as appropriate. Center has preference.
    :param mask: The input LSF mask. It is a tuple instead of a numpy array to allow for hashing and thus caching
    :return: The Tuple of finite difference derivatives per sample. It can be None when estimate is impossible
    """
    # Determine derivatives type
    # RIGHT/LEFT in left/right boundaries. None if no valid neighbour and CENTER otherwise
    derivatives = []
    for i in range(len(mask)):
        # Masked LSF sample
        if mask[i]:
            derivatives.append(None)
        # Left boundary
        elif i == 0:
            if mask[i + 1]:
                derivatives.append(None)
            else:
                derivatives.append(Derivative.RIGHT)
        # Right Boundary
        elif i == len(mask) - 1:
            if mask[i - 1]:
                derivatives.append(None)
            else:
                derivatives.append(Derivative.LEFT)
        # Sample in between left/right boundaries
        elif mask[i - 1]:
            if mask[i + 1]:
                derivatives.append(None)
            else:
                derivatives.append(Derivative.RIGHT)
        elif mask[i + 1]:
            derivatives.append(Derivative.LEFT)
        else:
            derivatives.append(Derivative.CENTER)

    return tuple(derivatives)


@lru_cache()
def get_lsf_derivative_indices(n_samples: int, derivative: Derivative) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the left and right indices for estimating finite difference derivatives in an unmasked LSF
    :param n_samples: The LSF number of samples
    :param derivative: The desired type of finite difference derivative (can be different at interval boundaries)
    :return: The left and right finite difference derivative indices for each sample
    """
    derivatives = get_lsf_derivatives(n_samples, derivative)
    left_indices = []
    right_indices = []
    for i in range(n_samples):
        left_indices.append(i + derivatives[i].value[0])
        right_indices.append(i + derivatives[i].value[1])
    return np.array(left_indices, dtype=np.int), np.array(right_indices, dtype=np.int)


@lru_cache()
def get_masked_lsf_derivative_indices(derivatives: Tuple[Derivative])\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the valid, left and right indices for estimating finite difference derivatives in a masked LSF
    :param derivatives: The finite difference derivative type for each sample (it can be None)
    :return: The indices for valid samples and, for each of them, the left and right samples for finite difference
     derivative estimate
    """
    valid_indices = []
    left_indices = []
    right_indices = []
    for i in range(len(derivatives)):
        if derivatives[i] is not None:
            valid_indices.append(i)
            left_indices.append(i + derivatives[i].value[0])
            right_indices.append(i + derivatives[i].value[1])

    return np.array(valid_indices, dtype=int), np.array(left_indices, dtype=int), np.array(right_indices, dtype=int)


def get_psf_centroid_location_precision(
        psf: np.ndarray, read_out_noise: float, background: float, derivative: Derivative,
        collapse_to_lsf: bool = False, mask: np.ndarray = None) -> Tuple[float, float]:
    """
    Get the Cramer-Rao centroid location precision for a given input masked PSF
    :param psf: The 2D input PSF (e-)
    :param read_out_noise: The detector read-out noise (e-)
    :param background: The additional subtracted image background (e-)
    :param derivative: The finite difference derivative type. It will ignored (CENTER) for masked estimates
    :param collapse_to_lsf: If false (default), full 2D calculation. If true, an LSF in each direction is computed
     before estimating Cramer-Rao. Note the latter might have small effect in rectangular apertures but be significant
     for other telescope pupil geometries. It will be ignored (False) for masked estimates
    :param mask: The boolean mask for invalid PSF samples. It will supersede derivative and collapse_to_lsf_values.
     Any mask already present in the input PSF will be ignored
    :return: The Cramer-Rao lower bound for astrometric centroiding precision for the first and second indices (sample)
    """
    # Estimate derivatives from finite differences
    psf_derivative_first_index: np.ndarray
    psf_derivative_second_index: np.ndarray
    if mask is None:
        psf_derivative_first_index  = np.apply_along_axis(get_lsf_derivative, 0, psf, derivative)
        psf_derivative_second_index = np.apply_along_axis(get_lsf_derivative, 1, psf, derivative)
    else:
        masked_psf = np.ma.array(psf, mask=mask, keep_mask=False)
        psf_derivative_first_index  = np.apply_along_axis(get_masked_lsf_derivative, 0, masked_psf)
        psf_derivative_second_index = np.apply_along_axis(get_masked_lsf_derivative, 1, masked_psf)

    # Flattened PSF and derivatives. Binning is applied if unmasked and collapse_to_psf is True
    psf_1d_first_index:             np.ndarray
    psf_1d_second_index:            np.ndarray
    psf_derivative_1d_first_index:  np.ndarray
    psf_derivative_1d_second_index: np.ndarray
    # 2D computation (i.e. no binning)?
    if (not collapse_to_lsf) or (mask is not None):
        psf_1d_first_index  = np.reshape(psf, -1)
        psf_1d_second_index = psf_1d_first_index
        psf_derivative_1d_first_index  = np.reshape(psf_derivative_first_index, -1)
        psf_derivative_1d_second_index = np.reshape(psf_derivative_second_index, -1)
    else:
        psf_1d_first_index             = get_lsf(psf, 0)
        psf_1d_second_index            = get_lsf(psf, 1)
        psf_derivative_1d_first_index  = get_lsf(psf_derivative_first_index, 0)
        psf_derivative_1d_second_index = get_lsf(psf_derivative_second_index, 1)

    # Return output value
    precision_first_index  = get_cramer_rao_lower_bound(
        psf_1d_first_index,  psf_derivative_1d_first_index,  read_out_noise, background)
    precision_second_index = get_cramer_rao_lower_bound(
        psf_1d_second_index, psf_derivative_1d_second_index, read_out_noise, background)
    return precision_first_index, precision_second_index


def get_lsf(psf: np.ndarray, lsf_index: int) -> np.ndarray:
    """
    Collapse an input PSF into a 1D LSF by summing samples along the non-running index
    :param psf: The input PSF
    :param lsf_index: The index to preserve in the LSF calculation
    :return: The collapsed LSF
    """
    # Collapse 2D array second index
    if lsf_index == 0:
        return np.apply_along_axis(np.sum, 1, psf)
    elif lsf_index == 1:
        return np.apply_along_axis(np.sum, 0, psf)
    else:
        raise ValueError('Illegal LSF index: ' + str(lsf_index))


def get_lsf_derivative(lsf: np.ndarray, derivative: Derivative) -> np.ndarray:
    """
    Estimate the derivatives for a given unmasked LSF
    :param lsf: The input LSF
    :param derivative: The finite difference derivative type
    :return: The LSF derivatives
    """
    indices = get_lsf_derivative_indices(lsf.shape[0], derivative)
    return (lsf[indices[1]] - lsf[indices[0]]) / (indices[1] - indices[0])


def get_masked_lsf_derivative(lsf: np.ma.core.MaskedArray) -> np.ma.core.MaskedArray:
    """
    Estimate the derivatives for a given unmasked LSF
    :param lsf: The input masked LSF
    :return: The LSF derivatives, which can be nan when no estimate was possible
    """
    # Raise exception if input is not a 1D masked LSF
    if not isinstance(lsf, np.ma.core.MaskedArray):
        raise TypeError("Input needs to be a 1D Numpy Masked Array, but is: " + str(type(lsf)))
    elif len(lsf.shape) != 1:
        raise ValueError("Input needs to be a 1D Numpy Array, but has shape: " + str(lsf.shape))

    derivatives = get_masked_lsf_derivatives(tuple(lsf.mask))
    valid_indices, left_indices, right_indices = get_masked_lsf_derivative_indices(derivatives)
    values = np.full(lsf.shape, np.nan, dtype=float)
    values[valid_indices] = (lsf[right_indices] - lsf[left_indices]) / (right_indices - left_indices)
    lsf_derivative = np.ma.masked_invalid(values)

    return lsf_derivative


def get_cramer_rao_lower_bound(lsf: np.ndarray, lsf_derivative: np.ndarray,
                               read_out_noise: float, background: float) -> float:
    """
    Get the Cramer-Rao lower bound for astrometric precision coresponding to a given unmasked LSF
    :param lsf: The input LSF (e-)
    :param lsf_derivative: The LSF finite difference derivative estimate (e-)
    :param read_out_noise: The detector read-out noise (e-)
    :param background: The additional subtracted image background (e-)
    :return: The Cramer-Rao lower bound for astrometric centroiding precision for the first and second indices (sample)
    """
    try:
        return 1 / math.sqrt(np.nansum(lsf_derivative ** 2 / (lsf + background + read_out_noise ** 2)))
    except ValueError:
        return math.nan


def get_lsf_rms(lsf: np.ndarray) -> float:
    """
    Get the LSF RMS width. It is the standard deviation of the location of all electrons with respect to the average
    location
    :param lsf: The input LSF (e-)
    :return: The LSF RMS width (sample)
    """
    # Verify input is a 1D Numpy array
    if len(lsf.shape) != 1:
        raise TypeError("lsf is not a 1D Numpy array")

    # Generate indices array
    indices = range(lsf.shape[0])

    # Estimate RMS as weighted standard deviation
    mean = get_lsf_mean(lsf)
    rms = math.sqrt((lsf * (indices - mean) ** 2).sum() / lsf.sum())

    return rms


def get_lsf_mean(lsf: np.ndarray) -> float:
    """
    Get the LSF center of mass
    :param lsf: The input LSF (e-)
    :return: The LSF center of mass as a floating point array index
    """
    # Verify input is a 1D Numpy array
    if len(lsf.shape) != 1:
        raise TypeError("lsf is not a 1D Numpy array. lsf shape:" + str(lsf.shape))

    # Generate indices array
    indices = range(lsf.shape[0])

    # Estamte mean weighted index
    mean = (indices * lsf).sum() / lsf.sum()

    return mean


@lru_cache()
def get_psf_circular_mask(length1: int, length2: int) -> np.ndarray:
    """
    Get a mask for a rectangular PSF keeping the samples in the circle defined the larger axis.
    This is appropriate for Cramer-Rao estimates for a circular pupil sampled with square pixels.
    :param length1: The first axis length
    :param length2: The second axis length
    :return: The binary mask
    """
    index1, index2 = np.mgrid[:length1, :length2]
    diameter2_max = (max(length1, length2) - 1) * (max(length1, length2) - 1) + (min(length1, length2) + 1) % 2
    diameter2 = np.power(2 * index1 + 1 - length1, 2) + np.power(2 * index2 + 1 - length2, 2)
    return diameter2 > diameter2_max
