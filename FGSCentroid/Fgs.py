from dataclasses import dataclass, asdict
from enum import Enum
import json

from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import numpy as np
from scipy.stats import linregress

from Core.Utilities import ndarray_to_list

#
#
# Constants and definitions
#
#
star_status_columns = [
    '1_STAR_STATUS', '2_STAR_STATUS', '3_STAR_STATUS', '4_STAR_STATUS', '5_STAR_STATUS',
    '6_STAR_STATUS', '7_STAR_STATUS', '8_STAR_STATUS', '9_STAR_STATUS', '10_STAR_STATUS',
    '11_STAR_STATUS', '12_STAR_STATUS', '13_STAR_STATUS', '14_STAR_STATUS', '15_STAR_STATUS',
    '16_STAR_STATUS', '17_STAR_STATUS', '18_STAR_STATUS', '19_STAR_STATUS', '20_STAR_STATUS', ]
star_x_columns = [
    '1_STAR_X', '2_STAR_X', '3_STAR_X', '4_STAR_X', '5_STAR_X',
    '6_STAR_X', '7_STAR_X', '8_STAR_X', '9_STAR_X', '10_STAR_X',
    '11_STAR_X', '12_STAR_X', '13_STAR_X', '14_STAR_X', '15_STAR_X',
    '16_STAR_X', '17_STAR_X', '18_STAR_X', '19_STAR_X', '20_STAR_X', ]
star_y_columns = [
    '1_STAR_Y', '2_STAR_Y', '3_STAR_Y', '4_STAR_Y', '5_STAR_Y',
    '6_STAR_Y', '7_STAR_Y', '8_STAR_Y', '9_STAR_Y', '10_STAR_Y',
    '11_STAR_Y', '12_STAR_Y', '13_STAR_Y', '14_STAR_Y', '15_STAR_Y',
    '16_STAR_Y', '17_STAR_Y', '18_STAR_Y', '19_STAR_Y', '20_STAR_Y', ]
star_mag_columns = [
    '1_STAR_M', '2_STAR_M', '3_STAR_M', '4_STAR_M', '5_STAR_M',
    '6_STAR_M', '7_STAR_M', '8_STAR_M', '9_STAR_M', '10_STAR_M',
    '11_STAR_M', '12_STAR_M', '13_STAR_M', '14_STAR_M', '15_STAR_M',
    '16_STAR_M', '17_STAR_M', '18_STAR_M', '19_STAR_M', '20_STAR_M', ]
star_index_columns = [
    '1_STAR_INDEX', '2_STAR_INDEX', '3_STAR_INDEX', '4_STAR_INDEX', '5_STAR_INDEX',
    '6_STAR_INDEX', '7_STAR_INDEX', '8_STAR_INDEX', '9_STAR_INDEX', '10_STAR_INDEX',
    '11_STAR_INDEX', '12_STAR_INDEX', '13_STAR_INDEX', '14_STAR_INDEX', '15_STAR_INDEX',
    '16_STAR_INDEX', '17_STAR_INDEX', '18_STAR_INDEX', '19_STAR_INDEX', '20_STAR_INDEX', ]

def status2pec(val):
    o = []
    for v in val:
        o.append((int(float(v)) >> 14) & 0b11)
    return np.array(o)

def window2int(val):
    o = []
    for v in val:
        o.append((int(float(v)) >> 10) & 0b1111)
    return np.array(o)

def valid2int(val):
    o = []
    for v in val:
        o.append((int(float(v)) >> 9) & 0b1)
    return np.array(o)

#
#
# FGS Classes
#
#
class FgsPec(Enum):
    """Numerical values identifying the FGS PECs in the star status field"""
    R1 = 0
    R2 = 1
    L1 = 2
    L2 = 3

@dataclass
class FgsStarTrackingInterval:
    """Stars tracked by the FGS in a given time interval. Several definitions for which is a valid interval exist"""
    tdb: Time
    """UTC times (TDB) """
    pec1: FgsPec
    """PEC for first slot (A1)"""
    pec2: FgsPec
    """PEC for second slot (A2)"""
    indices_pec1: list[float]
    """PEC A1 star indices. Maximum length: 20 (2 stars x 10 windows), but 11 is already uncommon"""
    indices_pec2: list[float]
    """PEC A2 star indices. Maximum length: 20 (2 stars x 10 windows), but 11 is already uncommon"""
    windows_pec1: list[list[float]]
    """PEC A1 windows. Integer. Null value: -1. 1st index: star (ordered by index). 2nd index: time"""
    windows_pec2: list[list[float]]
    """PEC A2 windows. Integer. Null value: -1. 1st index: star (ordered by index). 2nd index: time"""
    x_pec1: list[list[float]]
    """PEC A1 x values (pix). 1st index: star (ordered by index). 2nd index: time"""
    x_pec2: list[list[float]]
    """PEC A2 x values (pix). 1st index: star (ordered by index). 2nd index: time"""
    y_pec1: list[list[float]]
    """PEC A1 y values (pix). 1st index: star (ordered by index). 2nd index: time"""
    y_pec2: list[list[float]]
    """PEC A1 y values (pix). 1st index: star (ordered by index). 2nd index: time"""
    mag_pec1: list[list[float]]
    """PEC A1 magnitudes. 1st index: star (ordered by index). 2nd index: time"""
    mag_pec2: list[list[float]]
    """PEC A2 magnitudes. 1st index: star (ordered by index). 2nd index: time"""

    @classmethod
    def from_webmust_table(cls, table: Table, pec1: FgsPec, pec2: FgsPec): #TODO Use window status to discard bad data. Needs cross-match
        """
        Instantiate from WebMUST housekeeping data
        :param table: WebMUST HK input table
        :param pec1: PEC for first slot (A1)
        :param pec2: PEC for first slot (A2)
        """
        
        tdb = Time(table['UTC_STRING'], scale='tdb')
        
        #Group column data by 1-20 star index
        pec_columns = [np.vectorize(lambda x: FgsPec(x))(status2pec(table[column].value))
                                    for column in star_status_columns]
        valid_columns = [valid2int(table[column].value) for column in star_status_columns]
        window_columns = [window2int(table[column].value) for column in star_status_columns] # TODO use
        x_columns = [table[column] for column in star_x_columns]
        y_columns = [table[column] for column in star_y_columns]
        mag_columns = [table[column] for column in star_mag_columns]
        index_columns = [table[column] for column in star_index_columns]
        
        # Determine valid star indices for each PEC
        pec1_valid_columns = [(pec == pec1) & (window_id > 0) & valid for pec, window_id, valid in zip(pec_columns, window_columns, valid_columns)]
        pec2_valid_columns = [(pec == pec2) & (window_id > 0) & valid for pec, window_id, valid in zip(pec_columns, window_columns, valid_columns)]
        indices_pec1 = ndarray_to_list(np.unique(np.concatenate([index[mask == True] for index, mask in zip(index_columns, pec1_valid_columns)])))
        indices_pec2 = ndarray_to_list(np.unique(np.concatenate([index[mask == True] for index, mask in zip(index_columns, pec2_valid_columns)])))
        indices_pec1_to_sequence = {index_pec1: i for i, index_pec1 in enumerate(indices_pec1)}
        indices_pec2_to_sequence = {index_pec1: i for i, index_pec1 in enumerate(indices_pec2)}
        
        # Fill valid data
        windows_pec1 = np.full((len(indices_pec1), len(table)), -1, dtype=np.int8)
        windows_pec2 = np.full((len(indices_pec2), len(table)), -1, dtype=np.int8)
        x_pec1 = np.full((len(indices_pec1), len(table)), np.nan)
        x_pec2 = np.full((len(indices_pec2), len(table)), np.nan)
        y_pec1 = np.full((len(indices_pec1), len(table)), np.nan)
        y_pec2 = np.full((len(indices_pec2), len(table)), np.nan)
        mag_pec1 = np.full((len(indices_pec1), len(table)), np.nan)
        mag_pec2 = np.full((len(indices_pec2), len(table)), np.nan)
        for window_column, x, y, mag, index, pec1_valid, pec2_valid in zip(
                window_columns, x_columns, y_columns, mag_columns, index_columns, pec1_valid_columns, pec2_valid_columns):
            for i, (window, star_x, star_y, star_mag, star_index, star_pec1_valid, star_pec2_valid) in enumerate(zip(
                window_column, x, y, mag, index, pec1_valid, pec2_valid)):
                if star_pec1_valid:
                    windows_pec1[indices_pec1_to_sequence[star_index], i] = window
                    x_pec1[indices_pec1_to_sequence[star_index], i] = star_x
                    y_pec1[indices_pec1_to_sequence[star_index], i] = star_y
                    mag_pec1[indices_pec1_to_sequence[star_index], i] = star_mag
                if star_pec2_valid:
                    windows_pec2[indices_pec2_to_sequence[star_index], i] = window
                    x_pec2[indices_pec2_to_sequence[star_index], i] = star_x
                    y_pec2[indices_pec2_to_sequence[star_index], i] = star_y
                    mag_pec2[indices_pec2_to_sequence[star_index], i] = star_mag
        
        return cls(tdb, pec1, pec2, indices_pec1, indices_pec2,
                   ndarray_to_list(windows_pec1), ndarray_to_list(windows_pec2),
                   ndarray_to_list(x_pec1), ndarray_to_list(x_pec2),
                   ndarray_to_list(y_pec1), ndarray_to_list(y_pec2),
                   ndarray_to_list(mag_pec1), ndarray_to_list(mag_pec2))
    
    def to_dict(self) -> dict:
        """
        Serialize to dictionary. Times and FgsPec are converted to string
        :return: the serialized dictionary
        """
        dictt = asdict(self)
        dictt['tdb'] = [str(time) for time in self.tdb]
        dictt['pec1'] = self.pec1.name
        dictt['pec2'] = self.pec2.name
        return dictt
    
    @classmethod
    def from_dict(cls, dictt: dict):
        """
        Deserialize from dictionary. Times are converted to an Astropy Time object.
        :param dictt: the input dictionary
        """
        dict_copy = {key: value for key, value in dictt.items()}
        dict_copy['tdb'] = Time(dict_copy['tdb'], scale='tdb')
        dict_copy['pec1'] = FgsPec[dict_copy['pec1']]
        dict_copy['pec2'] = FgsPec[dict_copy['pec2']]
        return cls(**dict_copy)
        
    def to_json(self):
        """
        JSON dump.
        :return: a JSON dump of the input object.
        """
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_document: str, file_name: str = None):
        """
        JSON deserialization.
        :param json_document: document
        :param file_name: The input JSON file name, if applicable
        :return: the deserialized object
        """
        deserialized = cls.from_dict(json.loads(json_document))
        return deserialized

BAD_CENTROID_THRESHOLD = 2
"""Maximum acceptable difference with respect to the median for any given stellar centroid to be considered valid."""

@dataclass
class FgsStarTrackingIntervalStatistics:
    """Basic statistics carried out over the stars tracked by the FGS during a time interval"""
    time_start: Time
    """Interval start (TDB)"""
    time_end: Time
    """Interval end (TDB)"""
    pec1: FgsPec
    """PEC for first slot (A1)"""
    pec2: FgsPec
    """PEC for second slot (A2)"""
    x_medians_pec1: list[float]
    """PEC A1 x median values (pix). Index: star"""
    x_medians_pec2: list[float]
    """PEC A2 x median values (pix). Index: star"""
    y_medians_pec1: list[float]
    """PEC A1 y median values (pix). Index: star"""
    y_medians_pec2: list[float]
    """PEC A2 y median values (pix). Index: star"""
    mags_pec1: list[float]
    """PEC A1 magintude median values. Index: star"""
    mags_pec2: list[float]
    """PEC A2 magintude median values. Index: star"""
    x_centroiding_errors_pec1: list[float]
    """PEC A1 x centroiding errors, standard deviation of x - x_median (pix). Index: star"""
    x_centroiding_errors_pec2: list[float]
    """PEC A2 x centroiding errors, standard deviation of x - x_median (pix). Index: star"""
    y_centroiding_errors_pec1: list[float]
    """PEC A1 y centroiding errors, standard deviation of y - y_median (pix). Index: star"""
    y_centroiding_errors_pec2: list[float]
    """PEC A2 y centroiding errors, standard deviation of y - y_median (pix). Index: star"""
    x_median_std_pec1: float
    """PEC A1 x standard deviation of median of excursions around median values (pix)"""
    x_median_std_pec2: float
    """PEC A2 x standard deviation of median of excursions around median values (pix)"""
    y_median_std_pec1: float
    """PEC A1 y standard deviation of median of excursions around median values (pix)"""
    y_median_std_pec2: float
    """PEC A2 y standard deviation of median of excursions around median values (pix)"""
    x_median_drift_pec1: float
    """PEC A1 x median temporal drift (pix/s)"""
    x_median_drift_pec2: float
    """PEC A2 x median temporal drift (pix/s)"""
    y_median_drift_pec1: float
    """PEC A1 y median temporal drift (pix/s)"""
    y_median_drift_pec2: float
    """PEC A2 y median temporal drift (pix/s)"""
    
    @classmethod
    def from_fgs_star_tracking_interval(cls, interval: FgsStarTrackingInterval):
       
        # Basic data and averages
        time_start = interval.tdb[0]
        time_end = interval.tdb[-1]
        elapsed_s = (interval.tdb - time_start).to(u.s).value
        
        # PEC1
        if len(interval.x_pec1) > 0:
            x_medians_pec1 = ndarray_to_list(np.nanmedian(interval.x_pec1, axis=1))
            y_medians_pec1 = ndarray_to_list(np.nanmedian(interval.y_pec1, axis=1))
            mags_pec1 = ndarray_to_list(np.nanmedian(interval.mag_pec1, axis=1))
            x_median_excursions_pec1 = (np.transpose(interval.x_pec1) - x_medians_pec1).T
            y_median_excursions_pec1 = (np.transpose(interval.y_pec1) - y_medians_pec1).T
            x_median_excursion_pec1 = np.nanmedian(x_median_excursions_pec1, axis=0)
            y_median_excursion_pec1 = np.nanmedian(y_median_excursions_pec1, axis=0)
            tmp = x_median_excursions_pec1 - x_median_excursion_pec1
            tmp[np.abs(tmp) > BAD_CENTROID_THRESHOLD] = np.nan
            x_centroiding_errors_pec1 = ndarray_to_list(np.nanstd(tmp, axis=1))
            tmp = y_median_excursions_pec1 - y_median_excursion_pec1
            tmp[np.abs(tmp) > BAD_CENTROID_THRESHOLD] = np.nan
            y_centroiding_errors_pec1 = ndarray_to_list(np.nanstd(tmp, axis=1))
            x_median_std_pec1 = np.nanstd(x_median_excursion_pec1)
            y_median_std_pec1 = np.nanstd(y_median_excursion_pec1)
            x_median_drift_pec1_linregress = linregress(elapsed_s, x_median_excursion_pec1)
            x_median_drift_pec1 = x_median_drift_pec1_linregress.slope
            y_median_drift_pec1_linregress = linregress(elapsed_s, y_median_excursion_pec1)
            y_median_drift_pec1 = y_median_drift_pec1_linregress.slope
        else:
            x_medians_pec1 = []
            y_medians_pec1 = []
            mags_pec1 = []
            x_centroiding_errors_pec1 = []
            y_centroiding_errors_pec1 = []
            x_median_std_pec1 = float('nan') 
            y_median_std_pec1 = float('nan') 
            x_median_drift_pec1 = float('nan') 
            y_median_drift_pec1 = float('nan') 
        
        # PEC2
        if len(interval.x_pec2) > 0:
            x_medians_pec2 = ndarray_to_list(np.nanmedian(interval.x_pec2, axis=1))
            y_medians_pec2 = ndarray_to_list(np.nanmedian(interval.y_pec2, axis=1))
            mags_pec2 = ndarray_to_list(np.nanmedian(interval.mag_pec2, axis=1))
            x_median_excursions_pec2 = (np.transpose(interval.x_pec2) - x_medians_pec2).T
            y_median_excursions_pec2 = (np.transpose(interval.y_pec2) - y_medians_pec2).T
            x_median_excursion_pec2 = np.nanmedian(x_median_excursions_pec2, axis=0)
            y_median_excursion_pec2 = np.nanmedian(y_median_excursions_pec2, axis=0)
            tmp = x_median_excursions_pec2 - x_median_excursion_pec2
            tmp[np.abs(tmp) > BAD_CENTROID_THRESHOLD] = np.nan
            x_centroiding_errors_pec2 = ndarray_to_list(np.nanstd(tmp, axis=1))
            tmp = y_median_excursions_pec2 - y_median_excursion_pec2
            tmp[np.abs(tmp) > BAD_CENTROID_THRESHOLD] = np.nan
            y_centroiding_errors_pec2 = ndarray_to_list(np.nanstd(tmp, axis=1))
            x_median_std_pec2 = np.nanstd(x_median_excursion_pec2)
            y_median_std_pec2 = np.nanstd(y_median_excursion_pec2)
            x_median_drift_pec2_linregress = linregress(elapsed_s, x_median_excursion_pec2)
            x_median_drift_pec2 = x_median_drift_pec2_linregress.slope
            y_median_drift_pec2_linregress = linregress(elapsed_s, y_median_excursion_pec2)
            y_median_drift_pec2 = y_median_drift_pec2_linregress.slope
        else:
            x_medians_pec2 = []
            y_medians_pec2 = []
            mags_pec2 = []
            x_centroiding_errors_pec2 = []
            y_centroiding_errors_pec2 = []
            x_median_std_pec2 = float('nan') 
            y_median_std_pec2 = float('nan') 
            x_median_drift_pec2 = float('nan') 
            y_median_drift_pec2 = float('nan') 
        
        return cls(time_start, time_end, interval.pec1, interval.pec2,
                   x_medians_pec1, x_medians_pec2, y_medians_pec1, y_medians_pec2, mags_pec1, mags_pec2,
                   x_centroiding_errors_pec1, x_centroiding_errors_pec2, y_centroiding_errors_pec1, y_centroiding_errors_pec2, 
                   x_median_std_pec1, x_median_std_pec2, y_median_std_pec1, y_median_std_pec2,
                   x_median_drift_pec1, x_median_drift_pec2, y_median_drift_pec1, y_median_drift_pec2)
    
    def to_dict(self) -> dict:
        """
        Serialize to dictionary. Times and FgsPec are converted to string
        :return: the serialized dictionary
        """
        dictt = asdict(self)
        dictt['time_start'] = str(self.time_start)
        dictt['time_end'] = str(self.time_end)
        dictt['pec1'] = self.pec1.name
        dictt['pec2'] = self.pec2.name
        return dictt
    
    @classmethod
    def from_dict(cls, dictt: dict):
        """
        Deserialize from dictionary. Times are converted to an Astropy Time object.
        :param dictt: the input dictionary
        """
        dict_copy = {key: value for key, value in dictt.items()}
        dict_copy['time_start'] = Time(dict_copy['time_start'], scale='tdb')
        dict_copy['time_end'] = Time(dict_copy['time_end'], scale='tdb')
        dict_copy['pec1'] = FgsPec[dict_copy['pec1']]
        dict_copy['pec2'] = FgsPec[dict_copy['pec2']]
        return cls(**dict_copy)
        
    def to_json(self):
        """
        JSON dump.
        :return: a JSON dump of the input object.
        """
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_document: str, file_name: str = None):
        """
        JSON deserialization.
        :param json_document: document
        :param file_name: The input JSON file name, if applicable
        :return: the deserialized object
        """
        deserialized = cls.from_dict(json.loads(json_document))
        return deserialized

#
#
# Utility functions
#
#
def fgs_star_tracking_intervals_to_json_file(intervals: list[FgsStarTrackingInterval], file_name: str):
    """
    Serialize to json file
    :param intervals: The FGS star tracking intervals
    :param file_name: The output JSON file
    """
    json_str = json.dumps([interval.to_dict() for interval in intervals], indent=4)
    with open(file_name, 'w') as output_file:
        output_file.write(json_str)

def fgs_star_tracking_intervals_from_json_file(file_name: str) -> list[FgsStarTrackingInterval]:
    """
    Deserialize from json file
    :param file_name: The input JSON file
    :return: The FGS star tracking intervals
    """
    with open(file_name, 'r') as input_file:
        intervals = [FgsStarTrackingInterval.from_dict(ddict) for ddict in json.load(input_file)]
    return intervals

def fgs_star_tracking_interval_statistics_to_json_file(interval_statistics: list[FgsStarTrackingIntervalStatistics], file_name: str):
    """
    Serialize to json file
    :param interval_statistics: The FGS star tracking interval statisticss
    :param file_name: The output JSON file
    """
    json_str = json.dumps([this_statistics.to_dict() for this_statistics in interval_statistics], indent=4)
    with open(file_name, 'w') as output_file:
        output_file.write(json_str)

def fgs_star_tracking_interval_statistics_from_json_file(file_name: str) -> list[FgsStarTrackingIntervalStatistics]:
    """
    Deserialize from json file
    :param file_name: The input JSON file
    :return: The FGS star tracking interval statistics
    """
    with open(file_name, 'r') as input_file:
        intervals = [FgsStarTrackingIntervalStatistics.from_dict(ddict) for ddict in json.load(input_file)]
    return intervals
