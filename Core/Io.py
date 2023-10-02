from astropy.time import Time
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy import ndimage
import sys

from Core.Utilities import TimeRange
from Core.Hms import Hms
from Core.Timeline import generateTable
from FGSCentroid.Fgs import FgsPec

star_params = [
    Hms.getParameterDef('FAAT9000', False),
    Hms.getParameterDef('FAAT9014', False),
    Hms.getParameterDef('FAAT9028', False),
    Hms.getParameterDef('FAAT9042', False),
    Hms.getParameterDef('FAAT9056', False),
    Hms.getParameterDef('FAAT9070', False),
    Hms.getParameterDef('FAAT9084', False),
    Hms.getParameterDef('FAAT9098', False),
    Hms.getParameterDef('FAAT9112', False),
    Hms.getParameterDef('FAAT9126', False),
    Hms.getParameterDef('FAAT9140', False),
    Hms.getParameterDef('FAAT9154', False),
    Hms.getParameterDef('FAAT9168', False),
    Hms.getParameterDef('FAAT9182', False),
    Hms.getParameterDef('FAAT9196', False),
    Hms.getParameterDef('FAAT9210', False),
    Hms.getParameterDef('FAAT9224', False),
    Hms.getParameterDef('FAAT9238', False),
    Hms.getParameterDef('FAAT9252', False),
    Hms.getParameterDef('FAAT9266', False),
    Hms.getParameterDef('FAAT9008', False),
    Hms.getParameterDef('FAAT9022', False),
    Hms.getParameterDef('FAAT9036', False),
    Hms.getParameterDef('FAAT9050', False),
    Hms.getParameterDef('FAAT9064', False),
    Hms.getParameterDef('FAAT9078', False),
    Hms.getParameterDef('FAAT9092', False),
    Hms.getParameterDef('FAAT9106', False),
    Hms.getParameterDef('FAAT9120', False),
    Hms.getParameterDef('FAAT9134', False),
    Hms.getParameterDef('FAAT9148', False),
    Hms.getParameterDef('FAAT9162', False),
    Hms.getParameterDef('FAAT9176', False),
    Hms.getParameterDef('FAAT9190', False),
    Hms.getParameterDef('FAAT9204', False),
    Hms.getParameterDef('FAAT9218', False),
    Hms.getParameterDef('FAAT9232', False),
    Hms.getParameterDef('FAAT9246', False),
    Hms.getParameterDef('FAAT9260', False),
    Hms.getParameterDef('FAAT9274', False),
    Hms.getParameterDef('FAAT9010', False),
    Hms.getParameterDef('FAAT9024', False),
    Hms.getParameterDef('FAAT9038', False),
    Hms.getParameterDef('FAAT9052', False),
    Hms.getParameterDef('FAAT9066', False),
    Hms.getParameterDef('FAAT9080', False),
    Hms.getParameterDef('FAAT9094', False),
    Hms.getParameterDef('FAAT9108', False),
    Hms.getParameterDef('FAAT9122', False),
    Hms.getParameterDef('FAAT9136', False),
    Hms.getParameterDef('FAAT9150', False),
    Hms.getParameterDef('FAAT9164', False),
    Hms.getParameterDef('FAAT9178', False),
    Hms.getParameterDef('FAAT9192', False),
    Hms.getParameterDef('FAAT9206', False),
    Hms.getParameterDef('FAAT9220', False),
    Hms.getParameterDef('FAAT9234', False),
    Hms.getParameterDef('FAAT9248', False),
    Hms.getParameterDef('FAAT9262', False),
    Hms.getParameterDef('FAAT9276', False),
    Hms.getParameterDef('FAAT9011', False),
    Hms.getParameterDef('FAAT9025', False),
    Hms.getParameterDef('FAAT9039', False),
    Hms.getParameterDef('FAAT9053', False),
    Hms.getParameterDef('FAAT9067', False),
    Hms.getParameterDef('FAAT9081', False),
    Hms.getParameterDef('FAAT9095', False),
    Hms.getParameterDef('FAAT9109', False),
    Hms.getParameterDef('FAAT9123', False),
    Hms.getParameterDef('FAAT9137', False),
    Hms.getParameterDef('FAAT9151', False),
    Hms.getParameterDef('FAAT9165', False),
    Hms.getParameterDef('FAAT9179', False),
    Hms.getParameterDef('FAAT9193', False),
    Hms.getParameterDef('FAAT9207', False),
    Hms.getParameterDef('FAAT9221', False),
    Hms.getParameterDef('FAAT9235', False),
    Hms.getParameterDef('FAAT9249', False),
    Hms.getParameterDef('FAAT9263', False),
    Hms.getParameterDef('FAAT9277', False),
    Hms.getParameterDef('FAAT9012', False),
    Hms.getParameterDef('FAAT9026', False),
    Hms.getParameterDef('FAAT9040', False),
    Hms.getParameterDef('FAAT9054', False),
    Hms.getParameterDef('FAAT9068', False),
    Hms.getParameterDef('FAAT9082', False),
    Hms.getParameterDef('FAAT9096', False),
    Hms.getParameterDef('FAAT9110', False),
    Hms.getParameterDef('FAAT9124', False),
    Hms.getParameterDef('FAAT9138', False),
    Hms.getParameterDef('FAAT9152', False),
    Hms.getParameterDef('FAAT9166', False),
    Hms.getParameterDef('FAAT9180', False),
    Hms.getParameterDef('FAAT9194', False),
    Hms.getParameterDef('FAAT9208', False),
    Hms.getParameterDef('FAAT9222', False),
    Hms.getParameterDef('FAAT9236', False),
    Hms.getParameterDef('FAAT9250', False),
    Hms.getParameterDef('FAAT9264', False),
    Hms.getParameterDef('FAAT9278', False)
]

pec_params = [
    Hms.getParameterDef('FJJT0239', True),
    Hms.getParameterDef('FJJT0243', True)
]


rpe_params = [
    Hms.getParameterDef('APPT1088', False),
    Hms.getParameterDef('APPT1089', False),
    Hms.getParameterDef('APPT1090', False)
]

fgkf_params = [
    Hms.getParameterDef('APPT0051', False),
    Hms.getParameterDef('APPT0052', False),
    Hms.getParameterDef('APPT0053', False)
]

qoff_params = [
    Hms.getParameterDef('APPT0601', False),
    Hms.getParameterDef('APPT0603', False),
    Hms.getParameterDef('APPT0605', False),
    Hms.getParameterDef('APPT0607', False)
]

fgs_mode_params = [ 
    Hms.getParameterDef('FAAT2010', True) 
]

aocs_state_params = [ 
    Hms.getParameterDef('APPT0838', True) 
]

fgs_use_angular_rate_params = [ 
    Hms.getParameterDef('FJJT0247', True) 
]

def refresh(days: int):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
        
    _ = generateTable(start, end, rpe_params, force=True)
    _ = generateTable(start, end, star_params, force=True)
    _ = generateTable(start, end, fgs_mode_params, force=True)
    _ = generateTable(start, end, fgs_use_angular_rate_params, force=True)
    _ = generateTable(start, end, pec_params, force=True)
    print('Complete')

def load_parameters(time_range: TimeRange, parameter_list: dict):
    table = generateTable(time_range.start, time_range.end, parameter_list)
    if len(table) == 0:
        raise Exception("Empty table")

    return table

def load_pointing_error_direct(time_range: TimeRange):
    table = generateTable(time_range.start, time_range.end, rpe_params)
    if len(table) == 0:
        raise Exception("Empty RPE table")

    return table

def load_fgs_kalman_direct(time_range: TimeRange):
    table = generateTable(time_range.start, time_range.end, fgkf_params)
    if len(table) == 0:
        raise Exception("Empty FGKF table")

    return table

def load_q_offset_direct(time_range: TimeRange):
    table = generateTable(time_range.start, time_range.end, qoff_params)
    if len(table) == 0:
        raise Exception("Empty Q_OFF table")

    return table


def load_stars_table_direct(time_range: TimeRange):
    star_table = generateTable(time_range.start, time_range.end, star_params)
    if len(star_table) == 0:
        raise Exception("Empty star table")

    return star_table

def fgs_is_tracking_direct(time_range: TimeRange):
    fgs_mode_table = generateTable(time_range.start, time_range.end, fgs_mode_params)
    if len(fgs_mode_table) == 0:
        raise Exception("Empty FGS mode table")

    # FGS tracking mode intervals
    tdb_fgs_mode = Time(fgs_mode_table['UTC_STRING'], scale='tdb')
    fgs_is_tracking = (fgs_mode_table['FGS_OPMODE'] == "ATM_TP") | (fgs_mode_table['FGS_OPMODE'] == "RTM_TP")
    fgs_is_tracking_indices = np.arange(len(fgs_is_tracking))
    fgs_is_tracking_labels_array, fgs_is_tracking_n_labels = ndimage.label(fgs_is_tracking)
    fgs_is_tracking_interval_min_indices, fgs_is_tracking_interval_max_indices, _, _ = \
        ndimage.extrema(fgs_is_tracking_indices, fgs_is_tracking_labels_array,
                        index=np.arange(1, fgs_is_tracking_n_labels + 1))

    return tdb_fgs_mode, fgs_is_tracking_interval_min_indices, fgs_is_tracking_interval_max_indices

def fgs_use_angular_rate_direct(time_range: TimeRange):
    fgs_use_angular_rate_table = generateTable(time_range.start, time_range.end, fgs_use_angular_rate_params)
    if len(fgs_use_angular_rate_table) == 0:
        raise Exception("Empty angular rate table")

    # Use angular rate intervals
    tdb_fgs_use_angular_rate = Time(fgs_use_angular_rate_table['UTC_STRING'], scale='tdb')
    fgs_use_angular_rate = fgs_use_angular_rate_table['FGS_MS1-UseCalcAngRate'] == "Calculated"
    fgs_use_angular_rate_indices = np.arange(len(fgs_use_angular_rate))
    fgs_use_angular_rate_labels_array, fgs_use_angular_rate_n_labels = ndimage.label(fgs_use_angular_rate)
    fgs_use_angular_rate_interval_min_indices, fgs_use_angular_rate_interval_max_indices, _, _ = \
        ndimage.extrema(fgs_use_angular_rate_indices, fgs_use_angular_rate_labels_array,
                        index=np.arange(1, fgs_use_angular_rate_n_labels + 1))

    return tdb_fgs_use_angular_rate, fgs_use_angular_rate_interval_min_indices, fgs_use_angular_rate_interval_max_indices

def fgs_active_pecs_direct(time_range: TimeRange):
    fgs_active_pecs_table = generateTable(time_range.start, time_range.end, pec_params)
    if len(fgs_active_pecs_table) == 0:
        raise Exception('Empty active PEC table')

    # Active PECs
    tdb_fgs_active_pec = Time(fgs_active_pecs_table['UTC_STRING'], scale='tdb')
    fgs_active_pec1 = np.array([FgsPec[pec_string[0:2]] for pec_string in fgs_active_pecs_table['FGS_MS1-PEC_A1']])
    fgs_active_pec2 = np.array([FgsPec[pec_string[0:2]] for pec_string in fgs_active_pecs_table['FGS_MS1-PEC_A2']])

    return tdb_fgs_active_pec, fgs_active_pec1, fgs_active_pec2

