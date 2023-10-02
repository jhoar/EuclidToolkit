import astropy.units as u
from astropy.time import Time
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import time

from Core.Utilities import TimeRange
import Core.Io as Io
import FGSCentroid.Fgs as Fgs

def runAnalysis(time_range: TimeRange) -> tuple[ list[Fgs.FgsStarTrackingInterval], list[Fgs.FgsStarTrackingIntervalStatistics] ]:

    timing = False

    TIME_MINUS_INFINITY = Time('1900-01-01T00:00:00', scale='tdb')
    TIME_PLUS_INFINITY = Time('2100-01-01T00:00:00', scale='tdb')
    MIN_TIME_DIFFERENCE = 50 * u.s  # Interval duration is 2s larger

    start = time.time()
    mainclock = start
    if timing: print('Loading FGS tracking data: ', end="")
    tdb_fgs_mode, fgs_is_tracking_interval_min_indices, fgs_is_tracking_interval_max_indices = Io.fgs_is_tracking_direct(time_range)
    if timing: print(f'{(time.time() - start):.2f}s')

    start = time.time()
    if timing: print('Loading angular rate data: ', end="")
    tdb_fgs_use_angular_rate, fgs_use_angular_rate_interval_min_indices, fgs_use_angular_rate_interval_max_indices = Io.fgs_use_angular_rate_direct(time_range)
    if timing: print(f'{(time.time() - start):.2f}s')

    start = time.time()
    if timing: print('Loading active PEC data: ', end="")
    tdb_fgs_active_pec, fgs_active_pec1, fgs_active_pec2 = Io.fgs_active_pecs_direct(time_range)
    if timing: print(f'{(time.time() - start):.2f}s')

    start = time.time()
    if timing: print('Loading star data: ', end="")
    stars_table = Io.load_stars_table_direct(time_range)
    if timing: print(f'{(time.time() - start):.2f}s')

    # Create mask. Initially, all values are good
    start = time.time()
    if timing: print('Masking invalid time ranges')
    tdb_stars = Time(stars_table['UTC_STRING'], scale='tdb')
    tdb_stars_valid = np.full_like(tdb_stars, True, dtype=bool)
    if timing: print(f'Initial number of star data valid values: {np.count_nonzero(tdb_stars_valid)}')

    # Mask values when FGS is not tracking
    for time_min, time_max in zip([TIME_MINUS_INFINITY, *tdb_fgs_mode[fgs_is_tracking_interval_max_indices]],
                                [*tdb_fgs_mode[fgs_is_tracking_interval_min_indices], TIME_PLUS_INFINITY]):
        tdb_stars_valid[(tdb_stars > time_min) & (tdb_stars < time_max)] = False
    print(f'Current number of star data valid values after tracking filter: {np.count_nonzero(tdb_stars_valid)}')

    # Mask values when angular rate is not used
    for time_min, time_max in zip([TIME_MINUS_INFINITY, *tdb_fgs_use_angular_rate[fgs_use_angular_rate_interval_max_indices]],
                                [*tdb_fgs_use_angular_rate[fgs_use_angular_rate_interval_min_indices], TIME_PLUS_INFINITY]):
        tdb_stars_valid[(tdb_stars > time_min) & (tdb_stars < time_max)] = False
    print(f'Current number of star data valid values after angular rate filter: {np.count_nonzero(tdb_stars_valid)}')
    if timing: print(f'Masking complete in {(time.time() - start):.2f}s')

    # Find good data intervals
    if timing: print('Creating good time intervals: ', end="")
    start = time.time()
    tdb_stars_indices = np.arange(len(tdb_stars_valid))
    tdb_stars_labels_array, tdb_stars_n_labels = ndimage.label(tdb_stars_valid)
    tdb_stars_interval_min_indices, tdb_stars_interval_max_indices, _, _ = \
        ndimage.extrema(tdb_stars_indices, tdb_stars_labels_array,
                        index=np.arange(1, tdb_stars_n_labels + 1))

    # Filter by minimum time difference
    long_intervals = [tdb_stars[tdb_stars_interval_max_index] -  tdb_stars[tdb_stars_interval_min_index] >= MIN_TIME_DIFFERENCE
                    for tdb_stars_interval_min_index, tdb_stars_interval_max_index
                    in zip(tdb_stars_interval_min_indices, tdb_stars_interval_max_indices)]
    tdb_stars_long_interval_min_indices = tdb_stars_interval_min_indices[long_intervals]
    tdb_stars_long_interval_max_indices = tdb_stars_interval_max_indices[long_intervals]
    if timing: print(f'{(time.time() - start):.2f}s')
    print(f'Intervals longer than the minimum threshold {MIN_TIME_DIFFERENCE}: {np.count_nonzero(long_intervals)}')
   
    # Active PEC determination
    start = time.time()
    if timing: print('Determining active PECs: ', end="")
    tdb_stars_long_interval_pec1 = [
        fgs_active_pec1[(tdb_fgs_active_pec >= tdb_min) & (tdb_fgs_active_pec <= tdb_max)][0]
        for tdb_min, tdb_max in zip(
        tdb_stars[tdb_stars_long_interval_min_indices], tdb_stars[tdb_stars_long_interval_max_indices])]
    tdb_stars_long_interval_pec2 = [
        fgs_active_pec2[(tdb_fgs_active_pec >= tdb_min) & (tdb_fgs_active_pec <= tdb_max)][0]
        for tdb_min, tdb_max in zip(
        tdb_stars[tdb_stars_long_interval_min_indices], tdb_stars[tdb_stars_long_interval_max_indices])]
    if timing: print(f'{(time.time() - start):.2f}s')

    # Rearrange data into intervals
    start = time.time()
    if timing: print('Creating tracking intervals: ', end="")
    intervals = [
        Fgs.FgsStarTrackingInterval.from_webmust_table(stars_table[interval_min_index:interval_max_index+1], pec1, pec2)    
        for pec1, pec2, interval_min_index, interval_max_index in zip(
            tdb_stars_long_interval_pec1, tdb_stars_long_interval_pec2,
            tdb_stars_long_interval_min_indices, tdb_stars_long_interval_max_indices)
    ]
    if timing: print(f'{(time.time() - start):.2f}s')

    # Generate statistics
    start = time.time()
    if timing: print('Generating statistics: ', end="")
    interval_statistics = [
        Fgs.FgsStarTrackingIntervalStatistics.from_fgs_star_tracking_interval(interval)
        for interval in intervals
    ]
    if timing: print(f'{(time.time() - start):.2f}s')

    print(f'Completed in {(time.time() - mainclock):.2f}s')
    # return intervals and stats
    return intervals, interval_statistics

def create_plots(interval_statistics: list[Fgs.FgsStarTrackingIntervalStatistics]):
    x_median_std_pec1 = [stats.x_median_std_pec1 for stats in interval_statistics]
    x_median_std_pec2 = [stats.x_median_std_pec2 for stats in interval_statistics]
    y_median_std_pec1 = [stats.y_median_std_pec1 for stats in interval_statistics]
    y_median_std_pec2 = [stats.y_median_std_pec2 for stats in interval_statistics]
    x_median_drift_pec1 = [abs(stats.x_median_drift_pec1) for stats in interval_statistics]
    x_median_drift_pec2 = [abs(stats.x_median_drift_pec2) for stats in interval_statistics]
    y_median_drift_pec1 = [abs(stats.y_median_drift_pec1) for stats in interval_statistics]
    y_median_drift_pec2 = [abs(stats.y_median_drift_pec2) for stats in interval_statistics]
    duration_s = [(stats.time_end - stats.time_start).to(u.s).value for stats in interval_statistics]
    n_stars = [len(stats.x_medians_pec1) + len(stats.x_medians_pec2) for stats in interval_statistics]

    plt.figure()
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x_median_std_pec1, x_median_std_pec2, '.')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('x_median_std_pec1')
    axs[0].set_ylabel('x_median_std_pec2')
    axs[1].plot(y_median_std_pec1, y_median_std_pec2, '.')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('y_median_std_pec1')
    axs[1].set_ylabel('y_median_std_pec2')
    plt.savefig('plots/std_PEC1_x_PEC2.png', format='png', dpi=300)

    plt.figure()
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x_median_drift_pec1, x_median_drift_pec2, '.')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('x_median_drift_pec1')
    axs[0].set_ylabel('x_median_drift_pec2')
    axs[1].plot(y_median_drift_pec1, y_median_drift_pec2, '.')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('y_median_drift_pec1')
    axs[1].set_ylabel('y_median_drift_pec2')
    plt.savefig('plots/drift_PEC1_x_PEC2.png', format='png', dpi=300)

    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(x_median_std_pec1, x_median_drift_pec1, '.')
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')
    axs[0,1].plot(x_median_std_pec2, x_median_drift_pec2, '.')
    axs[0,1].set_xscale('log')
    axs[0,1].set_yscale('log')
    axs[1,0].plot(y_median_std_pec1, y_median_drift_pec1, '.')
    axs[1,0].set_xscale('log')
    axs[1,0].set_yscale('log')
    axs[1,1].plot(y_median_std_pec2, y_median_drift_pec2, '.')
    axs[1,1].set_xscale('log')
    axs[1,1].set_yscale('log')
    plt.savefig('plots/std_x_drift.png', format='png', dpi=300)

    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(duration_s, x_median_drift_pec1, '.')
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')
    axs[0,1].plot(duration_s, x_median_drift_pec2, '.')
    axs[0,1].set_xscale('log')
    axs[0,1].set_yscale('log')
    axs[1,0].plot(duration_s, y_median_drift_pec1, '.')
    axs[1,0].set_xscale('log')
    axs[1,0].set_yscale('log')
    axs[1,1].plot(duration_s, y_median_drift_pec2, '.')
    axs[1,1].set_xscale('log')
    axs[1,1].set_yscale('log')
    plt.savefig('plots/duration_x_drift.png', format='png', dpi=300)

    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(duration_s, x_median_std_pec1, '.')
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')
    axs[0,1].plot(duration_s, x_median_std_pec2, '.')
    axs[0,1].set_xscale('log')
    axs[0,1].set_yscale('log')
    axs[1,0].plot(duration_s, y_median_std_pec1, '.')
    axs[1,0].set_xscale('log')
    axs[1,0].set_yscale('log')
    axs[1,1].plot(duration_s, y_median_std_pec2, '.')
    axs[1,1].set_xscale('log')
    axs[1,1].set_yscale('log')
    plt.savefig('plots/duration_x_std.png', format='png', dpi=300)

    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(n_stars, x_median_std_pec1, '.')
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')
    axs[0,1].plot(n_stars, x_median_std_pec2, '.')
    axs[0,1].set_xscale('log')
    axs[0,1].set_yscale('log')
    axs[1,0].plot(n_stars, y_median_std_pec1, '.')
    axs[1,0].set_xscale('log')
    axs[1,0].set_yscale('log')
    axs[1,1].plot(n_stars, y_median_std_pec2, '.')
    axs[1,1].set_xscale('log')
    axs[1,1].set_yscale('log')
    plt.savefig('plots/n_stars_x_std.png', format='png', dpi=300)

    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(n_stars, x_median_drift_pec1, '.')
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')
    axs[0,1].plot(n_stars, x_median_drift_pec2, '.')
    axs[0,1].set_xscale('log')
    axs[0,1].set_yscale('log')
    axs[1,0].plot(n_stars, y_median_drift_pec1, '.')
    axs[1,0].set_xscale('log')
    axs[1,0].set_yscale('log')
    axs[1,1].plot(n_stars, y_median_drift_pec2, '.')
    axs[1,1].set_xscale('log')
    axs[1,1].set_yscale('log')
    plt.savefig('plots/n_stars_x_drift.png', format='png', dpi=300)

    plt.figure()
    plt.hist(duration_s, bins=np.geomspace(10,10000, 121))
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('plots/duration_histo.png', format='png', dpi=300)
