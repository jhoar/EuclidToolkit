import argparse
import sys
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from astropy.table import Table
from astropy.time import Time

from numpy import ndarray

from Core.Utilities import TimeRange, RAD_TO_MAS
import FGSCentroid.Analysis as Analysis
import Core.Io as Io
import FGSCentroid.Fgs as Fgs

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable






def plot(name: str, pairs: list[list[str]]):

    params = {
        'axes.labelsize': 'x-small',
        'figure.titlesize': 'x-small', 
        'axes.titlesize':'x-small',
        'ytick.labelsize': 'x-small',
        'xtick.labelsize': 'x-small',
        'legend.fontsize': 'x-small'
        }

    pylab.rcParams.update(params)
    plt.clf()

    nSeries = len(pairs)
    fig, axs = plt.subplots(nSeries, 1, sharex='all')
    fig.suptitle(str(len(pairs[0][0])) + ' samples')
    for plotnum in range(0, len(pairs)):
        t = pairs[plotnum][0]
        s = pairs[plotnum][1]
        n = pairs[plotnum][2]
        size = np.full(len(t), 1)
        axs[plotnum].scatter(t, s[n], s=size, label=n)
        axs[plotnum].legend(loc="upper right")
        axs[plotnum].grid()

    plt.savefig(name, format='png', dpi=300)
    plt.close()





def commonTimeline(a_array: ndarray, b_array: ndarray, a_name: str, b_name: str) -> dict:
    a_array['POSIX_S'] = np.array(a_array['POSIX'] / 1000.0).astype(int) 
    b_array['POSIX_S'] = np.array(b_array['POSIX'] / 1000.0).astype(int)

    ts = {}

    for row in a_array:
        ts[row['POSIX_S']] = {}

    for row in b_array:
        ts[row['POSIX_S']] = {}

    for row in a_array:
        ts[row['POSIX_S']][a_name] = row[a_name]

    for row in b_array:
        ts[row['POSIX_S']][b_name] = row[b_name]

    return ts





def function(a_array: ndarray, b_array: ndarray, a_name: str, b_name: str, func):
    ts = commonTimeline(a_array, b_array, a_name, b_name)

    timeStamp = []
    result = []

    for row in ts:
        v = ts[row]
        if a_name in v and b_name in v:
            timeStamp.append(row)
            result.append(func(v[a_name], v[b_name]))

    return Table([timeStamp, result], names=['POSIX_S', 'RESULT'])






def toDateTime(vals: list[str]) -> datetime:
    d = datetime.strptime(str(vals[0]) + ' ' +  str(vals[1]) + ' ' +  str(vals[2]) + ' ' +  str(vals[3]) + ' ' +  str(vals[4]), '%Y %j %H %M %S')
    return d.replace(tzinfo=timezone.utc)






def runAnalysis(in_dir: str, glob: str) -> None:

    all_stats = []

    for path in Path(in_dir).rglob(glob):
        statistics_filename = 'statistics/' + path.name.replace('AOC', 'FGS').replace('txt','json')
        intervals_filename = 'intervals/' + path.name.replace('AOC', 'FGS').replace('txt','json')

        with open(str(path), 'r') as input_file:
            lines = list(csv.reader(input_file))

            start = toDateTime(lines[0])
            end = toDateTime(lines[1])
            time_range = TimeRange(start=start, end=end)

            print(f'Running analysis between {time_range.start} and {time_range.end}')
            intervals, interval_statistics = Analysis.runAnalysis(time_range)

            Fgs.fgs_star_tracking_intervals_to_json_file(intervals, intervals_filename)
            Fgs.fgs_star_tracking_interval_statistics_to_json_file(interval_statistics, statistics_filename)
            
            all_stats.extend(interval_statistics)

        Analysis.create_plots(all_stats)    






def createPlots(in_dir: str, glob: str) -> None:

    xsig = []
    ysig = []
    zsig = []
    cols = []

    for path in Path(in_dir).rglob(glob):

        print(str(path))

        with open(str(path), 'r') as input_file:

            lines = list(csv.reader(input_file))

            start = toDateTime(lines[0])
            end = toDateTime(lines[1])
            time_range = TimeRange(start=start, end=end)
            utc = time_range.start.strftime('%Y-%m-%dT%H:%M:%SZ')

            delta_s = (end - start).total_seconds()
            if delta_s < 1.0:
                continue

            try:
                con = Io.load_pointing_error_direct(time_range)
                fgkf = Io.load_fgs_kalman_direct(time_range)
                q_off = Io.load_q_offset_direct(time_range)
            except Exception as e:
                print(e, 'Failed to load data', str(path), time_range.start.strftime('%Y-%m-%dT%H:%M:%SZ'), time_range.end.strftime('%Y-%m-%dT%H:%M:%SZ') )
                continue

            con['CONT_ERR_X'] = np.array(con['DB_AOCS_AHK_C_AX_ER_A_SP']).astype(float) * -RAD_TO_MAS
            con['CONT_ERR_Y'] = np.array(con['DB_AOCS_AHK_C_AY_ER_A_SP']).astype(float) * -RAD_TO_MAS
            con['CONT_ERR_Z'] = np.array(con['DB_AOCS_AHK_C_AZ_ER_A_SP']).astype(float) * -RAD_TO_MAS
            conTime = Time(con['POSIX'].astype(float) / 1000.0, format='unix_tai')

            fgkf['FGKF_X'] = np.array(fgkf['DB_AOCS_AHK_FGKF_A1_L2A']).astype(float) * RAD_TO_MAS
            fgkf['FGKF_Y'] = np.array(fgkf['DB_AOCS_AHK_FGKF_A2_L2A']).astype(float) * RAD_TO_MAS
            fgkf['FGKF_Z'] = np.array(fgkf['DB_AOCS_AHK_FGKF_A3_L2A']).astype(float) * RAD_TO_MAS
            fgkfTime = Time(fgkf['POSIX'].astype(float) / 1000.0, format='unix_tai')

            q_off['DELTA_Q_X'] = np.array(q_off['DB_AOCS_AHK_FGS_QX_L2A']).astype(float) * 360000000.1486299
            q_off['DELTA_Q_Y'] = np.array(q_off['DB_AOCS_AHK_FGS_QY_L2A']).astype(float) * 360000000.1486299
            q_off['DELTA_Q_Z'] = np.array(q_off['DB_AOCS_AHK_FGS_QZ_L2A']).astype(float) * 360000000.1486299
            q_off['DELTA_Q_S'] = np.array(q_off['DB_AOCS_AHK_FGS_QS_L2A']).astype(float) 
            qTime = Time(q_off['POSIX'].astype(float) / 1000.0, format='unix_tai')

            plot('plots/' + str(path.name) + '_X.png', 
                 (
                    (qTime.mjd, q_off, 'DELTA_Q_X'), 
                    (fgkfTime.mjd, fgkf, 'FGKF_X'),
                    (conTime.mjd, con, 'CONT_ERR_X') 
                 )
            )

            plot('plots/' + str(path.name) + '_Y.png', 
                 (
                    (qTime.mjd, q_off, 'DELTA_Q_Y'), 
                    (fgkfTime.mjd, fgkf, 'FGKF_Y'),
                    (conTime.mjd, con, 'CONT_ERR_Y') 
                 )
            )

            plot('plots/' + str(path.name) + '_Z.png', 
                 (
                    (qTime.mjd, q_off, 'DELTA_Q_Z'), 
                    (fgkfTime.mjd, fgkf, 'FGKF_Z'),
                    (conTime.mjd, con, 'CONT_ERR_Z') 
                 )
            )

            sub = lambda x, y: x - y
            X_DIFF = function(q_off, con, 'DELTA_Q_X', 'CONT_ERR_X', sub)
            Y_DIFF = function(q_off, con, 'DELTA_Q_Y', 'CONT_ERR_Y', sub)
            Z_DIFF = function(q_off, con, 'DELTA_Q_Z', 'CONT_ERR_Z', sub)
            diffTime = Time(X_DIFF['POSIX_S'].astype(float), format='unix_tai')


            x_sigma = np.std(con['CONT_ERR_X'])
            y_sigma = np.std(con['CONT_ERR_Y'])
            z_sigma = np.std(con['CONT_ERR_Z'])

            success = x_sigma < 25.0 and y_sigma < 25.0 and z_sigma < 500

            if len(con['CONT_ERR_X']) > 30 and success:
                # xsig.append(25.0 / x_sigma)
                # ysig.append(25.0 / y_sigma)
                # zsig.append(500.0 / z_sigma)
                xsig.append(x_sigma)
                ysig.append(y_sigma)
                zsig.append(z_sigma)
                cols.append(len(con['CONT_ERR_X']))

            name = str(path.name) + '.png'

            params = {
                'axes.labelsize': 'x-small',
                'figure.titlesize': 'x-small', 
                'axes.titlesize':'x-small',
                'ytick.labelsize': 'x-small',
                'xtick.labelsize': 'x-small',
                'legend.fontsize': 'x-small'
            }

            fgkf_s = np.full(len(fgkf['FGKF_X']), 1)
            con_s = np.full(len(con['CONT_ERR_X']), 1)
            q_off_s = np.full(len(q_off['DELTA_Q_X']), 1)
            diff_x_s = np.full(len(X_DIFF['RESULT']), 1)
            diff_y_s = np.full(len(Y_DIFF['RESULT']), 1)
            diff_z_s = np.full(len(Z_DIFF['RESULT']), 1)

            pylab.rcParams.update(params)
            plt.clf()

            fig, axs = plt.subplots(4, 3, sharex='all', sharey='col')
            fig.suptitle(str(path.name) + " " + utc + ' ' + str(len(fgkf['FGKF_X'])) + ' samples')

            axs[0,0].set_title('σ(q(x)) = ' + str(np.std(q_off['DELTA_Q_X'])))
            axs[0,0].scatter(qTime.mjd, q_off['DELTA_Q_X'], s=q_off_s, label='δQ(X)')
            axs[0,0].legend(loc="upper right")

            axs[0,1].set_title('σ(q(y)) = ' + str(np.std(q_off['DELTA_Q_Y'])))
            axs[0,1].scatter(qTime.mjd, q_off['DELTA_Q_Y'], s=q_off_s, label='δQ(Y)')
            axs[0,1].legend(loc="upper right")

            axs[0,2].set_title('σ(q(z)) = ' + str(np.std(q_off['DELTA_Q_Z'])))
            axs[0,2].scatter(qTime.mjd, q_off['DELTA_Q_Z'], s=q_off_s, label='δQ(Z)')
            axs[0,2].legend(loc="upper right")

            axs[1,0].set_title('σ(fgkf(x)) = ' + str(np.std(fgkf['FGKF_X'])))
            axs[1,0].scatter(fgkfTime.mjd, fgkf['FGKF_X'], s=fgkf_s, label='FGKF(X)')
            axs[1,0].legend(loc="upper right")

            axs[1,1].set_title('σ(fgkf(y)) = ' + str(np.std(fgkf['FGKF_Y'])))
            axs[1,1].scatter(fgkfTime.mjd, fgkf['FGKF_Y'], s=fgkf_s, label='FGKF(Y)')
            axs[1,1].legend(loc="upper right")

            axs[1,2].set_title('σ(fgkf(z)) = ' + str(np.std(fgkf['FGKF_Z'])))
            axs[1,2].scatter(fgkfTime.mjd, fgkf['FGKF_Z'], s=fgkf_s, label='FGKF(Z)')
            axs[1,2].legend(loc="upper right")

            axs[2,0].set_title('σ(con(x)) = ' + str(np.std(con['CONT_ERR_X'])))
            axs[2,0].scatter(conTime.mjd, con['CONT_ERR_X'], s=con_s, label='-CON(X)')
            axs[2,0].legend(loc="upper right")

            axs[2,1].set_title('σ(con(y)) = ' + str(np.std(con['CONT_ERR_Y'])))
            axs[2,1].scatter(conTime.mjd, con['CONT_ERR_Y'], s=con_s, label='-CON(Y)')
            axs[2,1].legend(loc="upper right")

            axs[2,2].set_title('σ(con(z)) = ' + str(np.std(con['CONT_ERR_Z'])))
            axs[2,2].scatter(conTime.mjd, con['CONT_ERR_Z'], s=con_s, label='-CON(Z)')
            axs[2,2].legend(loc="upper right")

            axs[3,0].set_title('σ(diff(x)) = ' + str(np.std(X_DIFF['RESULT'])))
            axs[3,0].scatter(diffTime.mjd, X_DIFF['RESULT'], s=diff_x_s, label='δQ(X)-CON(X)')
            axs[3,0].legend(loc="upper right")

            axs[3,1].set_title('σ(diff(y)) = ' + str(np.std(Y_DIFF['RESULT'])))
            axs[3,1].scatter(diffTime.mjd, Y_DIFF['RESULT'], s=diff_y_s, label='δQ(Y)-CON(Y)')
            axs[3,1].legend(loc="upper right")

            axs[3,2].set_title('σ(diff(z)) = ' + str(np.std(Z_DIFF['RESULT'])))
            axs[3,2].scatter(diffTime.mjd, Z_DIFF['RESULT'], s=diff_z_s, label='δQ(Z)-CON(Z)')
            axs[3,2].legend(loc="upper right")

            plt.savefig('plots/' + name, format='png', dpi=300)

            plt.close()

    _, axs = plt.subplots(2, 2)
    nbins=50

    axs[0,0].hist(xsig, bins=nbins)
    axs[0,0].set_title('σ(RPE(CON(X))) at 25mas')

    axs[0,1].hist(ysig, bins=nbins)
    axs[0,1].set_title('σ(RPE(CON(Y))) at 25mas')

    axs[1,0].hist(zsig, bins=nbins)
    axs[1,0].set_title('σ(RPE(CON(Z))) at 500mas')

    scatter = axs[1,1].scatter(xsig, ysig, c=cols, s=np.full(len(xsig), 4), cmap='gist_rainbow')
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(scatter,cax=cax)

    axs[1,1].set_aspect('equal')
    axs[1,1].set_title('X vs Y, colour(exp_dur)')

    plt.savefig('histo_sig.png', format='png', dpi=300)


def main() -> int:

    # Create parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-d", "--dir", help = "Directory to process")

    # Read arguments from command line
    args = parser.parse_args()
    
    in_dir = 'data'

    if args.dir:
        in_dir = args.dir

    if not os.path.exists(in_dir):
        print('Input directory not found')
        sys.exit(1)
 
    if not os.path.exists('statistics'):
        os.makedirs('statistics', exist_ok=True)

    if not os.path.exists('intervals'):
        os.makedirs('intervals', exist_ok=True)        
        
    if not os.path.exists('plots'):
        os.makedirs('plots', exist_ok=True)


    createPlots(in_dir, 'EUC_AOC_VIS-*-1-C_*.txt')
    # runAnalysis(in_dir, 'EUC_AOC_VIS-65744-1-C_20230924T160946.000000Z_01_01_01.00.txt')

    return 0

if __name__ == '__main__':
    sys.exit(main())



