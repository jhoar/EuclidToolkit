import argparse
import sys
from datetime import datetime, timezone, timedelta
import csv
import numpy as np
import math

from Core.Utilities import TimeRange
import FGSCentroid.Analysis as Analysis
import Core.Io as Io

RAD_TO_MAS = 3600.0 * 1000.0 * 180.0 / math.pi

def paramsToStd(x, y, z):
    x = np.std(np.array(x).astype(float)) * RAD_TO_MAS
    y = np.std(np.array(y).astype(float)) * RAD_TO_MAS
    z = np.std(np.array(z).astype(float)) * RAD_TO_MAS

    return x, y, z

def main() -> int:

    k_score = {}

    with open('kuijken_score.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            file = row[0]
            score = float(row[7])
            k_score[file] = {}
            k_score[file]['score'] = score

    with open('vis.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            exp = float(row['Data.ExposureTime.Value'])
            start = datetime.strptime(row['Data.ObservationDateTime.UTC'], '%Y-%m-%d %H:%M:%S.%f')
            end = start + timedelta(seconds=exp)

            file = row['Data.FrameFitsFile.DataContainer.FileName'][20:-17]
            if file in k_score:
                k_score[file]['start'] = start
                k_score[file]['end'] = end

    with open('calibrate.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        writer.writerow([
            'file',
            'start',
            'end',
            'score',
            'x_median_std_pec1', 
            'y_median_std_pec1', 
            'x_median_drift_pec1', 
            'y_median_drift_pec1', 
            'x_median_std_pec2', 
            'y_median_std_pec2', 
            'x_median_drift_pec2', 
            'y_median_drift_pec2',
            'x_std',
            'y_std',
            'z_std'
        ])

        for file in k_score:
            range = TimeRange(start=k_score[file]['start'], end=k_score[file]['end'])

            # And set the input here
            time_range = range

            rpe = Io.load_pointing_error_direct(time_range)
            x_std, y_std , z_std = paramsToStd(rpe['DB_AOCS_AHK_C_AX_ER_A_SP'], rpe['DB_AOCS_AHK_C_AY_ER_A_SP'], rpe['DB_AOCS_AHK_C_AZ_ER_A_SP'])

            row = []
            print(f'Running analysis between {time_range.start} and {time_range.end}')
            try:
                _, interval_statistics = Analysis.runAnalysis(time_range)
                if len(interval_statistics) > 0:
                    row = [
                        file,
                        k_score[file]['start'],
                        k_score[file]['end'],
                        k_score[file]['score'],
                        interval_statistics[0].x_median_std_pec1, 
                        interval_statistics[0].y_median_std_pec1, 
                        interval_statistics[0].x_median_drift_pec1, 
                        interval_statistics[0].y_median_drift_pec1, 
                        interval_statistics[0].x_median_std_pec2, 
                        interval_statistics[0].y_median_std_pec2, 
                        interval_statistics[0].x_median_drift_pec2, 
                        interval_statistics[0].y_median_drift_pec2,
                        x_std, 
                        y_std, 
                        z_std
                    ] 
                else:
                    print('****No internals:  with k_Score {0}'.format(k_score[file]['score']))
                    row = [
                        file,
                        k_score[file]['start'],
                        k_score[file]['end'],
                        k_score[file]['score'],
                        'nan', 
                        'nan', 
                        'nan', 
                        'nan', 
                        'nan', 
                        'nan', 
                        'nan', 
                        'nan', 
                        x_std, 
                        y_std, 
                        z_std
                    ] 

            except:
                print('****No data: with k_Score {0}'.format(k_score[file]['score']))
                row = [
                    file,
                    k_score[file]['start'],
                    k_score[file]['end'],
                    k_score[file]['score'],
                    'nan', 
                    'nan', 
                    'nan', 
                    'nan', 
                    'nan', 
                    'nan', 
                    'nan', 
                    'nan', 
                    x_std, 
                    y_std, 
                    z_std
                ]

            writer.writerow(row)

    return 0

if __name__ == '__main__':
    sys.exit(main())



