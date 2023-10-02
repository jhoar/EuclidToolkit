import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import os
import csv

from Core.Utilities import TimeRange
import Core.Io as Io

from datetime import datetime, timezone

from astropy.io import ascii

def toDateTime(vals: list[str]) -> datetime:
    d = datetime.strptime(str(vals[0]) + ' ' +  str(vals[1]) + ' ' +  str(vals[2]) + ' ' +  str(vals[3]) + ' ' +  str(vals[4]), '%Y %j %H %M %S')
    d.replace(tzinfo=timezone.utc)
    return d

def main() -> int:

    # Create parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-d", "--dir", help = "Directory to process")

    # Read arguments from command line
    args = parser.parse_args()
    
    in_dir = 'vis'

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

    for path in Path(in_dir).rglob('EUC_AOC_VIS-65716-1-C_20230924T160946.000000Z_01_01_01.00.txt'):

        print(str(path))

        with open(str(path), 'r') as input_file:

            in_file = str(path.name)

            lines = list(csv.reader(input_file))

            start = toDateTime(lines[0])
            end = toDateTime(lines[1])
            time_range = TimeRange(start=start, end=end)

            try:
                table = Io.load_parameters(time_range, Io.rpe_params)
                ascii.write(table, in_file + '.csv', format='csv', overwrite=True)  

            except Exception as e:
                print(e, 'Failed to load data', str(path), time_range.start.strftime('%Y-%m-%dT%H:%M:%SZ'), time_range.end.strftime('%Y-%m-%dT%H:%M:%SZ') )
                continue

    return 0

if __name__ == '__main__':
    sys.exit(main())



