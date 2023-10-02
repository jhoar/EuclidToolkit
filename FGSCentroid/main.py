import argparse
import sys
from datetime import datetime, timezone, timedelta

from Core.Utilities import TimeRange
import FGSCentroid.Fgs as Fgs
import FGSCentroid.Analysis as Analysis

def main() -> int:

    # Create parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-s", "--start", help = "Start time, defaults to one day ago")
    parser.add_argument("-e", "--end", help = "End time, defaults to now")
    parser.add_argument("-d", "--duration", help = "Delta time in seconds, overrides --end")
    parser.add_argument("-p", "--plot", help = "Show plots", action='store_true')
    parser.add_argument("-o", "--output", help = "Set filename for output statistics")
    
    # Read arguments from command line
    args = parser.parse_args()
    
    # Default values
    start = datetime.utcnow() - timedelta(days=1)
    end = datetime.utcnow() 
    delta = 0.0


    if args.start:
        start = datetime.fromisoformat(args.start)

    if args.duration:
        delta = float(args.duration)
        end = start + timedelta(seconds=delta)
    else:
        if args.end:
            end = datetime.fromisoformat(args.end)

    if start >= end:
        print(f'Times reversed: start={start} >= end={end}')
        return 1

    range = TimeRange(start=start, end=end)

    # Or pick your times
    all = TimeRange(start=datetime(2023, 8, 3, tzinfo=timezone.utc), end=datetime.utcnow())
    pvrestart = TimeRange(start=datetime(2023, 9, 6, tzinfo=timezone.utc), end=datetime.utcnow())
    original = TimeRange(start=datetime(2023, 9, 1, tzinfo=timezone.utc), end=datetime(2023, 9, 6, tzinfo=timezone.utc))

    # And set the input here
    time_range = range

    print(f'Running analysis between {time_range.start} and {time_range.end}')
    intervals, interval_statistics = Analysis.runAnalysis(time_range)

    filename =  'fgs_star_tracking_interval_statistics.json'
    if args.output:
        filename = args.output

    # Write output JSON
    # Fgs.fgs_star_tracking_intervals_to_json_file(intervals, 'fgs_star_tracking_intervals.json')
    Fgs.fgs_star_tracking_interval_statistics_to_json_file(interval_statistics, filename)

    if args.plot:
        Analysis.create_plots(interval_statistics)

    return 0

if __name__ == '__main__':
    sys.exit(main())



