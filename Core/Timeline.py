import os.path
from datetime import datetime, timezone, timedelta
import csv

from astropy.table import Table

import pandas as pd

from Core.Hms import Hms, ParamDefKeywords

from Core.Utilities import _ROOT

hms = Hms()

def generateTable(t_start: datetime, t_end: datetime, param_defs: list[dict], colName=ParamDefKeywords.description, force=False) -> Table:
    """Generate a Table from a set of ARES parameters sharing identical sets of timestamps, e.g. from the same TM packet"""

    mustInit = True
    timeStamps = None
    UTC = None
    timeSeries = {}

    # First populate the timestamps from ares
    for param_def in param_defs:
        generateCacheFile(t_start, t_end, param_def, force)
        times, utc, values = generateTimelineFromCacheFile(t_start, t_end, param_def)
        if mustInit:
            timeStamps = times
            UTC = utc

        timeSeries[param_def[colName]] = values

    columns = [timeStamps, UTC]
    columnNames = ['POSIX', 'UTC_STRING']

    columns.extend(timeSeries.values())
    columnNames.extend(timeSeries.keys())

    return Table(columns, names=columnNames)


def generateDataFrame(t_start: datetime, t_end: datetime, param_defs: list[dict], colName=ParamDefKeywords.description, force=False) -> pd.DataFrame:
    """Generate a Table from a set of ARES parameters sharing identical sets of timestamps, e.g. from the same TM packet"""

    mustInit = True
    timeStamps = None
    UTC = None
    timeSeries = {}

    # First populate the timestamps from ares
    for param_def in param_defs:
        generateCacheFile(t_start, t_end, param_def, force)
        times, utc, values = generateTimelineFromCacheFile(t_start, t_end, param_def)
        if mustInit:
            timeStamps = times
            UTC = utc

        timeSeries[param_def[colName]] = values

    d = dict()

    d['POSIX'] = timeStamps
    d['UTC_STRING'] = UTC
    for ts in timeSeries:
        d[ts] = timeSeries[ts]

    return pd.DataFrame(data=d)


#
#
#
#
#
def generateCacheFile(t_start: datetime, t_end: datetime, param_def: dict, force=False):

    n_s_tt = t_start.utctimetuple()
    n_e_tt = t_end.utctimetuple()

    normalized_start = datetime(n_s_tt.tm_year, n_s_tt.tm_mon, n_s_tt.tm_mday)
    normalized_end = datetime(n_e_tt.tm_year, n_e_tt.tm_mon, n_e_tt.tm_mday)
    
    delta = normalized_end - normalized_start

    ares_id = int(param_def[ParamDefKeywords.ares_id])
    calibrate = param_def[ParamDefKeywords.calibrate]

    for i in range(delta.days + 1):
        s_date = normalized_start + timedelta(days=i)
        e_date = normalized_start + timedelta(days=i+1)

        s_tt = s_date.utctimetuple()
        e_tt = e_date.utctimetuple()

        checkDir(year=s_tt.tm_year, doy=s_tt.tm_yday)
        filename = getFilename(ares_id, year=s_tt.tm_year, doy=s_tt.tm_yday)

        if force or not os.path.isfile(filename):

            jsondata = hms.request(ares_id, 
                            startyear=s_tt.tm_year, 
                            startdoy=s_tt.tm_yday, 
                            endyear=e_tt.tm_year, 
                            enddoy=e_tt.tm_yday, 
                            calibrate=calibrate)

            print('Writing ARES output to ' + filename)
            json_to_csv(jsondata, filename)

#
#
#
#
#
def generateTimelineFromCacheFile(t_start: datetime, t_end: datetime, param_def: dict) -> tuple[list, list, list]:

    n_s_tt = t_start.utctimetuple()
    n_e_tt = t_end.utctimetuple()

    normalized_start = datetime(n_s_tt.tm_year, n_s_tt.tm_mon, n_s_tt.tm_mday)
    normalized_end = datetime(n_e_tt.tm_year, n_e_tt.tm_mon, n_e_tt.tm_mday)

    posix_start = int(t_start.replace(tzinfo=timezone.utc).timestamp()) * 1000
    posix_end = int(t_end.replace(tzinfo=timezone.utc).timestamp()) * 1000
    
    delta = normalized_end - normalized_start

    times = list()
    utc = list()
    values = list()

    for i in range(delta.days + 1):
        s_date = normalized_start + timedelta(days=i)
        s_tt = s_date.utctimetuple()

        filename = getFilename(int(param_def[ParamDefKeywords.ares_id]), year=s_tt.tm_year, doy=s_tt.tm_yday)

        with open(filename, 'r', encoding='UTF8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                time = int(row[0])
                if time >= posix_start and time <= posix_end:
                    times.append(time)
                    utc.append(row[1])
                    if param_def[ParamDefKeywords.calibrate]:
                        values.append(row[3])
                    else:
                        values.append(row[2])
            
    return times, utc, values

def json_to_csv(jsondata: str, filename: str):
    with open(filename, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        for datum in jsondata[0]['data']:
            posix = int(datum['date'])
            utc = datetime.utcfromtimestamp(float(posix) / 1000.0).strftime('%Y-%m-%dT%H:%M:%S.%f')
            raw = datum['value']
            cal = datum['calibratedValue']
            d = [posix, utc, raw, cal]
            writer.writerow(d)

    file.close()

def getFilename(id: int, year: int, doy: int, extension: str='.csv'):
    s_year = str(year)
    s_doy = str(doy)
    s_id = str(id)
    return str(_ROOT / s_year / s_doy / s_id) + extension

def checkDir(year: int, doy: int):
    s_year = str(year)
    s_doy = str(doy)
    path = _ROOT / s_year / s_doy
    if not path.exists():
        os.makedirs(str(path), exist_ok=True)