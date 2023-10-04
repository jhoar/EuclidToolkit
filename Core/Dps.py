import requests
import pandas as pd

# Data Model Constants
PRJ_EUCLID = 'TEST'

# Data.ImgNumber=144
# Data.InstrumentMode=Science
# Data.ObservationMode=WIDE
# Data.PlanningId=12296

# VIS
# Data.VisSequence=Nominal/Short

# NISP
# Data.ExposureConfiguration.KeywordConfiguration=HomeCheckFilterWheel
# Data.FilterWheelPos=J
# Data.GrismWheelPos=OPEN
# Data.GrismWheelTilt=0

# Data.ObservationSeqeuence.CalblockId=OTHER
# Data.ObservationSeqeuence.CalblockVariant=PV-001
# Data.ObservationSeqeuence.DitherObservation=3
# Data.ObservationSeqeuence.Exposure=3
# Data.ObservationSeqeuence.FieldId=0
# Data.ObservationSeqeuence.IslandId=-100000
# Data.ObservationSeqeuence.ObservationId=65536
# Data.ObservationSeqeuence.PatchId=-100000
# Data.ObservationSeqeuence.PointingId=262331
# Data.ObservationSeqeuence.TotalExposure=5

# Data.CommandedAttitude.Alpha=0.0
# Data.CommandedAttitude.Beta=0.0
# Data.CommandedAttitude.Frame=None
# Data.CommandedAttitude.Latitude=0.0
# Data.CommandedAttitude.Longitude=0.0
# Data.CommandedAttitude.PositionAngle=0.0
# Data.CommandedAttitude.SolarAspectAngle=0.0

# Data.ObservationDateTime.MJD=60158.2760520685
# Data.ObservationDateTime.UTC=2023-08-02 06:37:30.898718
# Data.ExposureTime.Value=560.52




# DPD Types
CL_LE1_VIS_FRAME = 'DpdVisRawFrame'
CL_LE1_NISP_FRAME = 'DpdNispRawFrame'

CL_LE1_HKTM = 'DpdHKTMProduct'
CL_LE1_QLAREP = 'DpdQlaReport'
CL_LE1_RAWHK = 'DpdRawHKTM'
CL_LE1_RAWSCI = 'DpdRawScience'

# Metadata fields
# Headers
MD_PROD_SDC = 'Header.ProdSDC'
MD_CREATION_DATE = 'Header.CreationDate'
MD_DATASET = 'Header.DataSetRelease'

# Level 1
MD_LE1_OBSMODE = 'Data.ObservationMode'
MD_LE1_PLANNINGID = 'Data.PlanningId'
MD_LE1_VIS_DATAFILE = 'Data.FrameFitsFile.DataContainer.FileName'
MD_LE1_NISP_DATAFILE = 'Data.NispFrameFitsFile.DataContainer.FileName'
MD_LE1_CALBLOCK = 'Data.ObservationSequence.CalblockId'

# HK Prod
MD_HKPROD_FROMDATE = 'Data.DateTimeRange.FromDate'
MD_HKPROD_TODATE = 'Data.DateTimeRange.ToDate'
MD_HKPROD_FILENAMES = 'Data.HKTMContainer.File.FileName'

# RAW HK
MD_RAWHK_FILENAME = 'Data.File.FileName'

# RAW Science
MD_RAWSCI_FILENAME = 'Data.File.FileName'

# QLA
MD_QLAREP_FILENAME = 'Data.File.FileName'

# Filters
FL_VALID_DATA = 'Header.ManualValidationStatus!=INVALID'

#
# Query functions
#
_api_base_url = "https://eas-dps-rest-ops.esac.esa.int/REST"

# Query the DPS and return the results in a pandas dataframe
def queryToFrame(product, criteria, output, LDAP_USER, LDAP_PASS):
    url = generateUrl(product=product, criteria=criteria, output=output)
    response = request(url, LDAP_USER, LDAP_PASS)

    df = pd.DataFrame(columns=output)

    skipFirst = True
    for lines in response.splitlines():

        # Skip the DPS header line
        if skipFirst:
            skipFirst = False
            continue

        df.loc[len(df.index)] = lines.split(',')

    return df

# Query the DPS and return the results in a CSV file
def queryToFile(outFile, product, criteria, output, LDAP_USER, LDAP_PASS):
    url = generateUrl(product=product, criteria=criteria, output=output)
    response = request(url, LDAP_USER, LDAP_PASS)

    # Build pandas compatible header
    header = ','.join(output)

    with open(outFile,'w') as dps_file:
        skipFirst = True
        for lines in response.splitlines():

            # Replace the DPS header (with '.'s) with a pandas compatible one
            if skipFirst:
                dps_file.write(header)
                dps_file.write('\n')
                skipFirst = False
                continue

            dps_file.write(lines)
            dps_file.write('\n')

        dps_file.close()    

# Build a query URL
def generateUrl(product, output=['Header.ProductType', 'Header.ProductId.LimitedString'], criteria=[]):
    url = """{base_url}?{class_name}&{allow_array}&{valid_data}&{project}&{criteria_text}&{output_fields}""".format(**{
                        'base_url': _api_base_url,
                        'class_name': 'class_name=' + product,
                        'valid_data': FL_VALID_DATA, 
                        'project': 'project=' + PRJ_EUCLID, 
                        'allow_array': 'allow_array=' + str(True),
                        'criteria_text': '&'.join(criteria),
                        'output_fields': 'fields=' + ':'.join(output)})
    return url

# Execute a request and return the response
def request(url, username, password):
    try: 
        response = requests.get(url, auth=(username,password))
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

# Convert a python DateTime to the dt() format needed for a DPS query through the REST interface
def getEASTime(time):
    tt = time.timetuple()
    return """dt({year},{month},{day},{hour},{minute},{second})""".format(**{
                        'year': tt.tm_year,
                        'month': tt.tm_mon,
                        'day': tt.tm_mday, 
                        'hour': tt.tm_hour, 
                        'minute': tt.tm_min,
                        'second': tt.tm_sec})

# Generate a DF query compatible column name
def toPandas(field):
    return '`' + field + '`' 


