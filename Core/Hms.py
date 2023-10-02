import requests
import datetime
import os
import json

urlprefix = 'http://euclid.esac.esa.int/webclient-ares/mustlink/dataproviders/EUCLID/parameters/data?'
urlsuffix = '&aggregationFunction=FIRST&aggregation=None&aggregationValue=1&compressionError=0&delta=0'

            # "&aggregationFunction=FIRST&aggregation=None&aggregationValue=1&compressionError=0&chunkCount=1749&delta=0"

from Utilities import _ROOT    
    
class ParamDefKeywords: 
    name: str = 'param'
    description: str = 'descr'
    ares_id: str = 'gid'
    calibrate: str = 'calibrate'

class Hms:

    """Key used for HMS authentication"""
    key: str = None

    lookup = {}

    @classmethod
    def _getKey(self) -> str:

        if self.key is not None:
            return self.key

        try: 
            header = {'Content-Type': 'application/json'}
            data = {"username": "NISP_1", "password": "hMBm8pt("}
            url = 'http://euclid.esac.esa.int/webclient-ares/mustlink/auth/login'
            response = requests.post(url, headers=header, json=data)
            jsonResponse = response.json()
            token = jsonResponse["token"]
            self.key = token
            return self.key
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)

    # This should be deleted
    @classmethod
    def request(self, id: int, startyear: int, startdoy: int, endyear: int, enddoy: int, calibrate: bool=True):

        key = self._getKey()

        start = datetime.datetime.strptime(str(startyear) + ' ' +  str(startdoy), '%Y %j')
        end = datetime.datetime.strptime(str(endyear) + ' ' +  str(enddoy), '%Y %j')

        header = {'Authorization': key}
        content = 'key=id&values=' + str(id) + "&from=" + str(start) + "&to=" + str(end)

        if calibrate is True:
            url = urlprefix + content + urlsuffix + '&calibrate=true'
        else:
            url = urlprefix + content + urlsuffix + '&calibrate=false'

        try: 
            response = requests.get(url, headers=header)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)


    @classmethod
    def _loadParameterDefs(self):

        paramdef_filename = str(_ROOT / 'paramdefs.json')

        if not os.path.isfile(paramdef_filename):
            print('Querying parameter definitions from' + paramdef_filename)
            self._queryParameterDefs(self)
        else:
            print('Reading parameter definitions from ' + paramdef_filename)
            with open(paramdef_filename, "r") as outfile:
                self.lookup = json.load(outfile)


    def _queryParameterDefs(self):
        key = self._getKey()

        paramdef_filename = str(_ROOT / 'paramdefs.json')

        header = {'Authorization': key}
        url = 'http://euclid.esac.esa.int/webclient-ares/mustlink/web/metadata/tree?ds=EUCLID-tmparams&id=EUCLID-tmparams'

        try: 
            response = requests.get(url, headers=header)
            response.raise_for_status()
            partitions = response.json()

            for partition in partitions:
                url = 'http://euclid.esac.esa.int/webclient-ares/mustlink/web/metadata/tree?ds=EUCLID-tmparams_AAAT&id=EUCLID-tmparams_' + partition['data']['title']
                response = requests.get(url, headers=header)
                response.raise_for_status()
                params = response.json()
                for param in params:
                    title = param['data']['title']
                    if len(title) > 10:
                        if title[-10]=='(' and title[-1] == ')':
                            desc = title[0:-11]
                            id = title[-9:-1]
                            gid = param['attr']['id'].split('-')[1]
                            self.lookup[id] = { 
                                ParamDefKeywords.description: desc, 
                                ParamDefKeywords.name: id, 
                                ParamDefKeywords.ares_id: gid, 
                                ParamDefKeywords.calibrate: False
                                }

            json_formatted_str = json.dumps(self.lookup, indent=2)

            with open(paramdef_filename, "w") as outfile:
                    print('Writing ParamDef output to ' + paramdef_filename)
                    outfile.write(json_formatted_str)

        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
        
    @classmethod 
    def getParameterDef(self, id: str, calibrate: bool) -> dict:

        if len(self.lookup) == 0:
            self._loadParameterDefs()

        if id in self.lookup:
            p = self.lookup[id]
            p[ParamDefKeywords.calibrate] = calibrate
            return p
        else:
            return None

