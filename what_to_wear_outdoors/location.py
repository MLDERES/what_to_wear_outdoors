import re
from copy import deepcopy
import config
from what_to_wear_outdoors.utility import get_data_path
import pandas as pd


class Location:
    """
    Encapsulates a possible location for querying from Weather Underground
    """

    def __init__(self, location_name):
        """
        :param location_name:
        """
        self._lat = self._long = None
        self._zipcode = self._city = self._state = None
        # Deal with the case where we have gotten a location rather than a string
        if isinstance(location_name, Location):
            self.__dict__ = deepcopy(location_name.__dict__)
        elif isinstance(location_name, str):
            # Need to determine if this is City, ST - a zipcode or Lat/Long
            expression = r'(^\d{5}$)|(^[\w\s]+),\s*(\w{2}$)|(^[\w\s]+)'
            mo = re.match(expression, str(location_name))
            if mo and mo.group(2) is not None:
                # if we have matched City, State then we need to build the query as ST/City.json
                # otherwise we can just use City or Zip + .json
                self._city = str.strip(mo.group(2))
                self._state = str.strip(mo.group(3))
            else:
                self._zipcode = str.strip(location_name)
        else:
            self._zipcode = str(location_name)

    def repl(self, sep='/'):
        """
        Get the representation of this object using the appropriate separator for the city and state (if required)
        :param sep: The separator to use between the city and the state fields if they exist
        :return: either the zip code or the string represented by the state followed by the separator then the city
        """
        return sep.join([self._state, self._city]) if self._zipcode is None else self._zipcode

    @property
    def name(self):
        return ','.join([self._city, self._state]) if self._zipcode is None else self._zipcode

    @property
    def lat(self):
        if self._lat is None:
            self.get_latlong()
        return self._lat

    @property
    def long(self):
        if self._long is None:
            self.get_latlong()
        return self._long

    def get_latlong(self):
        loc_file = get_data_path(config.location_filename)
        if loc_file.exists():
            df = pd.read_csv(loc_file,index_col=0)
        else:
            # Somehow get the lat_long, save it to a file and return df
            df = pd.DataFrame(data={'location': ['72712', 'AR-Bentonville'], 'latitude': [36.37233, 36.37233],
                                    'longitude': [-94.20949, -94.20949]},
                              columns=['location', 'latitude', 'longitude'])
            df.set_index('location', drop=True, inplace=True)
            df.to_csv(loc_file)
        row = df.loc[self.repl('-')]
        self._lat = row.latitude
        self._long = row.longitude
        return (row.latitude, row.longitude)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '{0} ({1})'.format(object.__repr__(self), str(self))
