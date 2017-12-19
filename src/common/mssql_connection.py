import logging
import luigi
import pandas as pd

logger = logging.getLogger('luigi-interface')

import pymssql

try:
    from ConfigParser import NoSectionError
except ImportError:
    from configparser import NoSectionError


class MSSqlConnection():

    def __init__(self, host=None, database=None, user=None, password=None):
        """
        Initializes a MsSqlTarget instance.
        :param host: MsSql server address. Possibly a host:port string.
        :type host: str
        :param database: database name.
        :type database: str
        :param user: database user
        :type user: str
        :param password: password for specified user.
        :type password: str
        """
        config = self._get_mssql_config()
        
        if host:
            config['host'] = host
        if database:
            config['database'] = database
        if user:
            config['user'] = user
        if password:
            config['password'] = password

        self._config = config

    def connect(self):
        """
        Create a SQL Server connection and return a connection object
        """
        connection = pymssql.connect(user=self._config.get('user'),
                                    password=self._config.get('password'),
                                    host=self._config.get('host'),
                                    database=self._config.get('database'))

        return connection

    def run_query(self, query):
        """
        Run a query on the MSSQL connection
        """
        return pd.read_sql(query, self.connect())

    def _get_mssql_config(self):
        """
        Get mssql config
        """        
        try:
            return dict(luigi.configuration.get_config().items('mssql'))
        except NoSectionError:
            return {}
