import logging
import luigi


logger = logging.getLogger('luigi-interface')

try:
    import _mssql
except ImportError as e:
    logger.warning("Loading MSSQL module without the python package pymssql. \
        This will crash at runtime if SQL Server functionality is used.")


class MSSqlQueryTask(luigi.Task):

    def __init__(self, host, database, user, password, query):
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
        if ':' in host:
            self.host, self.port = host.split(':')
            self.port = int(self.port)
        else:
            self.host = host
            self.port = 1433
        self.database = database
        self.user = user
        self.password = password
        self.query = query

    def connect(self):
        """
        Create a SQL Server connection and return a connection object
        """
        connection = _mssql.connect(user=self.user,
                                    password=self.password,
                                    server=self.host,
                                    port=self.port,
                                    database=self.database)

        return connection
