import os
import os.path

try:
    from ConfigParser import NoSectionError
except ImportError:
    from configparser import NoSectionError

import luigi
from luigi import configuration, six


class FileNotFoundException(luigi.target.FileSystemException):
    pass


class ADLClient(luigi.target.FileSystem):
    """
    Integration 
    """

    _adl = None

    def __init__(self, az_tenant_id=None, az_sp_client_id=None, az_sp_client_secret=None, **kwargs):
        """
        Initialize configuration for Azure Data Lake Client
        :param az_tenant_id: Active Directory TenantID
        :param az_sp_client_id: Azure Service Principal Id
        :param az_sp_client_secret: Azure Service Principal Secret Key
        :param kwargs: extra config options
        """
        options = self._get_adl_config()
        options.update(kwargs)
        if az_tenant_id:
            options['az_tenant_id'] = az_tenant_id
        if az_sp_client_id:
            options['az_sp_client_id'] = az_sp_client_id
        if az_sp_client_secret:
            options['az_sp_client_secret'] = az_sp_client_secret

        self._options = options

    @property
    def adl(self):
        """
        Create the Azure Data Lake client singleton with the provided configuration
        :return:
        """
        from azure.datalake.store import core, lib

        options = dict(self._options)

        if self._adl:
            return self._adl

        az_tenant_id = options.get('az_tenant_id')
        az_sp_client_id = options.get('az_sp_client_id')
        az_sp_client_secret = options.get('az_sp_client_secret')
        store_name = options.get('store_name')

        for key in ['az_tenant_id', 'az_sp_client_id', 'az_sp_client_secret']:
            if key in options:
                options.pop(key)

        token = lib.auth(tenant_id=az_tenant_id,
                 client_id=az_sp_client_id,
                 client_secret=az_sp_client_secret,
                 **options)
        self._adl = core.AzureDLFileSystem(token, store_name=store_name)
        return self._adl

    @adl.setter
    def adl(self, value):
        self._adl = value

    def exists(self, path):
        """
        Does the path exist
        :param path: path to file for luigi to check
        :return: boolean (file exists)
        """
        return self.adl.exists(path, invalidate_cache=True)

    def listdir(self, path):
        """
        List contents of directory provided by path
        :param path: directory to list
        :return: array of file/directory names at this path
        """
        return self.adl.ls(path, detail=True, invalidate_cache=True)

    def open(self, path, mode='rb', blocksize=2**25, delimiter=None):
        """
        Open a file for reading or writing in bytes mode
        :param path:
        :param mode:
        :param blocksize:
        :param delimiter:
        :return:
        """
        return self.adl.open(path, mode=mode, blocksize=blocksize, delimiter=delimiter)

    def put(self, source_path, destination_path, delimiter=None):
        """
        Upload a file from local path to Azure Data Lake store
        :param source_path: path to local file
        :param destination_path: path to destination file on ADL
        :param delimiter: file delimiter (e.g. ',' for csv)
        """
        self.adl.put(source_path, destination_path, delimiter=delimiter)

    def put_multipart(self, source_path, destination_path, thread_count=1,
                      overwrite=False, chunksize=2**28, buffersize=2**22,
                      blocksize=2**22, show_progress_bar=False):
        """
        Use multithread uploader with progress callback for very large files
        :param source_path: path to local file
        :param destination_path: path to destination file on ADL
        :param thread_count: threads to use
        :param overwrite: overwrite if file exists (defaults to False)
        :param chunksize: file chunksize
        :param buffersize:
        :param blocksize:
        :param show_progress_bar: Show a progress bar with Azure cli's controller
        """

        if show_progress_bar:
            try:
                import cli
            except:
                raise ImportError('Please install the azure cli pip package to show upload progress')

            from azure.cli.core.application import APPLICATION

            def _update_progress(current, total):
                hook = APPLICATION.get_progress_controller(det=True)
                hook.add(message='Alive', value=current, total_val=total)
                if total == current:
                    hook.end()
        else:
            def _update_progress(current, total):
                print('{}% complete'.format(round(current / total, 2)))

        from azure.datalake.store.multithread import ADLUploader

        ADLUploader(self.adl, destination_path, source_path, thread_count, overwrite=overwrite,
                    chunksize=chunksize,
                    buffersize = buffersize,
                    blocksize = blocksize,
                    progress_callback = _update_progress
        )

    def remove(self, path, recursive=True, skip_trash=True):
        return self.adl.rm(path, recursive=recursive)

    def _get_adl_config(self, key=None):
        defaults = dict(configuration.get_config().defaults())
        try:
            config = dict(configuration.get_config().items('adl'))
        except NoSectionError:
            return {}
        # So what ports etc can be read without us having to specify all dtypes
        for k, v in six.iteritems(config):
            try:
                config[k] = int(v)
            except ValueError:
                pass
        if key:
            return config.get(key)
        section_only = {k: v for k, v in config.items() if k not in defaults or v != defaults[k]}

        return section_only


class AtomicADLFile(luigi.target.AtomicLocalFile):
    """
    Writes to tmp file and puts to ADL on close
    """
    def __init__(self, path, client, **kwargs):
        self.client = client
        super(AtomicADLFile, self).__init__(path)
        self.options = kwargs

    def move_to_final_destination(self):
        self.client.put_multipart(self.tmp_path, self.path, **self.options)


class ADLTarget(luigi.target.FileSystemTarget):
    """
    Target Azure Data Lake file object
    """

    fs = None

    def __init__(self, path, format=None, client=None, **kwargs):
        super(ADLTarget, self).__init__(path)
        if format is None:
            format = luigi.format.get_default_format()

        # print('PATH IN ADLTarget CLASS: ', path)

        self.path = path
        self.format = format
        self.fs = client or ADLClient()
        self.adl_options = kwargs

    def open(self, mode='r'):
        if mode not in ('r', 'w'):
            raise ValueError("Unsupported open mode '%s'" % mode)

        if mode == 'r':
            if not self.fs.exists(self.path):
                raise FileNotFoundException("Could not find file at %s" % self.path)

            return self.fs.open(self.path)
        else:
            return self.format.pipe_writer(AtomicADLFile(self.path, self.fs, **self.adl_options))


class ADLFlagTarget(ADLTarget):
    """
    Defines a target directory with a flag-file (defaults to `_SUCCESS`) used
    to signify job success.
    This checks for two things:
    * the path exists (just like the ADLTarget)
    * the _SUCCESS file exists within the directory.
    """

    fs = None

    def __init__(self, path, format=None, client=None, flag='_SUCCESS'):
        """
        Initializes a ADLFlagTarget.
        :param path: the directory where the files are stored.
        :type path: str
        :param format: see the luigi.format module for options
        :type format: luigi.format.[Text|UTF8|Nop]
        :param client:
        :type client:
        :param flag:
        :type flag: str
        """
        if format is None:
            format = luigi.format.get_default_format()

        if path[-1] != "/":
            raise ValueError("ADLFlagTarget requires the path to be to a "
                             "directory.  It must end with a slash ( / ).")
        super(ADLFlagTarget, self).__init__(path + flag, format, client)
        self.flag = flag

    def exists(self):
        hadoopSemaphore = self.path + self.flag
        return self.fs.exists(hadoopSemaphore)


class ADLPathTask(luigi.ExternalTask):
    """
    An external task that requires existence of a path in Azure Data Lake store.
    """
    path = luigi.Parameter()

    def output(self):
        return ADLTarget(self.path)


class ADLFlagTask(luigi.Task):
    """
    An external task that requires the existence of 'Hadoop' like output
    _SUCCESS flag file in the directory specified by path
    """
    path = luigi.Parameter()
    flag = luigi.Parameter(default='_SUCCESS')

    def output(self):
        return ADLFlagTarget(self.path, flag=self.flag)
