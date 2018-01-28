import argparse
from azure.datalake.store import core, lib, multithread


def download_data(az_tenant_id, az_sp_client_id, az_sp_client_secret, datalake_store_name):
    """
    Download data directory from Azure Data Lake store
    """
    token = lib.auth(tenant_id=az_tenant_id,
                client_id=az_sp_client_id,
                client_secret=az_sp_client_secret)
    adl = core.AzureDLFileSystem(token, store_name=datalake_store_name)
    multithread.ADLDownloader(adl, "", "./data", 4, 2**24)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--az-tenant-id', help='Azure Tenant Id', required=True, default=None)
    parser.add_argument('--az-sp-client-id', help='Azure Service Principal Client Id', required=True, default=None)
    parser.add_argument('--az-sp-client-secret', help='Azure Service Principal Client Secret', required=True, default=None)
    parser.add_argument('--datalake-store-name', help='Datalake store name', required=True, default=None)    
    args = vars(parser.parse_args())
    download_data(args['az_tenant_id'], args['az_sp_client_id'], args['az_sp_client_secret'], args['datalake_store_name'])
